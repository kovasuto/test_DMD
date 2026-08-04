"""
EnSight Gold を読み込み、指定ROI・指定変数だけを抽出して
HDF5にストリーミング書き出しするモジュール。

全タイムステップを同時にメモリへ載せないことがポイント。
1ステップ読み込み -> 変数抽出 -> HDF5へ追記 -> メモリ解放、を繰り返す。

読み込み・変数抽出(CPU/パース処理が重い部分)は ProcessPoolExecutor で
並列化できる(config.ensight_parallel_workers)。
HDF5への書き込みは並列化せず、メインプロセスに集約して順番通りに行う
(h5pyの単一ファイルへの並列書き込みは基本的に安全ではないため)。
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h5py
import pyvista as pv

from config import Config


def _extract_field(mesh: "pv.DataSet", cfg: Config) -> np.ndarray:
    """1つのメッシュ(1タイムステップ分)から対象変数の1次元配列を取り出す"""

    if cfg.variable_name in mesh.point_data:
        data = mesh.point_data[cfg.variable_name]
    elif cfg.variable_name in mesh.cell_data:
        data = mesh.cell_data[cfg.variable_name]
    else:
        raise KeyError(
            f"変数 '{cfg.variable_name}' がメッシュ内に見つかりません。"
            f"利用可能な変数: point={list(mesh.point_data.keys())}, "
            f"cell={list(mesh.cell_data.keys())}"
        )

    data = np.asarray(data)

    if data.ndim == 2:
        # ベクトル量 (N, 3) の場合は成分を選ぶ
        if cfg.vector_component is None:
            raise ValueError(
                f"'{cfg.variable_name}' はベクトル量です。"
                f"config.vector_component (0=x,1=y,2=z) を指定してください。"
            )
        data = data[:, cfg.vector_component]

    return data.astype(np.float32).ravel()


def _apply_roi(mesh: "pv.DataSet", cfg: Config) -> "pv.DataSet":
    """ROIバウンディングボックスでメッシュを絞り込む(指定があれば)"""

    if cfg.roi_bounds is None:
        return mesh
    return mesh.clip_box(cfg.roi_bounds, invert=False)


def _select_part(reader: "pv.EnSightReader", cfg: Config) -> None:
    """複数パートがある場合、対象パートだけを有効化する"""

    if cfg.target_part_name is None:
        return

    try:
        reader.disable_all_element_arrays()  # 存在すれば
    except AttributeError:
        pass

    if hasattr(reader, "part_names"):
        for name in reader.part_names:
            active = name == cfg.target_part_name
            try:
                reader.set_active_part(name, active)
            except AttributeError:
                pass


def _read_and_extract(reader: "pv.EnSightReader", time_value: float, cfg: Config):
    """1タイムステップ分を読み込み、ROI抽出・変数抽出まで行う共通処理"""

    reader.set_active_time_value(time_value)
    mesh = reader.read()

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    mesh = _apply_roi(mesh, cfg)
    coords = np.asarray(mesh.points, dtype=np.float32)
    field = _extract_field(mesh, cfg)

    return coords, field


# ----------------------------------------------------------------------
# ProcessPoolExecutor用のワーカー: プロセスごとに1回だけEnSightリーダーを
# 開き(initializerで実行)、以降のタスクではそのリーダーを使い回す。
# タスクごとにリーダーを開き直すと、ケースファイルの再パースが毎回発生し
# 並列化のメリットが薄れるため。
# ----------------------------------------------------------------------

_worker_reader = None
_worker_cfg = None


def _init_worker(case_path: str, cfg: Config):
    global _worker_reader, _worker_cfg
    _worker_reader = pv.get_reader(case_path)
    _select_part(_worker_reader, cfg)
    _worker_cfg = cfg


def _worker_task(time_value: float):
    global _worker_reader, _worker_cfg
    coords, field = _read_and_extract(_worker_reader, time_value, _worker_cfg)
    return coords, field


def stream_ensight_to_hdf5(cfg: Config) -> dict:
    """
    EnSight Goldをストリーミング読み込みし、間引き・ROI抽出・変数抽出を行った上で
    HDF5にスナップショット行列 (n_points, n_snapshots) を逐次追記する。

    cfg.ensight_parallel_workers が1より大きい場合、読み込み・変数抽出を
    ProcessPoolExecutorで並列実行する(HDF5書き込みはメインプロセスに集約)。

    Returns
    -------
    meta : dict
        coords (n_points, 3), dt_actual, n_snapshots, times などのメタ情報
    """

    os.makedirs(cfg.work_dir, exist_ok=True)
    h5_path = os.path.join(cfg.work_dir, cfg.snapshot_hdf5_name)

    reader = pv.get_reader(cfg.ensight_case_path)
    _select_part(reader, cfg)

    all_times = np.asarray(reader.time_values)
    if all_times.size == 0:
        raise RuntimeError("EnSightケースにタイムステップが見つかりません。")

    # --- 間引き(decimation)係数の計算 -------------------------------
    stride = max(1, round((1.0 / cfg.dt_export) / cfg.fs_target))
    dt_actual = cfg.dt_export * stride
    fs_actual = 1.0 / dt_actual

    # --- 時間窓の切り出し(助走区間を除外) ----------------------------
    t_end = cfg.t_start + cfg.time_window
    idx_all = np.arange(all_times.size)
    mask = (all_times >= cfg.t_start) & (all_times <= t_end)
    idx_selected = idx_all[mask][::stride]

    if idx_selected.size < 2:
        raise RuntimeError(
            "選択されたタイムステップ数が不足しています。"
            "dt_export / fs_target / time_window / t_start を見直してください。"
        )

    print(f"[INFO] 全タイムステップ数         : {all_times.size}")
    print(f"[INFO] 間引き係数 stride           : {stride}")
    print(f"[INFO] 実効サンプリング周波数 fs   : {fs_actual:.3f} Hz")
    print(f"[INFO] 実効サンプリング時間刻み dt : {dt_actual:.6f} s")
    print(f"[INFO] 使用スナップショット数 N    : {idx_selected.size}")
    print(f"[INFO] 実効時間窓 Tw               : {idx_selected.size * dt_actual:.3f} s")

    n_workers = cfg.ensight_parallel_workers
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, idx_selected.size))

    time_values = [float(all_times[idx]) for idx in idx_selected]

    coords = None
    n_points = None
    dset = None

    time_read_total = 0.0
    time_write_total = 0.0
    t_overall_start = time.time()

    with h5py.File(h5_path, "w") as h5f:

        def _write_result(out_i: int, coords_i: np.ndarray, field: np.ndarray):
            nonlocal coords, n_points, dset

            t_b = time.time()

            if coords is None:
                coords = coords_i
                n_points = coords.shape[0]
                dset = h5f.create_dataset(
                    "X",
                    shape=(n_points, idx_selected.size),
                    dtype="float32",
                    chunks=(min(n_points, 100_000), 1),
                )
                h5f.create_dataset("coords", data=coords)
                h5f.create_dataset("times", shape=(idx_selected.size,), dtype="float64")

            if field.shape[0] != n_points:
                raise RuntimeError(
                    f"タイムステップ間で点数が変化しています "
                    f"({field.shape[0]} != {n_points})。移動格子や適応格子の場合は"
                    f"別途、固定格子への補間処理を追加してください。"
                )

            dset[:, out_i] = field
            h5f["times"][out_i] = time_values[out_i]

            return time.time() - t_b

        if n_workers <= 1:
            # ---- 逐次処理 (従来通り) ----
            print("[INFO] EnSight読み込み: 逐次処理 (ensight_parallel_workers<=1)")
            for out_i, tval in enumerate(time_values):
                t_a = time.time()
                coords_i, field = _read_and_extract(reader, tval, cfg)
                t_read = time.time() - t_a
                time_read_total += t_read

                time_write_total += _write_result(out_i, coords_i, field)

                del field
                if out_i % 10 == 0 or out_i == idx_selected.size - 1:
                    n_done = out_i + 1
                    print(
                        f"  ... {n_done}/{idx_selected.size} 処理済み  "
                        f"(平均: 読込+抽出 {time_read_total/n_done*1000:.1f} ms/件, "
                        f"HDF5書込 {time_write_total/n_done*1000:.1f} ms/件)"
                    )
        else:
            # ---- 並列処理 ----
            print(
                f"[INFO] EnSight読み込み: 並列処理 ({n_workers} プロセス)"
            )
            t_read_start = time.time()

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(cfg.ensight_case_path, cfg),
            ) as executor:
                # executor.map は入力順を保持して結果を返すため、
                # HDF5への書き込み順(時系列順)がそのまま保たれる。
                for out_i, (coords_i, field) in enumerate(
                    executor.map(_worker_task, time_values, chunksize=1)
                ):
                    time_write_total += _write_result(out_i, coords_i, field)

                    del field
                    if out_i % 10 == 0 or out_i == idx_selected.size - 1:
                        n_done = out_i + 1
                        elapsed = time.time() - t_read_start
                        print(
                            f"  ... {n_done}/{idx_selected.size} 処理済み  "
                            f"(経過 {elapsed:.1f} s, "
                            f"平均 {elapsed/n_done*1000:.1f} ms/件[並列合計])"
                        )

            time_read_total = time.time() - t_read_start - time_write_total

        h5f.attrs["dt"] = dt_actual
        h5f.attrs["fs"] = fs_actual
        h5f.attrs["n_snapshots"] = idx_selected.size
        h5f.attrs["n_points"] = n_points

    total_elapsed = time.time() - t_overall_start
    print(f"[INFO] HDF5書き出し完了: {h5_path}")
    print(
        f"[INFO] EnSight読込+抽出 合計(概算): {time_read_total:.1f} s / "
        f"HDF5書込 合計: {time_write_total:.1f} s / "
        f"Step1全体: {total_elapsed:.1f} s (workers={n_workers})"
    )

    return {
        "h5_path": h5_path,
        "coords": coords,
        "dt": dt_actual,
        "fs": fs_actual,
        "n_snapshots": int(idx_selected.size),
        "n_points": int(n_points),
        "time_read_total": time_read_total,
        "time_write_total": time_write_total,
    }


# ----------------------------------------------------------------------
# メッシュトポロジー(座標+セル接続情報)の参照ファイル
# ----------------------------------------------------------------------
#
# これまでのVTK出力(export_modes.py)は、座標だけのPolyData点群として
# 保存していたため、セル(面/要素)のつながりが失われ、ParaView等では
# 「点」しか表示できなかった。
#
# CFDの解析対象メッシュは時間によって変化しない(移動格子でない)ことを
# 前提としているため(stream_ensight_to_hdf5内で点数不一致をエラーにして
# いる箇所と同じ前提)、メッシュの接続情報は1回だけ読み込んで保存しておけば
# 十分である。ここで保存する参照メッシュ(mesh_topology.vtu)を、
# 後段のexport_modes.py・reconstruct.pyが読み込み、モード値・再構築値を
# 属性として載せることで、セル情報を保持したVTKを出力できるようになる。


def get_topology_path(cfg: Config) -> str:
    """
    参照メッシュ(トポロジー)ファイルのパスを返す(I/Oは行わない)。
    ファイルが存在するかどうかに関わらず、常に同じパスを返す。
    """
    return os.path.join(cfg.work_dir, cfg.mesh_topology_name)


def ensure_reference_topology(cfg: Config) -> str:
    """
    参照メッシュ(座標+セル接続情報、変数値は持たない)をUnstructuredGrid
    形式(.vtu)で保存する。既に保存済みならEnSightへの再アクセスは行わない。

    PolyData(表面メッシュ)の場合もUnstructuredGridにキャストして統一
    フォーマットで保存する(export_modes.py・reconstruct.pyが拡張子を
    意識せず一貫して扱えるようにするため)。

    Returns
    -------
    topology_path : str
    """

    os.makedirs(cfg.work_dir, exist_ok=True)
    topology_path = get_topology_path(cfg)

    if os.path.exists(topology_path):
        print(f"[INFO] 参照メッシュ(トポロジー)は既に存在します: {topology_path}")
        return topology_path

    print("[INFO] 参照メッシュ(トポロジー)を作成します(EnSightから1回だけ読み込み)...")

    reader = pv.get_reader(cfg.ensight_case_path)
    _select_part(reader, cfg)

    all_times = np.asarray(reader.time_values)
    if all_times.size == 0:
        raise RuntimeError("EnSightケースにタイムステップが見つかりません。")

    # メッシュは時間不変という前提のため、どのタイムステップを使ってもよい。
    # 最初のタイムステップを使う。
    reader.set_active_time_value(float(all_times[0]))
    mesh = reader.read()

    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    mesh = _apply_roi(mesh, cfg)

    # 変数データ(圧力・渦度等)は不要なので取り除き、幾何・接続情報だけ残す。
    topo = mesh.copy()
    for key in list(topo.point_data.keys()):
        del topo.point_data[key]
    for key in list(topo.cell_data.keys()):
        del topo.cell_data[key]

    # PolyData(表面メッシュ)の場合はUnstructuredGridにキャストし、
    # 拡張子・フォーマットを.vtuに統一する。
    if isinstance(topo, pv.PolyData):
        topo = topo.cast_to_unstructured_grid()

    topo.save(topology_path)
    print(
        f"[INFO] 参照メッシュを保存しました: {topology_path} "
        f"(点数={topo.n_points}, セル数={topo.n_cells})"
    )

    return topology_path
