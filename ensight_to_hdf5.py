"""
EnSight Gold を1ステップずつ読み込み、指定ROI・指定変数だけを抽出して
HDF5にストリーミング書き出しするモジュール。

全タイムステップを同時にメモリへ載せないことがポイント。
1ステップ読み込み -> 変数抽出 -> HDF5へ追記 -> メモリ解放、を繰り返す。
"""

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

    # pyvistaのEnSightReaderはブロック(パート)を選択的に有効化できる
    all_names = reader.point_arrays  # 参考: 属性APIはバージョンで差異があるため
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


def stream_ensight_to_hdf5(cfg: Config) -> dict:
    """
    EnSight Goldをストリーミング読み込みし、間引き・ROI抽出・変数抽出を行った上で
    HDF5にスナップショット行列 (n_points, n_snapshots) を逐次追記する。

    Returns
    -------
    meta : dict
        coords (n_points, 3), dt_actual, n_snapshots, times などのメタ情報
    """

    import os
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

    coords = None
    n_points = None

    with h5py.File(h5_path, "w") as h5f:
        dset = None

        for out_i, idx in enumerate(idx_selected):
            mesh = reader.set_active_time_value(all_times[idx])
            mesh = reader.read()

            # 複数パートがマルチブロックの場合は結合する
            if isinstance(mesh, pv.MultiBlock):
                mesh = mesh.combine()

            mesh = _apply_roi(mesh, cfg)

            if coords is None:
                coords = np.asarray(mesh.points, dtype=np.float32)
                n_points = coords.shape[0]
                dset = h5f.create_dataset(
                    "X",
                    shape=(n_points, idx_selected.size),
                    dtype="float32",
                    chunks=(min(n_points, 100_000), 1),
                )
                h5f.create_dataset("coords", data=coords)
                h5f.create_dataset("times", shape=(idx_selected.size,), dtype="float64")

            field = _extract_field(mesh, cfg)

            if field.shape[0] != n_points:
                raise RuntimeError(
                    f"タイムステップ間で点数が変化しています "
                    f"({field.shape[0]} != {n_points})。移動格子や適応格子の場合は"
                    f"別途、固定格子への補間処理を追加してください。"
                )

            dset[:, out_i] = field
            h5f["times"][out_i] = all_times[idx]

            # メモリ解放を明示
            del mesh, field

            if out_i % 100 == 0:
                print(f"  ... {out_i + 1}/{idx_selected.size} スナップショット処理済み")

        h5f.attrs["dt"] = dt_actual
        h5f.attrs["fs"] = fs_actual
        h5f.attrs["n_snapshots"] = idx_selected.size
        h5f.attrs["n_points"] = n_points

    print(f"[INFO] HDF5書き出し完了: {h5_path}")

    return {
        "h5_path": h5_path,
        "coords": coords,
        "dt": dt_actual,
        "fs": fs_actual,
        "n_snapshots": int(idx_selected.size),
        "n_points": int(n_points),
    }
