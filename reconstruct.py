"""
DMDモードから時系列方向のフィールドを再構築するモジュール。

x(t) ≈ mean_field + Re{ sum_k Phi_k * b_k * exp(omega_k * t) }

を、config.reconstruction_n_modes で指定した数のモード(select_modesで選ばれた
重要度上位のもの)だけを使って計算する。

物理空間の再構築結果 (m × n_times) は m が大きいと再びメモリを圧迫しうるため、
時間方向にチャンク分割しながらHDF5へ逐次書き出す(全時刻を同時に保持しない)。

再構築する時間範囲は、学習に使った時間窓と同じにすることも、
それより先(未来)まで延長して外挿・予測に使うこともできる
(config.reconstruction_extrapolate_duration で指定)。
"""

import os
import numpy as np
import h5py
import pyvista as pv

from config import Config


def _build_time_array(cfg: Config, dt_training: float, n_snapshots_training: int) -> np.ndarray:
    """再構築対象の時刻配列を作る"""

    if cfg.reconstruction_dt is not None:
        dt_out = cfg.reconstruction_dt
    else:
        dt_out = dt_training

    t_start = cfg.reconstruction_t_start

    if cfg.reconstruction_extrapolate_duration is not None:
        duration = cfg.reconstruction_extrapolate_duration
    else:
        duration = dt_training * n_snapshots_training

    n_out = int(round(duration / dt_out)) + 1
    times = t_start + np.arange(n_out) * dt_out
    return times


def reconstruct_timeseries(
    result: dict, selection: dict, cfg: Config
) -> dict:
    """
    選択されたモードを使って時系列を再構築し、HDF5に逐次書き出す。

    Returns
    -------
    meta : dict
        out_path, times, n_modes_used などのメタ情報
    """

    coords = result["coords"]
    dt_training = result["dt"]
    n_snapshots_training = result["n_snapshots"]
    mean_field = result["mean_field"]  # (m,) または None

    idx = selection["idx_selected"]
    b_full = selection["b_full"]

    Phi_selected = result["modes"][:, idx]                    # (m, k)
    b_selected = b_full[idx]                                   # (k,)
    growth_selected = result["growth_rate"][idx]
    freq_selected = result["freq_hz"][idx]
    omega_selected = growth_selected + 1j * 2.0 * np.pi * freq_selected  # (k,)

    times = _build_time_array(cfg, dt_training, n_snapshots_training)
    n_times = times.size
    n_points = coords.shape[0]
    k = idx.size

    print(
        f"[INFO] 時系列再構築: モード数={k}, 出力タイムステップ数={n_times}, "
        f"時間範囲=[{times[0]:.4f}, {times[-1]:.4f}] s"
    )

    os.makedirs(cfg.work_dir, exist_ok=True)
    out_path = os.path.join(cfg.work_dir, cfg.reconstruction_output_name)

    chunk = max(1, cfg.reconstruction_chunk_snapshots)

    with h5py.File(out_path, "w") as h5f:
        dset = h5f.create_dataset(
            "X_recon",
            shape=(n_points, n_times),
            dtype="float32",
            chunks=(min(n_points, 100_000), min(chunk, n_times)),
        )
        h5f.create_dataset("coords", data=coords)
        h5f.create_dataset("times", data=times)
        h5f.attrs["n_modes_used"] = k
        h5f.attrs["method"] = selection["method"]

        for t_start_i in range(0, n_times, chunk):
            t_end_i = min(t_start_i + chunk, n_times)
            t_block = times[t_start_i:t_end_i]                 # (c,)

            # exp(omega_k * t) for this block: (k, c)
            phase = np.exp(np.outer(omega_selected, t_block))

            # (m, k) @ (k, c) -> (m, c)
            field_block = Phi_selected @ (b_selected[:, None] * phase)
            field_block = field_block.real.astype(np.float32)

            if mean_field is not None:
                field_block = field_block + mean_field[:, None].astype(np.float32)

            dset[:, t_start_i:t_end_i] = field_block

            print(f"  ... {t_end_i}/{n_times} タイムステップ再構築済み")

    print(f"[INFO] 再構築結果を書き出しました: {out_path}")

    return {
        "out_path": out_path,
        "times": times,
        "n_modes_used": k,
    }


def compute_reconstruction_error(
    result: dict, selection: dict, h5_snapshot_path: str, cfg: Config
) -> float:
    """
    (オプション) 再構築結果を、元のスナップショットHDF5(訓練データ範囲)と比較し、
    全体相対誤差を計算する。訓練データ範囲を超える外挿区間は比較対象外。

    元データへの再アクセスが必要なため、dask経由でチャンク処理する
    (m×n全体を一括ロードしない)。
    """

    import dask.array as da

    dt_training = result["dt"]
    n_snapshots_training = result["n_snapshots"]

    if (
        cfg.reconstruction_extrapolate_duration is not None
        and cfg.reconstruction_extrapolate_duration
        > dt_training * n_snapshots_training * 1.001
    ):
        print(
            "[INFO] 外挿区間を含む再構築のため、"
            "誤差評価は訓練データ範囲内のみで行います。"
        )

    with h5py.File(h5_snapshot_path, "r") as h5f_orig:
        X_orig = da.from_array(h5f_orig["X"], chunks=h5f_orig["X"].chunks)

        with h5py.File(
            os.path.join(cfg.work_dir, cfg.reconstruction_output_name), "r"
        ) as h5f_recon:
            n_compare = min(
                X_orig.shape[1], h5f_recon["X_recon"].shape[1], n_snapshots_training
            )
            X_recon = da.from_array(
                h5f_recon["X_recon"], chunks=h5f_recon["X_recon"].chunks
            )[:, :n_compare]
            X_orig_trim = X_orig[:, :n_compare]

            num = da.sqrt(da.sum((X_orig_trim - X_recon) ** 2))
            den = da.sqrt(da.sum(X_orig_trim ** 2))
            rel_error = (num / den).compute()

    print(f"[INFO] 再構築の全体相対誤差 (訓練範囲内): {rel_error * 100:.3f} %")
    return float(rel_error)


# ============================================================================
# VTK時系列出力(セル接続情報付き、ParaViewでアニメーション再生可能)
# ============================================================================

def _select_frame_indices(n_times: int, cfg: Config) -> np.ndarray:
    """
    VTK出力するフレームのインデックスを、stride間引き・上限フレーム数の
    両方を考慮して決定する。
    """

    idx = np.arange(0, n_times, max(1, cfg.vtk_export_stride))

    max_frames = cfg.vtk_export_max_frames
    if max_frames is not None and idx.size > max_frames:
        # 上限を超える場合は、時間方向に均等になるようさらに間引く
        pick = np.linspace(0, idx.size - 1, max_frames).round().astype(int)
        idx = idx[pick]

    return idx


def _write_pvd_collection(pvd_path: str, frame_files: list, frame_times: list):
    """
    ParaViewのCollection(.pvd)ファイルを書き出す。
    これを開くと、複数の.vtuファイルが時間発展するアニメーションとして
    一度に読み込まれる。ファイルパスは.pvdと同じディレクトリからの
    相対パスで記述する(フォルダごと移動しても壊れないようにするため)。
    """

    lines = ['<?xml version="1.0"?>']
    lines.append('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">')
    lines.append("  <Collection>")
    for fname, t in zip(frame_files, frame_times):
        lines.append(f'    <DataSet timestep="{t:.8f}" part="0" file="{fname}"/>')
    lines.append("  </Collection>")
    lines.append("</VTKFile>")

    with open(pvd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_reconstruction_to_vtk_series(
    result: dict, selection: dict, cfg: Config, topology_path: str
) -> dict:
    """
    reconstruct_timeseries() が既にHDF5(reconstruction.h5)へ書き出した
    再構築結果を読み込み直し、参照メッシュ(セル接続情報付き)に値を載せて
    フレームごとの.vtuファイル + .pvd時系列コレクションとして出力する。

    HDF5から必要なフレームの列だけを読み込むため、再構築結果の全時刻を
    同時にメモリへ展開することはない。

    Parameters
    ----------
    topology_path : str
        ensight_to_hdf5.ensure_reference_topology() が保存した参照メッシュのパス。

    Returns
    -------
    meta : dict
        pvd_path, n_frames_written などのメタ情報
    """

    recon_path = os.path.join(cfg.work_dir, cfg.reconstruction_output_name)

    out_dir = os.path.join(cfg.work_dir, cfg.reconstruction_vtk_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    base_mesh = pv.read(topology_path)
    label = cfg.variable_name

    with h5py.File(recon_path, "r") as h5f:
        dset = h5f["X_recon"]
        times = h5f["times"][:]
        n_points, n_times = dset.shape

        if n_points != base_mesh.n_points:
            raise RuntimeError(
                f"再構築結果の点数({n_points})と参照メッシュの点数"
                f"({base_mesh.n_points})が一致しません。ROI設定や参照メッシュの"
                f"作り直しが必要な可能性があります(dmd_work内のmesh_topology.vtu"
                f"を削除して再実行してください)。"
            )

        frame_indices = _select_frame_indices(n_times, cfg)

        print(
            f"[INFO] VTK時系列出力: 全{n_times}フレーム中 {frame_indices.size}フレームを書き出します "
            f"(stride={cfg.vtk_export_stride}, max_frames={cfg.vtk_export_max_frames})"
        )

        frame_files = []
        frame_times = []

        for out_i, t_idx in enumerate(frame_indices):
            field = dset[:, t_idx]  # HDF5から必要な1列だけを読み込む
            base_mesh.point_data[label] = field.astype(np.float32)

            fname = f"frame_{out_i:04d}.vtu"
            base_mesh.save(os.path.join(out_dir, fname))

            frame_files.append(fname)
            frame_times.append(float(times[t_idx]))

            if out_i % 10 == 0 or out_i == frame_indices.size - 1:
                print(f"  ... {out_i + 1}/{frame_indices.size} フレーム書き出し済み")

    pvd_path = os.path.join(out_dir, "reconstruction.pvd")
    _write_pvd_collection(pvd_path, frame_files, frame_times)

    print(f"[INFO] VTK時系列コレクションを書き出しました: {pvd_path}")
    print("[INFO] ParaViewでこの.pvdファイルを開くと、アニメーションとして再生できます。")

    return {
        "pvd_path": pvd_path,
        "out_dir": out_dir,
        "n_frames_written": len(frame_files),
    }
