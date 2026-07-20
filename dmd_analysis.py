"""
HDF5に保存済みのスナップショット行列を読み込み、
config の use_pod_reduction 設定に応じて RDMD (POD的低ランク圧縮あり)
または 標準DMD (打ち切りなし) を実行するモジュール。
"""

import os
import numpy as np
import h5py

from pydmd import DMD, RDMD

from config import Config


def load_snapshot_matrix(h5_path: str, cfg: Config):
    """HDF5からXを読み込み、必要なら平均差し引きを行う"""

    with h5py.File(h5_path, "r") as h5f:
        X = h5f["X"][:]            # (n_points, n_snapshots)
        coords = h5f["coords"][:]  # (n_points, 3)
        times = h5f["times"][:]
        dt = float(h5f.attrs["dt"])

    mean_field = None
    if cfg.mean_subtraction:
        mean_field = X.mean(axis=1, keepdims=True)
        X = X - mean_field

    return X, coords, times, dt, mean_field


def run_dmd(X: np.ndarray, dt: float, cfg: Config):
    """
    config.use_pod_reduction に応じて RDMD または 標準DMD(打ち切りなし) を実行する。

    Returns
    -------
    dmd : pydmd.DMDBase
        fit済みのDMDオブジェクト
    """

    if cfg.use_pod_reduction:
        print(
            f"[INFO] RDMD (POD的低ランク圧縮 ON) "
            f"svd_rank={cfg.svd_rank}, oversampling={cfg.rdmd_oversampling}, "
            f"power_iters={cfg.rdmd_power_iters}"
        )
        dmd = RDMD(
            svd_rank=cfg.svd_rank,
            oversampling=cfg.rdmd_oversampling,
            power_iters=cfg.rdmd_power_iters,
        )
    else:
        print("[INFO] 標準DMD (打ち切りなし, svd_rank=-1) POD前処理 OFF")
        dmd = DMD(svd_rank=-1, exact=True)

    dmd.fit(X)
    dmd.original_time["dt"] = dt
    dmd.original_time["t0"] = 0.0

    return dmd


def analyze_modes(dmd, dt: float, cfg: Config) -> dict:
    """
    固有値から周波数・成長率を計算し、target_frequencyに近いモードを抽出する。
    """

    eigs = dmd.eigs  # 離散固有値 lambda
    # omega = ln(lambda)/dt  (連続時間の複素角周波数)
    omega = np.log(eigs) / dt
    growth_rate = omega.real          # 減衰(負)/成長(正)
    freq_hz = omega.imag / (2.0 * np.pi)

    # モードエネルギー: 各モードの時間係数(振幅)のRMSベースで評価
    # dmd.amplitudes はt=0での複素振幅
    amplitudes = dmd.amplitudes
    mode_energy = np.abs(amplitudes)

    # 周波数は符号対称に出るため、正の周波数側のみを対象にする
    positive_mask = freq_hz > 0

    idx_sorted = np.argsort(np.abs(freq_hz[positive_mask] - cfg.target_frequency))
    idx_positive = np.where(positive_mask)[0]
    idx_ranked = idx_positive[idx_sorted]

    within_tol = idx_ranked[
        np.abs(freq_hz[idx_ranked] - cfg.target_frequency) <= cfg.frequency_tolerance
    ]

    print(f"[INFO] 全モード数: {eigs.size} (うち正の周波数側: {positive_mask.sum()})")
    print(
        f"[INFO] {cfg.target_frequency:.3f} Hz ± {cfg.frequency_tolerance:.3f} Hz "
        f"に該当するモード数: {within_tol.size}"
    )

    print("\n[近傍モード一覧] (周波数近い順)")
    print(f"{'rank':>4} {'freq[Hz]':>10} {'growth_rate':>12} {'|amplitude|':>12}")
    for rank, i in enumerate(idx_ranked[: max(cfg.n_modes_to_export, within_tol.size)]):
        print(
            f"{rank:>4} {freq_hz[i]:>10.4f} {growth_rate[i]:>12.4e} "
            f"{mode_energy[i]:>12.4e}"
        )

    return {
        "eigs": eigs,
        "freq_hz": freq_hz,
        "growth_rate": growth_rate,
        "mode_energy": mode_energy,
        "idx_ranked": idx_ranked,
        "idx_within_tol": within_tol,
    }


def save_results(dmd, analysis: dict, coords: np.ndarray, cfg: Config, out_path: str):
    """固有値・モード解析結果をnpzに保存する"""

    modes = dmd.modes  # (n_points, n_modes) 複素モード形状

    np.savez_compressed(
        out_path,
        coords=coords,
        eigs=analysis["eigs"],
        freq_hz=analysis["freq_hz"],
        growth_rate=analysis["growth_rate"],
        mode_energy=analysis["mode_energy"],
        modes=modes,
        idx_ranked=analysis["idx_ranked"],
        idx_within_tol=analysis["idx_within_tol"],
    )
    print(f"[INFO] 解析結果を保存しました: {out_path}")
