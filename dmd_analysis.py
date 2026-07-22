"""
dask_rdmd.run_dask_rdmd() の出力 (result dict) を受け取り、
目標周波数(例: 1Hz)近傍のモードを抽出・整理するモジュール。

計算そのもの (SVD/RDMD/DMD) は dask_rdmd.py 側でアウトオブコアに実行済みで、
ここで扱うのは m×r 程度に圧縮済みの小さい配列のみ。
"""

import numpy as np

from config import Config


def analyze_modes(result: dict, cfg: Config) -> dict:
    """
    固有値から周波数・成長率を計算し、target_frequencyに近いモードを抽出する。

    Parameters
    ----------
    result : dict
        dask_rdmd.run_dask_rdmd() の戻り値
        (eigs, freq_hz, growth_rate, mode_energy を含む)
    """

    eigs = result["eigs"]
    freq_hz = result["freq_hz"]
    growth_rate = result["growth_rate"]
    mode_energy = result["mode_energy"]

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


def save_results(result: dict, analysis: dict, cfg: Config, out_path: str):
    """固有値・モード解析結果をnpzに保存する"""

    np.savez_compressed(
        out_path,
        coords=result["coords"],
        eigs=analysis["eigs"],
        freq_hz=analysis["freq_hz"],
        growth_rate=analysis["growth_rate"],
        mode_energy=analysis["mode_energy"],
        modes=result["modes"],           # (n_points, r) 複素モード形状
        idx_ranked=analysis["idx_ranked"],
        idx_within_tol=analysis["idx_within_tol"],
        rank=result["rank"],
    )
    print(f"[INFO] 解析結果を保存しました: {out_path}")
