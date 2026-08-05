"""
dask_rdmd.run_dask_rdmd() の出力 (result dict) を受け取り、
DMDスペクトル(全モードの周波数・成長率・振幅)を一覧表示するモジュール。

特定の周波数を優先的に抽出するようなフィルタは持たない。
どのモードが重要かの判定は mode_selection.py の汎用手法
(amplitude / integrated_energy / greedy / spdmd) に一本化している。
"""

import numpy as np

from config import Config
from nondim import freq_to_strouhal


def analyze_modes(result: dict, cfg: Config) -> dict:
    """
    固有値から周波数・成長率を計算し、スペクトル全体を周波数順に整理する。
    (特定周波数への絞り込みは行わない。重要モードの選定は select_modes を使う)

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
    st_number = freq_to_strouhal(freq_hz, cfg)  # St = f * L / U (config.reference_length/velocity)

    # 周波数は符号対称に出るため、正の周波数側のみを表示対象にする
    positive_mask = freq_hz > 0
    idx_positive = np.where(positive_mask)[0]
    idx_by_freq = idx_positive[np.argsort(freq_hz[idx_positive])]

    print(f"[INFO] 全モード数: {eigs.size} (うち正の周波数側: {positive_mask.sum()})")
    print(
        f"[INFO] St数の定義: St = f * L / U "
        f"(L={cfg.reference_length} m, U={cfg.reference_velocity} m/s)"
    )
    print("\n[DMDスペクトル一覧] (周波数昇順)")
    print(f"{'idx':>5} {'freq[Hz]':>10} {'St':>10} {'growth_rate':>12} {'|amplitude|':>12}")
    for i in idx_by_freq:
        print(
            f"{i:>5} {freq_hz[i]:>10.4f} {st_number[i]:>10.4f} "
            f"{growth_rate[i]:>12.4e} {mode_energy[i]:>12.4e}"
        )

    return {
        "eigs": eigs,
        "freq_hz": freq_hz,
        "st_number": st_number,
        "growth_rate": growth_rate,
        "mode_energy": mode_energy,
        "idx_by_freq": idx_by_freq,
    }


def save_results(result: dict, analysis: dict, selection: dict, cfg: Config, out_path: str):
    """固有値・モード解析結果・選択されたモードをnpzに保存する"""

    np.savez_compressed(
        out_path,
        coords=result["coords"],
        eigs=analysis["eigs"],
        freq_hz=analysis["freq_hz"],
        st_number=analysis["st_number"],
        growth_rate=analysis["growth_rate"],
        mode_energy=analysis["mode_energy"],
        modes=result["modes"],              # (n_points, r) 複素モード形状
        idx_by_freq=analysis["idx_by_freq"],
        idx_selected=selection["idx_selected"],   # mode_selectionで選ばれた重要モード
        b_full=selection["b_full"],
        selection_method=selection["method"],
        rank=result["rank"],
        reference_length=cfg.reference_length,
        reference_velocity=cfg.reference_velocity,
    )
    print(f"[INFO] 解析結果を保存しました: {out_path}")
