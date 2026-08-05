"""
DMD/RDMD結果の標準的な診断プロットを生成するモジュール。

DMD解析でよく使われる代表的な3種類のプロットを出力する:

  1. 特異値スペクトル・累積エネルギープロット
     -> POD的な低ランク圧縮が、どの程度のランクでエネルギーを
        説明できているかを確認する(Kutz et al. "Dynamic Mode
        Decomposition" 等の教科書で定番の図)。

  2. 固有値の単位円プロット (複素平面上に mu_k をプロットし、
     |mu|=1 の単位円を重ねる)
     -> 単位円に近いモードは減衰も成長もしない周期的な構造、
        単位円の内側は減衰、外側は成長(非物理的・発散の疑いあり)。
        DMD結果の健全性を一目で確認するための最も標準的な図。

  3. モード振幅スペクトル (周波数 vs |b_k| のstemプロット)
     -> どの周波数にエネルギーが集中しているかを可視化する。
        mode_selectionで選ばれた重要モードを強調表示する。

全て matplotlib で PNG として cfg.work_dir/diagnostics_dir_name に保存する。
(m×n の大規模データには依存しない、r(ランク)程度の小さい配列だけを使うため、
 この処理自体はメモリ・計算コストの両面で軽い)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面表示なし、ファイル保存専用のバックエンド
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import Config
from nondim import freq_to_strouhal


def _setup_japanese_font():
    """
    利用可能な日本語フォントを自動検出して設定する。
    見つからない場合はデフォルトフォントのまま続行する(文字化けするが処理は止めない)。
    """

    candidates = [
        "Yu Gothic", "Meiryo", "MS Gothic",       # Windows
        "Hiragino Sans", "Hiragino Kaku Gothic ProN",  # macOS
        "Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic",  # Linux
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}

    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return

    print(
        "[WARN] 日本語フォントが見つかりませんでした。"
        "診断プロットのラベルが文字化けする可能性がありますが、処理は続行します。"
    )


_setup_japanese_font()


def _ensure_dir(cfg: Config) -> str:
    out_dir = os.path.join(cfg.work_dir, cfg.diagnostics_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_singular_value_spectrum(result: dict, cfg: Config, out_dir: str) -> str:
    """特異値の大きさと累積エネルギーの2段組プロット"""

    s = result["singular_values"]
    energy = s ** 2
    cum_energy = np.cumsum(energy) / np.sum(energy)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].semilogy(np.arange(1, s.size + 1), s, "o-", ms=3)
    axes[0].set_xlabel("モード番号")
    axes[0].set_ylabel("特異値 (log scale)")
    axes[0].set_title("特異値スペクトル")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(np.arange(1, s.size + 1), cum_energy * 100, "o-", ms=3, color="tab:orange")
    axes[1].axhline(99.0, color="gray", linestyle="--", linewidth=1, label="99%")
    axes[1].axhline(99.9, color="gray", linestyle=":", linewidth=1, label="99.9%")
    axes[1].set_xlabel("モード番号(累積)")
    axes[1].set_ylabel("累積エネルギー [%]")
    axes[1].set_title("累積エネルギー")
    axes[1].set_ylim(0, 102)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "singular_value_spectrum.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] 診断プロットを保存しました: {out_path}")
    return out_path


def plot_eigenvalues_unit_circle(
    result: dict, selection: dict, cfg: Config, out_dir: str
) -> str:
    """固有値を複素平面にプロットし、単位円を重ねる (標準的なDMD健全性チェック図)"""

    eigs = result["eigs"]
    idx_selected = set(selection["idx_selected"].tolist())

    theta = np.linspace(0, 2 * np.pi, 200)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(unit_circle_x, unit_circle_y, "k--", linewidth=1, label="単位円 |μ|=1")

    is_selected = np.array([i in idx_selected for i in range(eigs.size)])

    ax.scatter(
        eigs.real[~is_selected],
        eigs.imag[~is_selected],
        c="lightgray",
        s=25,
        label="非選択モード",
        edgecolors="gray",
        linewidths=0.5,
    )
    ax.scatter(
        eigs.real[is_selected],
        eigs.imag[is_selected],
        c="tab:red",
        s=60,
        label="選択された重要モード",
        edgecolors="k",
        linewidths=0.5,
        zorder=5,
    )

    ax.set_xlabel("Re(μ)")
    ax.set_ylabel("Im(μ)")
    ax.set_title("DMD固有値の複素平面プロット\n(単位円の内側=減衰, 外側=成長)")
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    margin = 1.3
    lim = max(margin, np.max(np.abs(eigs)) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "eigenvalues_unit_circle.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] 診断プロットを保存しました: {out_path}")
    return out_path


def plot_mode_amplitude_spectrum(
    result: dict, selection: dict, cfg: Config, out_dir: str
) -> str:
    """周波数 vs |振幅| のstemプロット。選択された重要モードを強調表示する。"""

    freq_hz = result["freq_hz"]
    b_full = selection["b_full"]
    idx_selected = set(selection["idx_selected"].tolist())

    positive_mask = freq_hz > 0
    idx_positive = np.where(positive_mask)[0]

    freqs = freq_hz[idx_positive]
    amps = np.abs(b_full[idx_positive])
    is_selected = np.array([i in idx_selected for i in idx_positive])

    order = np.argsort(freqs)
    freqs, amps, is_selected = freqs[order], amps[order], is_selected[order]

    fig, ax = plt.subplots(figsize=(8, 4))

    if freqs.size == 0:
        ax.text(
            0.5, 0.5, "正の周波数を持つモードがありません",
            ha="center", va="center", transform=ax.transAxes,
        )
    else:
        if (~is_selected).any():
            markerline, stemlines, baseline = ax.stem(
                freqs[~is_selected],
                amps[~is_selected],
                linefmt="lightgray",
                markerfmt="o",
                basefmt=" ",
            )
            plt.setp(markerline, color="lightgray", markersize=4)
            plt.setp(stemlines, color="lightgray", linewidth=1)

        if is_selected.any():
            markerline2, stemlines2, _ = ax.stem(
                freqs[is_selected],
                amps[is_selected],
                linefmt="tab:red",
                markerfmt="o",
                basefmt=" ",
            )
            plt.setp(markerline2, color="tab:red", markersize=7)
            plt.setp(stemlines2, color="tab:red", linewidth=1.5)

    ax.set_xlabel("周波数 [Hz]")
    ax.set_ylabel("|振幅 b|")
    ax.set_yscale("log")
    ax.set_title(f"モード振幅スペクトル (赤: {selection['method']}で選択された重要モード)")
    ax.grid(True, which="both", alpha=0.3)

    # 上部にSt数の副軸を追加 (St = f * L / U)。下軸(Hz)と連動して自動更新される。
    def _hz_to_st(f):
        return f * cfg.reference_length / cfg.reference_velocity

    def _st_to_hz(st):
        return st * cfg.reference_velocity / cfg.reference_length

    ax_st = ax.secondary_xaxis("top", functions=(_hz_to_st, _st_to_hz))
    ax_st.set_xlabel("St = f・L/U")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "mode_amplitude_spectrum.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] 診断プロットを保存しました: {out_path}")
    return out_path


def generate_diagnostic_plots(result: dict, selection: dict, cfg: Config) -> dict:
    """標準的なDMD診断プロットをまとめて生成する"""

    out_dir = _ensure_dir(cfg)

    paths = {
        "singular_value_spectrum": plot_singular_value_spectrum(result, cfg, out_dir),
        "eigenvalues_unit_circle": plot_eigenvalues_unit_circle(
            result, selection, cfg, out_dir
        ),
        "mode_amplitude_spectrum": plot_mode_amplitude_spectrum(
            result, selection, cfg, out_dir
        ),
    }
    return paths