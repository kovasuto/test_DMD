"""
DMDモードの重要度判定・選択を行うモジュール。

全て「縮約空間」(POD係数 Y: r×n, 固有ベクトル W: r×r)の上で計算するため、
巨大な物理空間データ(m×n)への再アクセスは一切発生しない。
dask_rdmd.run_dask_rdmd() が既に計算済みの Y_reduced, W, eigs を使うだけで良い。

提供する4つの重要度判定方法:

  1. "amplitude"         : 単純に t=0 での振幅 |b_k| でランキング
  2. "integrated_energy" : Kou & Zhang (2017) の重み付け。
                           |b_k| に加えて、モードの成長/減衰を時間窓全体で
                           積分した寄与量で評価する。振動が減衰していくモードや、
                           逆に成長していくモードを、単純な初期振幅だけでは
                           過小/過大評価してしまう問題を補正する。デフォルト推奨。
  3. "greedy"             : Orthogonal Matching Pursuit 型の貪欲法。
                           縮約データYへのフィット誤差を最も減らすモードを
                           1つずつ追加していく。指定したモード数になるまで反復。
  4. "spdmd"              : Sparsity-Promoting DMD (Jovanovic, Schmid, Nichols, 2014)。
                           ADMMで L1正則化付き最小二乗問題を解き、
                           スパースな(=重要なものだけが非ゼロの)振幅集合を得る。
                           gamma(正則化強度)で有効モード数を制御する。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import Config


# ======================================================================
# 縮約空間での最小二乗振幅 (全スナップショットに対するフィット)
# ======================================================================

def _vandermonde(eigs: np.ndarray, n_snapshots: int) -> np.ndarray:
    """mu_k^n (k=モード, n=0..n_snapshots-1) の (r, n_snapshots) 行列"""
    n_idx = np.arange(n_snapshots)
    return eigs[:, None] ** n_idx[None, :]


def compute_optimal_amplitudes(
    Y: np.ndarray, W: np.ndarray, eigs: np.ndarray
) -> np.ndarray:
    """
    全スナップショット Y (r×n) に対して、モード W の線形結合
        Y ≈ W diag(b) Vand(mu)
    を最小二乗で満たす振幅 b (r,) を求める (Jovanovic et al. 2014 の定式化)。

    t=0の1スナップショットだけへのフィットよりも頑健で、
    spDMD/貪欲法の出発点として使う。
    """

    n_snapshots = Y.shape[1]
    Vand = _vandermonde(eigs, n_snapshots)  # (r, n)

    # P = (W^H W) ∘ conj(Vand Vand^H)   (アダマール積)
    P = (W.conj().T @ W) * np.conj(Vand @ Vand.conj().T)
    # q = conj(diag(Vand Y^H W))
    q = np.conj(np.diag(Vand @ Y.conj().T @ W))

    # 正則化を少量加えて数値的に安定化 (P がほぼ特異になる場合への対策)
    reg = 1e-10 * np.trace(P).real / P.shape[0]
    b = np.linalg.solve(P + reg * np.eye(P.shape[0]), q)

    return b


# ======================================================================
# 1. 単純振幅ランキング
# ======================================================================

def rank_by_amplitude(b: np.ndarray) -> np.ndarray:
    """|b_k| が大きい順のインデックスを返す"""
    return np.argsort(-np.abs(b))


# ======================================================================
# 2. Kou-Zhang 時間積分エネルギー
# ======================================================================

def rank_by_integrated_energy(
    b: np.ndarray, eigs: np.ndarray, n_snapshots: int
) -> np.ndarray:
    """
    d_k = |b_k| * sqrt( sum_{n=0}^{N-1} |mu_k|^{2n} )

    成長/減衰するモードが時間窓全体でどれだけ寄与するかを積分評価する。
    純粋に周期的なモード(|mu_k|=1)では d_k = |b_k| * sqrt(N) となり、
    amplitudeランキングと単調に同じ順序になる。
    """

    abs_mu = np.abs(eigs)
    # |mu|^{2n} の等比級数の和。|mu|≈1のときは特異にならないよう分岐。
    with np.errstate(divide="ignore", invalid="ignore"):
        series_sum = np.where(
            np.isclose(abs_mu, 1.0),
            float(n_snapshots),
            (1.0 - abs_mu ** (2 * n_snapshots)) / (1.0 - abs_mu ** 2),
        )
    d = np.abs(b) * np.sqrt(np.abs(series_sum))
    return np.argsort(-d), d


# ======================================================================
# 3. 貪欲法 (Orthogonal Matching Pursuit 型)
# ======================================================================

def greedy_select(
    Y: np.ndarray, W: np.ndarray, eigs: np.ndarray, n_modes: int
) -> "tuple[np.ndarray, np.ndarray]":
    """
    縮約データ Y (r×n) を最もよく再現するモードを1つずつ貪欲に追加していく。

    各ステップで:
      1. 現在選択済みのモード集合 S に対し、最小二乗で振幅 b_S を再フィット
      2. 残差 R = Y - W[:,S] diag(b_S) Vand[S,:] を計算
      3. 残差に最も強く相関する(まだ選ばれていない)モードを追加

    Returns
    -------
    selected_idx : 選ばれたモードのインデックス (重要度順)
    b_full       : 全モード分の振幅配列。選ばれなかったモードは0。
    """

    r, n_snapshots = Y.shape[0], Y.shape[1]
    Vand = _vandermonde(eigs, n_snapshots)  # (r, n)

    remaining = list(range(r))
    selected: list = []
    residual = Y.copy()

    n_modes = min(n_modes, r)

    for _ in range(n_modes):
        # 残差との相関が最大のモードを選ぶ
        # score_k = || W[:,k] (Vand[k,:] . conj(residual)) の射影 ||
        # 簡易的に、各候補モードだけを使って残差にフィットしたときの
        # 二乗誤差減少量で評価する (計算コストは r×n 程度で軽い)
        best_k, best_score = None, -np.inf
        for k in remaining:
            wk = W[:, k : k + 1]                 # (r, 1)
            vk = Vand[k : k + 1, :]               # (1, n)
            basis = wk @ vk                       # (r, n) このモード単体の寄与
            # 残差とこの基底の内積(フロベニウス内積)の大きさをスコアにする
            score = np.abs(np.vdot(basis, residual))
            if score > best_score:
                best_score = score
                best_k = k

        selected.append(best_k)
        remaining.remove(best_k)

        # 選択済みモード集合で再フィット (小さい最小二乗問題)
        W_S = W[:, selected]                      # (r, |S|)
        Vand_S = Vand[selected, :]                 # (|S|, n)

        # Y ≈ W_S diag(b_S) Vand_S を b_S について解く
        # -> 列ごとに Vand_S の対角化された形にするため、
        #    A = W_S * Vand_S の各時刻を並べたシステムを最小二乗で解く。
        # ここでは全時刻同時に解く定式化 (Jovanovic流のP,q) を使う。
        b_S = compute_optimal_amplitudes(Y, W_S, eigs[selected])

        Vand_full_S = _vandermonde(eigs[selected], n_snapshots)
        residual = Y - (W_S @ np.diag(b_S) @ Vand_full_S)

    b_full = np.zeros(r, dtype=complex)
    b_full[selected] = b_S

    return np.array(selected), b_full


# ======================================================================
# 4. Sparsity-Promoting DMD (spDMD, ADMM)
# ======================================================================

def spdmd_select(
    Y: np.ndarray,
    W: np.ndarray,
    eigs: np.ndarray,
    gamma: float,
    admm_iters: int = 200,
    rho: float = 1.0,
    tol: float = 1e-6,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Jovanovic, Schmid, Nichols (2014) "Sparsity-promoting dynamic mode
    decomposition" の ADMM解法。

        minimize  (1/2) b^H P b - Re(q^H b) + const   +   gamma * ||b||_1

    を ADMM で解き、多くの成分が厳密にゼロになるスパースな振幅 b を得る。
    最後に非ゼロ成分だけで再度制約なし最小二乗フィット(polishing)を行い、
    振幅の偏りを補正する。

    Parameters
    ----------
    gamma : float
        正則化強度。大きいほど非ゼロモード数が減る。

    Returns
    -------
    selected_idx : 非ゼロになったモードのインデックス (|b|降順)
    b_full       : 全モード分の振幅 (非選択モードは0)
    """

    n_snapshots = Y.shape[1]
    Vand = _vandermonde(eigs, n_snapshots)

    P = (W.conj().T @ W) * np.conj(Vand @ Vand.conj().T)
    q = np.conj(np.diag(Vand @ Y.conj().T @ W))

    r = P.shape[0]
    reg = 1e-10 * np.trace(P).real / r
    P_reg = P + reg * np.eye(r)

    # ADMM変数初期化
    b = np.linalg.solve(P_reg, q)   # 制約なし解からスタート
    z = b.copy()
    u = np.zeros(r, dtype=complex)  # スケール済み双対変数

    # b更新用の行列を事前分解 (毎回同じ線形系を解くのでCholesky等でも良いが、
    # rが小さい(数百程度)ので毎回solveでも十分速い)
    lhs = P_reg + rho * np.eye(r)

    def soft_threshold_complex(x: np.ndarray, kappa: float) -> np.ndarray:
        """複素数版ソフト閾値作用素 (絶対値方向にのみ縮小)"""
        mag = np.abs(x)
        mag_shrunk = np.maximum(mag - kappa, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = np.where(mag > 0, x / np.where(mag == 0, 1, mag), 0.0)
        return direction * mag_shrunk

    for it in range(admm_iters):
        # b更新: (P + rho I) b = q + rho (z - u)
        rhs = q + rho * (z - u)
        b_new = np.linalg.solve(lhs, rhs)

        # z更新: 複素ソフト閾値
        z_new = soft_threshold_complex(b_new + u, gamma / rho)

        # 双対更新
        u = u + b_new - z_new

        diff = np.linalg.norm(b_new - b) / (np.linalg.norm(b) + 1e-12)
        b, z = b_new, z_new
        if diff < tol:
            break

    # スパースパターン(非ゼロ成分)を確定し、polishing(制約なし再フィット)
    support = np.where(np.abs(z) > 1e-8 * np.max(np.abs(z) + 1e-30))[0]

    if support.size == 0:
        # gammaが強すぎて全部ゼロになった場合、振幅最大の1モードだけ残す
        support = np.array([np.argmax(np.abs(b))])

    W_S = W[:, support]
    b_polished = compute_optimal_amplitudes(Y, W_S, eigs[support])

    b_full = np.zeros(r, dtype=complex)
    b_full[support] = b_polished

    # |b|降順で並べたインデックスを返す
    order = support[np.argsort(-np.abs(b_polished))]

    return order, b_full


def spdmd_autotune_gamma(
    Y: np.ndarray,
    W: np.ndarray,
    eigs: np.ndarray,
    target_n_modes: int,
    gamma_min: float = 1e-4,
    gamma_max: float = 1e4,
    n_search: int = 25,
    **admm_kwargs,
) -> "tuple[float, np.ndarray, np.ndarray]":
    """
    目標モード数 target_n_modes に近い有効モード数になる gamma を、
    対数スケールの二分探索で自動的に見つける。

    spDMDは「gammaを直接指定する」設計だが、
    実務上は「何モードにしたいか」で考えたいことが多いため用意した補助関数。
    """

    gammas = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_search)

    best = None
    for gamma in gammas:
        idx, b_full = spdmd_select(Y, W, eigs, gamma, **admm_kwargs)
        n_active = idx.size
        if best is None or abs(n_active - target_n_modes) < abs(
            best[1] - target_n_modes
        ):
            best = (gamma, n_active, idx, b_full)
        if n_active <= target_n_modes:
            # モード数が目標以下になった時点で、これ以上gammaを上げても
            # 減る一方なので探索を打ち切ってよい
            break

    gamma_best, n_active, idx, b_full = best
    print(
        f"[INFO] spDMD auto-tune: gamma={gamma_best:.4g} で "
        f"有効モード数={n_active} (目標{target_n_modes})"
    )
    return gamma_best, idx, b_full


# ======================================================================
# メインディスパッチ
# ======================================================================

def select_modes(result: dict, cfg: Config) -> dict:
    """
    config.reconstruction_method に応じてモード重要度判定・選択を行う。

    Returns
    -------
    selection : dict
        idx_selected : 選択された(または重要度順の)モードインデックス
        b_full       : 全モード分の振幅 (未選択は0になる手法もある)
        method       : 使用した手法名
    """

    Y = result["Y_reduced"]
    W = result["W"]
    eigs = result["eigs"]
    n_snapshots = result["n_snapshots"]

    method = cfg.reconstruction_method
    n_modes = cfg.reconstruction_n_modes

    if method == "amplitude":
        b_full = compute_optimal_amplitudes(Y, W, eigs)
        idx_ranked = rank_by_amplitude(b_full)
        idx_selected = idx_ranked[:n_modes]

    elif method == "integrated_energy":
        b_full = compute_optimal_amplitudes(Y, W, eigs)
        idx_ranked, _ = rank_by_integrated_energy(b_full, eigs, n_snapshots)
        idx_selected = idx_ranked[:n_modes]

    elif method == "greedy":
        idx_selected, b_full = greedy_select(Y, W, eigs, n_modes)

    elif method == "spdmd":
        if cfg.spdmd_auto_tune:
            gamma, idx_selected, b_full = spdmd_autotune_gamma(
                Y,
                W,
                eigs,
                target_n_modes=n_modes,
                admm_iters=cfg.spdmd_admm_iters,
                rho=cfg.spdmd_rho,
            )
        else:
            idx_selected, b_full = spdmd_select(
                Y,
                W,
                eigs,
                gamma=cfg.spdmd_gamma,
                admm_iters=cfg.spdmd_admm_iters,
                rho=cfg.spdmd_rho,
            )
            if idx_selected.size > n_modes:
                # gamma固定で指定より多く残った場合は、上位n_modesだけ使う
                order = np.argsort(-np.abs(b_full[idx_selected]))
                idx_selected = idx_selected[order][:n_modes]

    else:
        raise ValueError(f"未知の reconstruction_method: {method}")

    print(f"\n[モード選択] 方法 = {method}, 選択モード数 = {idx_selected.size}")
    print(f"{'rank':>4} {'idx':>5} {'freq[Hz]':>10} {'growth_rate':>12} {'|b|':>12}")
    for rank, i in enumerate(idx_selected):
        print(
            f"{rank:>4} {i:>5} {result['freq_hz'][i]:>10.4f} "
            f"{result['growth_rate'][i]:>12.4e} {np.abs(b_full[i]):>12.4e}"
        )

    return {
        "idx_selected": np.asarray(idx_selected),
        "b_full": b_full,
        "method": method,
    }
