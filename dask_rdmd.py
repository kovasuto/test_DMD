"""
真にアウトオブコアな RDMD 実装 (Dask版)。

前バージョン (dmd_analysis.py) は
    X = h5f["X"][:]
としてスナップショット行列を一括でRAMに読み込んでいたため、
「ストリーミングでHDF5に書き出した」意味が計算段階で失われていた。

本モジュールでは、HDF5上のXを dask.array としてチャンク単位のまま扱い、
一度も m×n 全体をメモリに展開せずに DMD 固有値・モードを計算する。

アルゴリズムの流れ (標準DMDのSVDベース手法, 論文Eq.7-9 の低ランク近似版):

  X1 = X[:, :-1], X2 = X[:, 1:]                       (どちらも dask array, 遅延評価)

  X1 ≈ U * diag(s) * V^T                                (ランダム化SVD, dask.array.linalg.svd_compressed)
      -> U, s, V は小さい (m×r, r, (n-1)×r) ので .compute() してnumpy化してよい
      -> ここで X1 への「本読み込み」が発生する (内部でpower_iterationsの回数だけ複数パス)

  Atilde = U^T X2 V diag(1/s)                            (r×r, 小さい)
      -> U^T X2 の計算に X2 への1回のパスが発生する (dask経由、m×(n-1)を一度にRAMに載せない)

  Atilde の固有分解 -> mu (離散固有値), W (固有ベクトル, r×r)

  Phi = X2 V diag(1/s) W                                 (exact DMD modes, m×r)
      -> X2 への追加1回のパスが発生する (dask経由)

X1・X2それぞれへのアクセスは合計でも数回のパスに収まり、
どの時点でも同時にメモリに保持するのは「小さい行列 (m×r, r×r, r×n など)」だけ。

use_pod_reduction=False の場合は rank を n-1 (打ち切りなし) にして同じ経路で計算する。
ただしこれは事実上フルSVDに近い計算量・メモリになるため、
大規模データでは use_pod_reduction=True (低ランク) を強く推奨する。
"""

import numpy as np
import h5py
import dask.array as da
from dask.array.linalg import svd_compressed

from config import Config


def _open_dask_matrices(h5_path: str, mean_subtraction: bool):
    """
    HDF5上のXデータセットをdask.arrayとして遅延ロードし、
    X1=X[:, :-1], X2=X[:, 1:] を返す。

    mean_subtraction=True の場合、列方向平均(時間平均)を計算して差し引く。
    平均計算自体もdaskのchunk処理で行われ、Xを一括ロードしない。
    """

    h5f = h5py.File(h5_path, "r")  # クローズせず開いたままdask経由で参照させる
    dset = h5f["X"]
    coords = h5f["coords"][:]      # 座標は小さいので通常ロードでよい
    dt = float(h5f.attrs["dt"])

    # HDF5のchunk形状に合わせてdask arrayのchunkを揃えると効率が良い
    chunks = dset.chunks if dset.chunks is not None else "auto"
    X = da.from_array(dset, chunks=chunks)

    mean_field = None
    if mean_subtraction:
        # 列方向(時間方向)平均。chunk処理で計算され、全体を一括保持しない。
        mean_field = X.mean(axis=1, keepdims=True)
        X = X - mean_field  # 遅延演算。実際にはSVD計算時に初めて評価される

    X1 = X[:, :-1]
    X2 = X[:, 1:]

    return X1, X2, coords, dt, mean_field, h5f


def run_dask_rdmd(h5_path: str, cfg: Config):
    """
    Dask版アウトオブコア RDMD/DMD を実行する。

    Returns
    -------
    result : dict
        eigs, freq_hz, growth_rate, mode_energy, modes(m×r, numpy),
        coords, dt などを含む
    """

    X1, X2, coords, dt, mean_field, h5f = _open_dask_matrices(
        h5_path, cfg.mean_subtraction
    )

    n_points, n_minus_1 = X1.shape
    print(f"[INFO] X1 shape (m, n-1) = {X1.shape}  (dask遅延配列, 未ロード)")

    # --- ランク決定 -----------------------------------------------------
    if cfg.use_pod_reduction:
        if isinstance(cfg.svd_rank, float) and 0.0 < cfg.svd_rank < 1.0:
            # 累積エネルギー基準は事前ランク指定ができないため、
            # まず十分大きめの上限ランクで計算し、事後にエネルギー基準で
            # 有効ランクを絞り込む二段構えにする。
            rank_cap = min(n_minus_1, max(50, int(0.1 * n_minus_1)))
            print(
                f"[INFO] svd_rank={cfg.svd_rank} (累積エネルギー基準) のため、"
                f"上限ランク {rank_cap} でSVDを計算し、事後に絞り込みます。"
            )
            target_rank = rank_cap
            energy_ratio = cfg.svd_rank
        else:
            target_rank = int(cfg.svd_rank) if cfg.svd_rank > 0 else min(
                n_points, n_minus_1
            )
            energy_ratio = None
        print(
            f"[INFO] RDMD (POD的低ランク圧縮 ON) target_rank={target_rank}, "
            f"oversampling={cfg.rdmd_oversampling}, power_iters={cfg.rdmd_power_iters}"
        )
    else:
        target_rank = min(n_points, n_minus_1)
        energy_ratio = None
        print(
            "[INFO] POD前処理 OFF: 打ち切りなし (rank = min(m, n-1)) で計算します。"
            "大規模データでは非常に重くなる可能性があります。"
        )

    # --- Step 1: X1 のランダム化SVD (dask, out-of-core) ------------------
    U, s, Vh = svd_compressed(
        X1,
        k=target_rank,
        n_power_iter=cfg.rdmd_power_iters,
        n_oversamples=cfg.rdmd_oversampling,
        compute=False,
    )

    # U, s, Vh はここで初めて実データを読みに行く (X1への実質的な本読み込み)
    U, s, Vh = da.compute(U, s, Vh)
    V = Vh.conj().T  # (n-1, r)

    # --- エネルギー基準でのランク絞り込み (事後) --------------------------
    if energy_ratio is not None:
        cum_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        r_eff = int(np.searchsorted(cum_energy, energy_ratio) + 1)
        r_eff = min(r_eff, s.size)
        print(
            f"[INFO] 累積エネルギー{energy_ratio*100:.2f}%を満たす実効ランク: {r_eff} "
            f"(上限ランク{target_rank}のうち)"
        )
        U = U[:, :r_eff]
        s = s[:r_eff]
        V = V[:, :r_eff]

    r = s.size
    print(f"[INFO] 最終的に使用するランク r = {r}")

    # --- Step 2: Atilde = U^T X2 V diag(1/s)  (X2への1回のdaskパス) -------
    U_da = da.from_array(U, chunks=U.shape)  # 小さいのでchunk分割不要
    UT_X2 = da.matmul(U_da.T, X2)            # (r, n-1) だが計算はchunk単位で実施
    UT_X2 = UT_X2.compute()                  # ここでX2への実読み込みが発生 (1パス)

    Atilde = UT_X2 @ V @ np.diag(1.0 / s)    # 小さい行列同士の演算 (r×r)

    # --- Step 3: Atildeの固有分解 (小さいので通常のnumpyでよい) -----------
    mu, W = np.linalg.eig(Atilde)

    # --- Step 4: exact DMD modes Phi = X2 V diag(1/s) W (X2への追加1パス) --
    # 小さい行列 (V diag(1/s) W) を先に作ってから X2 に掛けることで、
    # 大きい行列同士の掛け算を避ける (m×(n-1) @ (n-1)×r で済む)
    small_mat = V @ np.diag(1.0 / s) @ W     # (n-1, r)
    small_mat_da = da.from_array(small_mat, chunks=small_mat.shape)
    Phi = da.matmul(X2, small_mat_da)        # (m, r), chunk単位で計算
    Phi = Phi.compute()                       # ここでX2への実読み込みが発生 (2パス目)

    h5f.close()

    # --- 周波数・成長率・振幅の計算 ---------------------------------------
    omega = np.log(mu) / dt
    growth_rate = omega.real
    freq_hz = omega.imag / (2.0 * np.pi)

    # 振幅 b: 初期スナップショット x1 を Phi で最小二乗フィット (小さい問題)
    x1_0 = np.asarray(X1[:, 0].compute()) if hasattr(X1, "compute") else X1[:, 0]
    amplitudes, *_ = np.linalg.lstsq(Phi, x1_0, rcond=None)
    mode_energy = np.abs(amplitudes)

    return {
        "eigs": mu,
        "freq_hz": freq_hz,
        "growth_rate": growth_rate,
        "mode_energy": mode_energy,
        "modes": Phi,
        "amplitudes": amplitudes,
        "coords": coords,
        "dt": dt,
        "rank": r,
    }
