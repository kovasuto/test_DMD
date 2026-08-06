# -*- coding: utf-8 -*-
"""
karman_cfd.py

2次元・非圧縮性 Navier-Stokes ソルバー(分数ステップ法 / MAC格子)による
円柱まわりのカルマン渦列(Karman vortex street)シミュレーション。

- 空間離散化: MAC(Marker-And-Cell)スタッガード格子, 2次central difference
- 時間積分  : 陽解法オイラー法(移流項+粘性項)
- 圧力解法  : 圧力ポアソン方程式を疎行列LU分解(scipy.sparse.linalg.splu)で解く
- 円柱      : マスキング(direct forcing)による埋め込み境界(immersed boundary)
- 出力形式  : EnSight Gold (ASCII) 形式
              - cylinder.case  : ケースファイル
              - cylinder.geo   : 格子ジオメトリ(iblank付き構造格子)
              - pressure.scl****, vorticity.scl**** : スカラー(節点)
              - velocity.vec****                    : ベクトル(節点)

使い方:
    python3 karman_cfd.py

出力は ./ensight_output/ 以下に生成される。
ParaView / EnSight で cylinder.case を開くことで、圧力・渦度・速度場の
時間発展(カルマン渦列)を確認できる。
"""

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# 出力(EnSightファイル書き出し)を並列化するワーカー数。
# 時間積分ループ自体は前ステップの結果に依存する(逐次計算)ため並列化できないが、
# 各フレームのファイル書き出しは互いに独立なので、別プロセスに投げて
# 「計算」と「I/O」をオーバーラップさせることで実質的な高速化を狙う。
N_IO_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# start method に"spawn"(Windows/macOSのデフォルト)が使われると、子プロセスが
# このスクリプトをモジュールとして再インポートし、LU分解などトップレベルの重い処理を
# 子プロセスごとに再実行してしまう。"fork"(Linux既定)を明示指定してこれを防ぐ。
# ※ Windowsではforkが使えないため、その場合は本並列化部分は無効化して逐次書き出しにフォールバックする。
try:
    _MP_CTX = multiprocessing.get_context("fork")
except ValueError:
    _MP_CTX = None

# =====================================================================
# 1. パラメータ設定
# =====================================================================

# --- 流体・幾何パラメータ ---
D = 1.0                 # 円柱直径
U_inf = 1.0             # 一様流速度
Re = 150.0              # レイノルズ数 (Re=100〜200程度で明瞭なカルマン渦列)
nu = U_inf * D / Re     # 動粘性係数
rho = 1.0               # 密度(規格化)

# --- 計算領域 ---
Lx = 16.0 * D           # 流れ方向長さ
Ly = 8.0 * D            # 横方向長さ
xc_cyl = 4.0 * D         # 円柱中心 x 座標(流入端からの距離)
# 円柱中心 y 座標: 完全に対称な配置だと渦放出(カルマン渦)が数値誤差だけでは
# なかなか誘発されないため、わずかにオフセットして非対称性を与える
yc_cyl = Ly / 2.0 + 0.05 * D

# --- 格子 ---
nx = 400                # x方向セル数
ny = 200                # y方向セル数
dx = Lx / nx
dy = Ly / ny

# --- 時間積分 ---
T_end = 600.0            # 総計算時間(無次元時間, D/U_inf 単位)

# DMDベンチマーク用: スナップショット(出力)間隔を明示的な"きれいな"値に固定する。
# 積分の時間刻み dt は、この dt_snapshot をちょうど割り切る値として
# 安定条件(移流CFL・拡散条件)を満たす範囲で自動的に決定する。
# -> 出力される各フレームの時刻は必ず k * dt_snapshot (k=0,1,2,...) に厳密一致し、
#    DMD側で「サンプリング間隔 dt = dt_snapshot」として一意に使える。
dt_snapshot = 0.20      # DMDスナップショット間隔(無次元時間)。用途に応じて変更可

dt_stable = 0.15 * min(dx, dy) / U_inf          # 移流CFLベースの上限
dt_stable = min(dt_stable, 0.2 * min(dx, dy) ** 2 / nu)  # 拡散安定条件も満たす上限

# dt_snapshot を n_sub 等分した値を実際の積分dtとする(n_subは安定上限を満たす最小の整数)
n_sub = max(1, int(np.ceil(dt_snapshot / dt_stable)))
dt = dt_snapshot / n_sub                         # 実際の積分時間刻み(dt_snapshotの厳密な約数)
output_every = n_sub                             # この刻みで出力すればちょうどdt_snapshot間隔になる

n_frames_target = int(round(T_end / dt_snapshot)) + 1   # 出力フレーム数(t=0を含む)
n_steps = (n_frames_target - 1) * output_every           # ちょうどn_frames_target個のフレームが出るように総ステップ数を決定
T_end = (n_frames_target - 1) * dt_snapshot               # 実際の終了時刻(dt_snapshotの整数倍に調整)

# --- 出力設定 ---
out_dir = "ensight_output"
print(f"dx={dx:.4f} dy={dy:.4f}")
print(f"dt(積分)={dt:.6f}  dt_snapshot(DMD用の出力間隔)={dt_snapshot:.6f}  "
      f"(dt_snapshot = {output_every} x dt, 厳密一致)")
print(f"n_steps={n_steps}  n_frames={n_frames_target}  T_end={T_end:.4f}")

os.makedirs(out_dir, exist_ok=True)


# =====================================================================
# 2. MAC格子の定義
#    p        : セル中心 (nx, ny)          位置 ((i+0.5)dx, (j+0.5)dy)
#    u        : 縦フェイス (nx+1, ny)       位置 (i*dx, (j+0.5)dy)
#    v        : 横フェイス (nx, ny+1)       位置 ((i+0.5)dx, j*dy)
# =====================================================================

u = np.zeros((nx + 1, ny))
v = np.zeros((nx, ny + 1))
p = np.zeros((nx, ny))

u[:, :] = U_inf   # 初期条件: 一様流

# セル中心座標
xc = (np.arange(nx) + 0.5) * dx
yc = (np.arange(ny) + 0.5) * dy
Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

# u節点(縦フェイス)座標
xu = np.arange(nx + 1) * dx
yu = (np.arange(ny) + 0.5) * dy
Xu, Yu = np.meshgrid(xu, yu, indexing="ij")

# v節点(横フェイス)座標
xv = (np.arange(nx) + 0.5) * dx
yv = np.arange(ny + 1) * dy
Xv, Yv = np.meshgrid(xv, yv, indexing="ij")

# --- 円柱マスク(True=固体内部) ---
r2 = (D / 2.0) ** 2
mask_p = ((Xc - xc_cyl) ** 2 + (Yc - yc_cyl) ** 2) < r2
mask_u = ((Xu - xc_cyl) ** 2 + (Yu - yc_cyl) ** 2) < r2
mask_v = ((Xv - xc_cyl) ** 2 + (Yv - yc_cyl) ** 2) < r2

# =====================================================================
# 3. 圧力ポアソン方程式の係数行列(LU分解を1回だけ実施し使い回す)
#    -Lap(p) = -(rho/dt) * div(u*)
#    境界条件: 流入・上下壁: ノイマン(dp/dn=0), 流出: ディリクレ p=0
# =====================================================================

def build_pressure_matrix(nx, ny, dx, dy):
    N = nx * ny

    def idx(i, j):
        return i * ny + j

    rows, cols, vals = [], [], []
    b_extra = np.zeros(N)  # ディリクレ境界からの寄与(定数項側で処理)

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            diag = 0.0

            # --- x方向(左右) ---
            if i == 0:
                # 流入端: ノイマン dp/dx=0 -> ミラー(隣接セルのみ)
                diag -= 1.0 / dx**2
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(1.0 / dx**2)
            elif i == nx - 1:
                # 流出端: ディリクレ p=0(仮想セル p_ghost=-p_i として扱う
                # -> (p_ghost - 2p_i + p_{i-1})/dx^2 = (p_{i-1} - 3p_i)/dx^2)
                diag -= 3.0 / dx**2
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(1.0 / dx**2)
            else:
                diag -= 2.0 / dx**2
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(1.0 / dx**2)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(1.0 / dx**2)

            # --- y方向(上下壁, 自由すべり: ノイマン) ---
            if j == 0:
                diag -= 1.0 / dy**2
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(1.0 / dy**2)
            elif j == ny - 1:
                diag -= 1.0 / dy**2
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(1.0 / dy**2)
            else:
                diag -= 2.0 / dy**2
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(1.0 / dy**2)
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(1.0 / dy**2)

            rows.append(k); cols.append(k); vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return A

print("圧力ポアソン方程式の係数行列を構築中...")
A = build_pressure_matrix(nx, ny, dx, dy)
lu = spla.splu(A.tocsc())
print("LU分解完了。時間積分を開始します。")

# =====================================================================
# 4. 補助関数
# =====================================================================

def apply_velocity_bc(u, v):
    # 流入(左端): u=U_inf, v=0
    u[0, :] = U_inf
    # 流出(右端): 対流流出条件(ゼロ勾配で近似)
    u[-1, :] = u[-2, :]
    v[-1, :] = v[-2, :]
    # 上下壁(自由すべり): 法線方向速度ゼロ, 接線方向はミラー(勾配ゼロ)
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    return u, v


def apply_cylinder_mask(u, v):
    u[mask_u] = 0.0
    v[mask_v] = 0.0
    return u, v


def compute_rhs_u(u, v):
    """u方程式の移流項(風上差分)+拡散項(中心差分)"""
    rhs = np.zeros_like(u)

    # 内部(1..nx-1)のみ更新(境界はBCで上書き)
    ue = u[1:-1, :]  # 内部 u

    # x方向隣接値
    uw = u[:-2, :]
    uee = u[2:, :]

    # y方向: 上下端はゴーストとしてミラー(勾配ゼロ)で近似
    u_pad = np.empty((u.shape[0], u.shape[1] + 2))
    u_pad[:, 1:-1] = u
    u_pad[:, 0] = u[:, 0]
    u_pad[:, -1] = u[:, -1]

    un = u_pad[1:-1, 2:]
    us = u_pad[1:-1, :-2]
    d2udy2 = (un - 2 * ue + us) / dy**2
    d2udx2 = (uee - 2 * ue + uw) / dx**2

    # v を u節点上に補間: まずy方向平均でu節点のy位置に合わせ((nx,ny)),
    # 次にx方向平均でu内部節点(i=1..nx-1)のx位置に合わせる
    v_y_avg = 0.5 * (v[:, :-1] + v[:, 1:])          # (nx, ny)  x位置は(i+0.5)dx
    v_at_u = 0.5 * (v_y_avg[:-1, :] + v_y_avg[1:, :])  # (nx-1, ny) x位置はi*dx (i=1..nx-1)

    # 移流項: 風上差分(1次)。局所セル・ペクレ数が大きいため中心差分は不安定。
    dudx_up = np.where(ue >= 0, (ue - uw) / dx, (uee - ue) / dx)
    dudy_up = np.where(v_at_u >= 0, (ue - us) / dy, (un - ue) / dy)

    conv = ue * dudx_up + v_at_u * dudy_up
    diff = nu * (d2udx2 + d2udy2)

    rhs[1:-1, :] = -conv + diff
    return rhs


def compute_rhs_v(u, v):
    """v方程式の移流項(風上差分)+拡散項(中心差分)"""
    rhs = np.zeros_like(v)

    ve = v[:, 1:-1]
    vs = v[:, :-2]
    vn = v[:, 2:]

    d2vdy2 = (vn - 2 * ve + vs) / dy**2

    v_pad = np.empty((v.shape[0] + 2, v.shape[1]))
    v_pad[1:-1, :] = v
    v_pad[0, :] = v[0, :]
    v_pad[-1, :] = v[-1, :]
    vw = v_pad[:-2, 1:-1]
    vee = v_pad[2:, 1:-1]
    d2vdx2 = (vee - 2 * ve + vw) / dx**2

    # u を v節点上に補間: まずx方向平均でv節点のx位置に合わせ((nx,ny)),
    # 次にy方向平均でv内部節点(j=1..ny-1)のy位置に合わせる
    u_x_avg = 0.5 * (u[:-1, :] + u[1:, :])          # (nx, ny)  y位置は(j+0.5)dy
    u_at_v = 0.5 * (u_x_avg[:, :-1] + u_x_avg[:, 1:])  # (nx, ny-1) y位置はj*dy (j=1..ny-1)

    # 移流項: 風上差分(1次)
    dvdx_up = np.where(u_at_v >= 0, (ve - vw) / dx, (vee - ve) / dx)
    dvdy_up = np.where(ve >= 0, (ve - vs) / dy, (vn - ve) / dy)

    conv = u_at_v * dvdx_up + ve * dvdy_up
    diff = nu * (d2vdx2 + d2vdy2)

    rhs[:, 1:-1] = -conv + diff
    return rhs


def divergence(u, v):
    dudx = (u[1:, :] - u[:-1, :]) / dx
    dvdy = (v[:, 1:] - v[:, :-1]) / dy
    return dudx + dvdy


def vorticity_at_cell_center(u, v):
    """渦度 omega = dv/dx - du/dy をセル中心で評価"""
    # v, u をそれぞれセル中心へ補間((nx, ny), p格子と同じ位置)
    v_cc = 0.5 * (v[:, :-1] + v[:, 1:])  # (nx, ny)
    u_cc = 0.5 * (u[:-1, :] + u[1:, :])  # (nx, ny)

    dvdx = np.zeros_like(v_cc)
    dvdx[1:-1, :] = (v_cc[2:, :] - v_cc[:-2, :]) / (2 * dx)
    dvdx[0, :] = (v_cc[1, :] - v_cc[0, :]) / dx
    dvdx[-1, :] = (v_cc[-1, :] - v_cc[-2, :]) / dx

    dudy = np.zeros_like(u_cc)
    dudy[:, 1:-1] = (u_cc[:, 2:] - u_cc[:, :-2]) / (2 * dy)
    dudy[:, 0] = (u_cc[:, 1] - u_cc[:, 0]) / dy
    dudy[:, -1] = (u_cc[:, -1] - u_cc[:, -2]) / dy

    return dvdx - dudy


# =====================================================================
# 5. EnSight Gold 出力関数
# =====================================================================

def write_geo_file(path):
    """構造格子(iblank付き)のジオメトリファイルを書き出す(1回のみ)"""
    nz = 1
    iblank = np.where(mask_p, 0, 1).astype(np.int32)  # 0=blanked(固体内), 1=active

    with open(path, "w") as f:
        f.write("EnSight Gold geometry file - Karman vortex around a cylinder\n")
        f.write("2D flow past a circular cylinder (MAC staggered grid)\n")
        f.write("node id off\n")
        f.write("element id off\n")
        f.write("part\n")
        f.write("%10d\n" % 1)
        f.write("cylinder_wake\n")
        f.write("block iblanked\n")
        f.write("%10d%10d%10d\n" % (nx, ny, nz))

        # x, y, z 座標(節点=セル中心とみなす), Fortran(列優先: i最速)順
        Zc = np.zeros_like(Xc)
        for arr in (Xc, Yc, Zc):
            flat = arr.flatten(order="F")
            for val in flat:
                f.write("%12.5e\n" % val)

        for val in iblank.flatten(order="F"):
            f.write("%10d\n" % int(val))


def write_scalar_file(path, field, description):
    with open(path, "w") as f:
        f.write(f"{description}\n")
        f.write("part\n")
        f.write("%10d\n" % 1)
        f.write("block\n")
        # 1値ずつのPythonループはボトルネックになるため、np.savetxtでまとめて出力
        np.savetxt(f, field.flatten(order="F"), fmt="%12.5e")


def write_vector_file(path, u_field, v_field, description):
    w_field = np.zeros_like(u_field)
    data = np.concatenate([
        u_field.flatten(order="F"),
        v_field.flatten(order="F"),
        w_field.flatten(order="F"),
    ])
    with open(path, "w") as f:
        f.write(f"{description}\n")
        f.write("part\n")
        f.write("%10d\n" % 1)
        f.write("block\n")
        np.savetxt(f, data, fmt="%12.5e")


def write_frame_outputs(out_dir, frame, p_out, omega, u_cc, v_cc):
    """1フレーム分(pressure/vorticity/velocity)のEnSightファイルをまとめて書き出す。
    ProcessPoolExecutorのワーカープロセスから呼ばれる、独立した(副作用が引数と戻り値の
    やり取りだけで完結する)トップレベル関数である必要がある。
    """
    write_scalar_file(os.path.join(out_dir, f"pressure.{frame:04d}"), p_out, "pressure")
    write_scalar_file(os.path.join(out_dir, f"vorticity.{frame:04d}"), omega, "vorticity")
    write_vector_file(os.path.join(out_dir, f"velocity.{frame:04d}"), u_cc, v_cc, "velocity")
    return frame


def write_case_file(path, n_frames, dt_out, prefix_p, prefix_w, prefix_v, geo_name):
    with open(path, "w") as f:
        f.write("FORMAT\n")
        f.write("type: ensight gold\n\n")
        f.write("GEOMETRY\n")
        f.write(f"model: {geo_name}\n\n")
        f.write("VARIABLE\n")
        f.write(f"scalar per node: 1 pressure {prefix_p}.****\n")
        f.write(f"scalar per node: 1 vorticity {prefix_w}.****\n")
        f.write(f"vector per node: 1 velocity {prefix_v}.****\n\n")
        f.write("TIME\n")
        f.write("time set: 1\n")
        f.write(f"number of steps: {n_frames}\n")
        f.write("filename start number: 0\n")
        f.write("filename increment: 1\n")
        f.write("time values:\n")
        for k in range(n_frames):
            f.write("%12.5e\n" % (k * dt_out))


# ジオメトリは時間不変なので最初に一度だけ書き出す
write_geo_file(os.path.join(out_dir, "cylinder.geo"))

# =====================================================================
# 6. 時間積分ループ
# =====================================================================

u, v = apply_velocity_bc(u, v)
u, v = apply_cylinder_mask(u, v)

frame = 0
t = 0.0
t0 = time.time()

if _MP_CTX is not None:
    io_executor = ProcessPoolExecutor(max_workers=N_IO_WORKERS, mp_context=_MP_CTX)
    print(f"出力I/Oを{N_IO_WORKERS}並列ワーカー(fork)で計算とオーバーラップさせます。")
else:
    io_executor = None
    print("この環境ではfork方式が使えないため、出力I/Oは逐次実行にフォールバックします。")
io_futures = []

for step in range(n_steps):
    # --- 1) 移流+拡散項による中間速度の計算(陽的オイラー法) ---
    rhs_u = compute_rhs_u(u, v)
    rhs_v = compute_rhs_v(u, v)

    u_star = u + dt * rhs_u
    v_star = v + dt * rhs_v

    u_star, v_star = apply_velocity_bc(u_star, v_star)
    u_star, v_star = apply_cylinder_mask(u_star, v_star)

    # --- 2) 圧力ポアソン方程式を解く ---
    div_star = divergence(u_star, v_star)
    rhs_p = (rho / dt) * div_star
    # build_pressure_matrix の idx(i,j)=i*ny+j はC順(行優先)なので、
    # ここも同じ順序で平坦化・復元する必要がある
    p_new = lu.solve(rhs_p.flatten(order="C")).reshape((nx, ny), order="C")

    # --- 3) 速度場を圧力勾配で補正 ---
    dpdx = np.zeros_like(u)
    dpdx[1:-1, :] = (p_new[1:, :] - p_new[:-1, :]) / dx
    dpdy = np.zeros_like(v)
    dpdy[:, 1:-1] = (p_new[:, 1:] - p_new[:, :-1]) / dy

    u = u_star - dt / rho * dpdx
    v = v_star - dt / rho * dpdy

    u, v = apply_velocity_bc(u, v)
    u, v = apply_cylinder_mask(u, v)
    p = p_new

    t += dt

    # --- 4) EnSight Gold 出力 ---
    if step % output_every == 0:
        u_cc = 0.5 * (u[:-1, :] + u[1:, :])
        v_cc = 0.5 * (v[:, :-1] + v[:, 1:])
        u_cc[mask_p] = 0.0
        v_cc[mask_p] = 0.0
        omega = vorticity_at_cell_center(u, v)
        omega[mask_p] = 0.0
        p_out = p.copy()
        p_out[mask_p] = 0.0

        # メインプロセスは次ステップの計算に進み、ファイル書き出しはワーカープロセスに
        # 任せる(配列はここでコピー済みなので、以降uやpを更新しても書き出し内容に影響しない)。
        if io_executor is not None:
            io_futures.append(
                io_executor.submit(write_frame_outputs, out_dir, frame, p_out, omega, u_cc, v_cc)
            )
        else:
            write_frame_outputs(out_dir, frame, p_out, omega, u_cc, v_cc)

        if frame % 10 == 0:
            elapsed = time.time() - t0
            print(f"  step={step:6d}/{n_steps}  t={t:6.2f}  frame={frame:4d}  "
                  f"elapsed={elapsed:6.1f}s")
        frame += 1

# 全フレームの書き出し完了を待つ(例外があればここで送出される)
for fut in io_futures:
    fut.result()
if io_executor is not None:
    io_executor.shutdown(wait=True)

n_frames = frame
dt_out = dt_snapshot  # = output_every * dt (厳密に一致するよう時間刻みを設計済み)
assert abs(dt_out - output_every * dt) < 1e-12

write_case_file(
    os.path.join(out_dir, "cylinder.case"),
    n_frames, dt_out,
    "pressure", "vorticity", "velocity",
    "cylinder.geo",
)

# DMDベンチマーク用メタ情報(スナップショット間隔dtを明示)
with open(os.path.join(out_dir, "dmd_info.txt"), "w") as f:
    f.write("DMD snapshot metadata\n")
    f.write("======================\n")
    f.write(f"dt (snapshot interval, non-dimensional time D/U_inf) = {dt_snapshot:.6f}\n")
    f.write(f"number of snapshots                                   = {n_frames}\n")
    f.write(f"t_start                                                = 0.0\n")
    f.write(f"t_end                                                  = {T_end:.6f}\n")
    f.write(f"integration time step (CFD internal, substeps per snapshot={output_every}) = {dt:.6f}\n")
    f.write(f"Reynolds number Re = {Re}\n")
    f.write(f"cylinder diameter D = {D}\n")
    f.write(f"free-stream velocity U_inf = {U_inf}\n")
    f.write("Snapshot k corresponds to file suffix %04d (k=0..n_frames-1) and time t_k = k * dt\n")

print(f"\n完了: {n_frames} フレームを {out_dir}/ に出力しました。")
print(f"DMDスナップショット間隔 dt = {dt_snapshot} (厳密, 全フレーム共通) "
      f"-> {out_dir}/dmd_info.txt に記録")
print(f"総計算時間: {time.time() - t0:.1f} 秒")
print(f"ParaView等で {out_dir}/cylinder.case を開いてください。")