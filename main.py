import os
import glob
import numpy as np
import h5py
import shutil
import matplotlib
# 画面ポップアップを禁止し、画像が真っ白になるのを防ぐ
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from pydmd import DMD, SpDMD
from pydmd.plotter import plot_eigs, plot_summary

def write_vtk_universal_2d(filename, orig_shape, data_dict, is_vector):
    """
    (100, 400) などの構造格子データに、スカラーまたはベクトルを埋め込んでVTKにする汎用関数
    """
    ny, nx = orig_shape
    nz = 1
    n_points = nx * ny * nz

    # 格子点の座標を作成
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    with open(filename, 'w') as f:
        # 1. ヘッダー情報
        f.write("# vtk DataFile Version 3.0\n")
        f.write("DMD Reconstructed Flow Field (Universal)\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        
        # 2. 座標データの書き込み
        f.write(f"POINTS {n_points} float\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
        # 3. データセクション
        f.write(f"\nPOINT_DATA {n_points}\n")
        
        for field_name, val in data_dict.items():
            if is_vector:
                # 【ベクトルモード】 X成分とY成分を矢印として書き出し
                vec_x, vec_y = val
                vx = np.real(vec_x).flatten()
                vy = np.real(vec_y).flatten()
                vz = np.zeros_like(vx)
                
                f.write(f"VECTORS {field_name} float\n")
                for i in range(n_points):
                    f.write(f"{vx[i]:.6f} {vy[i]:.6f} {vz[i]:.6f}\n")
            else:
                # 【スカラーモード】 圧力をそのまま単一の値として書き出し
                scalar_val = np.real(val).flatten()
                f.write(f"SCALARS {field_name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for i in range(n_points):
                    f.write(f"{scalar_val[i]:.6f}\n")

def execute_universal_pydmd(file_pattern, base_name, zone_name, array_names, n_greedy_modes, dt, use_spdmd=False):
    """
    スカラー量(1変数)・ベクトル量(複数変数)のどちらのCFDデータにも自動対応する汎用DMD解析
    """
    # 変数の数を自動判別 (1つならスカラーモード、2つ以上ならベクトルモード)
    n_variables = len(array_names)
    is_vector = n_variables >= 2
    print(f"📊 [モード判別] 変数設定数: {n_variables} -> {'ベクトルDMD' if is_vector else 'スカラーDMD'} を実行します。")

    # 1. 時系列ファイルの収集
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        raise FileNotFoundError(f"ファイルが見つかりません: {file_pattern}")
    
    n_snapshots = len(file_list)
    print(f"📦 解析ファイル数: {n_snapshots}")

    with h5py.File(file_list[0], 'r') as f:
        data_paths = [f"{base_name}/{zone_name}/FlowSolution/{name}/ data" for name in array_names]
        for p in data_paths:
            if p not in f:
                raise ValueError(f"指定されたパス '{p}' が見つかりません。")
        orig_data_shape = f[data_paths[0]][:].shape
        print(f"📐 1変数あたりの空間形状: {orig_data_shape}")

    print("⏳ データ行列を構築中...")
    snapshots = []
    for file_path in file_list:
        with h5py.File(file_path, 'r') as f:
            # 複数変数があれば縦に連結、1つならそのままフラット化
            combined_frame = np.concatenate([f[p][:].flatten() for p in data_paths])
            snapshots.append(combined_frame)
    
    X_all = np.column_stack(snapshots)
    
    # 2. DMD モデルのフィッティング
    if use_spdmd:
        dmd_model = SpDMD(svd_rank=0.99, tlsq_rank=0, exact=True, opt=True, rho=10.0)
    else:
        dmd_model = DMD(svd_rank=0.99, tlsq_rank=0, exact=True, opt=True, sorted_eigs='abs')
    
    dmd_model.fit(X_all)
    
    eigenvalues = dmd_model.eigs
    Phi = dmd_model.modes
    amplitudes = dmd_model.amplitudes

    # 3. モードの抽出
    greedy_indices = np.argsort(np.abs(amplitudes))[::-1]
    selected_indices = greedy_indices[:n_greedy_modes] if not use_spdmd else np.arange(len(amplitudes))

    # --- グラフプロットセクション（累積値ハック） ---
    print("\n📊 重要プロットを生成中...")
    plot_dir = "C:/Users/Miyashitaryou/Downloads/dmd_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_eigs(dmd_model)
    plt.title("DMD Eigenvalues and Unit Circle")
    plt.savefig(os.path.join(plot_dir, "eigenvalues_unit_circle.png"), dpi=150)
    plt.close()
    
    if not use_spdmd and len(selected_indices) > 0:
        plot_summary(dmd_model, index_modes=tuple(selected_indices))
    else:
        plot_summary(dmd_model)
        
    fig = plt.gcf()
    axes = fig.get_axes()
    if len(axes) > 0:
        if hasattr(dmd_model, 'singular_values') and dmd_model.singular_values is not None:
            s = dmd_model.singular_values
        else:
            _, s, _ = np.linalg.svd(X_all, full_matrices=False)
            
        ax_svd = axes[0]
        ax_svd.clear()
        variance = (s ** 2) / np.sum(s ** 2) * 100
        cumulative_variance = np.cumsum(variance)
        ax_svd.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='tab:orange', lw=2)
        n_fit = len(selected_indices) if not use_spdmd else len(dmd_model.eigs)
        ax_svd.plot(range(1, min(n_fit + 1, len(cumulative_variance) + 1)), cumulative_variance[:n_fit], 'o', color='tab:orange', markersize=8, markeredgecolor='k')
        ax_svd.set_title("Cumulative Singular Values")
        ax_svd.set_ylabel("% Cumulative Variance")
        ax_svd.set_xlabel("Rank")
        ax_svd.set_ylim(-5, 105)
        ax_svd.grid(True, linestyle='--')

    plt.savefig(os.path.join(plot_dir, "dmd_summary.png"), dpi=150)
    plt.close()

    # 4. 時系列データの低ランク再構築 (Reconstruction)
    print("\n🎬 モード別の時系列データを計算中...")
    Phi_selected = Phi[:, selected_indices]
    eigs_selected = eigenvalues[selected_indices]
    b_selected = amplitudes[selected_indices]
    
    single_component_size = np.prod(orig_data_shape)
    vtk_output_dir = "C:/Users/Miyashitaryou/Downloads/dmd_reconstructed_vtk"
    os.makedirs(vtk_output_dir, exist_ok=True)
    print(f"💡 統合VTKを '{vtk_output_dir}' にエクスポート中...")

    # 各タイムステップごとにファイルを作成
    for t in range(n_snapshots):
        data_dict = {}
        
        # ① 選ばれた全上位モードの合計（全体の再構築）
        total_flow = np.zeros(X_all.shape[0], dtype=complex)
        for r_idx in range(len(selected_indices)):
            total_flow += Phi_selected[:, r_idx] * (eigs_selected[r_idx] ** t) * b_selected[r_idx]
            
        if is_vector:
            vx_total = total_flow[0 : single_component_size].reshape(orig_data_shape)
            vy_total = total_flow[single_component_size : 2 * single_component_size].reshape(orig_data_shape)
            data_dict["Velocity_Total_Recon"] = (vx_total, vy_total)
        else:
            s_total = total_flow.reshape(orig_data_shape)
            data_dict[f"{array_names[0]}_Total_Recon"] = s_total
        
        # ② 各モード単体の時系列変化
        for rank_num, idx in enumerate(selected_indices):
            single_mode_flow = Phi[:, idx] * (eigenvalues[idx] ** t) * amplitudes[idx]
            
            if is_vector:
                vx_single = single_mode_flow[0 : single_component_size].reshape(orig_data_shape)
                vy_single = single_mode_flow[single_component_size : 2 * single_component_size].reshape(orig_data_shape)
                data_dict[f"Velocity_Rank{rank_num+1}_Mode{idx}"] = (vx_single, vy_single)
            else:
                s_single = single_mode_flow.reshape(orig_data_shape)
                data_dict[f"{array_names[0]}_Rank{rank_num+1}_Mode{idx}"] = s_single
            
        # VTKファイルへ一括書き出し
        vtk_filename = os.path.join(vtk_output_dir, f"recon_flow_universal_{t:03d}.vtk")
        write_vtk_universal_2d(vtk_filename, orig_data_shape, data_dict, is_vector)
        
    print(f"✅ すべての統合VTKファイル（計 {n_snapshots} ステップ）の出力が完了しました。")

# --- 実行セクション ---
if __name__ == "__main__":
    FILE_PATTERN = "C:/Users/Miyashitaryou/vortex_cgns_results/vortex_*.cgns"
    BASE_NAME = "Base"
    ZONE_NAME = "Zone1"
    
    # 💡 テスト用設定切り替え
    # 【パターンA：圧力スカラー量で試す場合】
    # ARRAY_NAMES = ["Pressure"] 
    
    # 【パターンB：従来の流速ベクトル量で試す場合】
    ARRAY_NAMES = ["VelocityX", "VelocityY"] 
    
    DT = 0.00083 * 300 
    USE_SPDMD = False        
    N_GREEDY_MODES = 5      
    
    files = sorted(glob.glob(FILE_PATTERN))
    if len(files) > 0:
        execute_universal_pydmd(
            file_pattern=FILE_PATTERN,
            base_name=BASE_NAME,
            zone_name=ZONE_NAME,
            array_names=ARRAY_NAMES,
            n_greedy_modes=N_GREEDY_MODES,
            dt=DT,
            use_spdmd=USE_SPDMD
        )
