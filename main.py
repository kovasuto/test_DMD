import os
import glob
import pickle
import numpy as np
import meshio  # 💡 EnSight形式の読み込みと非構造VTKの書き出しに使用
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from pydmd import DMD
from pydmd.plotter import plot_eigs, plot_summary

def run_step1_ensight_train(case_file_path, variable_name, model_save_path):
    """
    【ステップ1】EnSight形式から時系列データを読み込み、DMDを計算してモデルを保存する
    """
    print("\n=== [ステップ1] EnSight Goldデータの読み込みとDMD学習を開始 ===")
    
    if not os.path.exists(case_file_path):
        raise FileNotFoundError(f".caseファイルが見つかりません: {case_file_path}")
    
    # 1. meshioでEnSight caseファイルを読み込む
    print(f"📖 EnSightマスターファイルを解析中: {case_file_path}")
    mesh_series = meshio.read(case_file_path)
    
    # meshioで読み込んだ時系列データのポイントデータから、対象の物理量(Pressure等)を抽出
    # ※meshioの時系列読み込み仕様はバージョンやフォーマットにより異なりますが、
    # 一般的にpoint_dataの辞書に時系列順、または辞書キーとして格納されます。
    print(f"🔍 物理量 '{variable_name}' の時系列スナップショットを探索中...")
    
    snapshots = []
    # 時系列データの数（タイムステップ数）を確認してループ
    # 一般的なmeshioのEnSight時系列読み込みでは、各ステップがリストで取得できるか、
    # もしくは `point_data` 内に 'Pressure_0', 'Pressure_1' のように展開されます。
    # ここでは一般的な時系列展開（keyにステップ名が含まれる、またはリスト構造）を想定したロジックにします
    
    # 表記規則が 'Variable_001' などの場合に対応するため、該当するキーをソートして収集
    all_keys = sorted([k for k in mesh_series.point_data.keys() if variable_name in k])
    
    if not all_keys:
        # 時系列が一つのキーにまとまっていないか、変数名そのものがキーの場合
        if variable_name in mesh_series.point_data:
            all_keys = [variable_name]
        else:
            raise ValueError(f"指定された変数 '{variable_name}' がpoint_dataに見つかりません。存在するキー: {list(mesh_series.point_data.keys())}")
            
    n_snapshots = len(all_keys)
    print(f"📦 検出されたタイムステップ数: {n_snapshots}")
    
    for key in all_keys:
        data_frame = mesh_series.point_data[key]
        # 3次元ベクトル（流速等）の場合はフラット化して結合、スカラーはそのままフラット化
        snapshots.append(data_frame.flatten())
        
    X_all = np.column_stack(snapshots)
    print(f"📐 DMDデータ行列のサイズ: {X_all.shape} (空間点数×変数ハック × ステップ数)")
    
    # 2. DMDの実行
    print("⏳ SVDおよびDMDの計算を実行中...")
    dmd_model = DMD(svd_rank=0.99, tlsq_rank=0, exact=True, opt=True, sorted_eigs='abs')
    dmd_model.fit(X_all)
    
    # 特異値の補完（プロット用）
    if not hasattr(dmd_model, 'singular_values') or dmd_model.singular_values is None:
        _, s, _ = np.linalg.svd(X_all, full_matrices=False)
        dmd_model.singular_values = s

    # 3. メッシュの幾何形状（構造）とDMDモデルを一緒に保存
    # 非構造格子は点の位置(points)と要素の繋がり(cells)が命なので、これを退避します
    save_data = {
        "dmd_model": dmd_model,
        "n_snapshots": n_snapshots,
        "variable_name": variable_name,
        "mesh_points": mesh_series.points,
        "mesh_cells": mesh_series.cells,
        "single_snapshot_shape": snapshots[0].shape # 1ステップ分のデータの長さ
    }
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"💾 非構造格子情報を含むDMDモデルを保存しました: {model_save_path}")
    print("※ メモリ上の巨大なデータ行列は解放されます。")

def run_step2_ensight_reconstruct(model_save_path, n_greedy_modes, plot_dir, vtk_output_dir):
    """
    【ステップ2】保存した非構造格子DMDモデルから、指定したモード数で非構造VTK(.vtu)を段階的に再構築する
    """
    print("\n=== [ステップ2] モデルの読み込みと非構造VTK(.vtu)再構築を開始 ===")
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"保存されたモデルが見つかりません: {model_save_path}")
        
    with open(model_save_path, 'rb') as f:
        save_data = pickle.load(f)
        
    dmd_model = save_data["dmd_model"]
    n_snapshots = save_data["n_snapshots"]
    variable_name = save_data["variable_name"]
    mesh_points = save_data["mesh_points"]
    mesh_cells = save_data["mesh_cells"]
    single_snapshot_shape = save_data["single_snapshot_shape"]
    
    eigenvalues = dmd_model.eigs
    Phi = dmd_model.modes
    amplitudes = dmd_model.amplitudes

    # モードをアンプリチュードの大きい順にソート
    greedy_indices = np.argsort(np.abs(amplitudes))[::-1]
    selected_indices = greedy_indices[:n_greedy_modes]

    # --- 📊 グラフプロットセクション ---
    print("📊 プロット画像を生成中...")
    os.makedirs(plot_dir, exist_ok=True)
    plot_eigs(dmd_model)
    plt.title("DMD Eigenvalues and Unit Circle")
    plt.savefig(os.path.join(plot_dir, "eigenvalues_unit_circle.png"), dpi=150)
    plt.close()
    
    plot_summary(dmd_model, index_modes=tuple(selected_indices))
    fig = plt.gcf()
    axes = fig.get_axes()
    if len(axes) > 0 and hasattr(dmd_model, 'singular_values'):
        ax_svd = axes[0]
        ax_svd.clear()
        s = dmd_model.singular_values
        variance = (s ** 2) / np.sum(s ** 2) * 100
        cumulative_variance = np.cumsum(variance)
        ax_svd.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='tab:orange', lw=2)
        ax_svd.plot(range(1, min(n_greedy_modes + 1, len(cumulative_variance) + 1)), cumulative_variance[:n_greedy_modes], 'o', color='tab:orange', markersize=8, markeredgecolor='k')
        ax_svd.set_title("Cumulative Singular Values")
        ax_svd.set_ylabel("% Cumulative Variance")
        ax_svd.set_xlabel("Rank")
        ax_svd.set_ylim(-5, 105)
        ax_svd.grid(True, linestyle='--')
    plt.savefig(os.path.join(plot_dir, "dmd_summary.png"), dpi=150)
    plt.close()

    # --- 🎬 時系列データの再構築と非構造VTK(VTU)エクスポート ---
    print(f"🎬 上位 {n_greedy_modes} モードを使って非構造格子(VTU)を再構築中...")
    Phi_selected = Phi[:, selected_indices]
    eigs_selected = eigenvalues[selected_indices]
    b_selected = amplitudes[selected_indices]
    
    os.makedirs(vtk_output_dir, exist_ok=True)

    # ベクトルかスカラーかの判定（データ長と格子点数の関係から推測）
    n_points = mesh_points.shape[0]
    is_vector = single_snapshot_shape[0] == n_points * 3

    for t in range(n_snapshots):
        # meshio用のpoint_data辞書を作成
        point_data_dict = {}
        
        # ① 選ばれた全上位モードの合計（再構築全流場）
        total_flow = np.zeros(Phi.shape[0], dtype=complex)
        for r_idx in range(len(selected_indices)):
            total_flow += Phi_selected[:, r_idx] * (eigs_selected[r_idx] ** t) * b_selected[r_idx]
            
        real_total = np.real(total_flow)
        if is_vector:
            point_data_dict[f"{variable_name}_Total_Recon"] = real_total.reshape((n_points, 3))
        else:
            point_data_dict[f"{variable_name}_Total_Recon"] = real_total
        
        # ② 各モード単体の時系列変化
        for rank_num, idx in enumerate(selected_indices):
            single_mode_flow = Phi[:, idx] * (eigenvalues[idx] ** t) * amplitudes[idx]
            real_single = np.real(single_mode_flow)
            if is_vector:
                point_data_dict[f"{variable_name}_Rank{rank_num+1}_Mode{idx}"] = real_single.reshape((n_points, 3))
            else:
                point_data_dict[f"{variable_name}_Rank{rank_num+1}_Mode{idx}"] = real_single
            
            # 💡 モードの「形状（位相を含んだ複素数）」そのものを確認したい場合（おまけ情報）
            # point_data_dict[f"{variable_name}_Mode{idx}_Spatial_Real"] = np.real(Phi[:, idx])
            
        # 💡 meshioを使って、STAR-CCM+の非構造格子のトポロジーを保ったまま.vtuファイルに書き出す
        vtu_filename = os.path.join(vtk_output_dir, f"recon_unstructured_{t:03d}.vtu")
        
        out_mesh = meshio.Mesh(
            points=mesh_points,
            cells=mesh_cells,
            point_data=point_data_dict
        )
        meshio.write(vtu_filename, out_mesh)
        
    print(f"✅ すべての非構造VTKファイル（.vtu 計 {n_snapshots} ステップ）の出力が完了しました。")

# --- 実行制御セクション ---
if __name__ == "__main__":
    # 💡 STAR-CCM+からエクスポートしたEnSight Goldのマスターファイルを指定
    CASE_FILE_PATH = "C:/Users/Miyashitaryou/starccm_exports/post_flow.case"
    
    # 💡 解析したい変数名（STAR-CCM+での出力名。例: "Pressure", "Velocity" など）
    # スカラー（Pressure等）でも、ベクトル（Velocity等）でも、内部で自動判定して対応します
    VARIABLE_NAME = "Pressure" 
    
    # 中間モデルおよび結果の出力先
    MODEL_SAVE_PATH = "C:/Users/Miyashitaryou/Downloads/dmd_models/starccm_ensight_dmd.pkl"
    PLOT_DIR = "C:/Users/Miyashitaryou/Downloads/dmd_plots"
    VTK_OUTPUT_DIR = "C:/Users/Miyashitaryou/Downloads/dmd_reconstructed_vtk"
    
    RUN_STEP_1 = True   # 初回計算時のみTrue
    RUN_STEP_2 = True   # モード数を変えて可視化ファイルだけ出し直したい時はここだけTrue
    
    if RUN_STEP_1:
        run_step1_train_and_save = run_step1_ensight_train(
            case_file_path=CASE_FILE_PATH,
            variable_name=VARIABLE_NAME,
            model_save_path=MODEL_SAVE_PATH
        )
        
    if RUN_STEP_2:
        N_GREEDY_MODES = 3  # ParaViewに出力したい主要なモード数
        run_step2_ensight_reconstruct(
            model_save_path=MODEL_SAVE_PATH,
            n_greedy_modes=N_GREEDY_MODES,
            plot_dir=PLOT_DIR,
            vtk_output_dir=VTK_OUTPUT_DIR
        )
