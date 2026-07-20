"""
EnSight Gold -> HDF5(ストリーミング) -> RDMD/DMD(POD前処理オンオフ可) -> モード解析・VTK出力

実行方法:
    python run_dmd_pipeline.py

事前準備 (このスクリプトを動かす環境で):
    pip install pyvista h5py pydmd numpy scipy --break-system-packages
    (pydmdのRDMD/CDMDクラスの有無はバージョンに依存するため、
     'from pydmd import RDMD' が失敗する場合はpydmdをアップデートしてください)

設定変更は config.py を編集してください。
    - use_pod_reduction : True  -> RDMD (POD的低ランク圧縮あり)
                          False -> 標準DMD (打ち切りなし)
    - svd_rank          : ランク or 累積エネルギー基準
    - mean_subtraction  : 平均差し引きDMDのオンオフ
    - dt_export / fs_target / time_window / t_start : サンプリング設定
"""

import os
import time

from config import Config
from ensight_to_hdf5 import stream_ensight_to_hdf5
from dmd_analysis import load_snapshot_matrix, run_dmd, analyze_modes, save_results
from export_modes import export_top_modes


def main():
    cfg = Config()
    os.makedirs(cfg.work_dir, exist_ok=True)

    t0 = time.time()

    # --------------------------------------------------------------
    # Step 1: EnSight Gold -> HDF5 (ストリーミング, ROI/変数抽出込み)
    # --------------------------------------------------------------
    h5_path_cache = os.path.join(cfg.work_dir, cfg.snapshot_hdf5_name)
    if os.path.exists(h5_path_cache):
        print(f"[INFO] 既存のHDF5を再利用します: {h5_path_cache}")
        print("       (作り直したい場合はこのファイルを削除してから再実行してください)")
        meta = {"h5_path": h5_path_cache}
    else:
        meta = stream_ensight_to_hdf5(cfg)

    # --------------------------------------------------------------
    # Step 2: スナップショット行列の読み込み (+ 平均差し引き)
    # --------------------------------------------------------------
    X, coords, times, dt, mean_field = load_snapshot_matrix(meta["h5_path"], cfg)
    print(f"[INFO] X shape = {X.shape} (n_points, n_snapshots), dt = {dt:.6f} s")

    # --------------------------------------------------------------
    # Step 3: DMD/RDMD 実行 (POD前処理オンオフは config.use_pod_reduction)
    # --------------------------------------------------------------
    dmd = run_dmd(X, dt, cfg)

    # --------------------------------------------------------------
    # Step 4: モード解析 (目標周波数近傍の抽出)
    # --------------------------------------------------------------
    analysis = analyze_modes(dmd, dt, cfg)

    # --------------------------------------------------------------
    # Step 5: 結果保存 (npz) + モード形状のVTK出力
    # --------------------------------------------------------------
    result_path = os.path.join(cfg.work_dir, cfg.result_npz_name)
    save_results(dmd, analysis, coords, cfg, result_path)
    export_top_modes(dmd, analysis, coords, cfg)

    # --------------------------------------------------------------
    # Step 6: 再構成誤差 (参考指標として表示)
    # --------------------------------------------------------------
    X_reconstructed = dmd.reconstructed_data.real
    rel_error = (
        (((X_reconstructed - X) ** 2).sum() ** 0.5) / ((X ** 2).sum() ** 0.5)
    )
    print(f"[INFO] 全体再構成相対誤差: {rel_error * 100:.3f} %")

    print(f"[INFO] 総処理時間: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
