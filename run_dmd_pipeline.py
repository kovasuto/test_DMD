"""
EnSight Gold -> HDF5(ストリーミング) -> Dask版アウトオブコア RDMD/DMD
(POD前処理オンオフ可) -> モード解析・VTK出力

前バージョンとの違い:
    旧: dmd_analysis.load_snapshot_matrix() が h5f["X"][:] で全データを一括ロードし、
        pydmd.RDMD/DMD に渡していた -> 大規模データではここが律速・メモリ超過の原因になる。
    新: dask_rdmd.run_dask_rdmd() が HDF5上のXをdask.arrayとしてchunk単位のまま扱い、
        SVD計算・行列積を全てdask経由で遅延評価することで、
        どの時点でも m×n 全体を同時にメモリへ展開しない。

実行方法:
    python run_dmd_pipeline.py

事前準備 (このスクリプトを動かす環境で):
    pip install pyvista h5py "dask[array]" numpy scipy --break-system-packages
    (pydmd は本パイプラインでは不要になりました)

設定変更は config.py を編集してください。
    - use_pod_reduction : True  -> RDMD (POD的低ランク圧縮あり, アウトオブコア)
                          False -> 打ち切りなし (大規模データでは非推奨、動作確認用)
    - svd_rank          : ランク(整数) or 累積エネルギー基準(0〜1の小数)
    - rdmd_oversampling / rdmd_power_iters : ランダム化SVDの精度パラメータ
      (power_itersを増やすほど精度は上がるが、大規模データへのアクセス回数が増える)
    - mean_subtraction  : 平均差し引きDMDのオンオフ
    - dt_export / fs_target / time_window / t_start : サンプリング設定
"""

import os
import time

from config import Config
from ensight_to_hdf5 import stream_ensight_to_hdf5
from dask_rdmd import run_dask_rdmd
from dmd_analysis import analyze_modes, save_results
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
        h5_path = h5_path_cache
    else:
        meta = stream_ensight_to_hdf5(cfg)
        h5_path = meta["h5_path"]

    # --------------------------------------------------------------
    # Step 2+3: Dask版アウトオブコア RDMD/DMD 実行
    #   (HDF5の読み込み・平均差し引き・SVD・DMD計算を全てここで行う。
    #    m×n全体を一括ロードすることはない)
    # --------------------------------------------------------------
    result = run_dask_rdmd(h5_path, cfg)

    # --------------------------------------------------------------
    # Step 4: モード解析 (目標周波数近傍の抽出)
    # --------------------------------------------------------------
    analysis = analyze_modes(result, cfg)

    # --------------------------------------------------------------
    # Step 5: 結果保存 (npz) + モード形状のVTK出力
    # --------------------------------------------------------------
    result_path = os.path.join(cfg.work_dir, cfg.result_npz_name)
    save_results(result, analysis, cfg, result_path)
    export_top_modes(result, analysis, cfg)

    print(f"[INFO] 使用ランク r = {result['rank']}")
    print(f"[INFO] 総処理時間: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
