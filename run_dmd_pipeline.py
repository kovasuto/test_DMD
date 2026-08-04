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
    pip install pyvista h5py "dask[array,distributed]" numpy scipy matplotlib --break-system-packages
    (pydmd は本パイプラインでは不要になりました)

並列処理について:
    - EnSight読み込み(Step1): config.ensight_parallel_workers で
      ProcessPoolExecutorによる並列読み込みを行う(None=CPU数を自動使用)。
    - DMD計算(Step2+3のSVD等): config.use_dask_distributed=True にすると
      dask.distributedのマルチプロセスLocalClusterを使う。
      Windowsではスレッドよりプロセスの方がGILの影響を受けずCPUを
      複数コアで使えるため、計算が重い場合に有効。

設定変更は config.py を編集してください。
    - use_pod_reduction : True  -> RDMD (POD的低ランク圧縮あり, アウトオブコア)
                          False -> 打ち切りなし (大規模データでは非推奨、動作確認用)
    - svd_rank          : ランク(整数) or 累積エネルギー基準(0〜1の小数)
    - rdmd_oversampling / rdmd_power_iters : ランダム化SVDの精度パラメータ
      (power_itersを増やすほど精度は上がるが、大規模データへのアクセス回数が増える)
    - mean_subtraction  : 平均差し引きDMDのオンオフ
    - dt_export / fs_target / time_window / t_start : サンプリング設定
    - ensight_parallel_workers / use_dask_distributed / dask_n_workers : 並列化設定
"""

import os
import time
import h5py

from config import Config
from ensight_to_hdf5 import stream_ensight_to_hdf5
from dask_rdmd import run_dask_rdmd
from dmd_analysis import analyze_modes, save_results
from export_modes import export_top_modes
from mode_selection import select_modes
from reconstruct import reconstruct_timeseries, compute_reconstruction_error
from diagnostics import generate_diagnostic_plots


def main():
    cfg = Config()
    os.makedirs(cfg.work_dir, exist_ok=True)

    t0 = time.time()
    timings = {}  # ステップ名 -> 所要秒数 (最後に内訳を表示するため)

    def _tick():
        return time.time()

    def _tock(label, t_start):
        elapsed = time.time() - t_start
        timings[label] = elapsed
        print(f"[TIME] {label}: {elapsed:.1f} s")
        return time.time()

    t_step = _tick()

    # --------------------------------------------------------------
    # Step 0: dask.distributed クラスタの起動 (config.use_dask_distributed=True の場合)
    #   これにより、Step2+3のSVD/行列積計算がマルチプロセスで並列実行される。
    #   Windowsではスレッドスケジューラよりも、GILの影響を受けないプロセス
    #   ベースのスケジューラの方がCPUを複数コア分活用しやすい。
    # --------------------------------------------------------------
    dask_client = None
    if cfg.use_dask_distributed:
        from dask.distributed import Client, LocalCluster

        n_workers = cfg.dask_n_workers or (os.cpu_count() or 1)
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=cfg.dask_threads_per_worker,
            memory_limit=cfg.dask_memory_limit,
            processes=True,
        )
        dask_client = Client(cluster)
        print(
            f"[INFO] dask.distributed クラスタ起動: "
            f"n_workers={n_workers}, threads_per_worker={cfg.dask_threads_per_worker}"
        )
        print(f"[INFO] ダッシュボード: {dask_client.dashboard_link}")
        t_step = _tock("Step0_daskクラスタ起動", t_step)

    # --------------------------------------------------------------
    # Step 1: EnSight Gold -> HDF5 (ストリーミング, ROI/変数抽出込み)
    # --------------------------------------------------------------
    h5_path_cache = os.path.join(cfg.work_dir, cfg.snapshot_hdf5_name)
    h5_path = None

    if os.path.exists(h5_path_cache):
        # 以前の実行が途中で失敗・中断していると、"X"はあっても
        # dt等のattrs(ループ完了後にしか書かれない)が欠けた不完全なファイルが
        # 残っていることがある。再利用前に必ず検証する。
        valid_cache = False
        try:
            with h5py.File(h5_path_cache, "r") as h5f_check:
                if "X" in h5f_check and "dt" in h5f_check.attrs:
                    valid_cache = True
        except OSError:
            valid_cache = False

        if valid_cache:
            print(f"[INFO] 既存のHDF5を再利用します: {h5_path_cache}")
            print(
                "       (作り直したい場合はこのファイルを削除してから再実行してください)"
            )
            h5_path = h5_path_cache
        else:
            print(
                f"[WARN] 既存のHDF5が不完全です(以前の実行が途中で失敗した可能性): "
                f"{h5_path_cache}"
            )
            print("[WARN] 作り直します。")
            os.remove(h5_path_cache)

    if h5_path is None:
        meta = stream_ensight_to_hdf5(cfg)
        h5_path = meta["h5_path"]
        t_step = _tock("Step1_EnSight読み込み+HDF5書き出し", t_step)
    else:
        t_step = _tock("Step1_EnSight読み込み(キャッシュ再利用, 実質0秒のはず)", t_step)

    # --------------------------------------------------------------
    # Step 2+3: Dask版アウトオブコア RDMD/DMD 実行
    #   (HDF5の読み込み・平均差し引き・SVD・DMD計算を全てここで行う。
    #    m×n全体を一括ロードすることはない)
    # --------------------------------------------------------------
    result = run_dask_rdmd(h5_path, cfg)
    t_step = _tock("Step2+3_RDMD計算(SVD+固有分解+モード計算)", t_step)

    # --------------------------------------------------------------
    # Step 4: DMDスペクトル一覧表示 (全モードの周波数・成長率・振幅)
    #   特定周波数への絞り込みは行わない。
    # --------------------------------------------------------------
    analysis = analyze_modes(result, cfg)
    t_step = _tock("Step4_スペクトル一覧表示", t_step)

    # --------------------------------------------------------------
    # Step 5: モード重要度判定・選択
    #   (amplitude / integrated_energy / greedy / spdmd から選択)
    #   全て縮約空間(Y_reduced, W: r×r程度)上で行うため、
    #   物理空間の巨大データへは一切再アクセスしない。
    #   ここで選ばれたモードが、以降のVTK出力・時系列再構築で使う「重要モード」。
    # --------------------------------------------------------------
    selection = select_modes(result, cfg)
    t_step = _tock(f"Step5_モード選択({cfg.reconstruction_method})", t_step)

    # --------------------------------------------------------------
    # Step 6: 結果保存 (npz) + 選択モードのVTK出力
    # --------------------------------------------------------------
    result_path = os.path.join(cfg.work_dir, cfg.result_npz_name)
    save_results(result, analysis, selection, cfg, result_path)
    export_top_modes(result, selection, cfg)
    t_step = _tock("Step6_結果保存+VTK出力", t_step)

    # --------------------------------------------------------------
    # Step 6.5: DMD標準診断プロット
    #   (特異値スペクトル・累積エネルギー、固有値の単位円プロット、
    #    モード振幅スペクトル)
    #   全てr(ランク)程度の小さい配列だけを使うため、計算コストは軽い。
    # --------------------------------------------------------------
    if cfg.generate_diagnostics:
        generate_diagnostic_plots(result, selection, cfg)
        t_step = _tock("Step6.5_診断プロット生成", t_step)

    # --------------------------------------------------------------
    # Step 7: 選択モードによる時系列再構築 (HDF5へストリーミング書き出し)
    # --------------------------------------------------------------
    recon_meta = reconstruct_timeseries(result, selection, cfg)
    t_step = _tock("Step7_時系列再構築", t_step)

    if cfg.reconstruction_compute_error:
        compute_reconstruction_error(result, selection, h5_path, cfg)
        t_step = _tock("Step8_再構築誤差評価(元データへの再読込あり)", t_step)

    total = time.time() - t0
    print(f"\n[INFO] 使用ランク r = {result['rank']}")
    print(f"[INFO] 再構築ファイル: {recon_meta['out_path']}")

    print("\n[処理時間内訳] (合計に対する割合)")
    for label, elapsed in timings.items():
        pct = 100.0 * elapsed / total if total > 0 else 0.0
        print(f"  {label:<55}: {elapsed:>8.1f} s  ({pct:5.1f} %)")
    print(f"[INFO] 総処理時間: {total:.1f} s")

    if dask_client is not None:
        dask_client.close()


if __name__ == "__main__":
    main()
