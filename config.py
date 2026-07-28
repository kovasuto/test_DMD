"""
設定ファイル
------------
このファイルの値を編集してから run_dmd_pipeline.py を実行してください。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


@dataclass
class Config:
    # ------------------------------------------------------------------
    # 1. 入力データ (EnSight Gold)
    # ------------------------------------------------------------------
    # STAR-CCM+等からエクスポートしたEnSight Goldのcaseファイル(*.case / *.encas)
    ensight_case_path: str = r"C:\Users\Miyashitaryou\Downloads\ensight_output\cylinder.case"

    # ケース内に複数パート(part)がある場合、対象パート名を指定
    # (STAR-CCM+側で後流ROIだけを別パートとしてエクスポート済みなら
    #  ここでさらに絞る必要はない。None なら全パートを対象にする)
    target_part_name: Optional[str] = None

    # 抽出したい変数名 (EnSight側でエクスポートした変数名と一致させる)
    # 例: "Vorticity_Z", "Velocity", "Pressure" など
    variable_name: str = "pressure"

    # ベクトル量の場合、成分を1つ選ぶ (0=x, 1=y, 2=z)。スカラー量ならNoneのまま。
    vector_component: Optional[int] = None

    # ------------------------------------------------------------------
    # 2. ROI(関心領域)によるさらなる絞り込み (任意、Noneなら絞り込まない)
    #    STAR-CCM+側で既にROIパートを切り出し済みなら不要
    # ------------------------------------------------------------------
    roi_bounds: Optional[Tuple[float, float, float, float, float, float]] = None
    # 例: (xmin, xmax, ymin, ymax, zmin, zmax)

    # ------------------------------------------------------------------
    # 3. 時間方向のサンプリング設定
    # ------------------------------------------------------------------
    # EnSightにエクスポートされているタイムステップ間隔 [s]
    # (CFDソルバーのΔtそのものではなく、既に間引かれたエクスポート間隔の場合が多い。
    #  実際のCFD時間刻み幅で全ステップ出力している場合はソルバーのΔtを入れる)
    dt_export: float = 2.0e-1

    # DMDに使いたい目標サンプリング周波数 [Hz]
    # dt_exportに対する間引き係数(stride)は自動計算される
    fs_target: float = 150.0

    # 使用する解析対象の物理時間窓 [s] (計算可能なCFD時間長に合わせて設定)
    time_window: float = 30.0

    # 解析開始時刻 [s] (助走区間を除外するため、0より大きい値を推奨)
    t_start: float = 400.0

    # EnSightの読み込み・変数抽出を並列化するプロセス数。
    # None なら os.cpu_count() を自動使用。1以下を指定すると逐次処理になる。
    # 各プロセスが独立にEnSightリーダーを1回だけ開き、複数タイムステップを
    # 使い回して処理する(タスクごとの再オープンは行わない)。
    # HDF5への書き込みはメインプロセスに集約されるため、書き込み順序は
    # 時系列順のまま保たれる。
    ensight_parallel_workers: Optional[int] = None

    # ------------------------------------------------------------------
    # 3.5 dask計算(SVD/RDMD)の並列化設定
    # ------------------------------------------------------------------
    # True: dask.distributed の LocalCluster(マルチプロセス)を使う。
    #       Windowsではスレッドよりプロセスの方がGILの影響を受けずCPUを
    #       複数コアで使えるため、計算が重い場合に有効。
    # False: dask標準のスレッドスケジューラを使う(セットアップのオーバーヘッドなし)。
    use_dask_distributed: bool = True

    # dask.distributedのワーカープロセス数。None なら os.cpu_count() を自動使用。
    dask_n_workers: Optional[int] = None

    # ワーカー1プロセスあたりのスレッド数(通常1のままでよい。
    # numpyのBLASが既に内部で並列化している場合、増やすと逆に競合して遅くなることがある)
    dask_threads_per_worker: int = 1

    # ワーカー1プロセスあたりのメモリ上限 (例 "2GB")。None なら制限なし。
    dask_memory_limit: Optional[str] = None

    # ------------------------------------------------------------------
    # 4. DMD/POD設定
    # ------------------------------------------------------------------
    # True  : RDMD (dask版アウトオブコア ランダム化SVD) でPOD的な低ランク圧縮を行う。
    #         X全体を一括ロードせず、chunk単位で読みながら計算する (大規模データ向け推奨)。
    # False : 打ち切りなし (rank = min(m, n-1)) で計算。論文と同じ非打ち切り運用だが、
    #         大規模データでは事実上フルSVDに近くなり非常に重くなるため注意。
    use_pod_reduction: bool = True

    # svd_rank:
    #   0        -> 自動最適ランク推定
    #   正の整数  -> そのランク数で打ち切り
    #   0<r<1の小数 -> 累積エネルギー基準 (例 0.999) でランクを自動決定
    # use_pod_reduction=False の場合は無視され、常に打ち切りなし(-1)になる
    svd_rank: Union[int, float] = 0.999

    # RDMD用の追加パラメータ (ランダム化SVDの精度に影響)
    rdmd_oversampling: int = 10
    rdmd_power_iters: int = 2

    # SVD/DMD計算時にdask配列へ与えるチャンク幅(スナップショット数の単位)。
    # HDF5への保存チャンク(1列ずつ、ストリーミング書き込み用)とは別物。
    # 小さすぎるとチャンク数が増えてオーバーヘッド・メモリ膨張の原因になり、
    # (実際に n=294 で shape (m,1,294) の中間配列によるメモリエラーが発生した)
    # 大きすぎるとチャンクあたりのメモリ使用量が増える。
    #
    # None にすると、n_points(空間点数m)から target_chunk_mb を満たすように
    # 自動計算する(推奨)。この値は m にのみ依存し、スナップショット数nには
    # 依存しない(nが増えてもチャンクの「数」が増えるだけで、1チャンクの
    # メモリ使用量は変わらないため)。手動で固定したい場合のみ整数を指定する。
    dask_chunk_snapshots: Optional[int] = None

    # dask_chunk_snapshots=None のときの、1チャンクあたりの目標メモリ量 [MB]。
    # 目安: n_points × dask_chunk_snapshots × 4byte ≈ target_chunk_mb
    target_chunk_mb: float = 32.0

    # 平均差し引き (mean-subtracted DMD) を行うかどうか
    mean_subtraction: bool = True

    # ------------------------------------------------------------------
    # 5. 出力設定
    # ------------------------------------------------------------------
    # 相対パス: run_dmd_pipeline.py を実行したフォルダの直下に作成される。
    # Windowsで絶対パスを指定する場合は必ず r"C:\..." のように raw文字列にすること。
    work_dir: str = "dmd_work"
    snapshot_hdf5_name: str = "snapshots.h5"
    result_npz_name: str = "dmd_result.npz"
    mode_export_dir_name: str = "mode_vtk"
    # DMD標準診断プロット(特異値スペクトル・固有値単位円・振幅スペクトル)の出力先
    diagnostics_dir_name: str = "diagnostics"
    generate_diagnostics: bool = True
    # 可視化(VTK出力)するモード数は、下記 reconstruction_n_modes と同じ
    # (mode_selectionで選ばれた重要モード全てをVTK出力する)

    # ------------------------------------------------------------------
    # 6. モード重要度判定・選択 (時系列再構築に使うモードを絞り込む)
    # ------------------------------------------------------------------
    # "amplitude"         : 単純な初期振幅 |b_k| でランキング
    # "integrated_energy" : Kou-Zhang流の時間積分エネルギーでランキング (推奨デフォルト)
    # "greedy"            : 貪欲法 (Orthogonal Matching Pursuit 型)
    # "spdmd"             : Sparsity-Promoting DMD (ADMM)
    reconstruction_method: str = "greedy"

    # 再構築に使うモード数 (最終的な次数)。
    # "spdmd"かつ spdmd_auto_tune=True の場合は「目標モード数」として扱われる。
    reconstruction_n_modes: int = 10

    # --- spDMD専用パラメータ ---
    # True: reconstruction_n_modes に近い有効モード数になるよう gamma を自動探索する
    # False: spdmd_gamma を固定値として使う (多い場合は上位n_modesだけに絞る)
    spdmd_auto_tune: bool = True
    spdmd_gamma: float = 1.0
    spdmd_admm_iters: int = 200
    spdmd_rho: float = 1.0

    # ------------------------------------------------------------------
    # 7. 時系列再構築の設定
    # ------------------------------------------------------------------
    # 再構築するタイムステップ間隔 [s]。Noneなら学習時のdt(実効dt)をそのまま使う。
    reconstruction_dt: Optional[float] = None

    # 再構築を開始する時刻 [s] (学習データの開始時刻を基準とした相対時刻)
    reconstruction_t_start: float = 0.0

    # 再構築する総時間長 [s]。
    # Noneなら学習に使った時間窓 Tw と同じ長さ(訓練範囲の再現)。
    # 学習時間窓より長い値を指定すると、モデルによる将来予測(外挿)になる。
    # 注意: 外挿は特に成長モード(growth_rate>0)を含む場合、急速に発散しうる。
    reconstruction_extrapolate_duration: Optional[float] = None

    # 一度に計算・書き出すタイムステップ数(メモリ使用量の上限を決める)。
    # m(空間点数)が大きい場合は小さめに設定するとメモリを抑えられる。
    reconstruction_chunk_snapshots: int = 50

    reconstruction_output_name: str = "reconstruction.h5"

    # 再構築結果を元のスナップショットと比較して誤差を計算するか
    # (訓練データ範囲内のみ、元データへの追加dask読み込みが発生する)
    reconstruction_compute_error: bool = True
