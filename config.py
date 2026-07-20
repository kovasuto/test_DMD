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
    ensight_case_path: str = "/path/to/export/hustler_wake.case"

    # ケース内に複数パート(part)がある場合、対象パート名を指定
    # (STAR-CCM+側で後流ROIだけを別パートとしてエクスポート済みなら
    #  ここでさらに絞る必要はない。None なら全パートを対象にする)
    target_part_name: Optional[str] = None

    # 抽出したい変数名 (EnSight側でエクスポートした変数名と一致させる)
    # 例: "Vorticity_Z", "Velocity", "Pressure" など
    variable_name: str = "Vorticity_Z"

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
    dt_export: float = 1.0e-3

    # DMDに使いたい目標サンプリング周波数 [Hz]
    # dt_exportに対する間引き係数(stride)は自動計算される
    fs_target: float = 150.0

    # 使用する解析対象の物理時間窓 [s] (計算可能なCFD時間長に合わせて設定)
    time_window: float = 15.0

    # 解析開始時刻 [s] (助走区間を除外するため、0より大きい値を推奨)
    t_start: float = 2.0

    # ------------------------------------------------------------------
    # 4. DMD/POD設定
    # ------------------------------------------------------------------
    # True  : RDMD (ランダム化SVD) でPOD的な低ランク圧縮を行う
    # False : 標準DMD (打ち切りなし, svd_rank=-1) を使う。論文と同じ非打ち切り運用。
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

    # 平均差し引き (mean-subtracted DMD) を行うかどうか
    mean_subtraction: bool = True

    # 注目したい周波数 [Hz] とその許容誤差 [Hz] (結果の絞り込み表示に使用)
    target_frequency: float = 1.0
    frequency_tolerance: float = 0.3

    # ------------------------------------------------------------------
    # 5. 出力設定
    # ------------------------------------------------------------------
    work_dir: str = "/home/claude/dmd_work"
    snapshot_hdf5_name: str = "snapshots.h5"
    result_npz_name: str = "dmd_result.npz"
    mode_export_dir_name: str = "mode_vtk"

    # 上位いくつのモード(周波数一致順)を可視化用にVTK出力するか
    n_modes_to_export: int = 5
