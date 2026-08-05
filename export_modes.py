"""
DMDモード形状(複素数)の実部・虚部を、セル接続情報を保持したVTK
(UnstructuredGrid, .vtu)に書き出す。ParaView等で可視化し、
渦構造を目視確認するために使う。

以前のバージョンは座標だけのPolyData点群として書き出していたため、
セル(面/要素)のつながりが失われ、ParaViewでは点しか表示できなかった。
本バージョンでは、ensight_to_hdf5.ensure_reference_topology() が保存した
参照メッシュ(座標+セル接続情報)を読み込み、そこにモード値を属性として
載せることで、面として表示できるVTKを出力する。
"""

import os
import numpy as np
import pyvista as pv

from nondim import freq_to_strouhal


def export_mode_to_vtk(base_mesh: "pv.DataSet", mode_vector: np.ndarray, out_path: str, label: str):
    """
    base_mesh: 参照メッシュ(座標+セル接続情報を持つ、ensure_reference_topology()で作成)
    mode_vector: (n_points,) 複素数配列 (DMDモード形状 Φ の1列)

    注意: base_mesh は呼び出し側で使い回される前提のため、この関数内で
    point_data を上書きする(コピーは作らない)。呼び出し側は1回save()する
    たびに次のモードのデータで上書きしてよい。
    """

    base_mesh.point_data[f"{label}_real"] = mode_vector.real.astype(np.float32)
    base_mesh.point_data[f"{label}_imag"] = mode_vector.imag.astype(np.float32)
    base_mesh.point_data[f"{label}_abs"] = np.abs(mode_vector).astype(np.float32)

    base_mesh.save(out_path)
    print(f"[INFO] モード形状を書き出しました: {out_path}")


def export_top_modes(result: dict, selection: dict, cfg, topology_path: str):
    """
    mode_selection.select_modes() が選んだ重要モードをVTKに書き出す。
    特定周波数への絞り込みは行わない(selectionが既に重要度で絞り込み済み)。

    Parameters
    ----------
    topology_path : str
        ensight_to_hdf5.ensure_reference_topology() が保存した参照メッシュのパス。
    """

    out_dir = os.path.join(cfg.work_dir, cfg.mode_export_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # 参照メッシュは1回だけ読み込み、以降のモードごとにpoint_dataを
    # 上書きしながら使い回す(毎回ファイルから読み直さない)。
    base_mesh = pv.read(topology_path)

    modes = result["modes"]  # (n_points, r)
    freq_hz = result["freq_hz"]
    growth_rate = result["growth_rate"]

    idx_selected = selection["idx_selected"]

    if modes.shape[0] != base_mesh.n_points:
        raise RuntimeError(
            f"モード形状の点数({modes.shape[0]})と参照メッシュの点数"
            f"({base_mesh.n_points})が一致しません。ROI設定や参照メッシュの"
            f"作り直しが必要な可能性があります(dmd_work内のmesh_topology.vtu"
            f"を削除して再実行してください)。"
        )

    st_number = freq_to_strouhal(freq_hz, cfg)

    for rank, i in enumerate(idx_selected):
        freq = freq_hz[i]
        st = st_number[i]
        gr = growth_rate[i]
        label = f"mode{rank:02d}_f{freq:+.3f}Hz_St{st:+.4f}_gr{gr:+.3e}"
        out_path = os.path.join(out_dir, f"{label}.vtu")
        export_mode_to_vtk(base_mesh, modes[:, i], out_path, label="mode")
