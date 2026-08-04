"""
DMDモード形状(複素数)の実部・虚部を、点群データとしてVTK(.vtp)に書き出す。
ParaView等で可視化し、渦構造を目視確認するために使う。
"""

import os
import numpy as np
import pyvista as pv


def export_mode_to_vtk(coords: np.ndarray, mode_vector: np.ndarray, out_path: str, label: str):
    """
    coords: (n_points, 3)
    mode_vector: (n_points,) 複素数配列 (DMDモード形状 Φ の1列)
    """

    cloud = pv.PolyData(coords)
    cloud[f"{label}_real"] = mode_vector.real.astype(np.float32)
    cloud[f"{label}_imag"] = mode_vector.imag.astype(np.float32)
    cloud[f"{label}_abs"] = np.abs(mode_vector).astype(np.float32)

    cloud.save(out_path)
    print(f"[INFO] モード形状を書き出しました: {out_path}")


def export_top_modes(result: dict, selection: dict, cfg):
    """
    mode_selection.select_modes() が選んだ重要モードをVTKに書き出す。
    特定周波数への絞り込みは行わない(selectionが既に重要度で絞り込み済み)。
    """

    out_dir = os.path.join(cfg.work_dir, cfg.mode_export_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    coords = result["coords"]
    modes = result["modes"]  # (n_points, r)
    freq_hz = result["freq_hz"]
    growth_rate = result["growth_rate"]

    idx_selected = selection["idx_selected"]

    for rank, i in enumerate(idx_selected):
        freq = freq_hz[i]
        gr = growth_rate[i]
        label = f"mode{rank:02d}_f{freq:+.3f}Hz_gr{gr:+.3e}"
        out_path = os.path.join(out_dir, f"{label}.vtp")
        export_mode_to_vtk(coords, modes[:, i], out_path, label="mode")
