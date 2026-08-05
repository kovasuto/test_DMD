"""
無次元化(Strouhal数)計算の共通ヘルパー。

St = f * L / U

f: 周波数 [Hz]
L: 代表長さ [m] (config.reference_length)
U: 代表流速 [m/s] (config.reference_velocity)

複数のモジュール(dmd_analysis.py, mode_selection.py, export_modes.py,
diagnostics.py)から同じ定義で参照するため、ここに1箇所だけ定義する。
"""

import numpy as np

from config import Config


def freq_to_strouhal(freq_hz, cfg: Config):
    """
    周波数 [Hz] (スカラーまたはnumpy配列) をStrouhal数に変換する。
    St = f * L / U
    """
    return np.asarray(freq_hz) * cfg.reference_length / cfg.reference_velocity
