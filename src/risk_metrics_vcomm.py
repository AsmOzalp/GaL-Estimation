# -*- coding: utf-8 -*-
"""
Module: risk_metrics.py
Description: Construction of volume-triggered compounding returns and normalized thresholds.
"""

import numpy as np
import pandas as pd

def normalized_volume_series(trades, window):
    s = pd.Series(trades).astype(float)
    roll_mean = s.rolling(window=window, min_periods=max(5, window // 5)).mean()
    relative = (s / roll_mean).to_numpy()
    relative = np.clip(relative, 0.0, np.nanpercentile(relative[np.isfinite(relative)], 99.9)
                        if np.isfinite(relative).any() else 1.0)
    return relative

def construct_volume_triggered_returns_v2(returns, trades, percentile=95, window=None, verbose=True):
    relative_trades = normalized_volume_series(trades, window=window)
    threshold_rel = np.nanpercentile(relative_trades, percentile)
    
    if verbose:
        print(f" -> [Normalize esik] Pencere={window} bar, {percentile}. persentil "
              f"(goreli hacim) = {threshold_rel:.4f}")

    aggregate_S = []
    current_sum = 0.0
    n_triggers = 0
    for i in range(len(returns)):
        current_sum += returns[i]
        if relative_trades[i] > threshold_rel:
            aggregate_S.append(current_sum)
            current_sum = 0.0
            n_triggers += 1

    if current_sum != 0.0 and verbose:
        print(f" -> Not: Sona kalan tetiklenmemis kismi toplam ({current_sum:.4f}) "
              f"gercek bir hacim-tetiklemeli gozlem olmadigi icin ATILDI (dahil edilmedi).")

    return np.array(aggregate_S)