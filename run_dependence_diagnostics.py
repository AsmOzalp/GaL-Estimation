# -*- coding: utf-8 -*-
"""
Runner: Dependence diagnostics for the N ~ X independence assumption.
Addresses AE comment #3 on the AoAS submission.
"""
import time
import numpy as np
from src.data_loader import load_btc_binance_data
from src.garch_models import fit_garch_robust_v3
from src.dependence_diagnostics import run_full_dependence_report

def prep_std_residuals(interval):
    returns, trades = load_btc_binance_data(interval=interval, years=1)
    res = fit_garch_robust_v3(returns, dist='t', verbose=False)
    scale = getattr(res, 'scale', 1.0)
    sigma_t = res.conditional_volatility / scale
    resid_corrected = res.resid / scale
    std_residuals = resid_corrected / sigma_t
    return std_residuals, trades

if __name__ == "__main__":
    configs = [
        dict(interval="1d", percentile=90, window=30, n_perm=200),
        dict(interval="5m", percentile=98, window=2016, n_perm=50),
    ]
    for cfg in configs:
        t0 = time.time()
        print(f"\n\n>>> {cfg['interval']} icin std_residuals hazirlaniyor (GARCH filtreleme)...")
        std_residuals, trades = prep_std_residuals(cfg["interval"])
        print(f"    n_obs={len(std_residuals)}, hazirlik suresi={time.time()-t0:.1f}s")

        report = run_full_dependence_report(
            std_residuals, trades,
            percentile=cfg["percentile"], window=cfg["window"],
            label=cfg["interval"], n_perm=cfg["n_perm"]
        )
        print(f"\n[{cfg['interval']}] toplam sure: {time.time()-t0:.1f}s")