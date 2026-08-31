# -*- coding: utf-8 -*-
"""
Runner: Methodology diagnostics for AE comments #1 (necessity of geometric
compounding) and #2 (genuineness of the MDE optimization challenge).
"""
import time
from src.data_loader import load_btc_binance_data
from src.garch_models import fit_garch_robust_v3
from src.dependence_diagnostics import reconstruct_cycles
from src.methodology_diagnostics import nested_model_comparison, multistart_local_vs_global


def prep_crash_aggregates(interval, percentile, window):
    returns, trades = load_btc_binance_data(interval=interval, years=1)
    res = fit_garch_robust_v3(returns, dist='t', verbose=False)
    scale = getattr(res, 'scale', 1.0)
    sigma_t = res.conditional_volatility / scale
    resid_corrected = res.resid / scale
    std_residuals = resid_corrected / sigma_t
    cycles, threshold_rel = reconstruct_cycles(std_residuals, trades, percentile, window)
    return cycles['S'].to_numpy()


if __name__ == "__main__":
    configs = [
        dict(interval="1d", percentile=90, window=30, n_starts=30),
        dict(interval="5m", percentile=98, window=2016, n_starts=30),
    ]
    for cfg in configs:
        t0 = time.time()
        print(f"\n\n>>> {cfg['interval']} icin cokus (crash) toplamlari hazirlaniyor...")
        S = prep_crash_aggregates(cfg["interval"], cfg["percentile"], cfg["window"])
        print(f"    n_crash={len(S)}, hazirlik suresi={time.time() - t0:.1f}s")

        nested_model_comparison(S, levels=(0.05, 0.01), label=cfg["interval"])
        multistart_local_vs_global(S, n_starts=cfg["n_starts"], seed=42, label=cfg["interval"])
        print(f"\n[{cfg['interval']}] toplam sure: {time.time() - t0:.1f}s")