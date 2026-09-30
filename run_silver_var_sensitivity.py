# run_silver_var_sensitivity.py
import numpy as np
import pandas as pd
from src.data_loader import load_commodity_data
from src.garch_models import fit_garch_robust_v3
from src.dependence_diagnostics import reconstruct_cycles
from src.estimation import estimate_gal_parameters, exact_gal_quantile
from src.backtest import backtest_var

def test_silver_sensitivity():
    print(">>> SILVER %5 VaR ESIK-DUYARLILIK TESTI <<<")
    
    # 1. Veriyi çek ve GARCH'tan geçir
    returns, volume, *_ = load_commodity_data("SI=F", start_date="2010-01-01", end_date="2026-08-31")
    
    res = fit_garch_robust_v3(returns, dist='t', verbose=False)
    scale = getattr(res, 'scale', 1.0)
    std_residuals = (res.resid / scale) / (res.conditional_volatility / scale)
    
    thresholds = [85, 90, 95, 98]
    level = 0.05
    
    print(f"{'Threshold':<10} | {'N_crash':<7} | {'Expected':<8} | {'Violations':<10} | {'Kupiec p':<10} | {'Christoff CC p':<12}")
    print("-" * 75)
    
    for pct in thresholds:
        # Krizleri kurgula (Yeni yazdigimiz ve %100 guvenli calisan fonksiyonu kullaniyoruz)
        cycles_df, threshold_rel = reconstruct_cycles(std_residuals, volume, percentile=pct, window=30)
        crashes = cycles_df["S"].to_numpy()
        n = len(crashes)
        
        if n < 15:
            print(f"{pct}%".ljust(10) + f" | {n:<7} | {'-':<8} | {'Not enough data'}")
            continue
            
        # GaL tahmin et
        try:
            alpha, theta, p, *_ = estimate_gal_parameters(crashes, polish=False) # Hızlı olması için polish=False
            q_5pct = exact_gal_quantile(level, alpha, theta, p)
        except Exception:
            print(f"{pct}%".ljust(10) + f" | {n:<7} | {'-':<8} | {'Estimation Failed'}")
            continue
            
        # Backtest
        hits = (crashes < q_5pct).astype(int)
        violations, p_uc, p_cc, *_ = backtest_var(hits, level)
        expected = n * level
        
        # Formatli cikti
        uc_flag = "FAIL" if p_uc < 0.05 else "PASS"
        cc_flag = "FAIL" if p_cc < 0.05 else "PASS"
        print(f"{pct}%".ljust(10) + f" | {n:<7} | {expected:<8.2f} | {violations:<10} | {p_uc:.4f} {uc_flag} | {p_cc:.4f} {cc_flag}")

if __name__ == "__main__":
    test_silver_sensitivity()