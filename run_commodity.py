# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:46:07 2026

@author: m-a-o
"""

# -*- coding: utf-8 -*-
"""
Main pipeline for Commodity Risk Analysis (Gold & Silver)
Updated with v9 Advanced Methodology (Robust GARCH, Normalized Volume, OOS Backtesting)
"""

import numpy as np
from scipy.stats import norm, t, cauchy, laplace

# Lokal proje modülleri
from src.data_loader import load_commodity_data
from src.garch_models import fit_garch_robust_v3
from src.estimation import (
    estimate_al_parameters, estimate_gal_parameters, 
    estimate_gal_parameters_smooth, check_boundary_proximity,
    exact_gal_quantile
)
from src.risk_metrics_vcomm import construct_volume_triggered_returns_v2
from src.backtest import test_iid_assumption, out_of_sample_backtest_v2, backtest_var
# Reproducibility: fixed data cutoff for the manuscript
DATA_START_DATE = "2010-01-01"
DATA_END_DATE = "2026-08-31"
def run_sensitivity_analysis(std_residuals, volume, symbol, window=30):
    """
    Belirli hacim eşiklerinde (85, 90, 95, 98) GaL parametrelerinin değişimini 
    gözlemlemek için duyarlılık analizi raporu üretir.
    """
    print("\n" + "="*85)
    print(f" DUYARLILIK ANALİZİ: GaL Parametreleri Farklı Hacim Eşiklerinde ({symbol})")
    print("="*85)
    print(f" {'Eşik (Percentile)':<18} | {'N (Kriz)':<10} | {'Alpha (SE)':<16} | {'Theta (SE)':<16} | {'p (SE)':<16}")
    print("-" * 85)
    
    for pct in [85, 90, 95, 98]:
        agg_ret = construct_volume_triggered_returns_v2(
            std_residuals, volume, percentile=pct, window=window, verbose=False
        )
        if len(agg_ret) < 15:
            print(f" %{pct:<17} | {len(agg_ret):<10} | {'Yetersiz Gözlem (Tahmin Yapılamaz)':<50}")
            continue
        alpha, theta, p, se_a, se_t, se_p = estimate_gal_parameters(agg_ret)
        print(f" %{pct:<17} | {len(agg_ret):<10} | {alpha:.4f} ({se_a:.4f}) | {theta:.4f} ({se_t:.4f}) | {p:.4f} ({se_p:.4f})")
    print("="*85)

def run_commodity_risk_application(symbol="GC=F", name="GOLD"):
    """
    Emtia için ana risk analizi ve backtesting boru hattı.
    """
    returns, volume = load_commodity_data(
    symbol=symbol,
    start_date=DATA_START_DATE,
    end_date=DATA_END_DATE
    )
    n_obs = len(returns)

    print(f"\nRunning Robust GARCH(1,1) model for {name} (Daily)...")
    res = fit_garch_robust_v3(returns, dist='t')
    
    scale = getattr(res, 'scale', 1.0)
    print(f" [FIX #11 kontrolü] arch iç-ölçek çarpanı (res.scale) = {scale}")
    sigma_t = res.conditional_volatility / scale
    std_residuals = (res.resid / scale) / sigma_t
    
    print("\nEstimating standard distributions on standardized residuals...")
    n_mean, n_std = norm.fit(std_residuals)
    t_df, t_loc, t_scale = t.fit(std_residuals)
    c_loc, c_scale = cauchy.fit(std_residuals)
    l_loc, l_scale = laplace.fit(std_residuals)
    
    print("\nEstimating AL parameters for raw residuals...")
    alpha_al, theta_al, se_a_al, se_t_al = estimate_al_parameters(std_residuals)
    print(f" AL -> Alpha: {alpha_al:.4f} ({se_a_al:.4f}), Theta: {theta_al:.4f} ({se_t_al:.4f})")
    
    # Güncellenmiş normalize hacimli Duyarlılık Analizi (Pencere = 30 gün)
    run_sensitivity_analysis(std_residuals, volume, name, window=30)
    
    print(f"\nConstructing Volume-Triggered Cumulative Crashes for VaR ({name})...")
    aggregate_returns = construct_volume_triggered_returns_v2(std_residuals, volume, percentile=95, window=30)
    n_crash = len(aggregate_returns)
    
    print("\nEstimating GaL parameters for cumulative crashes...")
    alpha_gal, theta_gal, p_gal, se_a, se_t, se_p = estimate_gal_parameters(aggregate_returns)
    print(f" GaL -> Alpha: {alpha_gal:.4f} ({se_a:.4f}), Theta: {theta_gal:.4f} ({se_t:.4f}), p: {p_gal:.4f} ({se_p:.4f})")
    
    th_ratio, p_dist = check_boundary_proximity(alpha_gal, theta_gal, p_gal, label="GaL (Crash)")
    if th_ratio > 0.90 or p_dist < 0.02:
        print("  -> Kıyaslama için Smooth Reparametrizasyon da hesaplanıyor...")
        a_s, th_s, p_s, sa_s, st_s, sp_s = estimate_gal_parameters_smooth(aggregate_returns)
        print(f"  [Smooth] Alpha: {a_s:.4f} ({sa_s:.4f}), Theta: {th_s:.4f} ({st_s:.4f}), p: {p_s:.4f} ({sp_s:.4f})")

    # Tablo 1: Rutin VaR Testi
    print(f"\n=========================================================================================")
    print(f" TABLO 1: RUTİN VaR TESTİ ({name} - {n_obs} Gözlem)")
    print(f"=========================================================================================")
    for lvl in [0.05, 0.01]:
        print(f"\n LEVEL: {lvl} (Beklenen İhlal: {n_obs * lvl:.2f} / {n_obs})")
        print(f"-----------------------------------------------------------------------------------------")
        
        models = {
            "Normal": norm.ppf(lvl, loc=n_mean, scale=n_std),
            "Student-t": t.ppf(lvl, t_df, loc=t_loc, scale=t_scale),
            "Cauchy": cauchy.ppf(lvl, loc=c_loc, scale=c_scale),
            "Laplace": laplace.ppf(lvl, loc=l_loc, scale=l_scale),
            "AL (Linnik)": exact_gal_quantile(lvl, alpha_al, theta_al, 1.0)
        }
        
        for model_name, q_val in models.items():
            fails, p_uc, p_cc = backtest_var((returns < (sigma_t * q_val)).astype(int), lvl)
            print(f" {model_name:<15} | İhlal: {fails:<5} | UC: {p_uc:.4f} {'FAIL' if p_uc < 0.05 else 'PASS':<4} | CC: {p_cc:.4f} {'FAIL' if p_cc < 0.05 else 'PASS'}")

    # Tablo 2: Kriz Piyasası (GaL)
    print(f"\n=========================================================================================")
    print(f" TABLO 2: KÜMÜLATİF ÇÖKÜŞ VaR TESTİ ({name} - GaL Modeli - {n_crash} Çöküş Anı)")
    print(f"=========================================================================================")
    for lvl in [0.05, 0.01]:
        print(f"\n LEVEL: {lvl} (Beklenen İhlal: {n_crash * lvl:.2f} / {n_crash})")
        print(f"-----------------------------------------------------------------------------------------")
        fails_gal, p_uc_gal, p_cc_gal = backtest_var((aggregate_returns < exact_gal_quantile(lvl, alpha_gal, theta_gal, p_gal)).astype(int), lvl)
        print(f" {'GaL (SBM Crash)':<15} | İhlal: {fails_gal:<5} | UC: {p_uc_gal:.4f} {'FAIL' if p_uc_gal < 0.05 else 'PASS':<4} | CC: {p_cc_gal:.4f} {'FAIL' if p_cc_gal < 0.05 else 'PASS'}")

    # Ljung-Box Testi
    test_iid_assumption(aggregate_returns, lags=10, label=name)
    
    # OOS Backtesting 
    out_of_sample_backtest_v2(std_residuals, volume, percentile=95, train_ratio=0.7, window=30, n_blocks=10)

if __name__ == "__main__":
    print(">>> RUNNING COMMODITY RISK ANALYSIS (UPDATED WITH v9) <<<")
    run_commodity_risk_application(symbol="GC=F", name="GOLD")
    run_commodity_risk_application(symbol="SI=F", name="SILVER")