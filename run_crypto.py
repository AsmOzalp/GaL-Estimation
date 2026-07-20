# -*- coding: utf-8 -*-
"""
Main pipeline for Crypto Risk Analysis
"""

import numpy as np
from scipy.stats import norm, t, cauchy, laplace

# Lokal proje modülleri
from src.data_loader import load_btc_binance_data
from src.garch_models import fit_garch_robust_v3
from src.estimation import (
    estimate_al_parameters, estimate_gal_parameters, 
    estimate_gal_parameters_smooth, check_boundary_proximity,
    exact_gal_quantile
)
from src.risk_metrics import construct_volume_triggered_returns_v2
from src.backtest import test_iid_assumption, out_of_sample_backtest_v2, backtest_var


def run_btc_risk_application(interval="1d"):
    returns, trades = load_btc_binance_data(interval=interval, years=1)

    print(f"\nRunning GARCH(1,1) model for BTC ({interval})...")
    res = fit_garch_robust_v3(returns, dist='t')

    scale = getattr(res, 'scale', 1.0)
    print(f" [FIX #11 kontrolu] arch ic-olcek carpani (res.scale) = {scale}"
          + ("  (1.0 -> rescale bu veride tetiklenmedi/gerek yoktu)" if scale == 1.0
             else f"  *** MUTLAK sigma_t/resid {scale}x'e bolunerek duzeltiliyor ***"))
    
    sigma_t = res.conditional_volatility / scale
    resid_corrected = res.resid / scale
    std_residuals = resid_corrected / sigma_t
    n_obs = len(returns)

    resid_std = np.std(std_residuals)
    print(f" [FIX #1 kontrolu] std_residuals -> mean: {np.mean(std_residuals):.5f}, "
          f"std: {resid_std:.5f}  (0'a ve 1'e yakin olmali)")
    if not (0.5 <= resid_std <= 2.0):
        print(" *** UYARI: std_residuals std'si 1'den cok uzak -- GARCH filtresi "
              "supheli, asagidaki tum sonuclari dikkatle degerlendirin! ***")

    print("\nEstimating standard distributions on standardized residuals...")
    n_mean, n_std = norm.fit(std_residuals)
    t_df, t_loc, t_scale = t.fit(std_residuals)
    c_loc, c_scale = cauchy.fit(std_residuals)
    l_loc, l_scale = laplace.fit(std_residuals)

    print("\n--- Standard Distribution Parameters ---")
    print(f"Normal    -> Mean: {n_mean:.4f}, Std: {n_std:.4f}")
    print(f"Student-t -> df: {t_df:.4f}, Loc: {t_loc:.4f}, Scale: {t_scale:.4f}")
    print(f"Cauchy    -> Loc: {c_loc:.4f}, Scale: {c_scale:.4f}")
    print(f"Laplace   -> Loc: {l_loc:.4f}, Scale: {l_scale:.4f}")

    print("\nEstimating AL parameters for raw residuals...")
    alpha_al, theta_al, se_alpha_al, se_theta_al = estimate_al_parameters(std_residuals)
    print(f"AL  -> Alpha: {alpha_al:.4f} +/- {se_alpha_al:.4f}")
    print(f"      Theta: {theta_al:.4f} +/- {se_theta_al:.4f}")
    check_boundary_proximity(alpha_al, theta_al, 1.0, label="AL (routine)", check_p=False)

    window = 30 if interval == "1d" else 2016
    print(f"\nConstructing Volume-Triggered Cumulative Crashes (SBM Compounding, normalize esik)...")
    percentile_val = 90 if interval == "1d" else 98
    aggregate_returns = construct_volume_triggered_returns_v2(
        std_residuals, trades, percentile=percentile_val, window=window)

    print(f" -> Islenen ham gozlem sayisi (Rutin VaR icin): {len(std_residuals)}")
    print(f" -> Olusturulan kumulatif cokus (crash) sayisi (GaL VaR icin): {len(aggregate_returns)}")

    print("\nEstimating GaL parameters for cumulative crashes (sert-sinir yontemi)...")
    alpha_gal, theta_gal, p_gal, se_alpha_gal, se_theta_gal, se_p_gal = estimate_gal_parameters(aggregate_returns)
    print(
    f"GaL -> Alpha: {alpha_gal:.4f} +/- {se_alpha_gal:.4f}, "
    f"Theta: {theta_gal:.4f} +/- {se_theta_gal:.4f}, "
    f"p: {p_gal:.4f} +/- {se_p_gal:.4f}"
    )
    theta_ratio, p_dist = check_boundary_proximity(alpha_gal, theta_gal, p_gal, label="GaL (crash)")

    if theta_ratio > 0.90 or p_dist < 0.02:
        print("\n -> Sinira yakinlik tespit edildi; karsilastirma icin puruzsuz "
              "(tanh) reparametrizasyonlu tahminci de calistiriliyor...")
        alpha_s, theta_s, p_s, se_a_s, se_t_s, se_p_s = estimate_gal_parameters_smooth(aggregate_returns)
        print(f" [Smooth] Alpha: {alpha_s:.4f} +/- {se_a_s:.4f}, "
              f"Theta: {theta_s:.4f} +/- {se_t_s:.4f}, p: {p_s:.4f} +/- {se_p_s:.4f}")
        print(f" [Karsilastirma] |theta_hard - theta_smooth| = {abs(theta_gal - theta_s):.4f}, "
              f"|p_hard - p_smooth| = {abs(p_gal - p_s):.4f}")

    print(f"\n=========================================================================================")
    print(f" TABLO 1: RUTIN VaR TESTI (Standart Modeller - {n_obs} Gozlem Uzerinden)")
    print(f"=========================================================================================")
    levels = [0.05, 0.01]

    for lvl in levels:
        expected_fails = n_obs * lvl
        print(f"\n LEVEL: {lvl} (Beklenen Ihlal: {expected_fails:.2f} / {n_obs})")
        print(f"-----------------------------------------------------------------------------------------")
        print(f" {'Model':<15} | {'Ihlaller':<13} | {'Kupiec (UC) p-val':<18} | {'Christoffersen (CC) p-val':<25}")
        print(f"-----------------------------------------------------------------------------------------")

        q_norm = norm.ppf(lvl, loc=n_mean, scale=n_std)
        q_t = t.ppf(lvl, t_df, loc=t_loc, scale=t_scale)
        q_cauchy = cauchy.ppf(lvl, loc=c_loc, scale=c_scale)
        q_laplace = laplace.ppf(lvl, loc=l_loc, scale=l_scale)
        q_al = exact_gal_quantile(lvl, alpha_al, theta_al, 1.0)

        models = {
            "Normal": q_norm,
            "Student-t": q_t,
            "Cauchy": q_cauchy,
            "Laplace": q_laplace,
            "AL (Linnik)": q_al
        }

        for model_name, q_val in models.items():
            dynamic_var = sigma_t * q_val
            hits = (returns < dynamic_var).astype(int)
            fails, p_uc, p_cc = backtest_var(hits, lvl)

            flag_uc = "*FAIL*" if p_uc < 0.05 else "PASS"
            flag_cc = "*FAIL*" if p_cc < 0.05 else "PASS"

            fail_str = f"{fails} ({fails/n_obs:.4f})"
            print(f" {model_name:<15} | {fail_str:<13} | {p_uc:.4f} {flag_uc:<11} | {p_cc:.4f} {flag_cc}")

    n_crash = len(aggregate_returns)
    print(f"\n=========================================================================================")
    print(f" TABLO 2: KUMULATIF COKUS VaR TESTI (GaL Modeli - {n_crash} Cokus Ani Uzerinden)")
    print(f" * GaL Modeli sadece islem hacmi ile tetiklenmis gercek kriz anlarini olcer.")
    print(f"=========================================================================================")

    for lvl in levels:
        expected_fails_crash = n_crash * lvl
        print(f"\n LEVEL: {lvl} (Beklenen Ihlal: {expected_fails_crash:.2f} / {n_crash})")
        print(f"-----------------------------------------------------------------------------------------")
        print(f" {'Model':<15} | {'Ihlaller':<13} | {'Kupiec (UC) p-val':<18} | {'Christoffersen (CC) p-val':<25}")
        print(f"-----------------------------------------------------------------------------------------")

        q_gal = exact_gal_quantile(lvl, alpha_gal, theta_gal, p_gal)

        hits_gal = (aggregate_returns < q_gal).astype(int)
        fails_gal, p_uc_gal, p_cc_gal = backtest_var(hits_gal, lvl)

        flag_uc_gal = "*FAIL*" if p_uc_gal < 0.05 else "PASS"
        flag_cc_gal = "*FAIL*" if p_cc_gal < 0.05 else "PASS"

        fail_str_gal = f"{fails_gal} ({fails_gal/n_crash:.4f})"
        print(f" {'GaL (SBM Crash)':<15} | {fail_str_gal:<13} | {p_uc_gal:.4f} {flag_uc_gal:<11} | {p_cc_gal:.4f} {flag_cc_gal}")

    test_iid_assumption(aggregate_returns, lags=10, label=interval)

    out_of_sample_backtest_v2(
        std_residuals, trades, percentile=percentile_val, train_ratio=0.7,
        window=window, n_blocks=10
    )

if __name__ == "__main__":
    print(">>> RUNNING DAILY (1d) ANALYSIS <<<")
    run_btc_risk_application(interval="1d")

    print("\n\n>>> RUNNING HIGH-FREQUENCY (5m) ANALYSIS <<<")
    run_btc_risk_application(interval="5m")