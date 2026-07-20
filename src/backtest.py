# -*- coding: utf-8 -*-
"""
Module: backtest.py
Description: Backtesting utilities for Value-at-Risk models, including OOS splitting.
"""

import numpy as np
from scipy.stats import chi2
from statsmodels.stats.diagnostic import acorr_ljungbox
from .estimation import estimate_gal_parameters, exact_gal_quantile
from .risk_metrics import normalized_volume_series

def test_iid_assumption(aggregate_returns, lags=5, label=""):
    tag = f" ({label})" if label else ""
    print(f"\n=========================================================================================")
    print(f" Ljung-Box Test for Serial Independence (i.i.d. check of Crash Magnitudes){tag}")
    print("=========================================================================================")
    if len(aggregate_returns) > lags:
        lb_test = acorr_ljungbox(aggregate_returns, lags=[lags], return_df=True)
        p_val = lb_test['lb_pvalue'].iloc[0]
        print(f" Ljung-Box Test (Lag {lags}) p-value: {p_val:.4f}")
        if p_val > 0.05:
            print(" Sonuc: PASS. Seri bagimsizdir. (i.i.d. varsayimi ampirik olarak dogrulanmistir)")
        else:
            print(" Sonuc: FAIL. Seride otokorelasyon mevcuttur.")
    else:
        print(" Test icin yeterli gozlem (N) bulunmamaktadir.")

def stratified_time_split(n_obs, test_fraction=0.3, n_blocks=10):
    block_edges = np.linspace(0, n_obs, n_blocks + 1).astype(int)
    n_test_blocks = max(1, round(n_blocks * test_fraction))
    test_block_idx = np.unique(np.round(np.linspace(0, n_blocks - 1, n_test_blocks)).astype(int))

    test_mask = np.zeros(n_obs, dtype=bool)
    for b in test_block_idx:
        test_mask[block_edges[b]:block_edges[b + 1]] = True

    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx, test_block_idx

def _run_oos_once(std_residuals, trades, percentile, train_idx, test_idx,
                   window, split_label):
    train_std, train_trades = std_residuals[train_idx], trades[train_idx]
    test_std, test_trades = std_residuals[test_idx], trades[test_idx]

    print(f"\n --- OOS [{split_label}] --- (train n={len(train_idx)}, test n={len(test_idx)})")

    train_relative = normalized_volume_series(train_trades, window=window)
    threshold_rel = np.nanpercentile(train_relative, percentile)

    def _aggregate(std_arr, trades_arr):
        rel = normalized_volume_series(trades_arr, window=window)
        agg, cur = [], 0.0
        for i in range(len(std_arr)):
            cur += std_arr[i]
            if rel[i] > threshold_rel:
                agg.append(cur)
                cur = 0.0
        return np.array(agg)

    train_agg = _aggregate(train_std, train_trades)
    test_agg = _aggregate(test_std, test_trades)

    if len(train_agg) < 15 or len(test_agg) < 5:
        print(f"  -> Yetersiz gozlem (train_agg={len(train_agg)}, test_agg={len(test_agg)}), atlaniyor.")
        return

    alpha_gal, theta_gal, p_gal, se_a, se_t, se_p = estimate_gal_parameters(train_agg)
    print(f"  [Train] Alpha={alpha_gal:.4f}, Theta={theta_gal:.4f}, p={p_gal:.4f}  (n_crash_train={len(train_agg)})")
    print(f"  [Test]  N_crash={len(test_agg)}")

    for lvl in [0.05, 0.01]:
        q_gal = exact_gal_quantile(lvl, alpha_gal, theta_gal, p_gal)
        hits = (test_agg < q_gal).astype(int)
        fails, p_uc, p_cc = backtest_var(hits, lvl)
        flag_uc = "*FAIL*" if p_uc < 0.05 else "PASS"
        flag_cc = "*FAIL*" if p_cc < 0.05 else "PASS"
        print(f"  OOS VaR {lvl} | Ihlal: {fails}/{len(test_agg)} | UC: {p_uc:.4f} {flag_uc} | CC: {p_cc:.4f} {flag_cc}")

def out_of_sample_backtest_v2(std_residuals, trades, percentile=95, train_ratio=0.7,
                               window=None, n_blocks=10):
    print(f"\n=========================================================================================")
    print(f" OUT-OF-SAMPLE (OOS) BACKTESTING -- IKI SPLIT STRATEJISI KARSILASTIRMASI")
    print(f"=========================================================================================")

    n_obs = len(std_residuals)

    split_idx = int(n_obs * train_ratio)
    chrono_train_idx = np.arange(0, split_idx)
    chrono_test_idx = np.arange(split_idx, n_obs)
    _run_oos_once(std_residuals, trades, percentile, chrono_train_idx, chrono_test_idx,
                  window, split_label="Kronolojik (orijinal, son %30)")

    strat_train_idx, strat_test_idx, test_blocks = stratified_time_split(
        n_obs, test_fraction=(1 - train_ratio), n_blocks=n_blocks)
    print(f"\n [Stratified split] test bloklari (0-{n_blocks-1} arasi): {list(test_blocks)}")
    _run_oos_once(std_residuals, trades, percentile, strat_train_idx, strat_test_idx,
                  window, split_label=f"Stratified ({n_blocks} blok, dagitilmis)")

def backtest_var(hits, p_target):
    n_obs = len(hits)
    failures = np.sum(hits)

    if failures == 0:
        lr_uc = 0.0
        p_val_uc = 1.0
    else:
        p_fail = failures / n_obs
        eps = 1e-10
        log_L_target = failures * np.log(p_target + eps) + (n_obs - failures) * np.log(1 - p_target + eps)
        log_L_actual = failures * np.log(p_fail + eps) + (n_obs - failures) * np.log(1 - p_fail + eps)
        lr_uc = max(0.0, -2 * (log_L_target - log_L_actual))
        p_val_uc = chi2.sf(lr_uc, df=1)

    n00 = n01 = n10 = n11 = 0
    for i in range(1, n_obs):
        if hits[i-1] == 0 and hits[i] == 0: n00 += 1
        elif hits[i-1] == 0 and hits[i] == 1: n01 += 1
        elif hits[i-1] == 1 and hits[i] == 0: n10 += 1
        elif hits[i-1] == 1 and hits[i] == 1: n11 += 1

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_1 = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0

    if failures > 0 and (n00 + n01) > 0 and (n10 + n11) > 0:
        log_L_ind_null = (n00 + n10) * np.log(1 - pi_1 + eps) + (n01 + n11) * np.log(pi_1 + eps)
        log_L_ind_alt = n00 * np.log(1 - pi_01 + eps) + n01 * np.log(pi_01 + eps) + \
                        n10 * np.log(1 - pi_11 + eps) + n11 * np.log(pi_11 + eps)
        lr_ind = max(0.0, -2 * (log_L_ind_null - log_L_ind_alt))
    else:
        lr_ind = 0.0

    lr_cc = lr_uc + lr_ind
    p_val_cc = chi2.sf(lr_cc, df=2)

    return failures, p_val_uc, p_val_cc