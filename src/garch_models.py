# -*- coding: utf-8 -*-
"""
Module: garch_models.py
Description: Robust GARCH fitting and diagnostic tools to prevent oversmoothing.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def _sum_matching(params, prefix):
    return float(sum(v for k, v in params.items() if k.startswith(prefix)))

def diagnose_garch_fit(res, returns, label="", realized_window=20, verbose=True):
    params = res.params
    omega = float(params.get('omega', np.nan))
    alpha_sum = _sum_matching(params, 'alpha[')
    beta_sum = _sum_matching(params, 'beta[')
    persistence = alpha_sum + beta_sum

    scale = getattr(res, 'scale', 1.0)
    sigma_t = np.asarray(res.conditional_volatility) / scale
    realized_vol = pd.Series(np.abs(returns)).rolling(
        window=realized_window, min_periods=max(5, realized_window // 4)
    ).std().to_numpy()

    valid = ~np.isnan(realized_vol) & ~np.isnan(sigma_t)
    if valid.sum() > 10 and np.std(sigma_t[valid]) > 1e-6:
        corr = float(np.corrcoef(sigma_t[valid], realized_vol[valid])[0, 1])
        rng_sigma = np.nanmax(sigma_t[valid]) - np.nanmin(sigma_t[valid])
        rng_real = np.nanmax(realized_vol[valid]) - np.nanmin(realized_vol[valid])
        range_ratio = rng_sigma / (rng_real + 1e-12)
    else:
        corr, range_ratio = np.nan, np.nan

    try:
        lb_raw = acorr_ljungbox(returns ** 2, lags=[10], return_df=True)
        lb_raw_sq_p = float(lb_raw['lb_pvalue'].iloc[0])
    except Exception:
        lb_raw_sq_p = np.nan
    genuine_arch_present = (not np.isnan(lb_raw_sq_p)) and (lb_raw_sq_p < 0.05)

    std_resid = (res.resid / scale) / sigma_t
    sq_resid = std_resid ** 2
    try:
        lb_sq = acorr_ljungbox(sq_resid, lags=[10], return_df=True)
        lb_sq_p = float(lb_sq['lb_pvalue'].iloc[0])
    except Exception:
        lb_sq_p = np.nan

    warn_persist = persistence >= 0.98
    warn_corr = genuine_arch_present and (not np.isnan(corr)) and (corr < 0.30)
    warn_lb = (not np.isnan(lb_sq_p)) and (lb_sq_p < 0.05)
    oversmoothing_flag = warn_persist or warn_corr or warn_lb

    if verbose:
        tag = f" [{label}]" if label else ""
        print(f"\n --- GARCH TANI{tag} ---")
        print(f"  Ham getirilerde gercek ARCH etkisi (Ljung-Box, ham^2, lag10) p={lb_raw_sq_p:.4f}"
              f" -> {'ARCH ETKISI VAR' if genuine_arch_present else 'anlamli ARCH etkisi tespit edilmedi (korelasyon/aralik kontrolleri devre disi)'}")
        print(f"  omega={omega:.6g}, sum(alpha)={alpha_sum:.4f}, sum(beta)={beta_sum:.4f}, "
              f"persistence(alpha+beta)={persistence:.4f}"
              + (" *** ASIRI KALICI (>=0.98) ***" if warn_persist else ""))
        print(f"  sigma_t vs {realized_window}-bar |return| kayan-std korelasyonu: {corr:.4f}"
              + (" *** DUSUK KORELASYON (<0.30) ***" if warn_corr else ""))
        print(f"  sigma_t degisim araligi / realized-vol degisim araligi: {range_ratio:.4f}"
              "  (<<1 ise GARCH asiri-smooth ediyor olabilir)")
        print(f"  Kare-standardize kalintilar Ljung-Box (lag10) p={lb_sq_p:.4f}"
              + (" *** FAIL: hala ARCH etkisi var ***" if warn_lb else " (PASS)"))
        print(f"  -> Genel tani: {'ASIRI-SMOOTH SUPHESI VAR' if oversmoothing_flag else 'sorun tespit edilmedi'}")

    return oversmoothing_flag, dict(omega=omega, alpha=alpha_sum, beta=beta_sum,
                                     persistence=persistence, corr=corr,
                                     range_ratio=range_ratio, lb_sq_p=lb_sq_p,
                                     lb_raw_sq_p=lb_raw_sq_p,
                                     genuine_arch_present=genuine_arch_present)

def fit_garch_robust_v3(returns, dist='t', verbose=True):
    def _try(label, vol_kwargs):
        try:
            am = arch_model(returns, mean='Constant', dist=dist, rescale=True, **vol_kwargs)
            res = am.fit(disp='off', options={'maxiter': 3000}, show_warning=False)
            return res
        except Exception as e:
            if verbose:
                print(f" [{label}] basarisiz (exception: {e})")
            return None

    standard_attempts = [
        ("GARCH(1,1)-t", dict(vol='Garch', p=1, o=0, q=1)),
        ("GARCH(1,1)-normal", dict(vol='Garch', p=1, o=0, q=1)),
    ]
    alt_attempts = [
        ("EGARCH(1,1,1)", dict(vol='EGARCH', p=1, o=1, q=1)),
        ("GJR-GARCH(1,1)", dict(vol='Garch', p=1, o=1, q=1)),
        ("GARCH(2,2)", dict(vol='Garch', p=2, o=0, q=2)),
    ]

    best_res, best_label, best_score = None, None, np.inf
    clean_converged_candidate = None

    def _score(res):
        std_ratio = float(np.std(res.resid / res.conditional_volatility))
        conv_ok = (res.convergence_flag == 0)
        return abs(std_ratio - 1.0) + (0.0 if conv_ok else 10.0), std_ratio, conv_ok

    for i, (label, vk) in enumerate(standard_attempts, 1):
        vk = dict(vk, dist=dist if 'normal' not in label.lower() else 'normal')
        vk.pop('dist_placeholder', None)
        kwargs = {k: v for k, v in vk.items() if k != 'dist'}
        res = _try(label, dict(kwargs))
        if res is None:
            continue
        score, std_ratio, conv_ok = _score(res)
        if verbose:
            print(f" [GARCH deneme {i}: {label}] convergence_flag={res.convergence_flag} "
                  f"({'OK' if conv_ok else 'YAKINSAMADI'}), std(resid/sigma)={std_ratio:.4f}")
        if score < best_score:
            best_res, best_label, best_score = res, label, score

        if conv_ok and 0.5 <= std_ratio <= 2.0:
            oversmooth, _ = diagnose_garch_fit(res, returns, label=label, verbose=verbose)
            if not oversmooth:
                if verbose:
                    print(f" -> {label} kabul edildi (yakinsadi ve asiri-smooth degil).")
                return res
            elif clean_converged_candidate is None:
                clean_converged_candidate = res
                if verbose:
                    print(f" -> {label} yakinsadi ama ASIRI-SMOOTH suphesi var; "
                          f"alternatif volatilite modelleri deneniyor (Cozum 1B)...")

    for i, (label, vk) in enumerate(alt_attempts, 1):
        res = _try(label, vk)
        if res is None:
            continue
        score, std_ratio, conv_ok = _score(res)
        if verbose:
            print(f" [Alternatif {i}: {label}] convergence_flag={res.convergence_flag} "
                  f"({'OK' if conv_ok else 'YAKINSAMADI'}), std(resid/sigma)={std_ratio:.4f}")
        if score < best_score:
            best_res, best_label, best_score = res, label, score

        if conv_ok and 0.5 <= std_ratio <= 2.0:
            oversmooth, _ = diagnose_garch_fit(res, returns, label=label, verbose=verbose)
            if not oversmooth:
                if verbose:
                    print(f" -> {label} kabul edildi (alternatif model, asiri-smooth degil).")
                return res

    if clean_converged_candidate is not None:
        print(" *** UYARI: Hicbir model 'asiri-smooth degil' kriterini tam saglamadi. "
              "Standart GARCH(1,1) sonucu kullaniliyor -- lutfen diagnose_garch_fit "
              "ciktisini dikkatle inceleyin ve sonuclari supheyle degerlendirin. ***")
        return clean_converged_candidate

    print(f" *** UYARI: Hicbir spesifikasyon 'convergence_flag==0 VE makul std' kriterini "
          f"saglamadi. En iyi bulunan ({best_label}) kullaniliyor -- SONUCLARI DIKKATLE "
          f"DEGERLENDIRIN. ***")
    return best_res