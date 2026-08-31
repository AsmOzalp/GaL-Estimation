# -*- coding: utf-8 -*-
"""
Module: dependence_diagnostics.py
Description:
    Diagnostics for the N ⊥ X independence assumption underlying the
    Subordinated Brownian Motion / volume-triggered GaL compounding
    framework, and for whether the cycle length N is well described by
    a Geometric distribution.

    Directly addresses AE comment #3:
        "The distribution assumes that N is independent with X. This is
        usually not realistic in actual stock trading, since trading
        volume and returns may be highly correlated. It might be more
        reliable to include a discussion of how the results are
        affected when N and X are correlated. Whether trading volume
        follows the geometric distribution also need further
        clarification."

    Four complementary diagnostics are provided:
      1. volume_shock_correlation      -- contemporaneous & lagged
         correlation between relative trading volume and shock
         magnitude |z_t|.
      2. reconstruct_cycles /
         cycle_dependence_test         -- tests whether the *length* of
         a compounding cycle (N_k) is related to the magnitude of the
         shocks realized within that cycle.
      3. geometric_goodness_of_fit     -- chi-square GOF test of whether
         empirical cycle lengths N_k are consistent with a Geometric(p)
         law.
      4. dependence_sensitivity_circular_shift -- a circular-shift
         (phase-randomization) surrogate test: the volume series is
         time-shifted by a random amount (preserving its own
         autocorrelation / clustering, but destroying its temporal
         alignment with the shocks), the GaL MDE is refit on each
         surrogate, and the real (naturally-aligned) estimate is
         compared against the resulting null distribution. This
         quantifies how much of the estimated (alpha, theta, p) is
         attributable to genuine volume-shock dependence rather than
         to the marginal properties of the two series alone.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, chi2, geom

from .risk_metrics import normalized_volume_series
from .estimation import estimate_gal_parameters


# ---------------------------------------------------------------------------
# 1. Contemporaneous / lagged correlation between volume and shock magnitude
# ---------------------------------------------------------------------------
def volume_shock_correlation(std_residuals, trades, window, max_lag=3):
    """
    Tests whether |z_t| (shock magnitude) and relative trading volume are
    correlated at lags -max_lag..+max_lag. lag > 0 means volume LEADS the
    shock by `lag` periods (volume_t vs |z|_{t+lag}); lag < 0 means volume
    LAGS the shock.

    Returns a DataFrame with one row per lag.
    """
    rel_vol = pd.Series(normalized_volume_series(np.asarray(trades, dtype=float), window=window))
    abs_z = pd.Series(np.abs(np.asarray(std_residuals, dtype=float)))
    n = min(len(rel_vol), len(abs_z))
    rel_vol, abs_z = rel_vol.iloc[:n].reset_index(drop=True), abs_z.iloc[:n].reset_index(drop=True)

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x_shock = abs_z.iloc[lag:].reset_index(drop=True)
            y_vol = rel_vol.iloc[: n - lag].reset_index(drop=True)
        else:
            x_shock = abs_z.iloc[: n + lag].reset_index(drop=True)
            y_vol = rel_vol.iloc[-lag:].reset_index(drop=True)

        mask = x_shock.notna() & y_vol.notna() & np.isfinite(x_shock) & np.isfinite(y_vol)
        n_valid = int(mask.sum())
        if n_valid < 10:
            continue

        r_p, p_p = pearsonr(x_shock[mask], y_vol[mask])
        r_s, p_s = spearmanr(x_shock[mask], y_vol[mask])
        rows.append(
            dict(lag=lag, n=n_valid, pearson_r=r_p, pearson_p=p_p, spearman_r=r_s, spearman_p=p_s)
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Cycle reconstruction (vectorized) and N vs. shock-magnitude dependence
# ---------------------------------------------------------------------------
def reconstruct_cycles(std_residuals, trades, percentile, window):
    """
    Reconstructs the volume-triggered compounding cycles exactly as
    risk_metrics.construct_volume_triggered_returns_v2 does, but returns
    per-cycle diagnostics (N_k, S_k, mean/max |shock| within the cycle)
    instead of only the aggregate sums S_k.

    Vectorized (pandas groupby) so it stays fast on the 5-minute BTC
    series (n ~ 10^5) across many permutations.

    Returns (cycles_df, threshold_rel).
    """
    std_residuals = np.asarray(std_residuals, dtype=float)
    trades = np.asarray(trades, dtype=float)
    n = len(std_residuals)

    rel_vol = normalized_volume_series(trades, window=window)
    threshold_rel = np.nanpercentile(rel_vol, percentile)
    trigger = rel_vol > threshold_rel

    # group id = number of triggers strictly BEFORE index i (so the trigger
    # row itself belongs to the cycle it closes, matching the original
    # sequential-loop construction exactly)
    prev_trigger = np.zeros(n, dtype=bool)
    prev_trigger[1:] = trigger[:-1]
    group_id = np.cumsum(prev_trigger)

    df = pd.DataFrame({"shock": std_residuals, "trigger": trigger, "group": group_id})
    grouped = df.groupby("group")
    cycles = grouped.agg(
        N=("shock", "size"),
        S=("shock", "sum"),
        mean_abs_shock=("shock", lambda s: np.abs(s).mean()),
        max_abs_shock=("shock", lambda s: np.abs(s).max()),
        ends_in_trigger=("trigger", "max"),
    ).reset_index(drop=True)

    # discard the trailing, un-triggered partial cycle (same behaviour as
    # the original sequential construction, which drops the leftover sum)
    cycles = cycles[cycles["ends_in_trigger"]].drop(columns="ends_in_trigger").reset_index(drop=True)

    return cycles, threshold_rel


def cycle_dependence_test(cycles_df):
    """
    Spearman correlation between cycle length N_k and the magnitude of the
    shocks realized within that cycle. Under the N ⊥ X assumption, longer
    cycles should show NO systematic relationship with shock magnitude.
    """
    r_mean, p_mean = spearmanr(cycles_df["N"], cycles_df["mean_abs_shock"])
    r_max, p_max = spearmanr(cycles_df["N"], cycles_df["max_abs_shock"])
    return dict(
        n_cycles=len(cycles_df),
        rho_N_vs_mean_abs_shock=r_mean,
        p_N_vs_mean_abs_shock=p_mean,
        rho_N_vs_max_abs_shock=r_max,
        p_N_vs_max_abs_shock=p_max,
    )


# ---------------------------------------------------------------------------
# 3. Geometric goodness-of-fit test for cycle lengths N_k
# ---------------------------------------------------------------------------
def geometric_goodness_of_fit(N_values):
    """
    Chi-square goodness-of-fit test of H0: N_k ~ Geometric(p), with p
    estimated by MLE (p_hat = 1 / mean(N)). Adjacent right-tail bins are
    merged until each has expected count >= 5 (standard chi-square rule
    of thumb), and the theoretical Geometric tail mass beyond the largest
    observed bin is folded into the final bin so expected probabilities
    sum to 1.
    """
    N = np.asarray(N_values, dtype=int)
    N = N[N >= 1]
    n = len(N)
    if n < 20:
        return dict(n=n, p_hat=np.nan, chi2_stat=np.nan, dof=np.nan, p_value=np.nan,
                     note="n < 20: test not reliable, skipped")

    p_hat = 1.0 / np.mean(N)
    max_N = int(N.max())
    support = np.arange(1, max_N + 1)
    pmf = geom.pmf(support, p_hat)
    obs_counts = np.array([(N == k).sum() for k in support])
    exp_counts = pmf * n

    bins_obs, bins_exp = [], []
    acc_obs, acc_exp = 0, 0.0
    for o, e in zip(obs_counts, exp_counts):
        acc_obs += o
        acc_exp += e
        if acc_exp >= 5:
            bins_obs.append(acc_obs)
            bins_exp.append(acc_exp)
            acc_obs, acc_exp = 0, 0.0
    if acc_exp > 0:
        if bins_exp:
            bins_obs[-1] += acc_obs
            bins_exp[-1] += acc_exp
        else:
            bins_obs.append(acc_obs)
            bins_exp.append(acc_exp)

    bins_obs = np.array(bins_obs, dtype=float)
    bins_exp = np.array(bins_exp, dtype=float)
    # fold in the theoretical tail mass beyond max_N (Geometric has infinite support)
    tail_prob = 1.0 - geom.cdf(max_N, p_hat)
    bins_exp[-1] += tail_prob * n

    k_bins = len(bins_obs)
    dof = max(k_bins - 1 - 1, 1)  # -1 normalization, -1 estimated parameter p
    chi2_stat = float(np.sum((bins_obs - bins_exp) ** 2 / bins_exp))
    p_value = float(chi2.sf(chi2_stat, df=dof))

    return dict(n=n, mean_N=float(np.mean(N)), p_hat=p_hat, k_bins=k_bins,
                 dof=dof, chi2_stat=chi2_stat, p_value=p_value)


# ---------------------------------------------------------------------------
# 4. Circular-shift surrogate sensitivity test for the GaL MDE estimates
# ---------------------------------------------------------------------------
def dependence_sensitivity_circular_shift(std_residuals, trades, percentile, window,
                                           n_perm=100, seed=42, min_shift_frac=0.05,
                                           min_crash_obs=15, verbose=True):
    """
    Circular-shift (phase-randomization) surrogate test.

    The trading-volume series is time-shifted by a random circular offset
    (np.roll), which preserves its own autocorrelation / clustering
    structure and marginal distribution but destroys its temporal
    alignment with the return shocks. The full pipeline (volume-triggered
    compounding -> GaL MDE) is then re-run on each surrogate. Comparing
    the real (naturally aligned) parameter estimates against this null
    distribution isolates the effect of genuine N-X dependence in the
    data, holding both series' marginal/autocorrelation properties fixed.

    Returns (summary_dict, perm_arrays_dict).
    """
    std_residuals = np.asarray(std_residuals, dtype=float)
    trades = np.asarray(trades, dtype=float)
    n = len(trades)
    rng = np.random.default_rng(seed)

    real_cycles, _ = reconstruct_cycles(std_residuals, trades, percentile, window)
    S_real = real_cycles["S"].to_numpy()
    if len(S_real) < min_crash_obs:
        raise ValueError(
            f"Yeterli kriz gozlemi yok (n_crash={len(S_real)} < {min_crash_obs}); "
            "permutasyon testi guvenilir olmaz."
        )
    a_real, th_real, p_real, *_ = estimate_gal_parameters(S_real)

    min_shift = max(1, int(min_shift_frac * n))
    max_shift = n - min_shift
    perm_alpha, perm_theta, perm_p, perm_ncrash = [], [], [], []

    for b in range(n_perm):
        shift = int(rng.integers(min_shift, max_shift))
        trades_shifted = np.roll(trades, shift)
        cyc_b, _ = reconstruct_cycles(std_residuals, trades_shifted, percentile, window)
        S_b = cyc_b["S"].to_numpy()
        if len(S_b) < min_crash_obs:
            continue
        try:
            a_b, th_b, p_b, *_ = estimate_gal_parameters(S_b)
        except Exception:
            continue
        perm_alpha.append(a_b)
        perm_theta.append(th_b)
        perm_p.append(p_b)
        perm_ncrash.append(len(S_b))
        if verbose and (b + 1) % 20 == 0:
            print(f"  [circular-shift] {b + 1}/{n_perm} tamamlandi...")

    perm_alpha = np.array(perm_alpha)
    perm_theta = np.array(perm_theta)
    perm_p = np.array(perm_p)

    def _two_sided_pvalue(real_val, perm_vals):
        if len(perm_vals) == 0:
            return np.nan
        center = np.mean(perm_vals)
        return float(np.mean(np.abs(perm_vals - center) >= abs(real_val - center)))

    summary = dict(
        n_crash_real=len(S_real),
        n_valid_perm=len(perm_alpha),
        n_perm_requested=n_perm,
        alpha_real=a_real, alpha_perm_mean=float(np.mean(perm_alpha)) if len(perm_alpha) else np.nan,
        alpha_perm_sd=float(np.std(perm_alpha)) if len(perm_alpha) else np.nan,
        alpha_perm_pvalue=_two_sided_pvalue(a_real, perm_alpha),
        theta_real=th_real, theta_perm_mean=float(np.mean(perm_theta)) if len(perm_theta) else np.nan,
        theta_perm_sd=float(np.std(perm_theta)) if len(perm_theta) else np.nan,
        theta_perm_pvalue=_two_sided_pvalue(th_real, perm_theta),
        p_real=p_real, p_perm_mean=float(np.mean(perm_p)) if len(perm_p) else np.nan,
        p_perm_sd=float(np.std(perm_p)) if len(perm_p) else np.nan,
        p_perm_pvalue=_two_sided_pvalue(p_real, perm_p),
    )
    return summary, dict(alpha=perm_alpha, theta=perm_theta, p=perm_p, n_crash=np.array(perm_ncrash))


# ---------------------------------------------------------------------------
# Convenience: run all four diagnostics and print a manuscript-ready report
# ---------------------------------------------------------------------------
def run_full_dependence_report(std_residuals, trades, percentile, window,
                                label="", n_perm=100, seed=42, max_lag=3):
    print("\n" + "=" * 91)
    print(f" BAGIMLILIK TANI RAPORU (N vs X){f' [{label}]' if label else ''}")
    print(" AE Yorum #3: N-X bagimsizlik varsayimi ve hacmin geometrik dagilima uygunlugu")
    print("=" * 91)

    print("\n-- 1) Hacim vs |sok| korelasyonu (lag -{0}..+{0}) --".format(max_lag))
    corr_df = volume_shock_correlation(std_residuals, trades, window, max_lag=max_lag)
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cycles_df, threshold_rel = reconstruct_cycles(std_residuals, trades, percentile, window)
    print(f"\n-- 2) Cevrim (cycle) duzeyinde N vs sok buyuklugu (esik goreli hacim={threshold_rel:.4f}, "
          f"n_cycles={len(cycles_df)}) --")
    cyc_result = cycle_dependence_test(cycles_df)
    for k, v in cyc_result.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    print("\n-- 3) N_k icin Geometrik dagilim uyum testi (chi-square GOF) --")
    gof_result = geometric_goodness_of_fit(cycles_df["N"].to_numpy())
    for k, v in gof_result.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    print(f"\n-- 4) Dairesel-kaydirma (circular-shift) surrogate duyarlilik testi (n_perm={n_perm}) --")
    print("   (Gercek [dogal hizalanmis] GaL tahminleri, hacim serisinin rastgele dairesel "
          "kaydirilmasiyla uretilen surrogate tahmin dagilimiyla karsilastiriliyor.)")
    sens_summary, sens_perms = dependence_sensitivity_circular_shift(
        std_residuals, trades, percentile, window, n_perm=n_perm, seed=seed, verbose=True
    )
    for k, v in sens_summary.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    return dict(
        volume_shock_corr=corr_df,
        cycles=cycles_df,
        cycle_dependence=cyc_result,
        geometric_gof=gof_result,
        sensitivity_summary=sens_summary,
        sensitivity_perms=sens_perms,
    )