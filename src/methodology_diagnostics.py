# -*- coding: utf-8 -*-
"""
Module: methodology_diagnostics.py
Description:
    Diagnostics addressing AE comments #1 and #2 on the AoAS submission:

    #1 "further explanation is needed regarding the benefits of geometric
        composition ... how it can address problems that could not
        previously be solved by the asymmetric Linnik distributions."
    -> nested_model_comparison(): AL (p fixed at 1, nested inside GaL) is
       fit directly to the SAME crash-aggregate series as GaL, and both
       models' VaR quantiles are backtested (Kupiec/Christoffersen) against
       that series. This isolates the marginal contribution of the
       compounding parameter p.

    #2 "the approach ... seems to be relatively natural. It needs further
        elaborate on the specific innovative aspects and challenges of
        the method."
    -> multistart_local_vs_global(): a naive local optimizer (Nelder-Mead)
       is run from many random starting points and compared against the
       DIRECT global optimizer used in the paper. Divergence of local
       starts demonstrates that the MDE objective surface is genuinely
       non-convex/multi-modal for the GaL family, so global optimization
       is a substantive methodological requirement rather than an
       arbitrary implementation choice.
"""

import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, direct, Bounds
from numpy.polynomial.hermite import hermgauss

from .estimation import estimate_al_parameters, estimate_gal_parameters, exact_gal_quantile
from .backtest import backtest_var


# ---------------------------------------------------------------------------
# AE point #1
# ---------------------------------------------------------------------------
def nested_model_comparison(crash_aggregates, levels=(0.05, 0.01), label=""):
    S = np.asarray(crash_aggregates, dtype=float)
    n = len(S)

    alpha_al, theta_al, se_a_al, se_t_al = estimate_al_parameters(S)
    alpha_gal, theta_gal, p_gal, se_a, se_t, se_p = estimate_gal_parameters(S)

    print(f"\n{'=' * 88}")
    print(f" NESTED MODEL COMPARISON: AL (p=1, restricted) vs GaL (p free)"
          f"{f' [{label}]' if label else ''}")
    print(f"{'=' * 88}")
    print(f" n_crash = {n}")
    print(f" AL  : alpha={alpha_al:.4f} (se={se_a_al:.4f}), theta={theta_al:.4f} (se={se_t_al:.4f})")
    print(f" GaL : alpha={alpha_gal:.4f} (se={se_a:.4f}), theta={theta_gal:.4f} (se={se_t:.4f}), "
          f"p={p_gal:.4f} (se={se_p:.4f})")

    results = []
    for lvl in levels:
        q_al = exact_gal_quantile(lvl, alpha_al, theta_al, 1.0)
        q_gal = exact_gal_quantile(lvl, alpha_gal, theta_gal, p_gal)

        hits_al = (S < q_al).astype(int)
        hits_gal = (S < q_gal).astype(int)

        fails_al, p_uc_al, p_cc_al = backtest_var(hits_al, lvl)
        fails_gal, p_uc_gal, p_cc_gal = backtest_var(hits_gal, lvl)

        expected = lvl * n
        results.append(dict(
            level=lvl, n=n, expected=expected,
            al_quantile=q_al, al_violations=fails_al, al_uc_p=p_uc_al, al_cc_p=p_cc_al,
            gal_quantile=q_gal, gal_violations=fails_gal, gal_uc_p=p_uc_gal, gal_cc_p=p_cc_gal,
        ))
        print(f"\n level={lvl:.0%} | expected={expected:.2f}")
        print(f"   AL  (p=1): q={q_al:8.3f} | violations={fails_al:4d} | "
              f"Kupiec p={p_uc_al:.4f} | Christoffersen p={p_cc_al:.4f}")
        print(f"   GaL       : q={q_gal:8.3f} | violations={fails_gal:4d} | "
              f"Kupiec p={p_uc_gal:.4f} | Christoffersen p={p_cc_gal:.4f}")

    return dict(
        alpha_al=alpha_al, theta_al=theta_al, se_a_al=se_a_al, se_t_al=se_t_al,
        alpha_gal=alpha_gal, theta_gal=theta_gal, p_gal=p_gal,
        se_a_gal=se_a, se_t_gal=se_t, se_p_gal=se_p,
        backtest=pd.DataFrame(results),
    )


# ---------------------------------------------------------------------------
# AE point #2
# ---------------------------------------------------------------------------
def _build_ecf_objective(data, deg=50):
    t_nodes, weights = hermgauss(deg)
    t_data = np.outer(t_nodes, data)
    ecf_c = np.mean(np.cos(t_data), axis=1)
    ecf_s = np.mean(np.sin(t_data), axis=1)

    def objective(params):
        alpha, theta, p = params
        if not (0.01 <= alpha <= 2.0 and 0.01 <= p <= 1.0):
            return 1e10
        max_theta = min(np.pi * alpha / 2.0, np.pi - np.pi * alpha / 2.0)
        if abs(theta) > max_theta:
            return 1e10
        sgn_t = np.sign(t_nodes)
        abs_t_alpha = np.abs(t_nodes) ** alpha
        cos_theta_sgn = np.cos(theta * sgn_t)
        sin_theta_sgn = np.sin(theta * sgn_t)
        A = (p + cos_theta_sgn * abs_t_alpha) ** 2 + (sin_theta_sgn * abs_t_alpha) ** 2
        theory_c = p * (p + cos_theta_sgn * abs_t_alpha) / A
        theory_s = p * (sin_theta_sgn * abs_t_alpha) / A
        return np.sum(weights * ((ecf_c - theory_c) ** 2 + (ecf_s - theory_s) ** 2))

    return objective


def multistart_local_vs_global(crash_aggregates, n_starts=30, seed=42, label=""):
    S = np.asarray(crash_aggregates, dtype=float)
    objective = _build_ecf_objective(S)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    bounds = Bounds([0.01, -np.pi, 0.01], [2.0, np.pi, 1.0])
    direct_result = direct(objective, bounds=bounds, maxfun=2000)
    direct_time = time.time() - t0
    global_opt = direct_result.x
    global_val = direct_result.fun

    local_optima = []
    for i in range(n_starts):
        alpha0 = rng.uniform(0.2, 1.9)
        max_th0 = min(np.pi * alpha0 / 2.0, np.pi - np.pi * alpha0 / 2.0)
        theta0 = rng.uniform(-0.9 * max_th0, 0.9 * max_th0)
        p0 = rng.uniform(0.05, 0.95)
        res = minimize(objective, x0=[alpha0, theta0, p0], method="Nelder-Mead",
                        options=dict(xatol=1e-6, fatol=1e-10, maxiter=2000))
        local_optima.append(dict(
            alpha0=alpha0, theta0=theta0, p0=p0,
            alpha_hat=res.x[0], theta_hat=res.x[1], p_hat=res.x[2],
            obj_val=res.fun, converged=res.success,
        ))

    df = pd.DataFrame(local_optima)
    dist_to_global = np.sqrt(
        (df["alpha_hat"] - global_opt[0]) ** 2
        + (df["theta_hat"] - global_opt[1]) ** 2
        + (df["p_hat"] - global_opt[2]) ** 2
    )
    frac_converged_to_global = float(np.mean(dist_to_global < 0.05))
    obj_gap = df["obj_val"] - global_val
    n_distinct = (
        df.round({"alpha_hat": 2, "theta_hat": 2, "p_hat": 2})
        .drop_duplicates(subset=["alpha_hat", "theta_hat", "p_hat"])
        .shape[0]
    )

    print(f"\n{'=' * 88}")
    print(f" MULTI-START LOCAL vs GLOBAL OPTIMIZATION{f' [{label}]' if label else ''}")
    print(f"{'=' * 88}")
    print(f" n_crash={len(S)}, n_starts={n_starts}")
    print(f" DIRECT (global) optimum: alpha={global_opt[0]:.4f}, theta={global_opt[1]:.4f}, "
          f"p={global_opt[2]:.4f}, obj={global_val:.6e}  (time={direct_time:.2f}s)")
    print(f" Local starts landing within 0.05 (Euclidean) of global optimum: "
          f"{frac_converged_to_global:.1%}  ({int(round(frac_converged_to_global * n_starts))}/{n_starts})")
    print(f" Objective gap (local - global): mean={obj_gap.mean():.4e}, "
          f"median={obj_gap.median():.4e}, max={obj_gap.max():.4e}")
    print(f" Distinct local optima found (rounded to 2 dp): {n_distinct} / {n_starts}")

    return dict(
        global_opt=global_opt, global_val=global_val, direct_time=direct_time,
        local_starts=df, frac_converged_to_global=frac_converged_to_global,
        n_distinct_optima=n_distinct,
    )