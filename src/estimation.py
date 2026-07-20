# -*- coding: utf-8 -*-
"""
Module: estimation.py
Description: Parameter estimation for AL and GaL distributions, including exact quantiles.
"""

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy.optimize import direct, Bounds
from numpy.polynomial.hermite import hermgauss

def gal_characteristic_function(t_val, alpha, theta, p):
    t_arr = np.asarray(t_val, dtype=float)
    sgn_t = np.sign(t_arr)
    abs_t_alpha = np.abs(t_arr) ** alpha
    C = (1.0 / p) * np.exp(-1j * theta * sgn_t) * abs_t_alpha
    D = 1.0 + C
    return 1.0 / D

def gal_cf_gradient(t_val, alpha, theta, p):
    t_arr = np.asarray(t_val, dtype=float)
    sgn_t = np.sign(t_arr)
    abs_t = np.abs(t_arr)

    with np.errstate(divide='ignore'):
        ln_abs_t = np.log(abs_t)

    abs_t_alpha = abs_t ** alpha
    C = (1.0 / p) * np.exp(-1j * theta * sgn_t) * abs_t_alpha
    D = 1.0 + C
    phi = 1.0 / D

    dD_dalpha = C * ln_abs_t
    dD_dalpha = np.where(t_arr == 0, 0.0 + 0.0j, dD_dalpha)
    dD_dtheta = -1j * sgn_t * C
    dD_dp = -(1.0 / p) * C

    dphi_dalpha = -dD_dalpha / D ** 2
    dphi_dtheta = -dD_dtheta / D ** 2
    dphi_dp = -dD_dp / D ** 2

    return phi, dphi_dalpha, dphi_dtheta, dphi_dp

def compute_sandwich_se(t_nodes, weights, params, n_obs, fixed_p=None):
    if fixed_p is not None:
        alpha, theta = params
        p = fixed_p
        n_params = 2
    else:
        alpha, theta, p = params
        n_params = 3

    _, g_a, g_t, g_p = gal_cf_gradient(t_nodes, alpha, theta, p)
    if n_params == 2:
        grad = np.vstack([g_a, g_t])
    else:
        grad = np.vstack([g_a, g_t, g_p])

    H = 2.0 * np.real((grad * weights) @ grad.conj().T)
    diff_grid = t_nodes[:, None] - t_nodes[None, :]
    phi_diff = gal_characteristic_function(diff_grid, alpha, theta, p)
    phi_pos = gal_characteristic_function(t_nodes, alpha, theta, p)
    phi_neg = gal_characteristic_function(-t_nodes, alpha, theta, p)
    Omega = phi_diff - np.outer(phi_pos, phi_neg)

    W = np.outer(weights, weights)
    weighted_Omega = Omega * W
    J = 4.0 * np.real(grad @ weighted_Omega @ grad.conj().T)

    try:
        H_inv = np.linalg.inv(H)
        cov = H_inv @ J @ H_inv / n_obs
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    return se

def theta_bound(alpha):
    return min(np.pi * alpha / 2.0, np.pi - np.pi * alpha / 2.0)

def check_boundary_proximity(alpha, theta, p, label="", check_p=True):
    max_th = theta_bound(alpha)
    theta_ratio = abs(theta) / max_th if max_th > 0 else np.nan
    p_dist_to_bound = min(p - 0.0, 1.0 - p)

    warn_theta = theta_ratio > 0.90
    warn_p = check_p and (p_dist_to_bound < 0.02)

    tag = f" [{label}]" if label else ""
    print(f" [Sinir kontrolu{tag}] |theta|/max_theta = {theta_ratio:.3f}"
          + (" *** SINIRA COK YAKIN ***" if warn_theta else ""))
    if warn_p:
        print(f" [Sinir kontrolu{tag}] p={p:.4f}, sinira uzaklik={p_dist_to_bound:.4f}"
              " *** SINIRA COK YAKIN ***")

    if warn_theta or warn_p:
        print("  -> Bu tahmin icin standart Wald guven araliklari guvenilir "
              "olmayabilir. Karsilastirma icin estimate_*_smooth (tanh "
              "reparametrizasyonu) fonksiyonunu calistirip sonucu "
              "karsilastirin; makalede bu sinir-yakinligini acikca tartisin.")

    return theta_ratio, p_dist_to_bound

def estimate_al_parameters(data, deg=50):
    t_nodes, weights = hermgauss(deg)
    t_data = np.outer(t_nodes, data)
    ecf_c = np.mean(np.cos(t_data), axis=1)
    ecf_s = np.mean(np.sin(t_data), axis=1)
    n_obs = len(data)

    def objective(params):
        alpha, theta = params
        p = 1.0
        max_theta = min(np.pi * alpha / 2.0, np.pi - np.pi * alpha / 2.0)
        if abs(theta) > max_theta: return 1e10
        sgn_t = np.sign(t_nodes)
        abs_t_alpha = np.abs(t_nodes)**alpha
        cos_theta_sgn = np.cos(theta * sgn_t)
        sin_theta_sgn = np.sin(theta * sgn_t)
        A = (p + cos_theta_sgn * abs_t_alpha)**2 + (sin_theta_sgn * abs_t_alpha)**2
        theory_c = p * (p + cos_theta_sgn * abs_t_alpha) / A
        theory_s = p * (sin_theta_sgn * abs_t_alpha) / A
        return np.sum(weights * ((ecf_c - theory_c)**2 + (ecf_s - theory_s)**2))

    bounds = Bounds([0.01, -np.pi], [2.0, np.pi])
    result = direct(objective, bounds=bounds, maxfun=2000)
    opt_params = result.x

    try:
        se = compute_sandwich_se(t_nodes, weights, (opt_params[0], opt_params[1]),
                                  n_obs, fixed_p=1.0)
    except Exception:
        se = np.array([np.nan, np.nan])

    return opt_params[0], opt_params[1], se[0], se[1]

def estimate_gal_parameters(data, deg=50):
    t_nodes, weights = hermgauss(deg)
    t_data = np.outer(t_nodes, data)
    ecf_c = np.mean(np.cos(t_data), axis=1)
    ecf_s = np.mean(np.sin(t_data), axis=1)
    n_obs = len(data)

    def objective(params):
        alpha, theta, p = params
        if not (0.01 <= alpha <= 2.0 and 0.01 <= p <= 1.0):
            return 1e10
        max_theta = min(np.pi * alpha / 2.0, np.pi - np.pi * alpha / 2.0)
        if abs(theta) > max_theta:
            return 1e10
        sgn_t = np.sign(t_nodes)
        abs_t_alpha = np.abs(t_nodes)**alpha
        cos_theta_sgn = np.cos(theta * sgn_t)
        sin_theta_sgn = np.sin(theta * sgn_t)
        A = (p + cos_theta_sgn * abs_t_alpha)**2 + (sin_theta_sgn * abs_t_alpha)**2
        theory_c = p * (p + cos_theta_sgn * abs_t_alpha) / A
        theory_s = p * (sin_theta_sgn * abs_t_alpha) / A
        return np.sum(weights * ((ecf_c - theory_c)**2 + (ecf_s - theory_s)**2))

    bounds = Bounds([0.01, -np.pi, 0.01], [2.0, np.pi, 1.0])
    result = direct(objective, bounds=bounds, maxfun=2000)
    opt_params = result.x

    try:
        se = compute_sandwich_se(t_nodes, weights,
                                  (opt_params[0], opt_params[1], opt_params[2]),
                                  n_obs, fixed_p=None)
    except Exception:
        se = np.array([np.nan, np.nan, np.nan])

    return opt_params[0], opt_params[1], opt_params[2], se[0], se[1], se[2]

def estimate_gal_parameters_smooth(data, deg=50, maxfun=3000):
    t_nodes, weights = hermgauss(deg)
    t_data = np.outer(t_nodes, data)
    ecf_c = np.mean(np.cos(t_data), axis=1)
    ecf_s = np.mean(np.sin(t_data), axis=1)
    n_obs = len(data)

    def unpack(u):
        alpha_raw, psi, chi = u
        alpha = 0.01 + (2.0 - 0.01) / (1.0 + np.exp(-alpha_raw))
        max_th = theta_bound(alpha)
        theta = max_th * np.tanh(psi)
        p = 1.0 / (1.0 + np.exp(-chi))
        return alpha, theta, p

    def objective(u):
        alpha, theta, p = unpack(u)
        sgn_t = np.sign(t_nodes)
        abs_t_alpha = np.abs(t_nodes) ** alpha
        cos_theta_sgn = np.cos(theta * sgn_t)
        sin_theta_sgn = np.sin(theta * sgn_t)
        A = (p + cos_theta_sgn * abs_t_alpha) ** 2 + (sin_theta_sgn * abs_t_alpha) ** 2
        theory_c = p * (p + cos_theta_sgn * abs_t_alpha) / A
        theory_s = p * (sin_theta_sgn * abs_t_alpha) / A
        return np.sum(weights * ((ecf_c - theory_c) ** 2 + (ecf_s - theory_s) ** 2))

    bounds = Bounds([-8.0, -8.0, -8.0], [8.0, 8.0, 8.0])
    result = direct(objective, bounds=bounds, maxfun=maxfun)
    alpha_hat, theta_hat, p_hat = unpack(result.x)

    try:
        se = compute_sandwich_se(t_nodes, weights, (alpha_hat, theta_hat, p_hat),
                                  n_obs, fixed_p=None)
    except Exception:
        se = np.array([np.nan, np.nan, np.nan])

    return alpha_hat, theta_hat, p_hat, se[0], se[1], se[2]

def gal_cf(t_val, alpha, theta, p):
    sgn_t = np.sign(t_val)
    abs_t_alpha = np.abs(t_val)**alpha
    complex_term = np.exp(-1j * theta * sgn_t)
    return 1.0 / (1.0 + (1.0 / p) * complex_term * abs_t_alpha)

def exact_gal_cdf(x, alpha, theta, p):
    def integrand(t_val):
        if t_val == 0:
            return 0.0
        cf_val = gal_cf(t_val, alpha, theta, p)
        return np.imag(np.exp(-1j * t_val * x) * cf_val) / t_val

    integral_val, _ = integrate.quad(integrand, 1e-10, np.inf, limit=2000)
    return 0.5 - (1.0 / np.pi) * integral_val

def exact_gal_quantile(level, alpha, theta, p):
    objective = lambda x: exact_gal_cdf(x, alpha, theta, p) - level
    try:
        return optimize.brentq(objective, -100.0, 0.0)
    except ValueError:
        return optimize.brentq(objective, -2000.0, 100.0)