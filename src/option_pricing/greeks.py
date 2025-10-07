# src/option_pricing/greeks.py
from typing import Callable
import numpy as np

def finite_diff_greeks(price_func: Callable, S: float, bump: float = 1e-4, **kwargs) -> dict:
    """
    Generic finite-difference greeks with bump-and-revalue.
    price_func should accept S as first positional argument or via kwargs.
    Returns delta, gamma, vega (w.r.t sigma), theta (small dt), rho.
    """
    # delta (central)
    p_plus = price_func(S + bump, **kwargs)
    p_minus = price_func(S - bump, **kwargs)
    delta = (p_plus - p_minus) / (2 * bump)
    gamma = (p_plus - 2 * price_func(S, **kwargs) + p_minus) / (bump**2)

    # vega: bump sigma
    sigma = kwargs.get("sigma")
    if sigma is None:
        vega = None
    else:
        h = max(1e-4, sigma * 1e-3)
        kwargs_sigma_up = dict(kwargs, sigma=sigma + h)
        kwargs_sigma_down = dict(kwargs, sigma=sigma - h)
        vega = (price_func(S, **kwargs_sigma_up) - price_func(S, **kwargs_sigma_down)) / (2 * h)

    # theta: small time bump (reduce T)
    T = kwargs.get("T")
    if T is None or T <= 0:
        theta = None
    else:
        hT = min(1e-4, T * 1e-4)
        kwargs_T_up = dict(kwargs, T=max(1e-12, T - hT))
        theta = (price_func(S, **kwargs_T_up) - price_func(S, **kwargs)) / (-hT)  # per year

    # rho: bump r
    r = kwargs.get("r")
    if r is None:
        rho = None
    else:
        hr = 1e-5
        kwargs_r_up = dict(kwargs, r=r + hr)
        kwargs_r_dn = dict(kwargs, r=r - hr)
        rho = (price_func(S, **kwargs_r_up) - price_func(S, **kwargs_r_dn)) / (2 * hr)

    return {"delta": float(delta), "gamma": float(gamma), "vega": vega, "theta": theta, "rho": rho}
