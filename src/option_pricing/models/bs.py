# src/option_pricing/models/bs.py
import numpy as np
from scipy.stats import norm
from typing import Tuple
from ..utils import validate_positive

def _d1d2(S, K, r, q, sigma, T):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2

def bsm_price(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = "call") -> float:
    """
    Black-Scholes-Merton closed-form price for European call/put.
    option_type: "call" or "put"
    """
    validate_positive(S=S, K=K, sigma=sigma, T=T)
    if T == 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1, d2 = _d1d2(S, K, r, q, sigma, T)
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return float(price)

def bsm_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = "call") -> dict:
    """
    Returns a dict: delta, gamma, vega, theta, rho
    Theta reported per year.
    """
    validate_positive(S=S, K=K, sigma=sigma, T=T)
    if T == 0 or sigma == 0:
        # fallback to numerical or trivial
        return {"delta": None, "gamma": None, "vega": None, "theta": None, "rho": None}
    d1, d2 = _d1d2(S, K, r, q, sigma, T)
    pdf_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)

    delta = np.exp(-q * T) * N_d1 if option_type == "call" else -np.exp(-q * T) * N_neg_d1
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)  # per vol (not %)
    theta_call = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * N_d2
                  + q * S * np.exp(-q * T) * N_d1)
    theta_put = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * N_neg_d2
                 - q * S * np.exp(-q * T) * N_neg_d1)
    theta = theta_call if option_type == "call" else theta_put
    rho_call = K * T * np.exp(-r * T) * N_d2
    rho_put = -K * T * np.exp(-r * T) * N_neg_d2
    rho = rho_call if option_type == "call" else rho_put

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),  # per year
        "rho": float(rho)
    }
