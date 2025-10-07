# src/option_pricing/models/mc.py
import numpy as np
from typing import Literal, Tuple
from ..utils import seed_rng, payoff_call, payoff_put, validate_positive
from .bs import bsm_price  # analytic used for benchmarking

OptionType = Literal["call", "put"]

def mc_price(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: OptionType = "call",
             n_paths: int = 100000, antithetic: bool = True,
             control_variate: bool = True, seed: int = None) -> Tuple[float, float]:
    """
    Monte Carlo pricing for European options.
    Returns (price_estimate, standard_error)
    - antithetic: if True, uses antithetic variates
    - control_variate: if True, uses S_T (discounted) as control variate (known E)
    """
    validate_positive(S=S, K=K, sigma=sigma, T=T)
    rng = seed_rng(seed)
    dt = T

    if antithetic:
        # ensure even number
        if n_paths % 2 != 0:
            n_paths += 1
        half = n_paths // 2
        z = rng.standard_normal(size=half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(size=n_paths)

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    ST = S * np.exp(drift + diffusion)

    payoff = payoff_call(ST, K) if option_type == "call" else payoff_put(ST, K)
    discounted = np.exp(-r * T) * payoff

    if control_variate:
        # discounted ST has known expectation = S * exp(-q*T)
        control = np.exp(-r * T) * ST
        control_expectation = S * np.exp(-q * T)
        cov = np.cov(discounted, control, bias=True)[0,1]
        var_control = np.var(control)
        beta = cov / var_control if var_control > 0 else 0.0
        adjusted = discounted - beta * (control - control_expectation)
        est = adjusted.mean()
        stderr = adjusted.std(ddof=1) / np.sqrt(len(adjusted))
    else:
        est = discounted.mean()
        stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))

    # analytic price for reference (not returned)
    analytic = bsm_price(S, K, r, q, sigma, T, option_type=option_type)

    return float(est), float(stderr)
