# src/option_pricing/models/binomial.py
import numpy as np
from typing import Literal
from ..utils import payoff_call, payoff_put, validate_positive

OptionType = Literal["call", "put"]

def binomial_price(S: float, K: float, r: float, q: float, sigma: float, T: float, steps: int = 100, option_type: OptionType = "call") -> float:
    """
    CRR binomial tree pricing for European options.
    """
    validate_positive(S=S, K=K, sigma=sigma, T=T)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # terminal stock prices
    j = np.arange(0, steps + 1)
    ST = S * (u ** j) * (d ** (steps - j))
    payoff = payoff_call(ST, K) if option_type == "call" else payoff_put(ST, K)

    # backward induction
    for i in range(steps - 1, -1, -1):
        payoff = disc * (p * payoff[1:i + 2] + (1 - p) * payoff[0:i + 1])
    return float(payoff[0])
