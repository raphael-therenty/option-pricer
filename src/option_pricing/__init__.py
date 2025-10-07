# src/option_pricing/__init__.py
"""
Top-level package exports for option_pricing.

This file re-exports the main pricing functions so external imports can remain:
    from src.option_pricing import bsm_price, bsm_greeks, binomial_price, fd_price_cn, mc_price
even though model modules are located in src/option_pricing/models/.
"""

# re-export models from the new models subpackage
from .models.bs import bsm_price, bsm_greeks
from .models.binomial import binomial_price
from .models.fd import fd_price_cn
from .models.mc import mc_price

# helpers still in package root
from .greeks import finite_diff_greeks
from .utils import payoff_call, payoff_put, seed_rng

__all__ = [
    "bsm_price",
    "bsm_greeks",
    "binomial_price",
    "fd_price_cn",
    "mc_price",
    "finite_diff_greeks",
    "payoff_call",
    "payoff_put",
    "seed_rng",
]
