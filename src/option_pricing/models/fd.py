# src/option_pricing/models/fd.py
import numpy as np
from typing import Literal
from ..utils import payoff_call, payoff_put, validate_positive

OptionType = Literal["call", "put"]

def fd_price_cn(S: float, K: float, r: float, q: float, sigma: float, T: float,
                s_max_multiplier: float = 3.0, M: int = 400, N: int = 400,
                option_type: OptionType = "call") -> float:
    """
    Crank-Nicolson finite difference for European options.
    - S: spot
    - s_max_multiplier: maximum S considered = multiplier * K (or S)
    - M: number of price grid points
    - N: number of time steps
    Returns option price at S (interpolated).
    """
    validate_positive(S=S, K=K, sigma=sigma, T=T)
    S_max = max(S, K) * s_max_multiplier
    ds = S_max / M
    dt = T / N

    # grid: prices 0..S_max
    grid = np.zeros((M + 1, N + 1))
    S_grid = np.linspace(0, S_max, M + 1)

    # terminal payoff
    if option_type == "call":
        grid[:, -1] = payoff_call(S_grid, K)
    else:
        grid[:, -1] = payoff_put(S_grid, K)

    # boundary conditions for all t
    if option_type == "call":
        grid[-1, :] = S_max - K * np.exp(-r * dt * np.arange(N + 1))
        grid[0, :] = 0.0
    else:
        grid[0, :] = K * np.exp(-r * dt * np.arange(N + 1))  # deep in-the-money put
        grid[-1, :] = 0.0

    # coefficients for CN
    j = np.arange(1, M)
    sigma2 = sigma**2

    a = 0.25 * dt * (sigma2 * j**2 - (r - q) * j)
    b = -0.5 * dt * (sigma2 * j**2 + r)
    c = 0.25 * dt * (sigma2 * j**2 + (r - q) * j)

    A_diag = 1 - b
    A_sub = -a
    A_sup = -c

    B_diag = 1 + b
    B_sub = a
    B_sup = c

    # use scipy solve_banded
    from scipy.linalg import solve_banded

    # time-stepping
    for n in reversed(range(N)):
        rhs = B_sub * grid[0:-2, n+1] + B_diag * grid[1:-1, n+1] + B_sup * grid[2:, n+1]

        # boundaries
        rhs[0] += a[0] * grid[0, n]
        rhs[-1] += c[-1] * grid[-1, n]

        # solve A * x = rhs using banded representation
        ab = np.zeros((3, M-1))
        ab[0, 1:] = A_sub[1:]
        ab[1, :] = A_diag
        ab[2, :-1] = A_sup[:-1]
        x = solve_banded((1,1), ab, rhs)
        grid[1:-1, n] = x

    price = np.interp(S, S_grid, grid[:, 0])
    return float(price)
