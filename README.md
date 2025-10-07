# Option Pricing — Overview & Comparison

## Table of contents

- Features
- Quickstart (install & run)
- Usage (script + Streamlit)
- Project structure
- Development (tests, linting)
- Notes, assumptions and limitations
- Contributing

## Features

- Black-Scholes closed-form pricing + analytical Greeks
- CRR binomial tree pricing (European)
- Finite Difference (Crank–Nicolson) solver for European options
- Monte Carlo pricing with Antithetic variates and Control variate (discounted terminal price)
- Streamlit interactive app to compare pricing methods and visualize results
- Unit tests with `pytest`

## How each model works (simple explanation)

Below are brief, plain-language descriptions of each pricing method included in this project, with an intuitive explanation of how they work and their main trade-offs.

- Black-Scholes-Merton (BSM)
	- What it does: gives a closed-form formula (a direct calculation) for the price of European call and put options when the underlying follows a continuous log-normal diffusion with constant volatility and interest rate.
	- Intuition: it assumes the stock moves continuously and randomly; by solving the resulting partial differential equation you get a neat formula involving the normal distribution.
	- Pros: extremely fast, exact under the model's assumptions, gives analytical Greeks.
	- Cons: relies on idealized assumptions (constant volatility, no jumps, no dividends unless explicitly modeled).

- Cox-Ross-Rubinstein (CRR) Binomial tree
	- What it does: builds a discrete-time recombining tree of possible stock prices over N steps. At each step the price either goes up or down by fixed multipliers; option values are computed backward from expiry by risk-neutral expectation.
	- Intuition: think of time split into small slices; at each slice the stock moves up or down. By pricing from the end to the start you get a fair price today.
	- Pros: simple, flexible (can handle many payoffs), converges to BSM as N increases for European options.
	- Cons: for high accuracy you may need many steps (cost grows with N), American features require early-exercise checks.

- Finite Difference (Crank–Nicolson)
	- What it does: numerically solves the Black-Scholes PDE on a grid of stock prices and times using the Crank–Nicolson scheme, which blends explicit and implicit time-stepping for stability and accuracy.
	- Intuition: discretize the continuous PDE into a system of linear equations on a grid and march backward in time from expiry to today.
	- Pros: good accuracy for European options, flexible for boundary conditions and local volatility if extended.
	- Cons: requires building and solving linear systems, care is needed for boundary truncation, grid resolution choices affect runtime and error.

- Monte Carlo (plain, antithetic, control variate)
	- What it does: simulates many random future price paths for the underlying under the risk-neutral measure, computes the discounted payoff for each path, and averages results to estimate the option price.
	- Intuition: if you can't solve the model analytically, approximate the expected payoff by sampling many possible futures and averaging.
	- Variance reduction techniques included:
		- Antithetic variates: pair each random path with its mirror (negated random draws) so their errors partially cancel.
		- Control variate: use a related quantity with known expected value (e.g., Black-Scholes price of the same option) to reduce variance of the estimator.
	- Pros: extremely flexible (handles path-dependent payoffs, exotic features), straightforward to parallelize.
	- Cons: convergence is slow (error scales with 1/sqrt(N)); to get high precision you need many simulations unless variance reduction is used.

For practical use:

- Use Black-Scholes when the model assumptions are acceptable — it's fast and gives closed-form Greeks.
- Use Binomial for simple, robust checks and when flexibility for payoff forms is needed; increase steps until prices stabilize.
- Use Finite Difference when you need a grid-based PDE solution (or plan to handle local volatility or barriers) and want good accuracy for European payoffs.
- Use Monte Carlo for path-dependent/exotic payoffs or when other methods are infeasible; apply variance reduction and increased sample sizes for precision.


## Quickstart — install & run

Prerequisites:

- Python 3.10+ is recommended (the project was developed and tested on Python 3.11)
- Git (optional but recommended)

1) Create and activate a virtual environment (Windows PowerShell example):

Create the folder structure above.

```powershell
1. Save files with exact paths (e.g. `src/option_pricing/bs.py`, `app/streamlit_app.py`, etc.).
2. `python -m venv .venv 
3.  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
4.   `.venv\Scripts\Activate.ps1 (after opening again VSCode, just run from this line)
5. `pip install -r requirements.txt`
6. Run tests: `pytest`
7. Start the app: `streamlit run app/streamlit_app.py`
```


Recommended: create a virtual environment.


2) Run unit tests:

```powershell
pytest -q
```

3) Run the Streamlit app (interactive comparison):

```powershell
streamlit run app/streamlit_app.py
```

If you want a simple script interface, use the CLI helper:

```powershell
python cli\run.py --help
```

## Usage examples

From Python, you can import the library and price options directly. Example (pseudo):

```python
from src.option_pricing.models.bs import black_scholes_price

price = black_scholes_price(S=100, K=100, r=0.01, sigma=0.2, T=1.0, option_type="call")
print(price)
```

Use the Streamlit app to compare methods interactively. The app purposely exposes the main inputs (S, K, r, sigma, T) and lets you switch methods, number of steps/simulations, and toggles for variance reduction.

## Project structure

Top-level layout (important files/directories):

```
option-pricer/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ main.py                 # optional quick demo runner
├─ app/
│  └─ streamlit_app.py    # Streamlit UI
├─ cli/
│  └─ run.py              # CLI for batch pricing/comparisons
├─ src/
│  └─ option_pricing/
│     ├─ __init__.py
│     ├─ greeks.py
│     ├─ utils.py
│     ├─ viz.py
│     └─ models/
│        ├─ __init__.py
│        ├─ bs.py         # Black-Scholes-Merton implementation
│        ├─ binomial.py
│        ├─ fd.py
│        └─ mc.py
├─ tests/
│  ├─ test_bs.py
│  ├─ test_binomial.py
│  └─ test_mc.py
```

Files in `src/option_pricing` provide the computational core. `app/streamlit_app.py` is a thin UI layer intended for exploratory use.

## Development

- Run all tests: `pytest`
- Run a single test file: `pytest tests/test_bs.py -q`
- Linting: add `flake8`/`ruff` config if desired (not included by default)

If you modify numerical routines, add unit tests covering edge cases (deep ITM/OTM options, zero volatility, zero time to expiry, and negative or boundary interest rates) and numeric stability checks for large numbers of steps/simulations.

## Notes, assumptions and limitations

- Implementations focus on European vanilla options (no American exercise features)
- Binomial tree uses CRR branching and is intended for European payoffs
- Finite difference solver implements Crank–Nicolson for numerical stability; boundary conditions are simple Dirichlet/Neumann approximations depending on payoff
- Monte Carlo supports basic variance reduction (antithetic, control variate) but is not optimized for large-scale parallel runs



