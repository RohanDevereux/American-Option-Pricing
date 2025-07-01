# =====================================================================
# Monte-Carlo demo: pricing an American CALL on AAPL
# – pulls dividend yield, risk-free rate and ATM implied vol from Yahoo
# – simulates risk-neutral price paths
# – values the option with the Longstaff-Schwartz regression trick
# =====================================================================

# install: pip install yfinance pandas_market_calendars matplotlib 

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import math, warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------
# path generator: risk-neutral Geometric Brownian Motion
# ---------------------------------------------------------------------
def simulate_paths(S0, r, q, sigma, n_days, n_paths,
                   seed=42, antithetic=True) -> pd.DataFrame:
    """Return a DataFrame of size (n_days+1 , n_paths) with price paths."""
    if seed is not None:
        np.random.seed(seed)

    # cheap variance reduction: pair each normal shock with its negative
    if antithetic:
        n_paths += n_paths % 2                   # make even
        Z_half = np.random.normal(size=(n_days, n_paths // 2))
        Z = np.hstack([Z_half, -Z_half])
    else:
        Z = np.random.normal(size=(n_days, n_paths))

    dt = 1 / 252                                # one trading day
    drift = (r - q - 0.5 * sigma**2) * dt
    shocks = sigma * np.sqrt(dt) * Z
    log_paths = np.vstack([np.zeros(n_paths), drift + shocks]).cumsum(axis=0)

    return pd.DataFrame(S0 * np.exp(log_paths),
                        columns=[f"sim_{i}" for i in range(n_paths)])

# ---------------------------------------------------------------------
# Longstaff–Schwartz: regression-based early-exercise decision
# ---------------------------------------------------------------------
def ls_price(strike, paths, r, basis_fns):
    """Price an American call and return (value , 95%-CI half-width)."""
    dt = 1 / 252
    disc = math.exp(-r * dt)

    payoff = np.maximum(paths.values - strike, 0)
    cash = np.zeros_like(payoff)
    cash[-1] = payoff[-1]                       # must exercise at expiry

    # walk backwards through time
    for t in range(len(paths) - 2, 0, -1):
        itm = payoff[t] > 0                     # only ITM paths matter
        if not itm.any():
            continue
        X = np.column_stack([f(paths.values[t, itm]) for f in basis_fns])
        y = cash[t + 1, itm] * disc            # continuation if we wait one day
        coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
        cont = X @ coeff                       # fitted continuation value
        ex_now = payoff[t, itm] >= cont
        idx = np.where(itm)[0][ex_now]         # columns that exercise now
        cash[t, idx] = payoff[t, idx]
        cash[t + 1:, idx] = 0                  # cancel future cash-flows

    disc_vec = np.exp(-r * dt * np.arange(len(paths)))[:, None]
    pv = (cash * disc_vec).sum(axis=0)
    mean = pv.mean()
    se = pv.std(ddof=1) / np.sqrt(paths.shape[1])
    return mean, 1.96 * se

# ---------------------------------------------------------------------
# pull AAPL market data
# ---------------------------------------------------------------------
symbol = "AAPL"
tk = yf.Ticker(symbol)
spot = float(tk.history("1d")["Close"][-1])

# dividend yield comes back as “percent” (e.g. 0.52 for 52 %), convert to fraction
q = float(tk.info["dividendYield"]) / 100

# 3-month Treasury bill (^IRX) close also quoted in percent
r = float(yf.Ticker("^IRX").history("5d")["Close"].dropna()[-1]) / 100

# nearest listed option expiry at least 90 calendar days away
exp_dates = pd.to_datetime(tk.options)
today = pd.Timestamp.today().normalize()
expiry = exp_dates[exp_dates >= today + pd.Timedelta(days=90)].min()
expiry_str = expiry.strftime("%Y-%m-%d")

chain = tk.option_chain(expiry_str).calls
chain["diff"] = abs(chain["strike"] - spot)
atm_row = chain.loc[chain["diff"].idxmin()]
atm_strike = float(atm_row["strike"])
market_px = float(atm_row["lastPrice"])
sigma_iv = float(atm_row["impliedVolatility"])

# NYSE business-day count to expiry
nyse = mcal.get_calendar("NYSE")
n_days = len(mcal.date_range(nyse.schedule(today, expiry), frequency="1D"))

# ---------------------------------------------------------------------
# simulate paths and show a handful
# ---------------------------------------------------------------------
paths = simulate_paths(spot, r, q, sigma_iv, n_days, 20_000)

plt.figure(figsize=(9, 5))
for col in paths.columns[:50]:
    plt.plot(paths.index, paths[col], alpha=.4, lw=.8)
plt.title(f"Risk-neutral GBM paths  (S₀={spot:.2f}, σ={sigma_iv:.2%}, r={r:.2%}, q={q:.2%})")
plt.xlabel("trading day"); plt.ylabel("price"); plt.grid(True); plt.show()

# ---------------------------------------------------------------------
# price the ATM American call with two regression bases
# ---------------------------------------------------------------------
basis_lin = [lambda x: np.ones_like(x), lambda x: x]
basis_quad = basis_lin + [lambda x: x**2]

price_lin, ci_lin = ls_price(atm_strike, paths, r, basis_lin)
price_quad, ci_quad = ls_price(atm_strike, paths, r, basis_quad)

print(f"Spot         : {spot:.2f}")
print(f"Dividend yld : {q:.2%}")
print(f"Risk-free r  : {r:.2%}")
print(f"Expiry       : {expiry_str}  ({n_days} trading days)")
print(f"ATM strike   : {atm_strike}")
print(f"Market price : {market_px:.4f}")
print(f"LS linear    : {price_lin:.4f} ±{ci_lin:.4f}")
print(f"LS quadratic : {price_quad:.4f} ±{ci_quad:.4f}")

# ---------------------------------------------------------------------
# full smile comparison
# ---------------------------------------------------------------------
strikes = sorted(chain["strike"].unique())
mkt_px = [chain.loc[chain["strike"] == k, "lastPrice"].iloc[0] for k in strikes]
ls_lin_smile = [ls_price(k, paths, r, basis_lin)[0]  for k in strikes]
ls_quad_smile = [ls_price(k, paths, r, basis_quad)[0] for k in strikes]

plt.figure(figsize=(9, 5))
plt.plot(strikes, mkt_px, "o-", label="Market")
plt.plot(strikes, ls_lin_smile, "s-", label="LS linear")
plt.plot(strikes, ls_quad_smile, "^-", label="LS quadratic")
plt.axvline(atm_strike, color="gray", ls="--", lw=0.8)
plt.title(f"{symbol} call prices vs. strike — expiry {expiry_str}")
plt.xlabel("strike"); plt.ylabel("price"); plt.legend(); plt.grid(True); plt.show()
