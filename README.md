# Implied Volatility Surface

Builds the implied volatility surface of a listed equity or ETF (SPY by default) from its option chain and plots it in 3D across strike (or moneyness) and time to expiration.

[Live demo on Streamlit](https://volatilitysurface.streamlit.app/) · featured in a Coding Jesus video on YouTube

## How it works

1. **Option chain.** Pulls every listed expiry more than 7 days out from Yahoo Finance (`yfinance`) and keeps calls with a non-zero bid and ask. The mid price is the input.
2. **Implied volatility.** For each contract, solves the Black-Scholes call price for σ with Brent's method (`scipy.optimize.brentq`, bracket 1e-6 to 5), using a user-set risk-free rate and continuous dividend yield. Contracts with no root are dropped.
3. **Surface.** Linear interpolation (`scipy.interpolate.griddata`) onto a 50 x 50 grid of time to expiry and strike, rendered with Plotly.

Controls in the sidebar: ticker, risk-free rate, dividend yield, strike range as a % of spot, and strike vs moneyness on the y-axis.

## Run locally

```bash
pip install -r requirements.txt
streamlit run volatility_surface.py
```

## Limitations

- Calls only, European Black-Scholes pricing applied to American-style options, flat rate and dividend yield.
- Mid prices from delayed Yahoo data; deep OTM and illiquid strikes make the wings noisy.
- Linear interpolation, no arbitrage-free smoothing (e.g. SVI).

MIT License
