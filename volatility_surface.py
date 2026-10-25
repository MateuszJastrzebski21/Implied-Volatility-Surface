import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go


st.title('Implied Volatility Surface')

def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan

    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan

    return implied_vol

st.sidebar.header('Model Parameters')
st.sidebar.write('Adjust the parameters for the Black-Scholes model.')

risk_free_rate = st.sidebar.number_input(
    'Risk-Free Rate (e.g., 0.015 for 1.5%)',
    value=0.015,
    format="%.4f"
)

dividend_yield = st.sidebar.number_input(
    'Dividend Yield (e.g., 0.013 for 1.3%)',
    value=0.013,
    format="%.4f"
)

st.sidebar.header('Visualization Parameters')
y_axis_option = st.sidebar.selectbox(
    'Select Y-axis:',
    ('Strike Price ($)', 'Moneyness')
)

st.sidebar.header('Ticker Symbol')
ticker_symbol = st.sidebar.text_input(
    'Enter Ticker Symbol',
    value='SPY',
    max_chars=10
).upper()

st.sidebar.header('Strike Price Filter Parameters')

min_strike_pct = st.sidebar.number_input(
    'Minimum Strike Price (% of Spot Price)',
    min_value=50.0,
    max_value=199.0,
    value=80.0,
    step=1.0,
    format="%.1f"
)

max_strike_pct = st.sidebar.number_input(
    'Maximum Strike Price (% of Spot Price)',
    min_value=51.0,
    max_value=200.0,
    value=120.0,
    step=1.0,
    format="%.1f"
)

if min_strike_pct >= max_strike_pct:
    st.sidebar.error('Minimum percentage must be less than maximum percentage.')
    st.stop()

ticker = yf.Ticker(ticker_symbol)

today = pd.Timestamp('today').normalize()

try:
    expirations = ticker.options
except Exception as e:
    st.error(f'Error fetching options for {ticker_symbol}: {e}')
    st.stop()

exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]

if not exp_dates:
    st.error(f'No available option expiration dates for {ticker_symbol}.')
else:
    option_data = []

    for exp_date in exp_dates:
        try:
            opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
            calls = opt_chain.calls
        except Exception as e:
            st.warning(f'Failed to fetch option chain for {exp_date.date()}: {e}')
            continue

        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]

        for index, row in calls.iterrows():
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2

            option_data.append({
                'expirationDate': exp_date,
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'mid': mid_price
            })

    if not option_data:
        st.error('No option data available after filtering.')
    else:
        options_df = pd.DataFrame(option_data)

        try:
            spot_history = ticker.history(period='5d')
            if spot_history.empty:
                st.error(f'Failed to retrieve spot price data for {ticker_symbol}.')
                st.stop()
            else:
                spot_price = spot_history['Close'].iloc[-1]
        except Exception as e:
            st.error(f'An error occurred while fetching spot price data: {e}')
            st.stop()

        options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
        options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

        options_df = options_df[
            (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
            (options_df['strike'] <= spot_price * (max_strike_pct / 100))
        ]

        options_df.reset_index(drop=True, inplace=True)

        with st.spinner('Calculating implied volatility...'):
            options_df['impliedVolatility'] = options_df.apply(
                lambda row: implied_volatility(
                    price=row['mid'],
                    S=spot_price,
                    K=row['strike'],
                    T=row['timeToExpiration'],
                    r=risk_free_rate,
                    q=dividend_yield
                ), axis=1
            )

        options_df.dropna(subset=['impliedVolatility'], inplace=True)

        options_df['impliedVolatility'] *= 100

        options_df.sort_values('strike', inplace=True)

        options_df['moneyness'] = options_df['strike'] / spot_price

        if y_axis_option == 'Strike Price ($)':
            Y = options_df['strike'].values
            y_label = 'Strike Price ($)'
        else:
            Y = options_df['moneyness'].values
            y_label = 'Moneyness (Strike / Spot)'

        X = options_df['timeToExpiration'].values
        Z = options_df['impliedVolatility'].values

        ti = np.linspace(X.min(), X.max(), 50)
        ki = np.linspace(Y.min(), Y.max(), 50)
        T, K = np.meshgrid(ti, ki)

        Zi = griddata((X, Y), Z, (T, K), method='linear')

        Zi = np.ma.array(Zi, mask=np.isnan(Zi))

        fig = go.Figure(data=[go.Surface(
            x=T, y=K, z=Zi,
            colorscale='Viridis',
            colorbar_title='Implied Volatility (%)'
        )])

        fig.update_layout(
            title=f'Implied Volatility Surface for {ticker_symbol} Options',
            scene=dict(
                xaxis_title='Time to Expiration (years)',
                yaxis_title=y_label,
                zaxis_title='Implied Volatility (%)'
            ),
            autosize=False,
            width=900,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        st.plotly_chart(fig)

        st.write("---")
        st.markdown(
            "Created by Mateusz Jastrzębski  |   [LinkedIn](https://www.linkedin.com/in/mateuszjastrzebski21/)"
        )
