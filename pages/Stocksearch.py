import streamlit as st
import pandas as pd
from tradingview_screener import Query, col

st.set_page_config("📈 Screener", layout="wide")
st.title("📈 Stock Screener")

# Sidebar filters
REL_VOLUME = st.sidebar.slider("Relative Volume Threshold", 0.5, 5.0, 1.2, 0.1)
RSI_VALUE = st.sidebar.slider("RSI Threshold", 0, 100, 60)
PRICE_ABOVE_EMA = st.sidebar.checkbox("Price Above EMA", value=True)
EMA_CONSTRAINT_DAILY = st.sidebar.checkbox("EMA Constraint (Daily)", value=True)
EMA_CONSTRAINT_HOURLY = st.sidebar.checkbox("EMA Constraint (Hourly)", value=True)
EMA_CONSTRAINT_WEEKLY = st.sidebar.checkbox("EMA Constraint (Weekly)", value=False)
MACD_CONSTRAINT = st.sidebar.checkbox("MACD Constraint", value=True)

# Always refresh results
filters = [
    col('type') == 'stock',
    col('relative_volume_10d_calc') > REL_VOLUME,
    col('RSI') < RSI_VALUE,
    col('exchange') != 'OTC',
]

if MACD_CONSTRAINT:
    filters += [col('MACD.macd') >= col('MACD.signal')]

if PRICE_ABOVE_EMA:
    filters += [
        col('close') > col('EMA9'),
        col('close') > col('EMA20'),
        col('close') > col('EMA50'),
        col('close') > col('EMA200'),
    ]

if EMA_CONSTRAINT_DAILY:
    filters += [
        col('EMA9') > col('EMA20'),
        col('EMA20') > col('EMA50'),
        col('EMA50') > col('EMA200'),
    ]

if EMA_CONSTRAINT_HOURLY:
    filters += [
        col('EMA9|60') > col('EMA20|60'),
        col('EMA20|60') > col('EMA50|60'),
        col('EMA50|60') > col('EMA200|60'),
    ]

if EMA_CONSTRAINT_WEEKLY:
    filters += [
        col('EMA9|1W') > col('EMA20|1W'),
        col('EMA20|1W') > col('EMA50|1W'),
        col('EMA50|1W') > col('EMA200|1W'),
    ]

# Query results on every change
with st.spinner("Running stock query..."):
    try:
        result = (
            Query()
            .select(
                'name', "close",
                "relative_volume_10d_calc",
                "EMA9", "EMA20", "EMA50", "EMA200",
                "EMA9|60", "EMA20|60", "EMA50|60", "EMA200|60", 
                "EMA9|240", "EMA20|240", "EMA50|240", "EMA200|240", 
                "EMA9|1W", "EMA20|1W", "EMA50|1W", "EMA200|1W", 
                "RSI7", "RSI",
                'change|60', 'change|240', 'change', 'change|1W', 'change|1M',
                'MACD.macd', 'MACD.signal',
                "market_cap_basic",
                "High.1M", "High.3M", "High.6M", "High.All",
                "Low.1M", "Low.3M", "Low.6M", "Low.All",
                'type', 'exchange',
                "Volatility.D",
                "Perf.W", "Perf.1M", "Perf.Y", "Perf.3Y", "Perf.5Y", "Perf.10Y", "Perf.YTD",
                "sector", "price_target_average"
            )
            .where(*filters)
            .order_by('change', ascending=False)
            .offset(5)
            .limit(25)
            .get_scanner_data()
        )

        if result is None or result[1] is None or len(result[1]) == 0:
            st.warning("❗ No results found. Try relaxing the filters.")
            st.stop()
        else:
            row_count, df = result

            # Post-processing
            df.rename(columns={"relative_volume_10d_calc": "rel_vol"}, inplace=True)
            df["target up/down"] = (((df["price_target_average"] / df["close"]) - 1) * 100).round(2)
            df = df.sort_values(by="target up/down", ascending=False)

            st.success(f"Showing {len(df)} filtered results.")
            st.dataframe(df[[
                "name", "close", "change", "rel_vol", "Perf.W",
                "Perf.1M", "Perf.Y", "Perf.3Y", "Perf.5Y", "Perf.10Y", "Perf.YTD",
                "Volatility.D", "exchange", "sector", "price_target_average", "target up/down"
            ]])
    except Exception as e:
        st.error(f"🚨 Error running screener: {e}")
        st.stop()

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from mplfinance.original_flavor import candlestick_ohlc

# Define charting functions
def kalman_filter(observations, process_noise=0.001, measurement_noise=0.1):
    estimates = []
    estimate = observations[0]
    error_cov = 1.0
    for z in observations:
        prior_estimate = estimate
        prior_error_cov = error_cov + process_noise
        kalman_gain = prior_error_cov / (prior_error_cov + measurement_noise)
        estimate = prior_estimate + kalman_gain * (z - prior_estimate)
        error_cov = (1 - kalman_gain) * prior_error_cov
        estimates.append(estimate)
    return estimates

def heikin_ashi(df, smoothing_length=10):
    ha = pd.DataFrame(index=df.index)
    ha['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha['Open'] = 0.0
    for i in range(len(df)):
        if i == 0:
            ha.iat[0, ha.columns.get_loc('Open')] = df['Open'].iloc[0]
        else:
            ha.iat[i, ha.columns.get_loc('Open')] = (ha['Open'].iloc[i-1] + ha['Close'].iloc[i-1]) / 2
    ha['High'] = ha[['Open', 'Close']].join(df['High']).max(axis=1)
    ha['Low'] = ha[['Open', 'Close']].join(df['Low']).min(axis=1)
    ha['Volume'] = df['Volume']
    ha = ha.rolling(window=smoothing_length).mean()
    return ha.dropna()

def calculate_macd(price, fast=12, slow=26, signal=9):
    exp1 = price.ewm(span=fast, adjust=False).mean()
    exp2 = price.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(price, period=14):
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if not df.empty:
    tickers_to_plot = st.multiselect("📉 Choose stocks to visualize", df.sort_values(by="name")["name"].tolist())

    if tickers_to_plot:
        st.subheader("📈 Advanced Visualizations")
        data = yf.download(tickers_to_plot, interval="1d", group_by="ticker", start="2025-01-01", progress=False)
        for ticker in tickers_to_plot:
            # try:
            st.markdown(f"### {ticker}")

            df_ticker = data[ticker].dropna()


            # === Analysis logic ===
            close = df_ticker["Close"]
            smoothed = pd.Series(kalman_filter(close), index=close.index)

            df_ha = heikin_ashi(df_ticker)
            df_ha.columns = ['Open', 'Close', 'High', 'Low', 'Volume']
            df_ha = df_ha[['Open', 'High', 'Low', 'Close', 'Volume']]
            macd_line, signal_line, hist = calculate_macd(close)
            rsi = calculate_rsi(close)

            # Align
            dates = df_ha.index
            close_aligned = close.loc[dates]
            smoothed_aligned = smoothed.loc[dates]
            macd_line = macd_line.loc[dates]
            signal_line = signal_line.loc[dates]
            hist = hist.loc[dates]
            rsi = rsi.loc[dates]

            # OHLC for plotting
            df_ha_ohlc = df_ha.copy()
            df_ha_ohlc['Date'] = mdates.date2num(df_ha_ohlc.index.to_pydatetime())
            quotes = [tuple(x) for x in df_ha_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values]

            # Plot
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 1], hspace=0.1)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)

            for quote in quotes:
                color = 'indianred' if quote[4] >= quote[1] else 'mediumseagreen'
                candlestick_ohlc(ax1, [quote], width=0.6, colorup=color, colordown=color)

            ax1.plot(dates, close_aligned, color='darkblue', linestyle=':', label='Price')
            ax1.plot(dates, smoothed_aligned, color='gray', label='Kalman Smoothed')
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(dates, rsi, label='RSI', color='darkblue')
            ax2.axhline(70, color='red', linestyle='--')
            ax2.axhline(30, color='green', linestyle='--')
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI")
            ax2.grid(True)

            ax3.plot(dates, macd_line, label='MACD', color='purple')
            ax3.plot(dates, signal_line, label='Signal', color='gray')
            ax3.bar(dates, hist, color=['green' if h >= 0 else 'red' for h in hist], width=0.6, alpha=0.3)
            ax3.set_ylabel("MACD")
            ax3.legend()
            ax3.grid(True)

            ax3.xaxis_date()
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.xticks(rotation=45)

            plt.tight_layout()
            st.pyplot(fig)

            # except Exception as e:
            #     st.error(f"❌ Error plotting {ticker}: {e}")
