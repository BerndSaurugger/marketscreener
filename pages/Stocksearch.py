import streamlit as st
import pandas as pd
from tradingview_screener import Query, col
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from mplfinance.original_flavor import candlestick_ohlc
from tools.loaddata import implemented_columns, all_columns
from tools.visualizations import visualize_stock_analysis

st.set_page_config("📈 Screener", layout="wide")
st.title("📈 Stock Screener")

# === VOLUME & MOMENTUM ===
st.sidebar.subheader("📊 Volume & Momentum Filters")
REL_VOLUME = st.sidebar.slider("Relative Volume Threshold", 0.5, 5.0, 1.2, 0.1)
RSI_VALUE = st.sidebar.slider("RSI Threshold", 0, 100, 60)

# === DRAWDOWN FILTERS ===
st.sidebar.subheader("📉 Dip from Highs")
ONE_MONTH_DIP = st.sidebar.slider("Price below 1M High (%)", 0, 100, 0)
THREE_MONTH_DIP = st.sidebar.slider("Price below 3M High (%)", 0, 100, 0)
SIX_MONTH_DIP = st.sidebar.slider("Price below 6M High (%)", 0, 100, 0)
ATH_DIP = st.sidebar.slider("Price below All-Time High (%)", 0, 100, 0)
ATH_DIP_STOP = st.sidebar.slider("Price not under All-Time High (%)", 0, 100, 99)

# === TREND FILTERS ===
st.sidebar.subheader("📈 EMA / Trend Filters")
PRICE_ABOVE_EMA = st.sidebar.checkbox("Price Above EMA", value=True)
EMA_CONSTRAINT_DAILY = st.sidebar.checkbox("EMA Constraint (Daily)", value=True)
EMA_CONSTRAINT_HOURLY = st.sidebar.checkbox("EMA Constraint (Hourly)", value=True)
EMA_CONSTRAINT_WEEKLY = st.sidebar.checkbox("EMA Constraint (Weekly)", value=False)

# === INDICATORS ===
st.sidebar.subheader("🔍 Indicator Filters")

# Initialize session state
if "MACD_CONSTRAINT" not in st.session_state:
    st.session_state.MACD_CONSTRAINT = False
if "CROSS_MACD_SIGNAL" not in st.session_state:
    st.session_state.CROSS_MACD_SIGNAL = True

# Draw checkboxes with mutual exclusion
CROSS_MACD_SIGNAL = st.sidebar.checkbox(
    "MACD Crosses Above Signal",
    value=st.session_state.CROSS_MACD_SIGNAL,
    disabled=st.session_state.MACD_CONSTRAINT,
    key="CROSS_MACD_SIGNAL"
)

MACD_CONSTRAINT = st.sidebar.checkbox(
    "MACD Above Signal",
    value=st.session_state.MACD_CONSTRAINT,
    disabled=st.session_state.CROSS_MACD_SIGNAL,
    key="MACD_CONSTRAINT"
)

# Screener filters
filters = [
    col('type') == 'stock',
    col('relative_volume_10d_calc') > REL_VOLUME,
    col('RSI') < RSI_VALUE,
    col('exchange') != 'OTC',
]

if MACD_CONSTRAINT and not CROSS_MACD_SIGNAL:
    filters += [col('MACD.macd') >= col('MACD.signal')]

if CROSS_MACD_SIGNAL:
    filters += [col('MACD.macd').crosses_above(col('MACD.signal'))]

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

# Query
with st.spinner("Running stock query..."):
    try:
        result = (
            Query()
            .select(
                *all_columns()
            )
            .where(*filters)
            .order_by('change', ascending=False)
            # .offset(0)
            .limit(500)
            .get_scanner_data()
        )

        if result is None or result[1] is None or len(result[1]) == 0:
            st.warning("❗ No results found. Try relaxing the filters.")
            st.stop()

        # Extract result
        row_count, df = result
        df.rename(columns={"relative_volume_10d_calc": "rel_vol"}, inplace=True)

        # Post-query: apply 1M dip filter
        df = df[(df["High.1M"] - df["close"]) / df["High.1M"] >= ONE_MONTH_DIP / 100]
        df = df[(df["High.3M"] - df["close"]) / df["High.3M"] >= THREE_MONTH_DIP / 100]
        df = df[(df["High.6M"] - df["close"]) / df["High.6M"] >= SIX_MONTH_DIP / 100]
        df = df[(df["High.All"] - df["close"]) / df["High.All"] >= ATH_DIP / 100]
        df = df[(df["High.All"] - df["close"]) / df["High.All"] <= ATH_DIP_STOP / 100]

        # Compute price target distance
        if "price_target_average" in df.columns:
            df["target up/down"] = (((df["price_target_average"] / df["close"]) - 1) * 100).round(2)

        df = df.sort_values(by="target up/down", ascending=False)

        st.success(f"Showing {len(df)} filtered results.")
        st.dataframe(df[[
            "name", "close","High.1M", "change", "rel_vol", "Perf.W",
            "Perf.1M", "Perf.Y", "Perf.3Y", "Perf.5Y", "Perf.10Y", "Perf.YTD",
            "Volatility.D", "exchange", "sector", "price_target_average", "target up/down"
        ]])
    except Exception as e:
        st.error(f"🚨 Error running screener: {e}")
        st.stop()


if not df.empty:
    tickers_to_plot = st.multiselect("📉 Choose stocks to visualize", df.sort_values(by="name")["name"].tolist())
    visualize_stock_analysis(tickers_to_plot)

