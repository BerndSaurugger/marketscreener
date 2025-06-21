import streamlit as st
import pandas as pd
from tools.visualizations import visualize_stock_analysis

WATCHLIST_FILE = "watchlist.csv"

def load_watchlist():
    try:
        return pd.read_csv(WATCHLIST_FILE)["ticker"].tolist()
    except Exception:
        return []

def save_watchlist(tickers):
    pd.DataFrame({"ticker": tickers}).to_csv(WATCHLIST_FILE, index=False)

def add_to_watchlist(ticker):
    tickers = load_watchlist()
    ticker = ticker.upper()
    if ticker and ticker not in tickers:
        tickers.append(ticker)
        save_watchlist(tickers)
        st.sidebar.success(f"Added {ticker} to your watchlist!")
    elif ticker in tickers:
        st.sidebar.warning(f"{ticker} is already in your watchlist.")

def remove_from_watchlist(ticker):
    tickers = load_watchlist()
    ticker = ticker.upper()
    if ticker in tickers:
        tickers.remove(ticker)
        save_watchlist(tickers)
        st.sidebar.success(f"Removed {ticker} from your watchlist.")
    else:
        st.sidebar.warning(f"{ticker} is not in your watchlist.")

# Sidebar
st.set_page_config("📌 Watchlist", layout="wide")
st.title("📌 My Watchlist")

with st.sidebar:
    st.subheader("➕ Add to Watchlist")
    new_ticker = st.text_input("Enter stock ticker", value="")

    if st.button("Add to Watchlist"):
        add_to_watchlist(new_ticker)

    st.subheader("🗑️ Remove from Watchlist")
    tickers = load_watchlist()
    ticker_to_remove = st.selectbox("Select ticker to remove", options=[""] + tickers)
    if ticker_to_remove and st.button("Remove from Watchlist"):
        remove_from_watchlist(ticker_to_remove)

# Main page
tickers = load_watchlist()

if tickers:
    selected_tickers = st.multiselect("Select stocks from your watchlist to analyze", options=tickers)
    if selected_tickers:
        visualize_stock_analysis(selected_tickers)
else:
    st.info("Your watchlist is empty. Add tickers using the sidebar.")
