import streamlit as st
import pandas as pd
from tools.visualizations import visualize_stock_analysis
from tools.loaddata import load_watchlist, add_to_watchlist, remove_from_watchlist


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
tickers = sorted(load_watchlist())
if tickers:
    selected_tickers = st.multiselect("Select stocks from your watchlist to analyze", options=tickers)
    if selected_tickers:
        visualize_stock_analysis(selected_tickers)
else:
    st.info("Your watchlist is empty. Add tickers using the sidebar.")
