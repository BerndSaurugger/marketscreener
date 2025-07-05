import streamlit as st
import pandas as pd
from tools.loaddata import load_watchlist, add_to_watchlist, remove_from_watchlist
from tools.model_utils import train_model_for_ticker, load_models_and_features, infer_signals
from tools.machinelearning import get_enriched_stock_data

st.set_page_config(page_title="Stock Signals", layout="wide")

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

    st.subheader("⚙️ Hyperparameters")

    schalter = st.toggle('Choose Watchist or Recommendations')

    if schalter:
        st.write('Use Watchist.')
        tickers = sorted(load_watchlist())
    else:
        st.write('Use Recommentdations.')
        tickers = pd.read_csv("nasdaq.csv")['Symbol'].tolist()
    long_change_perc = st.slider("Long Model Change Threshold (%)", 0, 200, 60, 1)
    long_change = long_change_perc / 100.0
    long_change_days = st.slider("Days for Long Model Change", 1, 60, 20, 1)

    short_change_perc_positive = st.slider("Short Model Change Threshold (%) (higher slider → more negative)", 0, 99, 15, 1)
    short_change = -short_change_perc_positive / 100.0
    short_change_days = st.slider("Days for Short Model Change", 1, 60, 20, 1)

st.title("📈 Signal Finder")

if st.button("🔁 Train all Models"):
    progress = st.progress(0)
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            train_model_for_ticker(ticker, long_change, long_change_days, short_change, short_change_days)
            # st.success(f"{ticker} trained.")
        except Exception as e:
            continue
            # st.error(f"{ticker} failed: {e}")
        progress.progress(idx / total)

if st.button("📊 Show Signals Table"):
    all_results = []
    progress = st.progress(0)
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            model_data = load_models_and_features(ticker)
            if model_data is None:
                df_inf, model_long, model_short, feature_cols = train_model_for_ticker(
                    ticker, long_change, long_change_days, short_change, short_change_days
                )
            else:
                model_long = model_data["model_long"]
                model_short = model_data["model_short"]
                feature_cols = model_data["feature_cols"]
                df_inf = get_enriched_stock_data(ticker, change=0, inference=True)

            result = infer_signals(ticker, df_inf, model_long, model_short, feature_cols)
            if result:
                all_results.append(result)

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")

        progress.progress(idx / total)

    if all_results:
        df_signals = pd.DataFrame(all_results)

        # Konvertiere Datumsspalten zu datetime
        if "Last Long Signal" in df_signals.columns:
            df_signals["Last Long Signal"] = pd.to_datetime(df_signals["Last Long Signal"], errors='coerce')
        if "Last Short Signal" in df_signals.columns:
            df_signals["Last Short Signal"] = pd.to_datetime(df_signals["Last Short Signal"], errors='coerce')

        today = pd.Timestamp.today()
        delta_days = pd.Timedelta(days=5)

        # Tabelle 1: Long-Signale in letzten 5 Tagen
        recent_longs = df_signals[df_signals["Last Long Signal"].notna() & (today - df_signals["Last Long Signal"] <= delta_days)]
        st.write("### 🟩 **Long-Signale in den letzten 5 Tagen**")
        st.dataframe(recent_longs)

        # Tabelle 2: Short-Signale in letzten 5 Tagen
        recent_shorts = df_signals[df_signals["Last Short Signal"].notna() & (today - df_signals["Last Short Signal"] <= delta_days)]
        st.write("### 🟥 **Short-Signale in den letzten 5 Tagen**")
        st.dataframe(recent_shorts)

        # Tabelle 3: Long nach Short
        long_after_short = df_signals[
            df_signals["Last Long Signal"].notna() &
            df_signals["Last Short Signal"].notna() &
            (df_signals["Last Long Signal"] > df_signals["Last Short Signal"])
        ]
        st.write("### 🟩 **Long nach Short** 📉➡️📈")
        st.dataframe(long_after_short)

        # Tabelle 4: Short nach Long
        short_after_long = df_signals[
            df_signals["Last Long Signal"].notna() &
            df_signals["Last Short Signal"].notna() &
            (df_signals["Last Short Signal"] > df_signals["Last Long Signal"])
        ]
        st.write("### 🟥 **Short nach Long** 📈➡️📉")
        st.dataframe(short_after_long)

    else:
        st.warning("Keine Signale gefunden. Bitte zuerst Modelle trainieren.")
