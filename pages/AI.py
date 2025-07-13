import streamlit as st
import pandas as pd
from tools.visualizations import visualize_stock_analysis
from tools.loaddata import load_watchlist, add_to_watchlist, remove_from_watchlist
from tools.machinelearning import get_enriched_stock_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import yfinance as yf


def display_metrics_and_explanation(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    labels_present = np.unique(np.concatenate((y_true, y_pred)))
    cm_df = pd.DataFrame(cm, index=[f"Actual {i}" for i in labels_present],
                            columns=[f"Predicted {i}" for i in labels_present])

    st.subheader(f"\U0001F4CA Prediction Quality – {label} Strategy")
    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")

    with st.expander(f"\U0001F4D8 How to interpret the {label} model performance"):
        st.write("- **Accuracy**: Overall how often the model is correct (both buy/short and hold).")
        st.write("- **Precision**: For the trades predicted by the model, how many were correct. Useful to avoid false buys/shorts.")
        st.write("- **Recall**: Out of all real profitable situations, how many the model successfully caught. Helps minimize missed trades.")

        if label.upper() == "LONG":
            st.write("\U0001F537 This model aims to predict **buy** signals. High precision helps avoid false buys. High recall ensures you're capturing uptrends.")
        elif label.upper() == "SHORT":
            st.write("\U0001F537 This model aims to predict **short** signals. High precision avoids false short entries. High recall means catching most downturns.")

    with st.expander(f"\U0001F9AE Confusion Matrix ({label} Strategy)"):
        st.write(cm_df)


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
        tickers = pd.read_csv("2025-07-05T18-36_export.csv")["Ticker"].tolist()

    long_change_perc = st.slider("Long Model Change Threshold (%)", 0, 200, 60, 1)
    long_change = long_change_perc / 100.0
    long_change_days = st.slider("Days for Long Model Change", 1, 60, 20, 1)

    short_change_perc_positive = st.slider("Short Model Change Threshold (%) (higher slider → more negative)", 0, 99, 15, 1)
    short_change = -short_change_perc_positive / 100.0
    short_change_days = st.slider("Days for Short Model Change", 1, 60, 20, 1)


if tickers:
    select_all = st.checkbox("Select all tickers")
    selected_tickers = st.multiselect("Select stocks to analyze (watchlist only)", options=tickers, default=tickers if select_all else [])

    if selected_tickers:
        for ticker in selected_tickers:
            st.header(f"Analysis for {ticker}")
            df = yf.download(ticker, start="2000-01-01", interval='1d', progress=False)

            df_inference = get_enriched_stock_data(ticker, change=0, inference=True, df=df)
            df_long = get_enriched_stock_data(ticker, change=long_change, change_next_days=long_change_days, df=df)
            df_short = get_enriched_stock_data(ticker, change=short_change, change_next_days=short_change_days, df=df)

            for label, df_feat in zip(["LONG", "SHORT"], [df_long, df_short]):
                y = df_feat["change"]
                X = df_feat.drop(columns=["change", "Ticker", "Date"], errors='ignore')

                split_index = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
                tscv = TimeSeriesSplit(n_splits=5)
                grid = GridSearchCV(model, param_grid, cv=tscv, scoring='precision', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)

                display_metrics_and_explanation(y_test, y_pred, label)

                if label == "LONG":
                    best_model_long = best_model
                else:
                    best_model_short = best_model

            df_plot = df_inference.reset_index()
            df_plot['Date'] = pd.to_datetime(df_plot['Date'])

            inference_features = df_inference.drop(columns=["Ticker", "Date"], errors='ignore')
            y_pred_long_inf = best_model_long.predict(inference_features)
            y_pred_short_inf = best_model_short.predict(inference_features)

            dates = df_plot['Date']
            close_aligned = df_plot['Close']
            buy_mask = (y_pred_long_inf == 1)
            sell_mask = (y_pred_short_inf == 0)

            last_buy_idx = np.where(buy_mask)[0][-1] if np.any(buy_mask) else None
            last_sell_idx = np.where(sell_mask)[0][-1] if np.any(sell_mask) else None

            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(dates, close_aligned, label='Close Price', color='lightseagreen', linewidth=1.5)

            for i in range(1, len(dates)):
                if buy_mask[i]:
                    ax.annotate('↑', (mdates.date2num(dates[i]), close_aligned.iloc[i]),
                                color='green', fontsize=14, ha='center', va='bottom', weight='bold')
                elif sell_mask[i]:
                    ax.annotate('↓', (mdates.date2num(dates[i]), close_aligned.iloc[i]),
                                color='red', fontsize=14, ha='center', va='top', weight='bold')

            last_buy_date_str = dates.iloc[last_buy_idx].strftime('%Y-%m-%d') if last_buy_idx is not None else "Keine"
            last_sell_date_str = dates.iloc[last_sell_idx].strftime('%Y-%m-%d') if last_sell_idx is not None else "Keine"

            if last_buy_idx is not None:
                ax.annotate(f"{last_buy_date_str}",
                            (mdates.date2num(dates[last_buy_idx]), close_aligned.iloc[last_buy_idx]),
                            textcoords="offset points", xytext=(0, 15),
                            ha='center', color='green', fontsize=10, weight='bold')

            if last_sell_idx is not None:
                ax.annotate(f"{last_sell_date_str}",
                            (mdates.date2num(dates[last_sell_idx]), close_aligned.iloc[last_sell_idx]),
                            textcoords="offset points", xytext=(0, -20),
                            ha='center', color='red', fontsize=10, weight='bold')

            ax.set_title(f"Buy & Short Signals on Close Price for {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            fig.autofmt_xdate(rotation=45)

            custom_lines = [
                Line2D([0], [0], color='blue', label='Close Price'),
                Line2D([0], [0], marker='^', color='green', linestyle='None', markersize=10, label=f"Letztes Buy-Datum: {last_buy_date_str}"),
                Line2D([0], [0], marker='v', color='red', linestyle='None', markersize=10, label=f"Letztes Short-Datum: {last_sell_date_str}"),
            ]
            ax.legend(handles=custom_lines, loc='upper left', fontsize=9)

            st.pyplot(fig)
            with st.expander(f"\U0001F4D8 Fundamental Analysis for ({ticker})"):
                visualize_stock_analysis([ticker], without_indicators=True)
else:
    st.write("Watchlist ist leer.")