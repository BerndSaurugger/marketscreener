import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from tools.visualizations import visualize_stock_analysis
from tools.loaddata import load_watchlist, add_to_watchlist, remove_from_watchlist
from tools.machinelearning import get_enriched_stock_data

# Sidebar
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
    schalter = st.toggle('Choose Watchlist or Recommendations')
    if schalter:
        st.write('Use Watchlist.')
        tickers = sorted(load_watchlist())
    else:
        st.write('Use Recommendations.')
        tickers = pd.read_csv("2025-07-05T18-36_export.csv")["Ticker"].tolist()
    days = st.slider("Days for Model Prediction", 1, 100, 20, 1)

# Main area
if tickers:
    select_all = st.checkbox("Select all tickers")
    selected_tickers = st.multiselect("Select stocks to analyze", options=tickers, default=tickers if select_all else [])

    if selected_tickers:
        for ticker in selected_tickers:
            st.header(f"Regression Forecast for {ticker}")

            df = yf.download(ticker, start="2000-01-01", interval='1d', progress=False)
            if df.empty:
                st.warning(f"No data available for {ticker}. Skipping...")
                continue

            df_features_full = get_enriched_stock_data(ticker, change=0, inference=True, df=df.copy())
            df_features_full.index.name = 'Date'
            X_inf = df_features_full.drop(columns=["Ticker", "Date", "Close"], errors='ignore')

            df_features = df_features_full.copy()
            df_features['target'] = df_features['Close'].shift(-days)
            # df_features['target'] = (df_features['Close'].shift(-days) - df_features['Close']) / df_features['Close']

            df_features = df_features.dropna()
            y = df_features['target']
            X = df_features.drop(columns=["Ticker", "Date", "Close", "target"], errors='ignore')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            split_index = int(len(X) * 0.80)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10, None],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }

            base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            random_search = RandomizedSearchCV(base_model, param_distributions=param_dist,
                                               scoring='neg_root_mean_squared_error',
                                               n_iter=25, cv=3, verbose=0, n_jobs=-1)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_

            y_pred_test = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            st.write(f"**RMSE (20-day ahead, Test set):** {rmse:.2f}")

            y_pred_full = best_model.predict(X_inf)
            y_pred_aligned = y_pred_full[:len(df_features)]

            df_plot = df_features.reset_index()
            df_plot['Prediction'] = y_pred_aligned

            last_year_mask = df_plot['Date'] >= (df_plot['Date'].max() - pd.Timedelta(days=365))
            df_plot_last_year = df_plot[last_year_mask]

            # future_features = X_inf.iloc[[-1]]
            # Predict the last 20 rows
          # Take the last 20 samples
          # Letzte 20 Eingaben und zugehörige Zeitstempel
          # Letzte 20 Inputs
            # Letzte 20 Zeilen aus X_inf
            # Vorhersage für die letzten 20 Tage in die Zukunft
            X_last_20 = X_inf[-days:]
            y_pred_last_20 = best_model.predict(X_last_20)
            y_pred_last_20 = np.array(y_pred_last_20).flatten()

            # Vorhersage für alle Samples (vollständig)
            y_pred_full = best_model.predict(X_inf)
            y_pred_full = np.array(y_pred_full).flatten()

            # Plot setup
            fig, ax = plt.subplots(figsize=(14, 6))

            # Tatsächliche Close-Werte der letzten 200 Tage (blau)
            recent_close = df['Close'].iloc[-200:]
            recent_dates = df.index[-200:]
            recent_close_180 = recent_close[days:]
            recent_dates_180 = recent_dates[days:]
            ax.plot(recent_dates_180, recent_close_180, label="Letzte 180 Tage Close-Werte (ohne erste 20)", color='blue', linewidth=2)

            # Verschobene Vorhersage (orange), +20 Börsentage
            shift_days = days
            recent_dates_180 = recent_dates[:200-days]                  # take first 180 dates
            dates_shifted = recent_dates_180 + pd.offsets.BDay(shift_days)
            y_pred_recent_180 = y_pred_full[-200:-days]
            ax.plot(dates_shifted, y_pred_recent_180, label="Vorhersage (vergangen, +20 BDay)", color='orange', linestyle='--', linewidth=2)

            # Zukunftsvorhersagen (rot)
            start_date = df.index[-2]
            future_dates = pd.bdate_range(start=start_date + pd.offsets.BDay(1), periods=days)
            ax.plot(future_dates, y_pred_last_20, label="Vorhergesagte Close-Werte (Zukunft)", color='red', linestyle='--', linewidth=2)

            # Min/Max der Zukunftsvorhersage annotieren
            min_idx = np.argmin(y_pred_last_20)
            min_date = future_dates[min_idx]
            min_value = y_pred_last_20[min_idx]
            ax.annotate(f"Min: {min_value:.2f}\n{min_date.date()}",
                        xy=(min_date, min_value),
                        xytext=(0, -40),
                        textcoords='offset points',
                        ha='center',
                        va='top',
                        fontsize=10,
                        color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))

            max_idx = np.argmax(y_pred_last_20)
            max_date = future_dates[max_idx]
            max_value = y_pred_last_20[max_idx]
            ax.annotate(f"Max: {max_value:.2f}\n{max_date.date()}",
                        xy=(max_date, max_value),
                        xytext=(0, 30),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))

            # Labels und Styling
            ax.set_title(f"📊 Verlauf der letzten 200 Tage & Vorhersage – {ticker}", fontsize=14)
            ax.set_xlabel("Datum")
            ax.set_ylabel("Close Price")
            ax.legend()
            fig.autofmt_xdate(rotation=45)

            # Streamlit plot
            st.pyplot(fig)


            with st.expander(f"📘 Fundamental Analysis for ({ticker})"):
                visualize_stock_analysis([ticker], without_indicators=True)
else:
    st.write("Watchlist ist leer.")