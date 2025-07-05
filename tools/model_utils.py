import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tools.machinelearning import get_enriched_stock_data


def model_path(ticker):
    return f"models/{ticker}_model.pkl"


def save_models_and_features(ticker, model_long, model_short, feature_cols):
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model_long": model_long,
        "model_short": model_short,
        "feature_cols": feature_cols
    }, model_path(ticker))


def load_models_and_features(ticker):
    path = model_path(ticker)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def train_model_for_ticker(ticker, long_change, long_days, short_change, short_days):
    df = get_enriched_stock_data(ticker, change=0, inference=True)
    df_long = get_enriched_stock_data(ticker, change=long_change, change_next_days=long_days)
    df_short = get_enriched_stock_data(ticker, change=short_change, change_next_days=short_days)

    feature_cols = df_long.columns.difference(['change', 'Ticker', 'Date', 'year', 'month', 'dayofmonth'])

    X_long = df_long[feature_cols]
    y_long = df_long["change"]

    X_short = df_short[feature_cols]
    y_short = df_short["change"]

    model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100],
        "max_depth": [10, None],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }
    tscv = TimeSeriesSplit(n_splits=5)

    grid_long = GridSearchCV(model, param_grid, cv=tscv, scoring='precision', n_jobs=-1)
    grid_long.fit(X_long, y_long)

    grid_short = GridSearchCV(model, param_grid, cv=tscv, scoring='precision', n_jobs=-1)
    grid_short.fit(X_short, y_short)

    save_models_and_features(ticker, grid_long.best_estimator_, grid_short.best_estimator_, list(feature_cols))
    return df, grid_long.best_estimator_, grid_short.best_estimator_, list(feature_cols)


def infer_signals(ticker, df_inference, model_long, model_short, feature_cols):
    if df_inference.empty:
        return None

    df_inference["Date"] = pd.to_datetime(dict(
        year=df_inference.year,
        month=df_inference.month,
        day=df_inference.dayofmonth
    ))

    X_inf = df_inference[feature_cols]
    dates = df_inference["Date"].reset_index(drop=True)

    y_pred_long = model_long.predict(X_inf)
    last_long_idx = (pd.Series(y_pred_long[::-1]) == 1).idxmax() if 1 in y_pred_long else None
    last_long_date = dates.iloc[-(last_long_idx + 1)] if last_long_idx is not None else None

    y_pred_short = model_short.predict(X_inf)
    last_short_idx = (pd.Series(y_pred_short[::-1]) == 0).idxmax() if 1 in y_pred_short else None
    last_short_date = dates.iloc[-(last_short_idx + 1)] if last_short_idx is not None else None

    return {
        "Ticker": ticker,
        "Last Long Signal": last_long_date,
        "Last Short Signal": last_short_date
    }
