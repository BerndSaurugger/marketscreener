import yfinance as yf
import pandas as pd
import numpy as np

def gaussian_filter(series: pd.Series, poles: int = 4, period: int = 144, reduced_lag: bool = False, fast_response: bool = False) -> pd.Series:
    x = series.to_numpy(dtype=np.float64)
    beta = (1 - np.cos(4 * np.arcsin(1) / period)) / (1.414 ** (2 / poles) - 1)
    alpha = -beta + np.sqrt(beta ** 2 + 2 * beta)
    lag = int((period - 1) / (2 * poles))
    if reduced_lag:
        x = x + (x - np.roll(x, lag))
    f = np.zeros_like(x)
    for i in range(poles):
        for t in range(poles, len(x)):
            f[t] = alpha * x[t] + (1 - alpha) * f[t - 1]
    if fast_response:
        f_avg = (f + np.roll(f, 1)) / 2
        f_avg[:poles] = f[:poles]
        return pd.Series(f_avg, index=series.index)
    else:
        return pd.Series(f, index=series.index)

def Generate_features(df, change=0.01, inference=False, change_next_days=10):
    if inference == False:
        df['change_raw'] = (df['Close'].shift(-change_next_days) - df['Close']) / df['Close']
        df = df.dropna(subset=['change_raw'])
        df['change'] = (df['change_raw'] > change).astype(int)
        df.drop(columns=['change_raw'], inplace=True)
    df['t-1'] = df['Close'].shift(1)
    df['t-2'] = df['Close'].shift(2)
    df['t-3'] = df['Close'].shift(3)
    df['t-4'] = df['Close'].shift(4)
    df['t-5'] = df['Close'].shift(5)

    df['mean_t5'] = df[['t-1','t-2','t-3','t-4','t-5']].mean(axis=1)

    df['EMA_12'] = df['t-1'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['t-1'].ewm(span=26, adjust=False).mean()

    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta = df['t-1'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['gaussian'] = gaussian_filter(df['t-1'], poles=4, period=144)

    df['date'] = pd.to_datetime(df.index)
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day

    df.dropna(inplace=True)
    df.drop(columns=['date'], inplace=True)
    return df

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
    ha['Close'] = ha['Close'].rolling(window=smoothing_length).mean()
    ha['Open'] = ha['Open'].rolling(window=smoothing_length).mean()
    ha['High'] = ha['High'].rolling(window=smoothing_length).mean()
    ha['Low'] = ha['Low'].rolling(window=smoothing_length).mean()
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

def get_enriched_stock_data(ticker, change=0.01, inference=False, change_next_days=10, df=None):
    if df is None:
        df = yf.download(ticker, start="2023-01-01", interval='1d', progress=False)
        if df.empty:
            raise ValueError("No data downloaded for ticker")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    df = df.reset_index()

    rename_map = {}
    for col in df.columns:
        low_col = col.lower()
        if "open" in low_col:
            rename_map[col] = "Open"
        elif "high" in low_col:
            rename_map[col] = "High"
        elif "low" in low_col:
            rename_map[col] = "Low"
        elif "close" in low_col:
            rename_map[col] = "Close"
        elif "volume" in low_col:
            rename_map[col] = "Volume"
    df.rename(columns=rename_map, inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    df['KalmanClose'] = kalman_filter(df['Close'].tolist())

    ha = heikin_ashi(df, smoothing_length=10)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f"{col}_HA"] = ha[col]

    macd_line, signal_line, histogram = calculate_macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = histogram

    df['RSI'] = calculate_rsi(df['Close'])

    df = df.dropna().reset_index(drop=True)

    df = df.set_index('Date') if 'Date' in df.columns else df.set_index(df.columns[0])
    df = Generate_features(df, change=change, inference=inference, change_next_days=change_next_days)

    df['Ticker'] = ticker
    return df