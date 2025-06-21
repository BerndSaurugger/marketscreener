import pandas as pd

def kalman_filter(observations, process_noise=0.001, measurement_noise=0.1):
    """
    Applies a Kalman filter to smooth a time series.
    
    Parameters:
        observations (list/pd.Series): Time series data to filter
        process_noise (float): Variance of process noise (Q)
        measurement_noise (float): Variance of measurement noise (R)
        
    Returns:
        list: Smoothed estimates of the time series
    """
    estimates = []
    estimate = observations[0]  # Initial state estimate
    error_cov = 1.0             # Initial error covariance
    
    for z in observations:
        # Prediction
        prior_estimate = estimate
        prior_error_cov = error_cov + process_noise
        
        # Update (Kalman Gain)
        kalman_gain = prior_error_cov / (prior_error_cov + measurement_noise)
        estimate = prior_estimate + kalman_gain * (z - prior_estimate)
        error_cov = (1 - kalman_gain) * prior_error_cov
        
        estimates.append(estimate)
    
    return estimates

def heikin_ashi(df, smoothing_length=10):
    ha = pd.DataFrame(index=df.index)

    # Raw HA values before smoothing
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

    # Apply smoothing
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
