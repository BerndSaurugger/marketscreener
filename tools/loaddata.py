
# from tradingview_screener import Query, col
def implemented_columns():
    """
    Returns a list of implemented columns in the database.
    """
    return ['name', "close",
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
                "sector", "price_target_average"]