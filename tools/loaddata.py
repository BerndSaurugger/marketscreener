
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

def all_columns():
    """
    Returns a comprehensive list of implemented and valid columns for TradingView Screener.
    """
    return [
        "name", "close", "type", "exchange", "sector",

        # Volume & Liquidity
        "volume", "average_volume_10d_calc", "relative_volume_10d_calc", "Volatility.D",

        # Price Change & Performance
        "change", "change|60", "change|240", "change|1W", "change|1M",
        "Perf.W", "Perf.1M", "Perf.3M", "Perf.Y", "Perf.3Y", "Perf.5Y", "Perf.10Y", "Perf.YTD",

        # Technical Indicators - RSI
        "RSI", "RSI7",

        # Technical Indicators - MACD
        "MACD.macd", "MACD.signal", "MACD.hist",

        # EMAs
        "EMA9", "EMA20", "EMA50", "EMA200",
        "EMA9|60", "EMA20|60", "EMA50|60", "EMA200|60",
        "EMA9|240", "EMA20|240", "EMA50|240", "EMA200|240",
        "EMA9|1W", "EMA20|1W", "EMA50|1W", "EMA200|1W",

        # Highs & Lows
        "High.1M", "High.3M", "High.6M", "High.All",
        "Low.1M", "Low.3M", "Low.6M", "Low.All",

        # Market Cap
        "market_cap_basic",

        # Valuation
        "price_target_average", "price_earnings_ttm", "earnings_per_share_basic_ttm",

        # Financial Ratios
        "quick_ratio", "current_ratio", "debt_to_equity", "gross_margin", "net_margin",

        # Fundamental Events
        "earnings_release_next_date",
        
        # Other
        "number_of_employees"
    ]
