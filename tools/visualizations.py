import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from mplfinance.original_flavor import candlestick_ohlc
from tools.calculations import kalman_filter, heikin_ashi, calculate_macd, calculate_rsi
from tools.loaddata import load_watchlist, save_watchlist



def visualize_stock_analysis(tickers_to_plot):
    if tickers_to_plot:
        st.subheader("\U0001F4C8 Advanced Visualizations")

        data = yf.download(tickers_to_plot, interval="1d", group_by="ticker", start="2025-01-01", progress=False)
        watchlist = load_watchlist()

        for ticker in tickers_to_plot:
            try:
                info = yf.Ticker(ticker).info

                # === Valuation & Profitability ===
                sector = info.get("sector", "N/A")
                industry = info.get("industry", "N/A")
                market_cap = info.get("marketCap", "N/A")
                eps = info.get("trailingEps", "N/A")
                pe_ratio = info.get("forwardPE", info.get("trailingPE", "N/A"))
                pb_ratio = info.get("priceToBook", "N/A")
                de_ratio = info.get("debtToEquity", "N/A")
                roe = info.get("returnOnEquity", "N/A")
                gross_margin = info.get("grossMargins", "N/A")
                net_margin = info.get("netMargins", "N/A")
                div_yield = info.get("dividendYield", "N/A")
                book_value = info.get("bookValue", "N/A")

                # === Financial Health ===
                total_cash = info.get("totalCash", "N/A")
                total_debt = info.get("totalDebt", "N/A")
                current_ratio = info.get("currentRatio", "N/A")
                quick_ratio = info.get("quickRatio", "N/A")
                op_cashflow = info.get("operatingCashflow", "N/A")
                free_cashflow = info.get("freeCashflow", "N/A")
                ebitda = info.get("ebitda", "N/A")
                total_revenue = info.get("totalRevenue", "N/A")

                target_mean = info.get("targetMeanPrice", "N/A")
                target_high = info.get("targetHighPrice", "N/A")
                target_low = info.get("targetLowPrice", "N/A")
                analyst_count = info.get("numberOfAnalystOpinions", "N/A")
                recommendation = info.get("recommendationKey", "N/A")
                recommendation_score = info.get("recommendationMean", "N/A")

                def fmt_large(n):
                    try:
                        n = float(n)
                        for unit in ['', 'K', 'M', 'B', 'T']:
                            if abs(n) < 1000:
                                return f"{n:.2f}{unit}"
                            n /= 1000
                        return f"{n:.2f}T"
                    except:
                        return n

                # === Fundamentals ===
                st.markdown(f"""
                #### \U0001F4CC **{ticker} — Fundamentals Overview**
                **Sector**: {sector} • **Industry**: {industry}  
                **Market Cap**: {fmt_large(market_cap)} • **EPS (TTM)**: {eps}  
                **P/E**: {pe_ratio} • **P/B**: {pb_ratio}  
                **D/E**: {de_ratio} • **ROE**: {roe}  
                **Margins**: Gross {gross_margin}, Net {net_margin}  
                **Dividend Yield**: {div_yield} • **Book Value**: {book_value}  
                """)

                with st.expander("\U0001F4B0 Show Liquidity & Cash Flow"):
                    st.markdown(f"""
                    **Total Cash**: {fmt_large(total_cash)}  
                    **Total Debt**: {fmt_large(total_debt)}  
                    **Current Ratio**: {current_ratio} • **Quick Ratio**: {quick_ratio}  
                    **Operating Cash Flow**: {fmt_large(op_cashflow)}  
                    **Free Cash Flow**: {fmt_large(free_cashflow)}  
                    **EBITDA**: {fmt_large(ebitda)}  
                    **Total Revenue**: {fmt_large(total_revenue)}
                    """)

                with st.expander("\U0001F3AF Analyst Price Targets"):
                    if all(isinstance(val, (int, float)) for val in [target_mean, target_high, target_low] if val != "N/A"):
                        try:
                            current_price = data[ticker]["Close"].dropna().iloc[-1]
                            if isinstance(current_price, (int, float)):
                                upside = ((target_mean / current_price) - 1) * 100
                                st.markdown(f"""
                                **Mean Target**: {target_mean}  
                                **High / Low**: {target_high} / {target_low}  
                                **# of Analysts**: {analyst_count}  
                                **Consensus**: `{recommendation.title()}` ({recommendation_score})  
                                **Upside Potential**: `{upside:.2f}%` from current price `{current_price:.2f}`
                                """)
                        except Exception as e:
                            st.warning(f"\u26A0\uFE0F Could not compute upside: {e}")
                    else:
                        st.info("\u2139\uFE0F No analyst price target data available.")

                # === Add/Remove to/from Watchlist ===
                col1, col2 = st.columns(2)
                with col1:
                    if ticker not in watchlist:
                        if st.button(f"➕ Add {ticker} to Watchlist"):
                            watchlist.append(ticker.upper())
                            save_watchlist(watchlist)
                            st.success(f"{ticker} added to watchlist.")
                with col2:
                    if ticker in watchlist:
                        if f"confirm_remove_{ticker}" not in st.session_state:
                            st.session_state[f"confirm_remove_{ticker}"] = False

                        if not st.session_state[f"confirm_remove_{ticker}"]:
                            if st.button(f"➖ Remove {ticker} from Watchlist", key=f"remove_{ticker}"):
                                st.session_state[f"confirm_remove_{ticker}"] = True
                        else:
                            col_confirm1, col_confirm2 = st.columns([2, 1])
                            with col_confirm1:
                                st.warning(f"Click to confirm removal of {ticker}")
                            with col_confirm2:
                                if st.button("✅ Confirm", key=f"confirm_button_{ticker}"):
                                    watchlist = [t for t in watchlist if t != ticker.upper()]
                                    save_watchlist(watchlist)
                                    st.success(f"{ticker} removed from watchlist.")
                                    st.session_state[f"confirm_remove_{ticker}"] = False  # reset state


            except Exception as e:
                st.warning(f"\u26A0\uFE0F Could not load fundamentals for {ticker}: {e}")

            df_ticker = data[ticker].dropna()
            close = df_ticker["Close"]
            smoothed = pd.Series(kalman_filter(close), index=close.index)

            df_ha = heikin_ashi(df_ticker)
            df_ha.columns = ['Open', 'Close', 'High', 'Low', 'Volume']
            df_ha = df_ha[['Open', 'High', 'Low', 'Close', 'Volume']]
            macd_line, signal_line, hist = calculate_macd(close)
            rsi = calculate_rsi(close)

            dates = df_ha.index
            close_aligned = close.loc[dates]
            smoothed_aligned = smoothed.loc[dates]
            macd_line = macd_line.loc[dates]
            signal_line = signal_line.loc[dates]
            hist = hist.loc[dates]
            rsi = rsi.loc[dates]

            df_ha_ohlc = df_ha.copy()
            df_ha_ohlc['Date'] = mdates.date2num(df_ha_ohlc.index.to_pydatetime())
            quotes = [tuple(x) for x in df_ha_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values]

            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 1], hspace=0.1)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)

            for quote in quotes:
                color = 'indianred' if quote[4] >= quote[1] else 'mediumseagreen'
                candlestick_ohlc(ax1, [quote], width=0.6, colorup=color, colordown=color)

            ax1.plot(dates, close_aligned, color='darkblue', linestyle=':', label='Price')
            ax1.plot(dates, smoothed_aligned, color='gray', label='Kalman Smoothed')
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(dates, rsi, label='RSI', color='darkblue')
            ax2.axhline(70, color='red', linestyle='--')
            ax2.axhline(30, color='green', linestyle='--')
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("RSI")
            ax2.grid(True)

            ax3.plot(dates, macd_line, label='MACD', color='purple')
            ax3.plot(dates, signal_line, label='Signal', color='gray')
            ax3.bar(dates, hist, color=['green' if h >= 0 else 'red' for h in hist], width=0.6, alpha=0.3)
            ax3.set_ylabel("MACD")
            ax3.legend()
            ax3.grid(True)

            ax3.xaxis_date()
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.xticks(rotation=45)

            plt.tight_layout()
            st.pyplot(fig)