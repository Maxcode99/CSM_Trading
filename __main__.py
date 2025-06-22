# from utils.selector import StockSelector
import yfinance as yf


if __name__ == "__main__":
    tickers = ['NVDA', "LLY", "WMT", "HMC", "JPM"]
    df = yf.download(tickers, start="2020-01-01")["Close"]
    df.to_csv("data/stocks_info02.csv")
