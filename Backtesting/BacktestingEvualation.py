import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from CSM_Trading.model.model_2 import Model_GBR





def get_backtesting(list_of_stocks, benchmark: str, start_date: str, end_date: str):

    weights = [1 / len(list_of_stocks) for i in list_of_stocks]
    df = yf.download(start=start_date, end=end_date, tickers=list_of_stocks)["Close"]
    df_bench = yf.download(start=start_date, end=end_date, tickers=benchmark)["Close"]
    rt = df.pct_change().dropna() * weights
    rt_bench = df_bench.pct_change().dropna()
    df["Portfolio"] = rt[list_of_stocks].sum(axis=1)
    df["Benchmark"] = rt_bench[benchmark]

    plt.figure(figsize=[12, 6])

    plt.plot(df.index, df["Portfolio"], label="Portfolio")
    plt.plot(df.index, df["Benchmark"], label="Benchmark")

    plt.title("Backtesting Results")
    plt.xlabel("Date")
    plt.ylabel("Prices")


    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)


if __name__ == "__main__":

    model = Model_GBR()
    print("📈 Full stock dataset:")
    print(model.stock_df_train)

    trained_model, top_assets, train_r2, test_r2 = model.get_model()

    print("\n🏆 Top Ranked Assets on Test Set:")
    print(top_assets.head(10))

    model.performance(trained_model)


    stocks = list(top_assets["ticker"].unique())
    benchmark = "VOO"
    get_backtesting(stocks, "VOO", start_date="2024-01-01", end_date="2024-01-15")
    # a = get_backtesting(stocks, "VOO", start_date="2024-01-01", end_date="2024-01-15")
    # print(a)
    print(stocks)