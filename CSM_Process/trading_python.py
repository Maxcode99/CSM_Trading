from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta as pa
import warnings
warnings.filterwarnings("ignore")


# sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
# sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-")
# symbols_list = sp500["Symbol"].unique().tolist()
#
# end_date = "2023-09-27"
# start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)
# df = yf.download(tickers=symbols_list,
#                  start=start_date,
#                  end=end_date)

# df = df.drop(columns=["Adj Close"])
# df.to_csv("info_snp500.csv")

df = pd.read_csv("../data/info_snp500.csv")


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)


# df["garman_klas_vol"] = ((np.log(df["high"]) - ((np.log(df["low"])) ** 2)

if __name__ == "__main__":
    print(df.head(10))
    print("")
