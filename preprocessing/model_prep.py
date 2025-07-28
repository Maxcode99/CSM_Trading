import pandas as pd
import yfinance as yf


class StockInfo():



    def __init__(self, start_period: str, end_period: str):

        """
        start_period : str

         Need to be in str american datetime format

            Example:
                "2020-01-01", 2021-01-01

        end_period : str

        Need to be in str american datetime format

            Example:
                "2020-01-01", 2021-01-01


        """

        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

        symbols_list = sp500['Symbol'].unique().tolist()

        self.df = yf.download(tickers=symbols_list,
                              start=start_period,
                              end=end_period)[["Close", "Volume"]].stack()

        self.df.index.names = ['date', 'ticker']

        self.df.columns = self.df.columns.str.lower()
        # self.df = self.df.drop(columns=["adj close"])


    def _calculate_returns(self, dataframe: pd.DataFrame,  period_to_predict: int = 10) -> pd.DataFrame:

        outlier_cutoff = 0.005
        lags = [1, 2, 3, 4, 5, 6, 7]

        for lag in lags:
            dataframe[f'return_{lag}day'] = (dataframe['close']
                                      .pct_change(lag)
                                      .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                             upper=x.quantile(1 - outlier_cutoff)))
                                      .add(1)
                                      .pow(1 / lag)
                                      .sub(1))

        dataframe["future_return"] = (dataframe['close']
                               .pct_change(period_to_predict)
                               .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                      upper=x.quantile(1 - outlier_cutoff)))
                               .add(1)
                               .pow(1 / period_to_predict)
                               .sub(1))


        return dataframe

    def get_stocks(self) -> pd.DataFrame:
        stock_df = self.df.groupby(level=1, group_keys=False).apply(self._calculate_returns).dropna()
        # stock_df = stock_df.drop(columns=["high", "low", "open", "volume", "close"])
        stock_df = stock_df.drop(columns=["volume", "close"])
        stock_df["qid"] = stock_df.index.get_level_values(0)
        stock_df["qid"] = pd.factorize(stock_df['qid'])[0]

        def map_relevance(group):
            if group <= 0:
                return 0
            elif 0 < group < 0.007:
                return 1
            else:
                return 2

        stock_df['relevance'] = stock_df['future_return'].apply(map_relevance)  # ← fixed



        return stock_df


    def saved_data(self, dataframe: pd.DataFrame, route: str) -> None:

        saved_stocks = dataframe
        saved_stocks.to_csv(route)





pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)





if __name__ == "__main__":

    # stock_info = StockInfo(start_period="2020-01-01", end_period="2024-01-01")
    stock_info2 = StockInfo(start_period="2024-01-02", end_period="2024-01-20")

    # info = stock_info.get_stocks()
    info2 = stock_info2.get_stocks()
    # stock_info.saved_data(info, route="../data/train_data/TrainData_2024_01_01.csv")
    stock_info2.saved_data(info2, route="../data/test_data/TestData_2024_01_10.csv")

    # print(info)
    print(info2)













