import pandas as pd
import yfinance as yf


class StockInfo():

    def __init__(self):

        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

        symbols_list = sp500['Symbol'].unique().tolist()

        end_date = '2023-09-27'

        start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * 8)

        self.df = yf.download(tickers=symbols_list,
                              start=start_date,
                              end=end_date).stack()

        self.df.index.names = ['date', 'ticker']

        self.df.columns = self.df.columns.str.lower()
        self.df = self.df.drop(columns=["adj close"])


    def _calculate_returns(self, dataframe: pd.DataFrame,  period_to_predict: int = 10) -> pd.DataFrame:

        outlier_cutoff = 0.005
        lags = [1, 2, 3, 4, 5]

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
        stock_df = stock_df.drop(columns=["high", "low", "open", "volume", "close"])
        stock_df["qid"] = stock_df.index.get_level_values(0)
        stock_df["qid"] = pd.factorize(stock_df['qid'])[0]

        def map_relevance(group):
            if group <= 0:
                return 0
            elif 0 < group < 0.05:
                return 1
            else:
                return 2

        stock_df['relevance'] = stock_df['future_return'].apply(map_relevance)  # ← fixed

        return stock_df






pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)





if __name__ == "__main__":

    stock_info = StockInfo()
    info = stock_info.get_stocks()
    print(info)












