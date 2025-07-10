import pandas as pd
import yfinance as yf


class CSM():


    def __init__(self):

        self.data_frame = pd.read_csv("../data/stocks_info02.csv")
        self.data_frame.drop("Date", axis=1, inplace=True)
        self.returns = self.data_frame.pct_change().dropna()








if __name__ == "__main__":

    csm = CSM()
    print(csm.returns)