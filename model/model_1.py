from CSM_Trading.preprocessing.model_prep import StockInfo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate


import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')





class Model_1():
    def __init__(self):

        stocks_info = StockInfo()
        self.stock_df = stocks_info.get_stocks()
        self.X = self.stock_df.drop(columns= "relevance")
        self.y = self.stock_df["relevance"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )





pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)



if __name__ == "__main__":
    # route = "../data/closing_prices_snp500.csv"

    model_1 = Model_1()
    print(model_1.stock_df)
