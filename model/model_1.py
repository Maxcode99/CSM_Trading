from CSM_Trading.preprocessing.model_prep import StockInfo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

import os
import pyltr
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import warnings
import pickle
import numpy as np
warnings.filterwarnings('ignore')





class Model_1():
    def __init__(self):

        stocks_info = StockInfo(start_period=2, end_period=1)
        self.stock_df = stocks_info.get_stocks()
        self.X = self.stock_df.drop(columns= ["relevance", "qid"]).values
        self.y = self.stock_df["relevance"].astype(int).values
        self.qids = self.stock_df["qid"].astype(int).values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )


    def qid_validation(self):
        valid_qids = self.stock_df.groupby('qid')['relevance'].nunique()
        print((valid_qids >= 2).mean())  # % of valid qids


    def get_model(self, save_path =  "../saved_models/pythonltr_best_model.pkl"):




        model_path = "../saved_models/pythonltr_best_model.pkl"

        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            print("Model loaded successfully!")

        else:
            metric = pyltr.metrics.NDCG(k=5)
            model = pyltr.models.LambdaMART(metric=metric, n_estimators=100, verbose=1)


            model.fit(self.X, self.y)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(model_path, "wb") as file:
                pickle.dump(model, file)
            print("Model saved successfully!")

            return train_scores, test_scores, model

        return model









pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)



if __name__ == "__main__":
    # route = "../data/closing_prices_snp500.csv"

    model_1 = Model_1()
    print(model_1.stock_df)
    print(model_1.qid_validation())

