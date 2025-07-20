# from preprocessing.preprocess import Preprocess
# from models.structure import Structure


import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pyltr


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV



class Model_1():



    def __init__(self, route: str):


        self.stock_df = pd.read_csv(route)
        self.stock_df.set_index("Date", inplace=True)


        # df_predict = self.stock_df.loc[:, ["Close"]].shift(15)









pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)



if __name__ == "__main__":
    route = "../data/closing_prices_snp500.csv"
    model_1 = Model_1(route)
    print(model_1.stock_df)
    print(model_1.stock_df.index)