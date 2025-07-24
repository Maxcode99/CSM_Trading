from CSM_Trading.preprocessing.model_prep import StockInfo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

import matplotlib.pyplot as plt
import os
import pyltr
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import warnings
import pickle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np
import random
from collections import defaultdict
from pyltr.util.group import check_qids
warnings.filterwarnings('ignore')


# # Create and sort training DataFrame
# train_df = pd.DataFrame(self.X_train)
# train_df["y"] = self.y_train
# train_df["qid"] = self.qids_train
# train_df = train_df.sort_values("qid").reset_index(drop=True)
#
# # Extract sorted arrays
# self.X_train = train_df.drop(columns=["y", "qid"]).values
# self.y_train = train_df["y"].values
# self.qids_train = train_df["qid"].values
#
# # Repeat for test set
# test_df = pd.DataFrame(self.X_test)
# test_df["y"] = self.y_test
# test_df["qid"] = self.qids_test
# test_df = test_df.sort_values("qid").reset_index(drop=True)
#
# self.X_test = test_df.drop(columns=["y", "qid"]).values
# self.y_test = test_df["y"].values
# self.qids_test = test_df["qid"].values


class Model_1():
    def __init__(self):

        stocks_info = StockInfo(start_period=2, end_period=1)
        self.stock_df = stocks_info.get_stocks()
        self.X = self.stock_df.drop(columns= ["relevance", "qid"]).values
        self.y = self.stock_df["relevance"].astype(int).values
        self.qids = self.stock_df["qid"].astype(int).values

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test, self.qids_train, self.qids_test = train_test_split(
            self.X, self.y, self.qids, test_size=0.2, random_state=42
        )

        # Ordenar por qid para cada subset
        train_order = np.argsort(self.qids_train)
        test_order = np.argsort(self.qids_test)

        self.X_train = self.X_train[train_order]
        self.y_train = self.y_train[train_order]
        self.qids_train = self.qids_train[train_order]

        self.X_test = self.X_test[test_order]
        self.y_test = self.y_test[test_order]
        self.qids_test = self.qids_test[test_order]

        print("Cheking quick usability")

        check_qids(self.qids_train)  # No lanza error = OK



    def qid_validation(self):
        valid_qids = self.stock_df.groupby('qid')['relevance'].nunique()
        print((valid_qids >= 2).mean())  # % of valid qids


    def get_model(self, save_path="../saved_models/pythonltr_best_model.pkl"):

        model_path = save_path

        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            print("Model loaded successfully!")
        else:
            metric = pyltr.metrics.NDCG(k=5)
            model = pyltr.models.LambdaMART(metric=metric, n_estimators=100, verbose=1)

            model.fit(self.X_train, self.y_train, self.qids_train)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(model_path, "wb") as file:
                pickle.dump(model, file)
            print("Model saved successfully!")

        return model



    # def get_hyperparameter(self, save_path="../saved_models/ltr/lambdamart_best_model.pkl"):
    #     """
    #     Performs manual hyperparameter tuning for LambdaMART, selects the best model based on NDCG@5,
    #     saves it, and returns it.
    #
    #     Returns
    #     -------
    #     model : pyltr.models.LambdaMART
    #         The best LambdaMART model found during manual tuning.
    #     """
    #
    #     param_grid = {
    #         "n_estimators": [50, 100, 150],
    #         "learning_rate": [0.01, 0.05, 0.1],
    #         "min_samples_split": [2, 5, 10],
    #     }
    #
    #     metric = pyltr.metrics.NDCG(k=5)
    #
    #     best_score = -np.inf
    #     best_model = None
    #     best_params = {}
    #
    #     for i in range(10):
    #         params = {
    #             "n_estimators": random.choice(param_grid["n_estimators"]),
    #             "learning_rate": random.choice(param_grid["learning_rate"]),
    #             "min_samples_split": random.choice(param_grid["min_samples_split"]),
    #         }
    #
    #         model = pyltr.models.LambdaMART(
    #             metric=metric,
    #             n_estimators=params["n_estimators"],
    #             learning_rate=params["learning_rate"],
    #             min_samples_split=params["min_samples_split"],
    #             verbose=0,
    #         )
    #
    #         model.fit(self.X_train, self.y_train, self.qids_train)
    #
    #         preds = model.predict(self.X_valid)
    #         score = metric.evaluate(self.y_valid, preds, self.qids_valid)
    #
    #         print(f"Trial {i + 1} - Params: {params} | NDCG@5: {score:.4f}")
    #
    #         if score > best_score:
    #             best_score = score
    #             best_model = model
    #             best_params = params
    #
    #     print("Best hyperparameters found:", best_params)
    #     print("Best NDCG@5:", best_score)
    #
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     with open(save_path, "wb") as f:
    #         pickle.dump(best_model, f)
    #     print(f"Best model saved at {save_path}")
    #
    #     return best_model
    #
    # def performance_plot(self, model=None, ks=[1, 3, 5, 10]):
    #     if model is None:
    #         model = self.get_model()
    #
    #     y_pred = model.predict(self.X_test)
    #
    #     scores = []
    #     for k in ks:
    #         score = pyltr.metrics.NDCG(k=k).evaluate(self.y_test, y_pred, self.qids_test)
    #         scores.append(score)
    #
    #     plt.figure(figsize=(6, 4))
    #     plt.bar([f'NDCG@{k}' for k in ks], scores, color='skyblue')
    #     plt.title('NDCG@k Scores')
    #     plt.ylabel('Score')
    #     plt.ylim(0, 1)
    #     plt.show()
    #
    #     return dict(zip(ks, scores))
    #
    #
    # def mean_average_precision(y_true, y_pred, qids):
    #     grouped = defaultdict(list)
    #     for yt, yp, qid in zip(y_true, y_pred, qids):
    #         grouped[qid].append((yt, yp))
    #
    #     average_precisions = []
    #     for qid, pairs in grouped.items():
    #         # Sort by predicted score descending
    #         sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    #         rels = [x[0] for x in sorted_pairs]
    #
    #         num_relevant = 0
    #         score = 0.0
    #         for i, rel in enumerate(rels):
    #             if rel > 0:
    #                 num_relevant += 1
    #                 score += num_relevant / (i + 1)
    #
    #         if num_relevant > 0:
    #             average_precisions.append(score / num_relevant)
    #
    #     return np.mean(average_precisions)
    #
    # def mean_reciprocal_rank(y_true, y_pred, qids):
    #     grouped = defaultdict(list)
    #     for yt, yp, qid in zip(y_true, y_pred, qids):
    #         grouped[qid].append((yt, yp))
    #
    #     mrr_total = 0.0
    #     for qid, pairs in grouped.items():
    #         sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    #         for i, (rel, _) in enumerate(sorted_pairs):
    #             if rel > 0:
    #                 mrr_total += 1.0 / (i + 1)
    #                 break
    #
    #     return mrr_total / len(grouped)
    #
    # def ranking_metrics(self, model=None):
    #     """
    #     Evaluates the ranking model using NDCG@5, MAP, and MRR for train and test sets.
    #
    #     Parameters
    #     ----------
    #     model : pyltr.models.LambdaMART, optional
    #         Trained ranking model. If None, loads via self.get_model().
    #
    #     Returns
    #     -------
    #     dict
    #         Nested dictionary with scores for 'train_metrics' and 'test_metrics'.
    #     """
    #     if model is None:
    #         model = self.get_model()
    #
    #     y_pred_train = model.predict(self.X_train)
    #     y_pred_test = model.predict(self.X_test)
    #
    #
    #     ndcg_metric = NDCG(k=5)
    #
    #     results = {
    #         "train_metrics": {
    #             "NDCG@5": ndcg_metric.evaluate(self.y_train, y_pred_train, self.qids_train),
    #             "MAP": self.mean_average_precision(self.y_train, y_pred_train, self.qids_train),
    #             "MRR": self.mean_reciprocal_rank(self.y_train, y_pred_train, self.qids_train),
    #         },
    #         "test_metrics": {
    #             "NDCG@5": ndcg_metric.evaluate(self.y_test, y_pred_test, self.qids_test),
    #             "MAP": self.mean_average_precision(self.y_test, y_pred_test, self.qids_test),
    #             "MRR": self.mean_reciprocal_rank(self.y_test, y_pred_test, self.qids_test),
    #         }
    #     }
    #
    #     for split, metrics in results.items():
    #         print(f"\n{split.upper()}:")
    #         for metric, score in metrics.items():
    #             print(f"  {metric}: {score:.4f}")
    #
    #     return results
    #
    # def get_prediction(self, data):
    #
    #     """
    #     Generates predictions for the given input data using the trained model.
    #
    #     Parameters
    #     ----------
    #     data : array-like or pd.DataFrame
    #         The input data for which predictions are to be generated.
    #
    #     Returns
    #     -------
    #     array
    #         Predicted labels for the input data.
    #     """
    #
    #     model = self.get_model()
    #     prediction = model.predict(data)
    #
    #     return prediction




pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)



if __name__ == "__main__":
    model = Model_1()
    print(model.stock_df)
    trained_model = model.get_model()

    # model.performance(trained_model)
    # metrics = model.confusion_matrix_and_metrics(trained_model)
    #
    # print("Performance Metrics:")
    # for metric, value in metrics.items():
    #     if metric != "confusion_matrix":
    #         print(f"{metric.capitalize()}: {value}")
