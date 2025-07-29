from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import os
import pandas as pd
import warnings
import pickle
import numpy as np
from CSM_Trading.preprocessing.model_prep import StockInfo


warnings.filterwarnings('ignore')

class Model_1():


    def __init__(self):

        stocks_info_train = StockInfo(start_period="2020-01-01", end_period="2024-01-01")
        self.stock_df_train = stocks_info_train.get_stocks()
        self.X_train = self.stock_df_train.drop(columns=["relevance", "qid"])
        self.y_train = self.stock_df_train["future_return"]
        self.qid_train = self.stock_df_train["qid"]
        self.ticker_train = self.stock_df_train.index.get_level_values("ticker")

        # Test data
        stocks_info_test = StockInfo(start_period="2024-01-02", end_period="2024-01-20")
        self.stock_df_test = stocks_info_test.get_stocks()
        self.X_test = self.stock_df_test.drop(columns=["relevance", "qid"])
        self.y_test = self.stock_df_test["future_return"]
        self.qid_test = self.stock_df_test["qid"]
        self.ticker_test = self.stock_df_test.index.get_level_values("ticker")

    def get_model(self, save_path="../saved_models/xgbregressor.pkl"):
        """
        Trains or loads a GradientBoostingRegressor model, evaluates it, and returns predictions on the test set.

        Parameters
        ----------
        save_path : str
            Path to save/load the model.

        Returns
        -------
        model : GradientBoostingRegressor
            The trained or loaded model.
        top_assets : pd.DataFrame
            DataFrame with top 5 ranked assets by predicted return for each qid in test set.
        train_scores : float
            Mean R² score on training folds.
        test_scores : float
            Mean R² score on validation folds.
        """
        model_path = save_path

        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            print("✅ Model loaded successfully!")
        else:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=42
            )

            # Use training data for CV
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_validate(
                model,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring="r2",
                return_train_score=True
            )

            train_scores = np.mean(cv_results["train_score"])
            test_scores = np.mean(cv_results["test_score"])

            print(f"📊 CV Train R²: {train_scores:.4f}")
            print(f"📊 CV Test  R²: {test_scores:.4f}")

            # Fit final model
            model.fit(self.X_train, self.y_train)

            # Save model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(model_path, "wb") as file:
                pickle.dump(model, file)
            print(f"💾 Model saved at {model_path}!")

        # Use test set for predictions and ranking
        self.stock_df_test["predicted_return"] = model.predict(self.X_test)
        self.stock_df_test["qid"] = self.qid_test.values
        self.stock_df_test["ticker"] = self.ticker_test.values

        self.stock_df_test["rank"] = self.stock_df_test.groupby("qid")["predicted_return"].rank(ascending=False)
        top_assets = self.stock_df_test[self.stock_df_test["rank"] <= 5]

        return model, top_assets, train_scores, test_scores

    def get_hyperparameter(self, save_path="../saved_models/gbregressor_best_model.pkl"):
        """
        Tunes the hyperparameters of a GradientBoostingRegressor using RandomizedSearchCV,
        saves the best model, and returns it along with the top ranked assets on the test set.

        Parameters
        ----------
        save_path : str, optional
            The file path where the best model will be saved.

        Returns
        -------
        best_model : GradientBoostingRegressor
            The regressor with the best hyperparameters obtained from RandomizedSearchCV.

        top_assets : pd.DataFrame
            Top 5 ranked assets by predicted return for each qid in the test set.
        """

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.6, 0.8, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }

        gbr = GradientBoostingRegressor(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=gbr,
            param_distributions=param_distributions,
            n_iter=50,
            scoring='neg_mean_squared_error',  # Can change to 'r2' if preferred
            cv=5,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(self.X_train, self.y_train)

        best_model = random_search.best_estimator_

        print("✅ Best Parameters:", random_search.best_params_)
        print(f"📉 Best Score (Negative MSE): {random_search.best_score_:.4f}")

        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as file:
            pickle.dump(best_model, file)
        print(f"💾 Best model saved at {save_path}!")

        # Predict on test set
        self.stock_df_test["predicted_return"] = best_model.predict(self.X_test)
        self.stock_df_test["qid"] = self.qid_test.values
        self.stock_df_test["ticker"] = self.ticker_test.values

        # Rank by predicted return
        self.stock_df_test["rank"] = self.stock_df_test.groupby("qid")["predicted_return"].rank(ascending=False)
        top_assets = self.stock_df_test[self.stock_df_test["rank"] <= 5]

        return best_model, top_assets

    def performance(self, model=None):
        """
        Evaluates the performance of a regression model by calculating R² and MSE on the test set.
        Also prints the model's configuration.

        Parameters
        ----------
        model : GradientBoostingRegressor or tuple
            The trained model. If None, it is retrieved using `self.get_model()`.

        Returns
        -------
        tuple
            The model and a dictionary with performance metrics.
        """

        # Ensure we extract the model if passed as (model, top_assets, ...)
        if model is None:
            model = self.get_model()[0]
        elif isinstance(model, tuple):
            model = model[0]

        # Predict on the test set
        preds = model.predict(self.X_test)

        # Metrics
        r2 = r2_score(self.y_test, preds)
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)

        # Output summary
        print("\n📦 Model Summary:")
        print(model)

        print("\n📊 Performance Metrics:")
        print(f"R² Score : {r2:.4f}")
        print(f"MSE      : {mse:.4f}")
        print(f"RMSE     : {rmse:.4f}")

        # Plot predicted vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, preds, alpha=0.3, label="Predictions")
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'r--', lw=2, label='Ideal fit')
        plt.xlabel("Actual Future Return")
        plt.ylabel("Predicted Future Return")
        plt.title("Predicted vs Actual Future Returns")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return model, {"r2": r2, "mse": mse, "rmse": rmse}

    def get_prediction(self, data):
        """
        Generates predictions for the given input data using the trained model.

        Parameters
        ----------
        data : array-like or pd.DataFrame
            The input data for which predictions are to be generated.

        Returns
        -------
        np.ndarray
            Predicted values for the input data.
        """
        model = self.get_model()[0]  # Extract the model from (model, top_assets, ...)
        prediction = model.predict(data)
        return prediction


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)


if __name__ == "__main__":
    model = Model_1()
    print("📈 Full stock dataset:")
    print(model.stock_df_train)

    # Train or load model and get predictions on test set
    # trained_model, top_assets, train_r2, test_r2 = model.get_model()

    # print("\n🏆 Top Ranked Assets on Test Set:")
    # print(top_assets.head(10))  # show top 10 rows for preview


    # Evaluate performance
    # model.performance(trained_model)

    # Best Model
    best_model, best_top_assets = model.get_hyperparameter()

    print("\n🏆 Top Ranked Assets on Test Set:")
    print(best_top_assets.head(10))  # show top 10 rows for preview

    # Evaluate performance best parameters
    model.performance(best_model)

