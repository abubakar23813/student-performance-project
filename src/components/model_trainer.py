import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

        #  MLflow setup (persistent tracking)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Student Performance Project")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train & test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            #  Models
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            #  Params
            params = {
                "Linear Regression": {
                    "fit_intercept": [True, False]
                },

                "Ridge": {
                    "alpha": [0.01, 0.1, 1, 10, 100],
                    "solver": ["auto", "svd", "cholesky", "lsqr"]
                },

                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10],
                    "max_iter": [5000, 10000, 20000]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest Regressor": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "GradientBoosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.7, 1]
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }
            }

            #  Run experiment for all models
            with mlflow.start_run(run_name="Model Comparison Run"):

                model_report, trained_models = evaluate_models(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    models=models,
                    param=params
                )

                #  Best model selection
                best_model_name = max(
                    model_report,
                    key=lambda x: model_report[x]["test_score"]
                )

                best_model_score = model_report[best_model_name]["test_score"]
                best_model = trained_models[best_model_name]

                if best_model_score < 0.6:
                    raise CustomException("No best model found", sys)

                logging.info(f"Best model: {best_model_name} | Score: {best_model_score}")

                #  Log best model separately
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_metric("best_model_test_score", best_model_score)

                mlflow.sklearn.log_model(
                    best_model,
                    artifact_path="best_model"
                )

                # Save model locally
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                predicted = best_model.predict(X_test)
                r2_square = r2_score(y_test, predicted)

                mlflow.log_metric("final_r2_score", r2_square)

                return r2_square

        except Exception as e:
            raise CustomException(e, sys)