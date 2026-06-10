import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import Lasso,Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self,train_array,test_array):
        try:

            logging.info("spliting train & test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
}
      

            params = {
                "Linear Regression": (
                    {
                        "fit_intercept": [True, False]
                    }
                ),

                "Ridge": (
                    {
                        "alpha": [0.01, 0.1, 1, 10, 100],
                        "solver": ["auto", "svd", "cholesky", "lsqr"]
                    }
                ),

                "Lasso": (
                    {
                        "alpha": [0.001, 0.01, 0.1, 1, 10],
                        "max_iter":[5000,10000,20000]
                    }
                ),

                "Decision Tree": (
            
                    {
                        "criterion": ["squared_error", "absolute_error","poisson"],
                        "splitter":["best","random"],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                ),

                "Random Forest Regressor": (
    
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2]
                    }
                ),

                "GradientBoosting": (
                    
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5]
                    }
                ),

                "XGBRegressor": (
                
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 5, 7],
                        "subsample": [0.7, 1]
                    }
                ),

                "K-Neighbors Regressor": (
                    
                    {
                        "n_neighbors": [3, 5, 7],
                        "weights": ["uniform", "distance"],
                        "metric": ["euclidean", "manhattan"]
                    }
                ),

                "CatBoosting Regressor" : (
                                        {
                            "iterations": [100, 200, 500],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "depth": [4, 6, 8, 10],
                            "l2_leaf_reg": [1, 3, 5, 7],
                            "loss_function": ["RMSE"],
                            "verbose": [0]
                        }

                    
                ),

                "AdaBoost Regressor": (
                     {
                                "n_estimators": [50, 100, 200],
                                "learning_rate": [0.01, 0.1, 1.0],
                                "loss": ["linear", "square", "exponential"]
                            }
                )


            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            # to get best model score from dict
            best_model_score= max(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score< 0.6:
                raise CustomException(error_detail=sys,error_message="no best model found")
            logging.info("best found model on both training & testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

            r2_squre = r2_score(y_test,predicted)
            return r2_squre
        except Exception as e:
            raise CustomException(e,sys)