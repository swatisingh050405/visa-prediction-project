from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_models_and_params():

    models = {

        "Logistic Regression": LogisticRegression(),

        "Decision Tree": DecisionTreeClassifier(),

        "Random Forest": RandomForestClassifier(),

        "Gradient Boosting": GradientBoostingClassifier(),

        "AdaBoost": AdaBoostClassifier(),

        "XGBoost": XGBClassifier(
            eval_metric="logloss"
        ),

        "CatBoost": CatBoostClassifier(
            verbose=False,
             
            allow_writing_files=False
        )

    }

    params = {

        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10]
        },

        "Decision Tree": {
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5, 10]
        },

        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5]
        },

        "Gradient Boosting": {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 200]
        },

        "AdaBoost": {
            "learning_rate": [0.01, 0.1, 1],
            "n_estimators": [50, 100]
        },

        "XGBoost": {
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [100, 200]
        },

        "CatBoost": {
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1],
            "iterations": [100, 200]
        }

    }

    return models, params