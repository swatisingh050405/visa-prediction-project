from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models_and_params():
    models = {
        "XGBoost": XGBClassifier(
            random_state=42, 
            eval_metric="logloss", 
            scale_pos_weight=1.5  # Certified vs Denied ratio imbalance handler
        ),
        "CatBoost": CatBoostClassifier(
            random_state=42, 
            verbose=0, 
            allow_writing_files=False, 
            auto_class_weights="Balanced"  # Automatic inverse class weighting
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42, 
            class_weight="balanced"  # Automatic inverse class frequency weighting
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    params = {
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        },
        "CatBoost": {
            "iterations": [200, 400, 600],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5, 7, 9]
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 1.0]
        }
    }

    return models, params