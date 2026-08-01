import sys
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from us_visa.model.model_factory import get_models_and_params
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from us_visa.utils.main_utils import load_numpy_array_data, save_object


def log_model_to_mlflow(tuned_model):
    if isinstance(tuned_model, XGBClassifier):
        mlflow.xgboost.log_model(xgb_model=tuned_model, artifact_path="model")
    elif isinstance(tuned_model, CatBoostClassifier):
        mlflow.catboost.log_model(cb_model=tuned_model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(sk_model=tuned_model, artifact_path="model")


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise USvisaException(e, sys)

    def optimize_threshold(self, model, x_test, y_test):
        """Finds optimal decision threshold maximizing F1 score."""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_test)[:, 1]
            best_thresh = 0.5
            best_f1 = 0
            for thresh in np.arange(0.3, 0.7, 0.02):
                preds = (probs >= thresh).astype(int)
                score = f1_score(y_test, preds, average="weighted")
                if score > best_f1:
                    best_f1 = score
                    best_thresh = thresh
            return best_thresh
        return 0.5

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading train and test array")
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models, params = get_models_and_params()
            mlflow.set_experiment("US Visa Prediction Optimization")

            best_model = None
            best_model_name = None
            best_model_score = 0
            best_threshold = 0.5

            for model_name, model in models.items():
                try:
                    with mlflow.start_run(run_name=model_name):
                        logging.info(f"Training and Tuning {model_name}")

                        random_search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=params[model_name],
                            n_iter=10,
                            cv=5,
                            scoring="f1_weighted",
                            random_state=42,
                            n_jobs=-1
                        )

                        random_search.fit(x_train, y_train)
                        tuned_model = random_search.best_estimator_

                        optimal_thresh = self.optimize_threshold(tuned_model, x_test, y_test)

                        if hasattr(tuned_model, "predict_proba"):
                            y_pred = (tuned_model.predict_proba(x_test)[:, 1] >= optimal_thresh).astype(int)
                        else:
                            y_pred = tuned_model.predict(x_test)

                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, average="weighted")

                        mlflow.log_params(random_search.best_params_)
                        mlflow.set_tag("model_name", model_name)
                        mlflow.log_metric("optimal_threshold", optimal_thresh)
                        mlflow.log_metrics({
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1
                        })

                        log_model_to_mlflow(tuned_model)
                        logging.info(f"{model_name} Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

                        if f1 > best_model_score:
                            best_model_score = f1
                            best_model = tuned_model
                            best_model_name = model_name
                            best_threshold = optimal_thresh

                except Exception as model_error:
                    logging.info(f"Skipping {model_name} due to error: {model_error}")
                    continue

            if best_model is None:
                raise Exception("No model trained successfully")

            # Final Predictions using optimized threshold
            if hasattr(best_model, "predict_proba"):
                train_preds = (best_model.predict_proba(x_train)[:, 1] >= best_threshold).astype(int)
                test_preds = (best_model.predict_proba(x_test)[:, 1] >= best_threshold).astype(int)
            else:
                train_preds = best_model.predict(x_train)
                test_preds = best_model.predict(x_test)

            train_metric_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy_score(y_train, train_preds),
                f1_score=f1_score(y_train, train_preds, average="weighted"),
                precision_score=precision_score(y_train, train_preds, zero_division=0),
                recall_score=recall_score(y_train, train_preds, zero_division=0)
            )

            test_metric_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy_score(y_test, test_preds),
                f1_score=f1_score(y_test, test_preds, average="weighted"),
                precision_score=precision_score(y_test, test_preds, zero_division=0),
                recall_score=recall_score(y_test, test_preds, zero_division=0)
            )

            logging.info(f"Best Model Selected: {best_model_name} with F1: {best_model_score:.4f} (Thresh: {best_threshold:.2f})")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
            )

        except Exception as e:
            raise USvisaException(e, sys)