import sys
import numpy as np
import mlflow
import mlflow.sklearn
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
from us_visa.utils.main_utils import ( load_numpy_array_data, save_object )



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

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Entered the initiate_model_trainer method of ModelTrainer class")

            logging.info("Loading train and test array")
            train_arr = load_numpy_array_data(
                    self.data_transformation_artifact.transformed_train_file_path
                )

            test_arr = load_numpy_array_data(
                    self.data_transformation_artifact.transformed_test_file_path
                )

            logging.info("Splitting training and test input and target feature")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models, params = get_models_and_params()

            logging.info("Got models and params")
            logging.info("Setting MLflow experiment")

            mlflow.set_experiment("US Visa Prediction")
            logging.info("Initiating model trainer")

            best_model = None
            best_model_name = None
            best_model_score = 0

            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    logging.info(f"Training {model_name}")

                    random_search = RandomizedSearchCV(

                        estimator=model,

                        param_distributions=params[model_name],

                        n_iter=5,

                        cv=3,

                        scoring="accuracy",

                        random_state=42,

                        n_jobs=-1

                    )

                    random_search.fit(x_train, y_train)

                    tuned_model = random_search.best_estimator_

                    mlflow.log_params(random_search.best_params_)
                    mlflow.log_param("model_name", model_name)

                    y_pred = tuned_model.predict(x_test)

                    

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)

                    mlflow.sklearn.log_model(
                    sk_model=tuned_model,
                    artifact_path="model"
                )

                    logging.info(
                        f"{model_name} Accuracy : {accuracy}"
                    )

                    if accuracy > best_model_score:

                        best_model_score = accuracy

                        best_model = tuned_model

                        best_model_name = model_name

            train_pred = best_model.predict(x_train)

            mlflow.log_metric(
            "train_accuracy",
            accuracy_score(y_train, train_pred)
        )

            train_metric_artifact = ClassificationMetricArtifact(
                 accuracy_score=accuracy_score(y_train, train_pred),
                f1_score=f1_score(y_train, train_pred),
                precision_score=precision_score(y_train, train_pred),
                recall_score=recall_score(y_train, train_pred)
            )

            test_pred = best_model.predict(x_test)

            test_metric_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy_score(y_test, test_pred),
                f1_score=f1_score(y_test, test_pred),
                precision_score=precision_score(y_test, test_pred),
                recall_score=recall_score(y_test, test_pred)
            )

            if best_model_score < self.model_trainer_config.expected_accuracy:
                 raise Exception(
                        f"No best model found with accuracy greater than {self.model_trainer_config.expected_accuracy}"
                    )

            logging.info(
                f"Best model found {best_model_name} with accuracy {best_model_score}"
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
                        )

            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys)
                


            