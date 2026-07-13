import sys
import os

from sklearn.metrics import accuracy_score

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)

from us_visa.utils.main_utils import (
    load_numpy_array_data,
    load_object
)

class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):

        try:

            self.model_eval_config = model_eval_config

            self.data_transformation_artifact = data_transformation_artifact

            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:

            logging.info("Loading transformed test data")

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_test = test_arr[:, :-1]

            y_test = test_arr[:, -1]

            logging.info("Loading newly trained model")

            trained_model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )

            if not os.path.exists(self.model_eval_config.production_model_file_path
            ):

                logging.info("No production model found.")

                return ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=0.0
                )
            logging.info("Loading production model")

            production_model = load_object(
                self.model_eval_config.production_model_file_path
            ) 
            trained_pred = trained_model.predict(x_test)

            trained_score = accuracy_score(
                y_test,
                trained_pred
            )
            production_pred = production_model.predict(x_test)

            production_score = accuracy_score(
                y_test,
                production_pred
            )
            improved_accuracy = trained_score - production_score

            logging.info(f"Production Model Accuracy : {production_score}")
            logging.info(f"Trained Model Accuracy : {trained_score}")
            logging.info(f"Improved Accuracy : {improved_accuracy}")

            is_model_accepted = False

            if improved_accuracy > self.model_eval_config.changed_threshold_score:

                is_model_accepted = True

            model_evaluation_artifact = ModelEvaluationArtifact(

            is_model_accepted=is_model_accepted,

            improved_accuracy=improved_accuracy,

            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")

            return model_evaluation_artifact

        except Exception as e:
            raise USvisaException(e, sys)


