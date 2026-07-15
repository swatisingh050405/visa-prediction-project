import os
import sys
import shutil

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import (
    SAVED_MODEL_DIR,
    PREPROCESSING_OBJECT_FILE_NAME
)

from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    DataTransformationArtifact

    
)


class ModelPusher:

    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
        data_transformation_artifact: DataTransformationArtifact
    ):

        try:
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_artifact = model_evaluation_artifact
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:

        try:

            if self.model_evaluation_artifact.is_model_accepted:

                logging.info("Model accepted. Pushing model to production.")

                os.makedirs(
                    os.path.dirname(self.model_pusher_config.production_model_file_path),
                    exist_ok=True
                )

                shutil.copy(
                    self.model_trainer_artifact.trained_model_file_path,
                    self.model_pusher_config.production_model_file_path
                )
                shutil.copy(
                self.data_transformation_artifact.transformed_object_file_path,
                os.path.join(
                    SAVED_MODEL_DIR,
                    PREPROCESSING_OBJECT_FILE_NAME
                )
            )

                logging.info("Model pushed successfully.")

            else:

                logging.info("Model rejected. Skipping model push.")

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=self.model_pusher_config.production_model_file_path
            )

            return model_pusher_artifact

        except Exception as e:
            raise USvisaException(e, sys)