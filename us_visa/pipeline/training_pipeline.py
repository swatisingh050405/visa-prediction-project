import sys
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_drift import DataDrift
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.model_trainer import ModelTrainer
from us_visa.components.model_evaluation import ModelEvaluation
from us_visa.components.model_pusher import ModelPusher

from us_visa.entity.config_entity import (DataIngestionConfig,
                                         DataValidationConfig,
                                         DataDriftConfig,
                                         DataTransformationConfig,
                                         ModelTrainerConfig,
                                         ModelEvaluationConfig,
                                         ModelPusherConfig
                                         )

from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataDriftArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact
                                            )


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_drift_config = DataDriftConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        


    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e


    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config
                                             )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e
        
    def start_data_drift(
    self,
    data_ingestion_artifact: DataIngestionArtifact
) -> DataDriftArtifact:

        try:
            data_drift = DataDrift(
                data_ingestion_artifact=data_ingestion_artifact,
                data_drift_config=self.data_drift_config
            )

            data_drift_artifact = data_drift.initiate_data_drift()

            return data_drift_artifact

        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def start_model_trainer(
    self,
    data_transformation_artifact: DataTransformationArtifact
) -> ModelTrainerArtifact:

        try:

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )

            return model_trainer.initiate_model_trainer()

        except Exception as e:
            raise USvisaException(e, sys)


    def start_model_evaluation(
    self,
    data_transformation_artifact: DataTransformationArtifact,
    model_trainer_artifact: ModelTrainerArtifact
) -> ModelEvaluationArtifact:

        try:

            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )

            return model_evaluation.initiate_model_evaluation()

        except Exception as e:
            raise USvisaException(e, sys)


    def start_model_pusher(
    self,
    model_trainer_artifact: ModelTrainerArtifact,
    model_evaluation_artifact: ModelEvaluationArtifact
) -> ModelPusherArtifact:

        try:

            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=model_evaluation_artifact
            )

            return model_pusher.initiate_model_pusher()

        except Exception as e:
            raise USvisaException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Pipeline Started")

            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data ingestion completed successfully")

            # Data Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            logging.info("Data validation completed successfully")

            # Data Drift
            data_drift_artifact = self.start_data_drift(
                data_ingestion_artifact=data_ingestion_artifact
            )
            logging.info("Data drift completed successfully")

            # Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )

            # Model Trainer
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info("Model trainer completed successfully")

            # Model Evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )

            logging.info("Model evaluation completed successfully")

            # Model Pusher
            model_pusher_artifact = self.start_model_pusher(
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=model_evaluation_artifact
            )

            logging.info("Model pusher completed successfully")

            logging.info("Pipeline Completed Successfully")

        except Exception as e:
            raise USvisaException(e, sys)