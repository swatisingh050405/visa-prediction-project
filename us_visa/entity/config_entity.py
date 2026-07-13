import os
from us_visa.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()



@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    random_state: int = DATA_INGESTION_RANDOM_STATE
    collection_name:str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_VALIDATION_DIR_NAME
    )
    

@dataclass
class DataDriftConfig:
    data_drift_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_DRIFT_DIR_NAME
    )

    report_dir: str = os.path.join(
        data_drift_dir,
        DATA_DRIFT_REPORT_DIR
    )

    report_file_path: str = os.path.join(
        report_dir,
        DATA_DRIFT_REPORT_FILE_NAME
    )

    report_json_file_path: str = os.path.join(
        report_dir,
        DATA_DRIFT_REPORT_JSON_FILE_NAME
    ) 

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCESSING_OBJECT_FILE_NAME)



@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR_NAME
    )

    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        TRAINED_MODEL_FILE_NAME
    )

    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE

    overfitting_underfitting_threshold: float = (
        MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
    )

@dataclass
class ModelEvaluationConfig:

    model_evaluation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_EVALUATION_DIR_NAME
    )
    production_model_file_path: str = os.path.join(
        SAVED_MODEL_DIR,
        PRODUCTION_MODEL_FILE_NAME
    )

    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE



@dataclass
class ModelPusherConfig:

    model_pusher_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_PUSHER_DIR_NAME
    )

    

    production_model_file_path: str = os.path.join(
        SAVED_MODEL_DIR,
        PRODUCTION_MODEL_FILE_NAME
    )