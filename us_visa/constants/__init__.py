import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()


DATABASE_NAME = "visa_database"

COLLECTION_NAME = "visa_data"

MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = "visa_prediction"
ARTIFACT_DIR: str = "artifact"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME: str = "usvisa.csv"
TRAINED_MODEL_FILE_NAME = "trained_model.pkl"
PRODUCTION_MODEL_FILE_NAME = "production_model.pkl"

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "visa_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_RANDOM_STATE = 42


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME = "data_validation"


"""
Data Drift Constants
"""

DATA_DRIFT_DIR_NAME = "data_drift"

DATA_DRIFT_REPORT_DIR = "report"

DATA_DRIFT_REPORT_FILE_NAME = "drift_report.html"

DATA_DRIFT_REPORT_JSON_FILE_NAME = "drift_report.json"


"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

'''
Model Trainer related constant start with MODEL_TRAINER VAR NAME'''

MODEL_TRAINER_DIR_NAME = "model_trainer"

MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"

MODEL_TRAINER_EXPECTED_SCORE = 0.6

MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD = 0.05

'''
Model Evaluation related constant start with MODEL_EVALUATION VAR NAME'''

MODEL_EVALUATION_DIR_NAME = "model_evaluation"

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02

'''
Model Pusher related constant start with MODEL_PUSHER VAR NAME
'''

MODEL_PUSHER_DIR_NAME = "model_pusher"

SAVED_MODEL_DIR = "saved_models"