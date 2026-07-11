import sys
import pandas as pd

from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:

        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column count correct: {status}")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:

        try:
            dataframe_columns = df.columns

            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (
                missing_numerical_columns or missing_categorical_columns
            )

        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:

        try:
            validation_error_msg = ""

            logging.info("Starting Data Validation")

            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )
            if train_df.empty:
                validation_error_msg += "Training dataframe is empty. "

            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )
            if test_df.empty:
                validation_error_msg += "Testing dataframe is empty. "

            # Validate number of columns
            if not self.validate_number_of_columns(train_df):
                validation_error_msg += "Training dataframe has incorrect number of columns. "

            if not self.validate_number_of_columns(test_df):
                validation_error_msg += "Testing dataframe has incorrect number of columns. "

            # Validate column names
            if not self.is_column_exist(train_df):
                validation_error_msg += "Training dataframe has missing columns. "

            if not self.is_column_exist(test_df):
                validation_error_msg += "Testing dataframe has missing columns. "

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                validation_error_msg = "Data validation successful."

            logging.info(validation_error_msg)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
            )

            logging.info(
                f"Data validation artifact: {data_validation_artifact}"
            )

            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys)