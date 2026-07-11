import os
import sys
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.config_entity import DataDriftConfig
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataDriftArtifact
)

class DataDrift:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_drift_config: DataDriftConfig
    ):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_drift_config = data_drift_config
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                raise USvisaException(e, sys)

    def initiate_data_drift(self) -> DataDriftArtifact:
        try:
            logging.info("Reading training and testing data for data drift analysis")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            logging.info("Creating data drift report using Evidently")
            report = Report(metrics=[DataDriftPreset()])
            my_eval = report.run(
                reference_data=train_df,
                current_data=test_df
            )

            logging.info("Saving data drift report to file")
            os.makedirs(
                self.data_drift_config.report_dir,
                exist_ok=True
            )
            my_eval.save_html(self.data_drift_config.report_file_path)
            my_eval.save_json(self.data_drift_config.report_json_file_path)

            logging.info("Data drift analysis completed successfully")
            return DataDriftArtifact(
                drift_report_file_path=self.data_drift_config.report_file_path,
                drift_report_json_file_path=self.data_drift_config.report_json_file_path,
                drift_status=False
            )

        except Exception as e:
            raise USvisaException(e, sys)