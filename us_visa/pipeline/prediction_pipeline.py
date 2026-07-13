import os
import sys
import pandas as pd

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.utils.main_utils import load_object

from us_visa.constants import (
    SAVED_MODEL_DIR,
    PRODUCTION_MODEL_FILE_NAME,
    PREPROCSSING_OBJECT_FILE_NAME
)


class USvisaData:

    def __init__(
        self,
        continent,
        education_of_employee,
        has_job_experience,
        requires_job_training,
        no_of_employees,
        yr_of_estab,
        region_of_employment,
        prevailing_wage,
        unit_of_wage,
        full_time_position
    ):
        self.continent = continent
        self.education_of_employee = education_of_employee
        self.has_job_experience = has_job_experience
        self.requires_job_training = requires_job_training
        self.no_of_employees = no_of_employees
        self.yr_of_estab = yr_of_estab
        self.region_of_employment = region_of_employment
        self.prevailing_wage = prevailing_wage
        self.unit_of_wage = unit_of_wage
        self.full_time_position = full_time_position


    def get_data_as_dataframe(self):
        try:
            usvisa_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "yr_of_estab": [self.yr_of_estab],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position]
            }
            return pd.DataFrame(usvisa_data)

        except Exception as e:
            raise USvisaException(e, sys)


class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join(SAVED_MODEL_DIR, PREPROCSSING_OBJECT_FILE_NAME)
        self.model_path = os.path.join(SAVED_MODEL_DIR, PRODUCTION_MODEL_FILE_NAME)
        self.preprocessor = load_object(self.preprocessor_path)
        self.model = load_object(self.model_path)

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logging.info("Loading preprocessing object and trained model")
        try:
            logging.info("Transforming input data")
            transformed_data = self.preprocessor.transform(dataframe)
            
            logging.info("Making prediction")
            return self.model.predict(transformed_data)
            
        except Exception as e:
            raise USvisaException(e, sys)