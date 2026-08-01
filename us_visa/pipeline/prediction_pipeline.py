import os
import sys
import pandas as pd
import numpy as np
import shap

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_object, read_yaml_file
from us_visa.constants import (
    SAVED_MODEL_DIR,
    PRODUCTION_MODEL_FILE_NAME,
    PREPROCESSING_OBJECT_FILE_NAME,
    SCHEMA_FILE_PATH
)
from us_visa.utils.feature_engineering import perform_feature_engineering


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
        self.preprocessor_path = os.path.join(
            SAVED_MODEL_DIR,
            PREPROCESSING_OBJECT_FILE_NAME
        )
        self.model_path = os.path.join(
            SAVED_MODEL_DIR,
            PRODUCTION_MODEL_FILE_NAME
        )

        self.preprocessor = load_object(self.preprocessor_path)
        self.model = load_object(self.model_path)
        self.schema = read_yaml_file(SCHEMA_FILE_PATH)

        # SHAP Explainer
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, dataframe: pd.DataFrame):

        try:
            logging.info("Performing feature engineering")
            drop_cols = self.schema["drop_columns"]

            dataframe = perform_feature_engineering(
                dataframe,
                drop_cols
            )

            logging.info("Applying preprocessing")
            transformed_data = self.preprocessor.transform(dataframe)

            logging.info("Making prediction")
            probability = self.model.predict_proba(transformed_data)
            
            # Probability for Class 1 (Certified)
            certified_prob = probability[0][1]

            # Applying CatBoost Optimal Decision Threshold (0.42)
            threshold = 0.42
            prediction = np.array([1 if certified_prob >= threshold else 0])

            # ---------------- SHAP Calculations ----------------
            raw_shap_values = self.explainer.shap_values(transformed_data)

            # Ensure we extract 1D SHAP array corresponding to Class 1 (Certified)
            if isinstance(raw_shap_values, list):
                # Multi-class output array format
                shap_row = raw_shap_values[1][0]
            elif len(np.array(raw_shap_values).shape) == 3:
                # Shape (1, n_features, 2)
                shap_row = raw_shap_values[0, :, 1]
            else:
                # 2D Array shape (1, n_features)
                shap_row = raw_shap_values[0]

            feature_names = self.preprocessor.get_feature_names_out()

            combined = {}

            mapping = {
                "continent": "Applicant Continent",
                "education_of_employee": "Education Level",
                "has_job_experience": "Previous Work Experience",
                "requires_job_training": "Training Required",
                "region_of_employment": "Job Location",
                "unit_of_wage": "Salary Frequency",
                "full_time_position": "Full-Time Position",
                "no_of_employees": "Employer Size",
                "company_age": "Employer Age",
                "prevailing_wage": "Offered Salary",
                "wage_to_region_ratio": "Regional Salary Ratio"
            }

            for name, value in zip(feature_names, shap_row):

                clean = (
                    name.replace("Transformer__", "")
                        .replace("StandardScaler__", "")
                        .replace("OrdinalEncoder__", "")
                        .replace("Ordinal_Encoder__", "")
                        .replace("OneHotEncoder__", "")
                        .replace("OneHot__", "")
                        .replace("remainder__", "")
                )

                for key, label in mapping.items():
                    if clean.startswith(key):
                        clean = label
                        break

                combined[clean] = combined.get(clean, 0.0) + float(value)

            feature_impacts = [
                {
                    "feature": feature,
                    "impact": impact
                }
                for feature, impact in combined.items()
            ]

            feature_impacts.sort(
                key=lambda x: abs(x["impact"]),
                reverse=True
            )

            feature_impacts = feature_impacts[:5]

            print("=" * 60)
            print("Prediction :", prediction)
            print("Probability (Denied / Certified) :", probability)
            print("Top SHAP Features")
            print(feature_impacts)
            print("=" * 60)

            return prediction, probability, feature_impacts

        except Exception as e:
            raise USvisaException(e, sys)