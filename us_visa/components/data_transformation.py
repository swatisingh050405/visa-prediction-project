import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from us_visa.utils.feature_engineering import perform_feature_engineering

# Certified = 1, Denied = 0
mapping = {
    "Certified": 1,
    "Denied": 0
}


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            oh_columns = self._schema_config.get('oh_columns', [])
            or_columns = self._schema_config.get('or_columns', [])
            transform_columns = self._schema_config.get('transform_columns', [])
            num_features = self._schema_config.get('num_features', [])

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            oh_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            transform_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("OneHotEncoder", oh_pipeline, oh_columns),
                    ("Ordinal_Encoder", ordinal_pipeline, or_columns),
                    ("Transformer", transform_pipeline, transform_columns),
                    ("StandardScaler", num_pipeline, num_features)
                ],
                remainder='passthrough'
            )

            logging.info("Created preprocessor object with Imputers")
            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                drop_cols = self._schema_config.get('drop_columns', [])

                # 1. Separate Input & Target Features
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN].map(mapping)

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN].map(mapping)

                # 2. Perform Feature Engineering
                input_feature_train_df = perform_feature_engineering(input_feature_train_df, drop_cols)
                input_feature_test_df = perform_feature_engineering(input_feature_test_df, drop_cols)

                # 3. Fit-Transform Features
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                # 4. Clean any residual NaN / Inf values
                input_feature_train_arr = np.nan_to_num(input_feature_train_arr, nan=0.0, posinf=0.0, neginf=0.0)
                input_feature_test_arr = np.nan_to_num(input_feature_test_arr, nan=0.0, posinf=0.0, neginf=0.0)

                # 5. Direct Assignment (SMOTEENN Bypass to prevent overfitting)
                logging.info("Using raw transformed arrays without SMOTEENN sampling")
                input_feature_train_final = input_feature_train_arr
                target_feature_train_final = target_feature_train_df

                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Successfully completed Data Transformation")
                return DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USvisaException(e, sys) from e