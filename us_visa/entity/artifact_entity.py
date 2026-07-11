from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str 
    feature_store_file_path:str



@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    

@dataclass
class DataDriftArtifact:
    drift_report_file_path: str
    drift_report_json_file_path: str
    drift_status: bool


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float