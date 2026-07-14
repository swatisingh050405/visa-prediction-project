import os
import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

dagshub.init(
    repo_owner="swatisingh050405",
    repo_name="us_visa_prediction",
    mlflow=True
)
print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", mlflow.get_experiment_by_name("US Visa Prediction"))

os.environ["MLFLOW_TRACKING_USERNAME"] = "swatisingh050405"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")