import os
from pathlib import Path

poject_name = "us_visa"

list_of_files = [
    ".github/workflows/.gitkeep",

    f"{poject_name}/__init__.py",
    f"{poject_name}/components/__init__.py",
    f"{poject_name}/components/data_ingestion.py",  
    f"{poject_name}/components/data_validation.py",
    f"{poject_name}/components/model_trainer.py",
    f"{poject_name}/components/model_evaluation.py",
    f"{poject_name}/components/model_pusher.py",
    f"{poject_name}/configuration/__init__.py",
    f"{poject_name}/constants/__init__.py",
    f"{poject_name}/entity/__init__.py",
    f"{poject_name}/entity/config_entity.py",
    f"{poject_name}/entity/artifact_entity.py",
    f"{poject_name}/exception/__init__.py",
    f"{poject_name}/logger/__init__.py",
    f"{poject_name}/pipeline/__init__.py",
    f"{poject_name}/pipeline/training_pipeline.py",
    f"{poject_name}/pipeline/prediction_pipeline.py",
    f"{poject_name}/utils/__init__.py",
    f"{poject_name}/utils/main_utils.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "main.py",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        print(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            print(f"Creating empty file: {filepath}")

    else:
        print(f"{filename} is already exists")