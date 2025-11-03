"""
Project Template Generator

"""

import os
from pathlib import Path
import logging
from datetime import datetime

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"project_setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# -----------------------------------------------------------------------------
# Project Name (customizable)
# -----------------------------------------------------------------------------
PROJECT_NAME = "text_classification"

# -----------------------------------------------------------------------------
# Files & Directories to Create
# -----------------------------------------------------------------------------
FILES_TO_CREATE = [
    # CI/CD
    ".github/workflows/ci_cd.yml",

    # Core directories
    f"src/{PROJECT_NAME}/__init__.py",
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/components/data_ingestion.py",
    f"src/{PROJECT_NAME}/components/data_preprocessing.py",
    f"src/{PROJECT_NAME}/components/model_trainer.py",
    f"src/{PROJECT_NAME}/components/model_evaluation.py",

    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/pipeline/01_data_ingestion_pipeline.py",
    f"src/{PROJECT_NAME}/pipeline/02_data_preprocessing_pipeline.py",
    f"src/{PROJECT_NAME}/pipeline/03_model_trainer_pipeline.py",
    f"src/{PROJECT_NAME}/pipeline/04_model_evaluation_pipeline.py",
    f"src/{PROJECT_NAME}/pipeline/05_prediction_pipeline.py",

    f"src/{PROJECT_NAME}/entity/__init__.py",
    f"src/{PROJECT_NAME}/entity/config_entity.py",

    f"src/{PROJECT_NAME}/constants/__init__.py",
    f"src/{PROJECT_NAME}/constants/constants.py",

    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/configuration.py",

    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/utils/common.py",

    f"src/{PROJECT_NAME}/logging/__init__.py",
    f"src/{PROJECT_NAME}/logging/logger.py",

    # Root-level configuration
    "config/config.yaml",
    "config/params.yaml",
    ".env.example",

    # Docs & Notebooks
    "README.md",
    "notebooks",

    # Application & Deployment
    "main.py",
    "app.py",
    "streamlit_app.py"
    "Dockerfile",
    "requirements.txt",
    "pyproject.toml",
]

# -----------------------------------------------------------------------------
# Function: Create Files and Directories
# -----------------------------------------------------------------------------
def create_project_structure(file_list):
    """Creates directories and files for the project structure."""
    for filepath in file_list:
        path = Path(filepath)
        dir_path = path.parent

        # Create directories
        if dir_path and not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")

        # Create empty files if they don't exist or are empty
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", encoding="utf-8") as f:
                pass
            logging.info(f"Created empty file: {path}")
        else:
            logging.info(f"File already exists: {path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Starting project structure setup...")
    create_project_structure(FILES_TO_CREATE)
    logging.info(f"Project '{PROJECT_NAME}' initialized successfully.")