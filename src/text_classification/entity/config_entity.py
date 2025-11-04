# ============================================================================
# src/text_classification/entity/config_entity.py
# ============================================================================
"""Configuration entities for the text classification pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Literal


# ==============================
# Data Ingestion
# ==============================
@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion stage."""
    root_dir: Path
    cache_dir: Path
    dataset_name: str
    config_name: Optional[str]
    split: str = "train"
    max_samples: Optional[int] = None
    test_size: float = 0.2


# ==============================
# Data Validation
# ==============================
@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation stage."""
    root_dir: Path
    status_file: Path
    data_dir: Path
    required_columns: List[str]
    min_samples: int


# ==============================
# Data Transformation
# ==============================
@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration for data transformation stage."""
    root_dir: Path
    data_dir: Path
    tokenizer_name: str
    max_length: int
    padding: Literal["max_length", "longest", "do_not_pad"]
    truncation: bool
    batch_size: int
    label_column: str
    text_column: str


# ==============================
# Model Trainer
# ==============================
@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for model training stage."""
    root_dir: Path
    data_dir: Path
    tokenizer_dir: Path
    model_name: str
    train_split: float
    seed: int


# ==============================
# Model Evaluation
# ==============================
@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Configuration for model evaluation stage."""
    root_dir: Path
    data_dir: Path
    model_dir: Path
    tokenizer_dir: Path
    metric_file: Path
