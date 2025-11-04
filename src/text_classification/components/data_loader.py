# ============================================================================
# src/text_classification/components/data_ingestion.py
# ============================================================================
"""Data ingestion component for loading datasets from HuggingFace."""

from pathlib import Path
from typing import Optional
from datasets import Dataset, load_dataset

from text_classification.logging.logger import logger
from text_classification.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """Handles data ingestion from HuggingFace datasets."""

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize data ingestion component.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration and create necessary directories."""
        if not self.config.dataset_name:
            raise ValueError("dataset_name must be provided in DataIngestionConfig")

        # Ensure directories exist
        for directory in [self.config.root_dir, self.config.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory}")

    def download_dataset(self) -> Dataset:
        """
        Download and optionally limit the dataset from HuggingFace.

        Returns:
            Dataset: HuggingFace dataset object.

        Raises:
            ImportError: If `datasets` library is not installed.
            RuntimeError: For dataset loading failures.
        """
        try:
            logger.info(f"Loading dataset: {self.config.dataset_name}")
            logger.info(f"Config: {self.config.config_name}, Split: {self.config.split}")

            dataset = load_dataset(
                path=self.config.dataset_name,
                name=self.config.config_name,
                split=self.config.split,
                cache_dir=str(self.config.cache_dir)
            )

            # Limit dataset size for faster iteration if max_samples is set
            if self.config.max_samples:
                original_len = len(dataset)
                dataset = dataset.select(range(min(self.config.max_samples, original_len)))
                logger.info(f"Dataset limited: {original_len} -> {len(dataset)} samples")

            # Save dataset to disk
            output_path = self.config.root_dir / "dataset"
            dataset.save_to_disk(str(output_path))
            logger.info(f"Dataset saved to: {output_path}")
            logger.info(f"Dataset features: {list(dataset.features.keys())}")

            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
