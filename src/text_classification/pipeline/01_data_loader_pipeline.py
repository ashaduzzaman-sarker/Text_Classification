# ============================================================================
# src/text_classification/pipeline/stage_01_data_ingestion.py
# ============================================================================
"""Pipeline stage for data ingestion."""

from datasets import Dataset

from text_classification.config.configuration import ConfigurationManager
from text_classification.components.data_loader import DataIngestion
from text_classification.logging.logger import logger


class DataIngestionPipeline:
    """Pipeline for the data ingestion stage."""

    def __init__(self):
        self.stage_name = "Data Ingestion"

    def run(self) -> Dataset:
        """
        Execute the data ingestion pipeline.

        Returns:
            Dataset: Loaded HuggingFace dataset.
        """
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")

            # Load configuration
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()

            # Run data ingestion
            data_ingestion = DataIngestion(config=data_ingestion_config)
            dataset = data_ingestion.download_dataset()

            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return dataset

        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise
