# ============================================================================
# src/text_classification/pipeline/stage_02_data_validation.py
# ============================================================================
"""Data validation pipeline stage."""

from text_classification.config.configuration import ConfigurationManager
from text_classification.components.data_validation import DataValidation
from text_classification.logging.logger import logger


class DataValidationPipeline:
    """Pipeline stage for validating ingested dataset."""

    def __init__(self):
        self.stage_name = "Data Validation"

    def run(self) -> bool:
        """
        Execute data validation pipeline.

        Returns:
            bool: True if all validations pass, False otherwise.
        """
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")

            # Load configuration
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()

            # Run validation
            validator = DataValidation(config=data_validation_config)
            validation_passed = validator.validate()

            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return validation_passed

        except Exception as e:
            logger.exception(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            raise
