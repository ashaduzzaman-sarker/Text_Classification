# ============================================================================
# src/text_classification/components/data_validation.py
# ============================================================================
"""Data validation component for checking dataset quality."""

from pathlib import Path
from typing import List
from datasets import load_from_disk, Dataset

from text_classification.logging.logger import logger
from text_classification.entity.config_entity import DataValidationConfig


class DataValidation:
    """Validates dataset structure, columns, and data quality."""

    def __init__(self, config: DataValidationConfig):
        """
        Initialize data validation component.

        Args:
            config (DataValidationConfig): Configuration for validation stage.
        """
        self.config = config
        self.validation_status: bool = True
        self.validation_errors: List[str] = []

    def validate_dataset_exists(self) -> bool:
        """Check if dataset directory exists."""
        if not self.config.data_dir.exists():
            error_msg = f"Dataset directory not found: {self.config.data_dir}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False

        logger.info(f"Dataset directory exists: {self.config.data_dir}")
        return True

    def validate_columns(self, dataset: Dataset) -> bool:
        """Check if all required columns exist in the dataset."""
        dataset_columns = list(dataset.features.keys())
        logger.info(f"Dataset columns: {dataset_columns}")

        missing_columns = [
            col for col in self.config.required_columns if col not in dataset_columns
        ]

        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False

        logger.info("All required columns are present")
        return True

    def validate_data_quality(self, dataset: Dataset) -> bool:
        """Perform data quality checks."""
        quality_passed = True
        num_samples = len(dataset)
        logger.info(f"Dataset size: {num_samples} samples")

        # Check minimum samples
        if num_samples < self.config.min_samples:
            error_msg = (
                f"Dataset too small: {num_samples} samples "
                f"(minimum required: {self.config.min_samples})"
            )
            logger.warning(error_msg)
            self.validation_errors.append(error_msg)
            quality_passed = False

        # Check for null or empty values in required columns
        for column in self.config.required_columns:
            # null_count = sum(1 for item in dataset[column] if not item)
            null_count = sum(1 for item in dataset[column] if item is None)
            null_percentage = (null_count / num_samples) * 100

            if null_count > 0:
                warning_msg = (
                    f"Column '{column}' has {null_count} null/empty values "
                    f"({null_percentage:.2f}%)"
                )
                logger.warning(warning_msg)
                self.validation_errors.append(warning_msg)

                # Fail if null values exceed 5%
                if null_percentage > 5.0:
                    quality_passed = False

        if quality_passed:
            logger.info("Data quality checks passed")

        return quality_passed

    def validate(self) -> bool:
        """Run all validation checks and write status to file."""
        try:
            logger.info("Starting data validation stage")

            # Check dataset existence
            if not self.validate_dataset_exists():
                self.validation_status = False
                self._write_status()
                return False

            # Load dataset from disk
            logger.info(f"Loading dataset from {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))

            # Validate columns and data quality
            if not self.validate_columns(dataset):
                self.validation_status = False
            if not self.validate_data_quality(dataset):
                self.validation_status = False

            # Write validation status
            self._write_status()

            if self.validation_status:
                logger.info("All validation checks passed successfully")
            else:
                logger.error(f"Validation failed with {len(self.validation_errors)} errors/warnings")

            return self.validation_status

        except Exception as e:
            logger.exception(f"Data validation encountered an exception: {e}")
            self.validation_status = False
            self.validation_errors.append(str(e))
            self._write_status()
            raise

    def _write_status(self) -> None:
        """Write validation status and errors/warnings to the status file."""
        status_content = f"Validation Status: {'PASSED' if self.validation_status else 'FAILED'}"

        if self.validation_errors:
            status_content += "\n\nErrors/Warnings:\n" + "\n".join(f"- {err}" for err in self.validation_errors)

        self.config.status_file.write_text(status_content)
        logger.info(f"Validation status written to {self.config.status_file}")
