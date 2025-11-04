# ============================================================================
# src/text_classification/components/data_preprocessing.py
# ============================================================================
"""Data transformation component for tokenizing text for classification."""

from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from text_classification.logging.logger import logger
from text_classification.entity.config_entity import DataTransformationConfig


class DataTransformation:
    """Handles tokenization and preprocessing of text data for classification."""

    def __init__(self, config: DataTransformationConfig):
        """Initialize with configuration.
        
        Args:
            config: DataTransformationConfig instance
        """
        self.config = config
        self.tokenizer = None

    def load_tokenizer(self):
        """Load pretrained tokenizer."""
        try:
            if self.tokenizer is None:
                logger.info(f"Loading tokenizer: {self.config.tokenizer_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
                logger.info("Tokenizer loaded successfully")
            return self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def preprocess_function(self, examples):
        """Tokenize text and attach labels for classification."""
        try:
            if self.config.text_column not in examples:
                raise KeyError(f"Missing text column '{self.config.text_column}' in examples")

            if self.tokenizer is None:
                self.load_tokenizer()

            # Tokenize text
            inputs = examples[self.config.text_column]
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation
            )

            # Safely attach labels if column exists
            if self.config.label_column in examples:
                model_inputs["labels"] = examples[self.config.label_column]

            return model_inputs

        except Exception as e:
            logger.error(f"Error in preprocess_function: {e}")
            raise

    def transform(self):
        """Transform dataset by tokenizing all examples."""
        try:
            logger.info("Starting data transformation")

            # Load tokenizer
            self.load_tokenizer()

            # Load dataset
            logger.info(f"Loading dataset from {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))
            logger.info(f"Dataset loaded successfully with {len(dataset)} samples")

            # Tokenize dataset
            logger.info("Tokenizing dataset...")
            tokenized_dataset = dataset.map(
                self.preprocess_function,
                batched=True,
                batch_size=self.config.batch_size,
                remove_columns=dataset.column_names,
                desc="Tokenizing"
            )

            logger.info(f"Tokenization complete: {len(tokenized_dataset)} samples")
            logger.info(f"Features: {list(tokenized_dataset.features.keys())}")

            # Save tokenized dataset
            output_path = self.config.root_dir / "tokenized_dataset"
            tokenized_dataset.save_to_disk(str(output_path))
            logger.info(f"Tokenized dataset saved to: {output_path}")

            # Save tokenizer for reproducibility
            tokenizer_path = self.config.root_dir / "tokenizer"
            self.tokenizer.save_pretrained(str(tokenizer_path))
            logger.info(f"Tokenizer saved to: {tokenizer_path}")

            return tokenized_dataset

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise
