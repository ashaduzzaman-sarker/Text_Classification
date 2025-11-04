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
            logger.info(f"Loading tokenizer: {self.config.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            logger.info("Tokenizer loaded successfully")
            return self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def preprocess_function(self, examples):
        """Tokenize text and attach labels for classification."""
        # Extract text
        inputs = [text for text in examples[self.config.text_column]]
        
        # Tokenize text
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation
        )
        
        # Attach labels (keep numeric labels as is)
        labels = examples[self.config.label_column]
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def transform(self):
        """Transform dataset by tokenizing all examples."""
        try:
            logger.info("Starting data transformation")
            
            # Load tokenizer
            if self.tokenizer is None:
                self.load_tokenizer()
            
            # Load dataset
            logger.info(f"Loading dataset from {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))
            logger.info(f"Dataset loaded: {len(dataset)} samples")
            
            # Apply tokenization
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
            
            # Save transformed dataset
            output_path = self.config.root_dir / "tokenized_dataset"
            tokenized_dataset.save_to_disk(str(output_path))
            logger.info(f"Transformed dataset saved to: {output_path}")
            
            # Save tokenizer for later use
            tokenizer_path = self.config.root_dir / "tokenizer"
            self.tokenizer.save_pretrained(str(tokenizer_path))
            logger.info(f"Tokenizer saved to: {tokenizer_path}")
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise
