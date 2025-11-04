# ============================================================================
# src/text_classification/pipeline/stage_03_data_preprocessing.py
# ============================================================================
"""Data preprocessing pipeline stage for text classification."""

from text_classification.config.configuration import ConfigurationManager
from text_classification.components.data_preprocessing import DataTransformation
from text_classification.logging.logger import logger
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset


class DataTransformationPipeline:
    """Pipeline for data transformation and tokenization for classification."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.stage_name = "Data Transformation"
    
    def run(self) -> DatasetDict:
        """Execute data transformation pipeline.
        
        Returns:
            tokenized_dataset: DatasetDict with 'train' and 'test' splits
        """
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")
            
            # Load configuration
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()
            
            # Initialize transformer
            data_transformer = DataTransformation(config=data_transformation_config)
            
            # Load dataset from disk
            logger.info(f"Loading dataset from {data_transformation_config.data_dir}")
            raw_dataset = Dataset.load_from_disk(str(data_transformation_config.data_dir))
            logger.info(f"Dataset loaded: {len(raw_dataset)} samples")
            
            # Split train/test
            test_size = config_manager.get_data_ingestion_config().test_size
            logger.info(f"Splitting dataset: test_size={test_size}")
            train_dict, test_dict = train_test_split(
                raw_dataset,
                test_size=test_size,
                stratify=raw_dataset[data_transformation_config.label_column]
                if data_transformation_config.label_column in raw_dataset.column_names
                else None,
                random_state=42
            )
            
            dataset_splits = DatasetDict({
                "train": Dataset.from_dict(train_dict),
                "test": Dataset.from_dict(test_dict)
            })
            logger.info(f"Dataset split: train={len(dataset_splits['train'])}, test={len(dataset_splits['test'])}")
            
            # Tokenize each split
            logger.info("Tokenizing train and test datasets...")
            tokenized_dataset = dataset_splits.map(
                data_transformer.preprocess_function,
                batched=True,
                batch_size=data_transformation_config.batch_size,
                remove_columns=raw_dataset.column_names,
                desc="Tokenizing"
            )
            
            # Save transformed dataset
            output_path = data_transformation_config.root_dir / "tokenized_dataset"
            tokenized_dataset.save_to_disk(str(output_path))
            logger.info(f"Tokenized dataset saved to: {output_path}")
            
            # Save tokenizer for later use
            tokenizer_path = data_transformation_config.root_dir / "tokenizer"
            data_transformer.load_tokenizer()
            data_transformer.tokenizer.save_pretrained(str(tokenizer_path))
            logger.info(f"Tokenizer saved to: {tokenizer_path}")
            
            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise
