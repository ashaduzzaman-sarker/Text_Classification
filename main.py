# ============================================================================
# main.py
# ============================================================================
"""Main execution script."""

from text_classification.pipeline.stage_01_data_loader import DataIngestionPipeline
# from text_classification.pipeline.stage_02_data_validation import DataValidationPipeline
# from text_classification.pipeline.stage_03_data_transformation import DataTransformationPipeline
# from text_classification.pipeline.stage_04_model_trainer import ModelTrainerPipeline
# from text_classification.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

from text_classification.logging.logger import logger


if __name__ == "__main__":
    # Stage 1: Data Ingestion
    try:
        logger.info("=" * 60)
        pipeline = DataIngestionPipeline()
        pipeline.run()
    except Exception as e:
        logger.error("Data Ingestion failed")
        raise
    
    # # Stage 2: Data Validation
    # try:
    #     logger.info("=" * 60)
    #     pipeline = DataValidationPipeline()
    #     validation_passed = pipeline.run()
        
    #     if not validation_passed:
    #         logger.error("Data validation failed - check status file")
    #         raise ValueError("Data validation checks failed")
        
    # except Exception as e:
    #     logger.error("Data Validation failed")
    #     raise
    
    # # Stage 3: Data Transformation
    # try:
    #     logger.info("=" * 60)
    #     pipeline = DataTransformationPipeline()
    #     pipeline.run()
    # except Exception as e:
    #     logger.error("Data Transformation failed")
    #     raise
    
    # # Stage 4: Model Training
    # try:
    #     logger.info("=" * 60)
    #     pipeline = ModelTrainerPipeline()
    #     pipeline.run()
    # except Exception as e:
    #     logger.error("Model Training failed")
    #     raise
    
    # # Stage 5: Model Evaluation
    # try:
    #     logger.info("=" * 60)
    #     pipeline = ModelEvaluationPipeline()
    #     metrics = pipeline.run()
        
    #     logger.info("\nFinal Evaluation Metrics:")
    #     for metric_name, score in metrics.items():
    #         logger.info(f"  {metric_name}: {score:.4f}")
        
    # except Exception as e:
    #     logger.error("Model Evaluation failed")
    #     raise
    
    logger.info("=" * 60)
    logger.info("All pipeline stages completed successfully!")
    # logger.info(f"Trained model: artifacts/model_trainer/final_model")
    # logger.info(f"Evaluation report: artifacts/model_evaluation/evaluation_report.txt")
    # logger.info(f"Metrics: artifacts/model_evaluation/metrics.json")