#!/usr/bin/env python3
"""
Standalone script for training football prediction models.
Used by GitHub Actions and Render cron jobs.
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.collector import FootballDataCollector
from src.data.processor import MatchDataProcessor
from src.features.engineer import AdvancedFeatureEngineer
from src.models.predictor import FootballPredictor
from src.utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles the complete model training pipeline."""
    
    def __init__(self):
        self.collector = FootballDataCollector()
        self.processor = MatchDataProcessor()
        self.engineer = AdvancedFeatureEngineer()
        self.predictor = FootballPredictor()
        
        # Create necessary directories
        os.makedirs(Config.DATA_RAW_PATH, exist_ok=True)
        os.makedirs(Config.DATA_PROCESSED_PATH, exist_ok=True)
        os.makedirs(Config.DATA_MODELS_PATH, exist_ok=True)
    
    def collect_data(self) -> bool:
        """Collect fresh training data."""
        try:
            logger.info("Starting data collection...")
            
            # Collect historical data for all supported competitions
            historical_data = self.collector.collect_historical_data()
            
            if historical_data.empty:
                logger.error("No historical data collected")
                return False
            
            # Save raw data
            raw_file = os.path.join(Config.DATA_RAW_PATH, "historical_matches.csv")
            self.collector.save_data(historical_data, "historical_matches.csv")
            
            logger.info(f"Collected {len(historical_data)} historical matches")
            logger.info(f"Data saved to {raw_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process and clean the collected data."""
        try:
            logger.info("Starting data processing...")
            
            # Process the raw data
            self.processor.process_and_save(
                "historical_matches.csv", 
                "processed_matches.csv"
            )
            
            processed_file = os.path.join(Config.DATA_PROCESSED_PATH, "processed_matches.csv")
            
            if not os.path.exists(processed_file):
                logger.error("Processed data file not created")
                return False
            
            logger.info(f"Data processing completed: {processed_file}")
            return True
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return False
    
    def engineer_features(self) -> bool:
        """Engineer features for model training."""
        try:
            logger.info("Starting feature engineering...")
            
            # Engineer all features
            self.engineer.engineer_all_features()
            
            features_file = os.path.join(Config.DATA_PROCESSED_PATH, "engineered_features.csv")
            
            if not os.path.exists(features_file):
                logger.error("Engineered features file not created")
                return False
            
            logger.info(f"Feature engineering completed: {features_file}")
            return True
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return False
    
    def train_models(self) -> bool:
        """Train all prediction models."""
        try:
            logger.info("Starting model training...")
            
            # Train models
            self.predictor.train_models()
            
            # Get performance metrics
            performance = self.predictor.get_model_performance()
            logger.info(f"Training completed. Performance: {performance}")
            
            # Validate performance
            if not self.validate_performance(performance):
                logger.error("Model performance validation failed")
                return False
            
            # Save models
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.predictor.save_models(version=model_version)
            
            # Save performance metrics
            self.save_performance_metrics(performance, model_version)
            
            logger.info(f"Models saved with version: {model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def validate_performance(self, performance: dict) -> bool:
        """Validate model performance against thresholds."""
        try:
            # Define minimum performance thresholds
            thresholds = {
                'result_accuracy': 0.45,  # 45% accuracy for result prediction
                'confidence_correlation': 0.3,  # Correlation for confidence scores
                'goals_mae': 2.0  # Maximum MAE for goals prediction
            }
            
            for metric, threshold in thresholds.items():
                value = performance.get(metric, 0)
                
                if metric == 'goals_mae':
                    # For MAE, lower is better
                    if value > threshold:
                        logger.error(f"{metric}: {value:.3f} exceeds threshold {threshold}")
                        return False
                else:
                    # For other metrics, higher is better
                    if value < threshold:
                        logger.error(f"{metric}: {value:.3f} below threshold {threshold}")
                        return False
                
                logger.info(f"{metric}: {value:.3f} ✓")
            
            logger.info("All performance validations passed")
            return True
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return False
    
    def save_performance_metrics(self, performance: dict, version: str):
        """Save performance metrics to file."""
        try:
            metrics_data = {
                'version': version,
                'performance': performance,
                'training_date': datetime.now().isoformat(),
                'training_mode': os.getenv('MODEL_TRAINING_MODE', 'manual')
            }
            
            metrics_file = os.path.join(Config.DATA_MODELS_PATH, f"performance_{version}.json")
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Also save as latest
            latest_file = os.path.join(Config.DATA_MODELS_PATH, "latest_performance.json")
            with open(latest_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Performance metrics saved: {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
    
    def run_full_pipeline(self) -> bool:
        """Run the complete training pipeline."""
        logger.info("Starting full model training pipeline...")
        
        steps = [
            ("Data Collection", self.collect_data),
            ("Data Processing", self.process_data),
            ("Feature Engineering", self.engineer_features),
            ("Model Training", self.train_models)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Executing: {step_name}")
            
            if not step_func():
                logger.error(f"Pipeline failed at: {step_name}")
                return False
            
            logger.info(f"Completed: {step_name}")
        
        logger.info("Full training pipeline completed successfully!")
        return True

def main():
    """Main training script entry point."""
    logger.info("Football Prediction Model Training Script")
    logger.info("=" * 50)
    
    # Check environment
    api_key = os.getenv('FOOTBALL_DATA_API_KEY')
    if not api_key:
        logger.error("FOOTBALL_DATA_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Run training pipeline
    success = trainer.run_full_pipeline()
    
    if success:
        logger.info("Training completed successfully!")
        
        # Send notification if webhook is configured
        webhook_url = os.getenv('NOTIFICATION_WEBHOOK')
        if webhook_url:
            try:
                import requests
                requests.post(webhook_url, json={
                    'message': 'Football prediction model training completed successfully',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                })
                logger.info("Notification sent")
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
        
        sys.exit(0)
    else:
        logger.error("Training failed!")
        
        # Send failure notification
        webhook_url = os.getenv('NOTIFICATION_WEBHOOK')
        if webhook_url:
            try:
                import requests
                requests.post(webhook_url, json={
                    'message': 'Football prediction model training FAILED',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed'
                })
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()