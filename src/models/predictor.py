"""Multi-output ML model for football match predictions with confidence scoring."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import Config
from src.features.engineer import AdvancedFeatureEngineer


class FootballPredictor:
    """Multi-output football match predictor with confidence scoring."""
    
    def __init__(self):
        self.config = Config()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Models for different prediction tasks
        self.result_model = None  # Win/Draw/Loss probabilities
        self.goals_model = None   # Goal predictions
        self.confidence_model = None  # Confidence scoring
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Model performance metrics
        self.performance_metrics = {}
        
    def load_training_data(self) -> pd.DataFrame:
        """Load engineered features for training."""
        features_path = os.path.join(Config.DATA_PROCESSED_PATH, "engineered_features.csv")
        
        if os.path.exists(features_path):
            df = pd.read_csv(features_path)
            df["date"] = pd.to_datetime(df["date"])
            return df
        else:
            print("No engineered features found. Running feature engineering...")
            return self.feature_engineer.engineer_all_features()
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare features and multiple target variables."""
        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns()
        
        # Filter available features
        available_features = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        print(f"Using {len(self.feature_columns)} features for training")
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)  # Fill missing values
        
        # Prepare targets
        targets = {
            # Match result probabilities (will be converted to probabilities)
            "result": df[["home_win", "draw", "away_win"]].values,
            
            # Goal predictions
            "goals": df[["home_goals_target", "away_goals_target", "total_goals_target"]].values,
            
            # Over/Under and BTTS
            "specials": df[["over_1_5", "over_2_5", "over_3_5", "btts"]].values
        }
        
        return X.values, targets
    
    def create_confidence_features(self, X: np.ndarray, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create features for confidence model based on predictions and input features."""
        confidence_features = []
        
        # Add original features (scaled)
        confidence_features.append(X)
        
        # Add prediction uncertainties
        if "result" in predictions:
            result_probs = predictions["result"]
            # Entropy as uncertainty measure
            entropy = -np.sum(result_probs * np.log(result_probs + 1e-8), axis=1).reshape(-1, 1)
            confidence_features.append(entropy)
            
            # Max probability (higher = more confident)
            max_prob = np.max(result_probs, axis=1).reshape(-1, 1)
            confidence_features.append(max_prob)
            
            # Probability spread (difference between top 2 probabilities)
            sorted_probs = np.sort(result_probs, axis=1)
            prob_spread = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
            confidence_features.append(prob_spread)
        
        return np.hstack(confidence_features)
    
    def train_models(self, test_size: float = 0.2, random_state: int = 42):
        """Train all prediction models."""
        print("Loading training data...")
        df = self.load_training_data()
        
        if df.empty:
            raise ValueError("No training data available")
        
        print(f"Training data shape: {df.shape}")
        
        # Prepare features and targets
        X, targets = self.prepare_features_and_targets(df)
        
        # Split data chronologically (more realistic for time series)
        split_date = df["date"].quantile(0.8)  # Use 80% for training
        train_mask = df["date"] <= split_date
        
        X_train, X_test = X[train_mask], X[~train_mask]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training models...")
        
        # 1. Train result prediction model (classification)
        print("  Training result prediction model...")
        y_result_train = targets["result"][train_mask]
        y_result_test = targets["result"][~train_mask]
        
        self.result_model = MultiOutputClassifier(
            RandomForestClassifier(**Config.MODEL_PARAMS["random_forest"])
        )
        self.result_model.fit(X_train_scaled, y_result_train)
        
        # Evaluate result model
        result_pred = self.result_model.predict(X_test_scaled)
        result_accuracy = accuracy_score(y_result_test.argmax(axis=1), result_pred.argmax(axis=1))
        self.performance_metrics["result_accuracy"] = result_accuracy
        
        # 2. Train goals prediction model (regression)
        print("  Training goals prediction model...")
        y_goals_train = targets["goals"][train_mask]
        y_goals_test = targets["goals"][~train_mask]
        
        self.goals_model = MultiOutputRegressor(
            RandomForestRegressor(**Config.MODEL_PARAMS["random_forest"])
        )
        self.goals_model.fit(X_train_scaled, y_goals_train)
        
        # Evaluate goals model
        goals_pred = self.goals_model.predict(X_test_scaled)
        goals_mse = mean_squared_error(y_goals_test, goals_pred)
        goals_r2 = r2_score(y_goals_test, goals_pred)
        self.performance_metrics["goals_mse"] = goals_mse
        self.performance_metrics["goals_r2"] = goals_r2
        
        # 3. Train confidence model
        print("  Training confidence model...")
        
        # Get predictions for confidence training
        result_probs_train = self.result_model.predict_proba(X_train_scaled)
        result_probs_test = self.result_model.predict_proba(X_test_scaled)
        
        # Convert to probabilities format
        result_probs_train_formatted = np.array([prob[:, 1] for prob in result_probs_train]).T
        result_probs_test_formatted = np.array([prob[:, 1] for prob in result_probs_test]).T
        
        # Create confidence features
        conf_features_train = self.create_confidence_features(
            X_train_scaled, {"result": result_probs_train_formatted}
        )
        conf_features_test = self.create_confidence_features(
            X_test_scaled, {"result": result_probs_test_formatted}
        )
        
        # Create confidence targets (based on prediction accuracy)
        conf_targets_train = self._calculate_confidence_targets(
            y_result_train, result_probs_train_formatted
        )
        conf_targets_test = self._calculate_confidence_targets(
            y_result_test, result_probs_test_formatted
        )
        
        self.confidence_model = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=random_state
        )
        self.confidence_model.fit(conf_features_train, conf_targets_train)
        
        # Evaluate confidence model
        conf_pred = self.confidence_model.predict(conf_features_test)
        conf_r2 = r2_score(conf_targets_test, conf_pred)
        self.performance_metrics["confidence_r2"] = conf_r2
        
        self.is_trained = True
        
        print("\nModel Training Complete!")
        print(f"Result Accuracy: {result_accuracy:.3f}")
        print(f"Goals MSE: {goals_mse:.3f}")
        print(f"Goals R²: {goals_r2:.3f}")
        print(f"Confidence R²: {conf_r2:.3f}")
    
    def _calculate_confidence_targets(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> np.ndarray:
        """Calculate confidence targets based on prediction accuracy."""
        # Convert true labels to class indices
        y_true_classes = y_true.argmax(axis=1)
        
        # Get predicted probabilities for true classes
        true_class_probs = y_pred_probs[np.arange(len(y_true_classes)), y_true_classes]
        
        # Confidence target is the probability assigned to the correct class
        # Scaled to 0-1 range
        confidence_targets = np.clip(true_class_probs, 0, 1)
        
        return confidence_targets
    
    def predict_match(self, home_team_features: Dict, away_team_features: Dict, 
                     match_context: Dict) -> Dict[str, Any]:
        """Predict a single match with confidence scores."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Prepare feature vector
        feature_vector = self._prepare_prediction_features(
            home_team_features, away_team_features, match_context
        )
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Get predictions
        predictions = {}
        
        # Result probabilities
        result_probs = self.result_model.predict_proba(feature_vector_scaled)
        result_probs_formatted = np.array([prob[:, 1] for prob in result_probs]).T
        
        predictions["result_probabilities"] = {
            "home_win": float(result_probs_formatted[0, 0]),
            "draw": float(result_probs_formatted[0, 1]),
            "away_win": float(result_probs_formatted[0, 2])
        }
        
        # Most likely result
        result_classes = ["home_win", "draw", "away_win"]
        most_likely_result = result_classes[np.argmax(result_probs_formatted[0])]
        predictions["most_likely_result"] = most_likely_result
        
        # Goal predictions
        goals_pred = self.goals_model.predict(feature_vector_scaled)
        predictions["goals"] = {
            "home_goals": max(0, round(float(goals_pred[0, 0]))),
            "away_goals": max(0, round(float(goals_pred[0, 1]))),
            "total_goals": max(0, round(float(goals_pred[0, 2])))
        }
        
        # Confidence score
        conf_features = self.create_confidence_features(
            feature_vector_scaled, {"result": result_probs_formatted}
        )
        confidence_score = self.confidence_model.predict(conf_features.reshape(1, -1))[0]
        predictions["confidence_score"] = float(np.clip(confidence_score, 0, 1))
        
        # Additional insights
        predictions["insights"] = self._generate_match_insights(predictions, match_context)
        
        return predictions
    
    def _prepare_prediction_features(self, home_features: Dict, away_features: Dict, 
                                   context: Dict) -> np.ndarray:
        """Prepare feature vector for prediction."""
        # This is a simplified version - in practice, you'd need to calculate
        # all the engineered features based on recent team performance
        
        feature_vector = np.zeros(len(self.feature_columns))
        
        # Map provided features to feature vector
        # This would need to be implemented based on your specific feature engineering
        
        return feature_vector
    
    def _generate_match_insights(self, predictions: Dict, context: Dict) -> List[str]:
        """Generate human-readable insights about the prediction."""
        insights = []
        
        result_probs = predictions["result_probabilities"]
        confidence = predictions["confidence_score"]
        
        # Confidence level
        if confidence > 0.8:
            insights.append("High confidence prediction")
        elif confidence > 0.6:
            insights.append("Moderate confidence prediction")
        else:
            insights.append("Low confidence prediction - unpredictable match")
        
        # Result analysis
        max_prob = max(result_probs.values())
        if max_prob > 0.6:
            insights.append(f"Strong favorite identified ({predictions['most_likely_result']})")
        elif max_prob < 0.4:
            insights.append("Very close match - any result possible")
        
        # Goals analysis
        total_goals = predictions["goals"]["total_goals"]
        if total_goals >= 3:
            insights.append("High-scoring match expected")
        elif total_goals <= 1:
            insights.append("Low-scoring match expected")
        
        return insights
    
    def save_models(self, model_dir: str = None):
        """Save trained models to disk."""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        if model_dir is None:
            model_dir = Config.MODELS_PATH
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.result_model, os.path.join(model_dir, "result_model.pkl"))
        joblib.dump(self.goals_model, os.path.join(model_dir, "goals_model.pkl"))
        joblib.dump(self.confidence_model, os.path.join(model_dir, "confidence_model.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
        
        # Save metadata
        metadata = {
            "feature_columns": self.feature_columns,
            "performance_metrics": self.performance_metrics,
            "training_date": datetime.now().isoformat(),
            "model_version": "1.0"
        }
        
        import json
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = None):
        """Load trained models from disk."""
        if model_dir is None:
            model_dir = Config.MODELS_PATH
        
        try:
            self.result_model = joblib.load(os.path.join(model_dir, "result_model.pkl"))
            self.goals_model = joblib.load(os.path.join(model_dir, "goals_model.pkl"))
            self.confidence_model = joblib.load(os.path.join(model_dir, "confidence_model.pkl"))
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            
            # Load metadata
            import json
            with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata["feature_columns"]
            self.performance_metrics = metadata["performance_metrics"]
            self.is_trained = True
            
            print(f"Models loaded from {model_dir}")
            print(f"Model version: {metadata.get('model_version', 'Unknown')}")
            print(f"Training date: {metadata.get('training_date', 'Unknown')}")
            
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            self.is_trained = False
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get model performance metrics."""
        return self.performance_metrics.copy()


if __name__ == "__main__":
    # Example usage
    predictor = FootballPredictor()
    
    print("Training models...")
    predictor.train_models()
    
    print("Saving models...")
    predictor.save_models()
    
    print("Model training and saving completed!")