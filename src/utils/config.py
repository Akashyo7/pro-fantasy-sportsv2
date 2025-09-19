"""Configuration management for the football prediction system."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the football prediction system."""
    
    # API Configuration
    FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
    FOOTBALL_DATA_BASE_URL = os.getenv("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
    
    # Model Configuration
    MODEL_UPDATE_FREQUENCY = os.getenv("MODEL_UPDATE_FREQUENCY", "weekly")
    PREDICTION_CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.6"))
    HISTORICAL_SEASONS = int(os.getenv("HISTORICAL_SEASONS", "3"))
    
    # API Server Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # Streamlit Configuration
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # Deployment
    RENDER_API_URL = os.getenv("RENDER_API_URL", "")
    
    # Data Paths
    DATA_RAW_PATH = "data/raw"
    DATA_PROCESSED_PATH = "data/processed"
    MODELS_PATH = "data/models"
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
    
    # Supported Competitions (using correct football-data.org IDs)
    COMPETITIONS = {
        "premier_league": {"id": 2021, "name": "Premier League", "code": "PL"},
        "la_liga": {"id": 2014, "name": "La Liga", "code": "PD"},
        "serie_a": {"id": 2019, "name": "Serie A", "code": "SA"},
        "bundesliga": {"id": 2002, "name": "Bundesliga", "code": "BL1"},
        "ligue_1": {"id": 2015, "name": "Ligue 1", "code": "FL1"},
        "champions_league": {"id": 2001, "name": "UEFA Champions League", "code": "CL"}
    }
    
    # Feature Engineering Parameters
    FORM_WINDOW = 5  # Last N matches for form calculation
    ROLLING_WINDOWS = [3, 5, 10]  # Different rolling averages
    
    # Model Parameters
    MODEL_PARAMS = {
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    }
    
    @classmethod
    def get_api_headers(cls) -> Dict[str, str]:
        """Get headers for football-data.org API requests."""
        return {
            "X-Auth-Token": cls.FOOTBALL_DATA_API_KEY,
            "Content-Type": "application/json"
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.FOOTBALL_DATA_API_KEY:
            raise ValueError("FOOTBALL_DATA_API_KEY is required")
        return True