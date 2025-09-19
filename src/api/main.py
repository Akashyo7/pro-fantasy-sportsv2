"""FastAPI backend for football match predictions."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import os
import asyncio
import logging

from src.models.predictor import FootballPredictor
from src.data.collector import FootballDataCollector
from src.data.processor import MatchDataProcessor
from src.features.engineer import AdvancedFeatureEngineer
from src.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Football Prediction API",
    description="AI-powered football match predictions with confidence scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
predictor = FootballPredictor()
data_collector = FootballDataCollector()
data_processor = MatchDataProcessor()
feature_engineer = AdvancedFeatureEngineer()

# Pydantic models for API
class MatchPredictionRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    competition_id: int
    match_date: Optional[str] = None
    
class TeamFeatures(BaseModel):
    recent_form: Optional[List[str]] = None
    goals_for_avg: Optional[float] = None
    goals_against_avg: Optional[float] = None
    league_position: Optional[int] = None
    
class PredictionResponse(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    competition: str
    predictions: Dict[str, Any]
    confidence_score: float
    insights: List[str]
    generated_at: str

class UpcomingMatchesResponse(BaseModel):
    matches: List[Dict[str, Any]]
    total_matches: int
    competition: str

class ModelStatusResponse(BaseModel):
    is_trained: bool
    last_training_date: Optional[str]
    performance_metrics: Dict[str, float]
    model_version: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and load data on startup."""
    logger.info("Starting Football Prediction API...")
    
    try:
        # Try to load existing models
        predictor.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
        logger.info("Models will need to be trained")

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Football Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "predict": "/predict/match",
            "upcoming": "/predict/upcoming/{competition}",
            "competitions": "/competitions"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint for production deployment."""
    try:
        # Check database connectivity
        from src.data.database import db_manager
        db_status = db_manager.health_check() if db_manager.client else False
        
        # Check if we can query the database
        try:
            if db_manager.client:
                # Simple query to test connection
                result = db_manager.client.table('matches').select('id').limit(1).execute()
                db_status = "connected"
            else:
                db_status = "disconnected"
        except Exception as e:
            logger.warning(f"Database connection issue: {e}")
            db_status = "disconnected"
        
        # Check model status
        model_status = "loaded" if predictor.is_trained else "not_loaded"
        
        # Overall health status
        overall_status = "healthy" if db_status == "connected" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": db_status,
                "model": model_status,
                "api": "running"
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "1.0.0"
        }

# Model status endpoint
@app.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get current model status and performance metrics."""
    return ModelStatusResponse(
        is_trained=predictor.is_trained,
        last_training_date=None,  # Would be loaded from metadata
        performance_metrics=predictor.get_model_performance(),
        model_version="1.0"
    )

# Prediction endpoints
@app.post("/predict/match", response_model=PredictionResponse)
async def predict_match(request: MatchPredictionRequest):
    """Predict outcome for a specific match."""
    if not predictor.is_trained:
        raise HTTPException(
            status_code=503, 
            detail="Model not trained. Please train the model first."
        )
    
    try:
        # Get team information (simplified - would need actual team data)
        home_team_features = await get_team_features(request.home_team_id)
        away_team_features = await get_team_features(request.away_team_id)
        
        match_context = {
            "competition_id": request.competition_id,
            "match_date": request.match_date or datetime.now().isoformat()
        }
        
        # Make prediction
        predictions = predictor.predict_match(
            home_team_features, away_team_features, match_context
        )
        
        return PredictionResponse(
            match_id=f"{request.home_team_id}_{request.away_team_id}_{datetime.now().strftime('%Y%m%d')}",
            home_team=f"Team {request.home_team_id}",  # Would be actual team name
            away_team=f"Team {request.away_team_id}",  # Would be actual team name
            competition=Config.COMPETITIONS.get(str(request.competition_id), {}).get("name", "Unknown"),
            predictions=predictions,
            confidence_score=predictions["confidence_score"],
            insights=predictions["insights"],
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/upcoming/{competition}")
async def predict_upcoming_matches(competition: str, days_ahead: int = 7):
    """Get predictions for upcoming matches in a competition."""
    if not predictor.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please train the model first."
        )
    
    # Get competition ID
    comp_info = None
    for comp_key, comp_data in Config.COMPETITIONS.items():
        if comp_key == competition.lower():
            comp_info = comp_data
            break
    
    if not comp_info:
        raise HTTPException(status_code=404, detail="Competition not found")
    
    try:
        # Get upcoming matches
        upcoming_matches = data_collector.get_upcoming_matches(
            comp_info["id"], days_ahead
        )
        
        if upcoming_matches.empty:
            return UpcomingMatchesResponse(
                matches=[],
                total_matches=0,
                competition=comp_info["name"]
            )
        
        # Generate predictions for each match
        predictions = []
        for _, match in upcoming_matches.iterrows():
            try:
                home_features = await get_team_features(match["home_team_id"])
                away_features = await get_team_features(match["away_team_id"])
                
                match_context = {
                    "competition_id": comp_info["id"],
                    "match_date": match["date"]
                }
                
                prediction = predictor.predict_match(
                    home_features, away_features, match_context
                )
                
                predictions.append({
                    "match_id": match["id"],
                    "date": match["date"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "predictions": prediction,
                    "confidence_score": prediction["confidence_score"]
                })
                
            except Exception as e:
                logger.warning(f"Failed to predict match {match['id']}: {e}")
                continue
        
        return UpcomingMatchesResponse(
            matches=predictions,
            total_matches=len(predictions),
            competition=comp_info["name"]
        )
        
    except Exception as e:
        logger.error(f"Error getting upcoming matches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")

# Data management endpoints
@app.post("/data/collect")
async def collect_data(background_tasks: BackgroundTasks):
    """Trigger data collection in the background."""
    background_tasks.add_task(run_data_collection)
    return {"message": "Data collection started in background"}

@app.post("/model/train")
async def train_model(background_tasks: BackgroundTasks):
    """Trigger model training in the background."""
    if not os.path.exists(os.path.join(Config.DATA_PROCESSED_PATH, "engineered_features.csv")):
        raise HTTPException(
            status_code=400,
            detail="No training data available. Please collect and process data first."
        )
    
    background_tasks.add_task(run_model_training)
    return {"message": "Model training started in background"}

@app.post("/data/update")
async def update_data(background_tasks: BackgroundTasks):
    """Update data and retrain models."""
    background_tasks.add_task(run_full_update)
    return {"message": "Full data update and model retraining started"}

# Utility endpoints
@app.get("/competitions")
async def get_competitions():
    """Get list of supported competitions."""
    return {
        "competitions": [
            {"key": key, "name": info["name"], "id": info["id"]}
            for key, info in Config.COMPETITIONS.items()
        ]
    }

@app.get("/teams/{competition}")
async def get_teams(competition: str):
    """Get teams for a specific competition."""
    # This would return actual team data from the database
    return {"message": f"Teams for {competition} - implementation needed"}

# Helper functions
async def get_team_features(team_id: int) -> Dict:
    """Get team features for prediction (simplified version)."""
    # In a real implementation, this would fetch recent team performance data
    # and calculate the required features
    return {
        "recent_form": ["W", "W", "D", "L", "W"],
        "goals_for_avg": 1.8,
        "goals_against_avg": 1.2,
        "league_position": 10
    }

async def run_data_collection():
    """Background task for data collection."""
    try:
        logger.info("Starting data collection...")
        
        # Collect historical data
        historical_data = data_collector.collect_historical_data()
        if not historical_data.empty:
            data_collector.save_data(historical_data, "historical_matches.csv")
        
        # Process data
        data_processor.process_and_save("historical_matches.csv", "processed_matches.csv")
        
        # Engineer features
        feature_engineer.engineer_all_features()
        
        logger.info("Data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")

async def run_model_training():
    """Background task for model training."""
    try:
        logger.info("Starting model training...")
        
        predictor.train_models()
        predictor.save_models()
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")

async def run_full_update():
    """Background task for full data update and model retraining."""
    try:
        await run_data_collection()
        await run_model_training()
        logger.info("Full update completed successfully")
        
    except Exception as e:
        logger.error(f"Full update failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD
    )