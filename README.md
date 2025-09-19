# ⚽ Football Predictions API

A comprehensive football prediction system using machine learning to predict Premier League and Champions League match outcomes, goals, and confidence scores.

## 🚀 Features

- **Multi-Output Predictions**: Win/Draw/Loss probabilities + goal predictions + confidence scores
- **Advanced Metrics**: xG, defensive stats, team form, head-to-head records
- **Automated Training**: Weekly model updates via GitHub Actions
- **REST API**: FastAPI backend hosted on Render
- **Interactive Dashboard**: Streamlit frontend for gameweek predictions
- **Free Setup**: 100% free using football-data.org API

## 🏗️ Architecture

```
Data Collection (football-data.org) → Feature Engineering → ML Model → FastAPI → Streamlit
                    ↓
            GitHub Actions (Weekly Training)
```

## 📊 Predictions Include

- Match outcome probabilities (Home/Draw/Away)
- Expected goals for both teams
- Most likely exact scores
- Prediction confidence (0-100%)
- Historical accuracy tracking

## 🛠️ Tech Stack

- **Backend**: FastAPI, scikit-learn, XGBoost
- **Frontend**: Streamlit
- **Data**: football-data.org API
- **Deployment**: Render (API) + Streamlit Cloud
- **Automation**: GitHub Actions

## 🚀 Quick Start

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Run data collection: `python src/data/collect_data.py`
5. Train model: `python src/models/train_model.py`
6. Start API: `uvicorn src.api.main:app --reload`
7. Launch dashboard: `streamlit run src/frontend/dashboard.py`

## 📈 Model Performance

- Target accuracy: >55% for match outcomes
- Goal prediction MAE: <0.8 goals
- Confidence calibration: Well-calibrated probabilities

## 🔄 Automated Updates

Weekly GitHub Actions workflow:
- Fetches latest match data
- Retrains model with new data
- Updates predictions for upcoming matches
- Deploys updated model to production