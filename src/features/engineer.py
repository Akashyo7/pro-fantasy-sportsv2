"""Advanced feature engineering for football match prediction."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

from src.utils.config import Config


class AdvancedFeatureEngineer:
    """Creates advanced features for football match prediction based on research insights."""
    
    def __init__(self):
        self.config = Config()
        self.form_window = Config.FORM_WINDOW
        self.rolling_windows = Config.ROLLING_WINDOWS
        
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed match data, standings, and team form."""
        matches_path = os.path.join(Config.DATA_PROCESSED_PATH, "processed_matches.csv")
        standings_path = os.path.join(Config.DATA_PROCESSED_PATH, "league_standings.csv")
        form_path = os.path.join(Config.DATA_PROCESSED_PATH, "team_form.csv")
        
        matches_df = pd.read_csv(matches_path) if os.path.exists(matches_path) else pd.DataFrame()
        standings_df = pd.read_csv(standings_path) if os.path.exists(standings_path) else pd.DataFrame()
        form_df = pd.read_csv(form_path) if os.path.exists(form_path) else pd.DataFrame()
        
        if not matches_df.empty:
            matches_df["date"] = pd.to_datetime(matches_df["date"])
        if not form_df.empty:
            form_df["date"] = pd.to_datetime(form_df["date"])
            
        return matches_df, standings_df, form_df
    
    def calculate_expected_goals_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate xG proxy based on goals scored and league averages."""
        # Since we don't have shot data, we'll create xG proxy based on:
        # 1. Goals scored vs league average
        # 2. Goals conceded vs league average
        # 3. Home advantage factor
        
        df = df.copy()
        
        # Calculate league averages by competition and season
        league_stats = df.groupby(["competition_id", "season"]).agg({
            "home_goals": "mean",
            "away_goals": "mean",
            "total_goals": "mean"
        }).reset_index()
        
        # Merge league averages
        df = df.merge(league_stats, on=["competition_id", "season"], suffixes=("", "_league_avg"))
        
        # Calculate xG proxy
        # Home team xG = (team's avg goals for * opponent's defensive weakness) * home advantage
        # Away team xG = (team's avg goals for * opponent's defensive weakness) * away factor
        
        # Home advantage factor (typically 1.3-1.4 for home teams)
        home_advantage = 1.35
        
        # Basic xG proxy (will be enhanced with more data)
        df["home_xg_proxy"] = df["home_goals_league_avg"] * home_advantage
        df["away_xg_proxy"] = df["away_goals_league_avg"]
        
        return df
    
    def calculate_team_strength_ratings(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """Calculate team strength ratings based on recent performance."""
        team_ratings = {}
        
        # Get unique teams
        all_teams = set(df["home_team_id"].unique()).union(set(df["away_team_id"].unique()))
        
        for team_id in all_teams:
            # Get team's recent matches (last 10)
            team_matches = self._get_recent_team_matches(df, team_id, limit=10)
            
            if len(team_matches) >= 5:  # Need minimum matches for rating
                # Calculate offensive and defensive ratings
                offensive_rating = team_matches["goals_for"].mean()
                defensive_rating = team_matches["goals_against"].mean()
                
                # Calculate form-based adjustments
                recent_form = team_matches.head(5)  # Last 5 matches
                form_points = recent_form["points"].mean()
                
                # Calculate strength rating (0-100 scale)
                strength_rating = min(100, max(0, 
                    50 + (offensive_rating - 1.5) * 10 - (defensive_rating - 1.5) * 10 + (form_points - 1) * 5
                ))
                
                team_ratings[team_id] = {
                    "offensive_rating": offensive_rating,
                    "defensive_rating": defensive_rating,
                    "strength_rating": strength_rating,
                    "form_points": form_points,
                    "matches_played": len(team_matches)
                }
        
        return team_ratings
    
    def _get_recent_team_matches(self, df: pd.DataFrame, team_id: int, limit: int = 10) -> pd.DataFrame:
        """Get recent matches for a team with results from team's perspective."""
        # Home matches
        home_matches = df[df["home_team_id"] == team_id].copy()
        home_matches["goals_for"] = home_matches["home_goals"]
        home_matches["goals_against"] = home_matches["away_goals"]
        home_matches["is_home"] = True
        home_matches["points"] = home_matches["result"].map({"H": 3, "D": 1, "A": 0})
        
        # Away matches
        away_matches = df[df["away_team_id"] == team_id].copy()
        away_matches["goals_for"] = away_matches["away_goals"]
        away_matches["goals_against"] = away_matches["home_goals"]
        away_matches["is_home"] = False
        away_matches["points"] = away_matches["result"].map({"A": 3, "D": 1, "H": 0})
        
        # Combine and sort by date (most recent first)
        all_matches = pd.concat([home_matches, away_matches])
        all_matches = all_matches.sort_values("date", ascending=False).head(limit)
        
        return all_matches[["date", "goals_for", "goals_against", "is_home", "points"]]
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and form features."""
        df = df.copy()
        
        # Sort by date
        df = df.sort_values("date")
        
        # Calculate rolling averages for different windows
        for window in self.rolling_windows:
            # Home team rolling stats
            df[f"home_goals_avg_{window}"] = df.groupby("home_team_id")["home_goals"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f"home_conceded_avg_{window}"] = df.groupby("home_team_id")["away_goals"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Away team rolling stats
            df[f"away_goals_avg_{window}"] = df.groupby("away_team_id")["away_goals"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f"away_conceded_avg_{window}"] = df.groupby("away_team_id")["home_goals"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        return df
    
    def calculate_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate head-to-head features between teams."""
        df = df.copy()
        
        # Initialize H2H features
        h2h_features = [
            "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_total_matches",
            "h2h_home_goals_avg", "h2h_away_goals_avg", "h2h_total_goals_avg"
        ]
        
        for feature in h2h_features:
            df[feature] = 0.0
        
        # Calculate H2H for each match
        for idx, match in df.iterrows():
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            match_date = match["date"]
            
            # Get historical matches between these teams (before current match)
            h2h_matches = df[
                (((df["home_team_id"] == home_id) & (df["away_team_id"] == away_id)) |
                 ((df["home_team_id"] == away_id) & (df["away_team_id"] == home_id))) &
                (df["date"] < match_date)
            ]
            
            if len(h2h_matches) > 0:
                # Calculate H2H statistics
                home_wins = len(h2h_matches[
                    ((h2h_matches["home_team_id"] == home_id) & (h2h_matches["result"] == "H")) |
                    ((h2h_matches["away_team_id"] == home_id) & (h2h_matches["result"] == "A"))
                ])
                
                away_wins = len(h2h_matches[
                    ((h2h_matches["home_team_id"] == away_id) & (h2h_matches["result"] == "H")) |
                    ((h2h_matches["away_team_id"] == away_id) & (h2h_matches["result"] == "A"))
                ])
                
                draws = len(h2h_matches[h2h_matches["result"] == "D"])
                
                # Update features
                df.at[idx, "h2h_home_wins"] = home_wins
                df.at[idx, "h2h_away_wins"] = away_wins
                df.at[idx, "h2h_draws"] = draws
                df.at[idx, "h2h_total_matches"] = len(h2h_matches)
                df.at[idx, "h2h_home_goals_avg"] = h2h_matches["home_goals"].mean()
                df.at[idx, "h2h_away_goals_avg"] = h2h_matches["away_goals"].mean()
                df.at[idx, "h2h_total_goals_avg"] = h2h_matches["total_goals"].mean()
        
        return df
    
    def calculate_league_position_features(self, df: pd.DataFrame, standings_df: pd.DataFrame) -> pd.DataFrame:
        """Add league position and standings-based features."""
        if standings_df.empty:
            return df
        
        df = df.copy()
        
        # Merge standings data
        df = df.merge(
            standings_df[["team_id", "position", "points", "goal_difference", "points_per_game"]],
            left_on="home_team_id", right_on="team_id", how="left", suffixes=("", "_home")
        ).drop("team_id", axis=1)
        
        df = df.merge(
            standings_df[["team_id", "position", "points", "goal_difference", "points_per_game"]],
            left_on="away_team_id", right_on="team_id", how="left", suffixes=("", "_away")
        ).drop("team_id", axis=1)
        
        # Calculate position difference (negative means home team is higher)
        df["position_difference"] = df["position"] - df["position_away"]
        df["points_difference"] = df["points"] - df["points_away"]
        df["goal_diff_difference"] = df["goal_difference"] - df["goal_difference_away"]
        
        # Rename columns for clarity
        df.rename(columns={
            "position": "home_position",
            "points": "home_points",
            "goal_difference": "home_goal_diff",
            "points_per_game": "home_ppg",
            "position_away": "away_position",
            "points_away": "away_points",
            "goal_difference_away": "away_goal_diff",
            "points_per_game_away": "away_ppg"
        }, inplace=True)
        
        return df
    
    def calculate_match_importance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features related to match importance."""
        df = df.copy()
        
        # Days since last match (rest days)
        df["home_rest_days"] = 7  # Default assumption
        df["away_rest_days"] = 7  # Default assumption
        
        # Match importance based on stage and competition
        importance_map = {
            "REGULAR_SEASON": 1.0,
            "GROUP_STAGE": 1.2,
            "ROUND_OF_16": 1.5,
            "QUARTER_FINALS": 1.8,
            "SEMI_FINALS": 2.0,
            "FINAL": 2.5
        }
        
        df["match_importance"] = df["stage"].map(importance_map).fillna(1.0)
        
        # Season progress (0-1, where 1 is end of season)
        df["season_progress"] = df.groupby(["competition_id", "season"])["matchday"].transform(
            lambda x: x / x.max() if x.max() > 0 else 0
        )
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for multi-output prediction."""
        df = df.copy()
        
        # Main prediction targets
        df["home_win"] = (df["result"] == "H").astype(int)
        df["draw"] = (df["result"] == "D").astype(int)
        df["away_win"] = (df["result"] == "A").astype(int)
        
        # Goal prediction targets
        df["total_goals_target"] = df["total_goals"]
        df["home_goals_target"] = df["home_goals"]
        df["away_goals_target"] = df["away_goals"]
        
        # Over/Under targets
        df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)
        df["over_1_5"] = (df["total_goals"] > 1.5).astype(int)
        df["over_3_5"] = (df["total_goals"] > 3.5).astype(int)
        
        # Both teams to score
        df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
        
        return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Engineer all features and create final dataset."""
        print("Loading processed data...")
        matches_df, standings_df, form_df = self.load_processed_data()
        
        if matches_df.empty:
            print("No processed data found. Please run data collection and processing first.")
            return pd.DataFrame()
        
        print("Calculating expected goals proxy...")
        matches_df = self.calculate_expected_goals_proxy(matches_df)
        
        print("Calculating momentum features...")
        matches_df = self.calculate_momentum_features(matches_df)
        
        print("Calculating head-to-head features...")
        matches_df = self.calculate_head_to_head_features(matches_df)
        
        print("Adding league position features...")
        matches_df = self.calculate_league_position_features(matches_df, standings_df)
        
        print("Calculating match importance features...")
        matches_df = self.calculate_match_importance_features(matches_df)
        
        print("Creating target variables...")
        matches_df = self.create_target_variables(matches_df)
        
        # Remove rows with missing critical features
        critical_features = ["home_goals_avg_5", "away_goals_avg_5", "home_position", "away_position"]
        matches_df = matches_df.dropna(subset=critical_features)
        
        print(f"Final dataset shape: {matches_df.shape}")
        
        # Save engineered features
        output_path = os.path.join(Config.DATA_PROCESSED_PATH, "engineered_features.csv")
        matches_df.to_csv(output_path, index=False)
        print(f"Engineered features saved to {output_path}")
        
        return matches_df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model training."""
        base_features = [
            # Team strength and ratings
            "home_position", "away_position", "position_difference",
            "home_points", "away_points", "points_difference",
            "home_goal_diff", "away_goal_diff", "goal_diff_difference",
            "home_ppg", "away_ppg",
            
            # Form and momentum (multiple windows)
            "home_goals_avg_3", "home_goals_avg_5", "home_goals_avg_10",
            "away_goals_avg_3", "away_goals_avg_5", "away_goals_avg_10",
            "home_conceded_avg_3", "home_conceded_avg_5", "home_conceded_avg_10",
            "away_conceded_avg_3", "away_conceded_avg_5", "away_conceded_avg_10",
            
            # Head-to-head
            "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_total_matches",
            "h2h_home_goals_avg", "h2h_away_goals_avg", "h2h_total_goals_avg",
            
            # Match context
            "match_importance", "season_progress", "matchday",
            
            # Expected goals proxy
            "home_xg_proxy", "away_xg_proxy"
        ]
        
        return base_features


if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    features_df = engineer.engineer_all_features()
    
    if not features_df.empty:
        print("\nFeature engineering completed successfully!")
        print(f"Available features: {len(engineer.get_feature_columns())}")
        print(f"Total samples: {len(features_df)}")
    else:
        print("Feature engineering failed. Check data availability.")