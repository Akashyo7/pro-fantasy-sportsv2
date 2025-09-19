"""Data processing module for cleaning and preparing football match data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

from src.utils.config import Config


class MatchDataProcessor:
    """Processes raw football match data for ML model training."""
    
    def __init__(self):
        self.config = Config()
        
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """Load raw data from CSV file."""
        filepath = os.path.join(Config.DATA_RAW_PATH, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def clean_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize match data."""
        if df.empty:
            return df
        
        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Remove matches without results (future matches or cancelled)
        df = df.dropna(subset=["home_goals", "away_goals"])
        
        # Convert goals to integers
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)
        
        # Create result columns
        df["total_goals"] = df["home_goals"] + df["away_goals"]
        df["goal_difference"] = df["home_goals"] - df["away_goals"]
        
        # Create match result (from home team perspective)
        df["result"] = df.apply(self._get_match_result, axis=1)
        
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        
        return df
    
    def _get_match_result(self, row) -> str:
        """Get match result from home team perspective."""
        if row["home_goals"] > row["away_goals"]:
            return "H"  # Home win
        elif row["home_goals"] < row["away_goals"]:
            return "A"  # Away win
        else:
            return "D"  # Draw
    
    def calculate_team_form(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate team form over last N matches."""
        team_form = {}
        
        # Get unique teams
        home_teams = set(df["home_team_id"].unique())
        away_teams = set(df["away_team_id"].unique())
        all_teams = home_teams.union(away_teams)
        
        for team_id in all_teams:
            # Get all matches for this team
            team_matches = self._get_team_matches_history(df, team_id)
            
            if len(team_matches) >= window:
                # Calculate rolling form metrics
                team_matches["points"] = team_matches["result"].map({"W": 3, "D": 1, "L": 0})
                team_matches["form_points"] = team_matches["points"].rolling(window=window, min_periods=1).sum()
                team_matches["form_goals_for"] = team_matches["goals_for"].rolling(window=window, min_periods=1).mean()
                team_matches["form_goals_against"] = team_matches["goals_against"].rolling(window=window, min_periods=1).mean()
                team_matches["form_goal_diff"] = team_matches["form_goals_for"] - team_matches["form_goals_against"]
                
                team_form[team_id] = team_matches
        
        return team_form
    
    def _get_team_matches_history(self, df: pd.DataFrame, team_id: int) -> pd.DataFrame:
        """Get match history for a specific team."""
        # Home matches
        home_matches = df[df["home_team_id"] == team_id].copy()
        home_matches["is_home"] = True
        home_matches["goals_for"] = home_matches["home_goals"]
        home_matches["goals_against"] = home_matches["away_goals"]
        home_matches["opponent_id"] = home_matches["away_team_id"]
        home_matches["result"] = home_matches["result"].map({"H": "W", "A": "L", "D": "D"})
        
        # Away matches
        away_matches = df[df["away_team_id"] == team_id].copy()
        away_matches["is_home"] = False
        away_matches["goals_for"] = away_matches["away_goals"]
        away_matches["goals_against"] = away_matches["home_goals"]
        away_matches["opponent_id"] = away_matches["home_team_id"]
        away_matches["result"] = away_matches["result"].map({"H": "L", "A": "W", "D": "D"})
        
        # Combine and sort by date
        all_matches = pd.concat([home_matches, away_matches])
        all_matches = all_matches.sort_values("date").reset_index(drop=True)
        
        return all_matches[["date", "is_home", "goals_for", "goals_against", "opponent_id", "result"]]
    
    def calculate_head_to_head(self, df: pd.DataFrame) -> Dict:
        """Calculate head-to-head statistics between teams."""
        h2h_stats = {}
        
        for _, match in df.iterrows():
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            
            # Create unique pair key (always smaller ID first)
            pair_key = tuple(sorted([home_id, away_id]))
            
            if pair_key not in h2h_stats:
                h2h_stats[pair_key] = {
                    "matches": 0,
                    "team1_wins": 0,  # wins for smaller ID team
                    "team2_wins": 0,  # wins for larger ID team
                    "draws": 0,
                    "team1_goals": 0,
                    "team2_goals": 0
                }
            
            stats = h2h_stats[pair_key]
            stats["matches"] += 1
            
            # Determine which team is team1 (smaller ID) and team2 (larger ID)
            if home_id == min(pair_key):
                # Home team is team1
                stats["team1_goals"] += match["home_goals"]
                stats["team2_goals"] += match["away_goals"]
                
                if match["result"] == "H":
                    stats["team1_wins"] += 1
                elif match["result"] == "A":
                    stats["team2_wins"] += 1
                else:
                    stats["draws"] += 1
            else:
                # Away team is team1
                stats["team1_goals"] += match["away_goals"]
                stats["team2_goals"] += match["home_goals"]
                
                if match["result"] == "A":
                    stats["team1_wins"] += 1
                elif match["result"] == "H":
                    stats["team2_wins"] += 1
                else:
                    stats["draws"] += 1
        
        return h2h_stats
    
    def calculate_league_standings(self, df: pd.DataFrame, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """Calculate league standings as of a specific date."""
        if as_of_date:
            df_filtered = df[df["date"] <= as_of_date]
        else:
            df_filtered = df
        
        standings = {}
        
        for _, match in df_filtered.iterrows():
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            home_goals = match["home_goals"]
            away_goals = match["away_goals"]
            
            # Initialize team stats if not exists
            for team_id in [home_id, away_id]:
                if team_id not in standings:
                    standings[team_id] = {
                        "matches": 0, "wins": 0, "draws": 0, "losses": 0,
                        "goals_for": 0, "goals_against": 0, "points": 0
                    }
            
            # Update home team stats
            standings[home_id]["matches"] += 1
            standings[home_id]["goals_for"] += home_goals
            standings[home_id]["goals_against"] += away_goals
            
            # Update away team stats
            standings[away_id]["matches"] += 1
            standings[away_id]["goals_for"] += away_goals
            standings[away_id]["goals_against"] += home_goals
            
            # Update results and points
            if home_goals > away_goals:  # Home win
                standings[home_id]["wins"] += 1
                standings[home_id]["points"] += 3
                standings[away_id]["losses"] += 1
            elif home_goals < away_goals:  # Away win
                standings[away_id]["wins"] += 1
                standings[away_id]["points"] += 3
                standings[home_id]["losses"] += 1
            else:  # Draw
                standings[home_id]["draws"] += 1
                standings[home_id]["points"] += 1
                standings[away_id]["draws"] += 1
                standings[away_id]["points"] += 1
        
        # Convert to DataFrame and calculate additional metrics
        standings_df = pd.DataFrame.from_dict(standings, orient="index")
        standings_df["goal_difference"] = standings_df["goals_for"] - standings_df["goals_against"]
        standings_df["points_per_game"] = standings_df["points"] / standings_df["matches"]
        standings_df["goals_per_game"] = standings_df["goals_for"] / standings_df["matches"]
        standings_df["goals_conceded_per_game"] = standings_df["goals_against"] / standings_df["matches"]
        
        # Sort by points, then goal difference, then goals for
        standings_df = standings_df.sort_values(
            ["points", "goal_difference", "goals_for"], 
            ascending=[False, False, False]
        ).reset_index()
        standings_df.rename(columns={"index": "team_id"}, inplace=True)
        standings_df["position"] = range(1, len(standings_df) + 1)
        
        return standings_df
    
    def process_and_save(self, raw_filename: str, processed_filename: str):
        """Process raw data and save cleaned version."""
        print(f"Processing {raw_filename}...")
        
        # Load and clean data
        raw_df = self.load_raw_data(raw_filename)
        if raw_df.empty:
            print("No data to process")
            return
        
        cleaned_df = self.clean_match_data(raw_df)
        
        # Calculate additional metrics
        print("Calculating team form...")
        team_form = self.calculate_team_form(cleaned_df)
        
        print("Calculating head-to-head statistics...")
        h2h_stats = self.calculate_head_to_head(cleaned_df)
        
        print("Calculating league standings...")
        standings = self.calculate_league_standings(cleaned_df)
        
        # Save processed data
        os.makedirs(Config.DATA_PROCESSED_PATH, exist_ok=True)
        
        # Save main processed data
        processed_path = os.path.join(Config.DATA_PROCESSED_PATH, processed_filename)
        cleaned_df.to_csv(processed_path, index=False)
        
        # Save additional data
        standings.to_csv(os.path.join(Config.DATA_PROCESSED_PATH, "league_standings.csv"), index=False)
        
        # Save team form data
        if team_form:
            form_data = []
            for team_id, form_df in team_form.items():
                form_df["team_id"] = team_id
                form_data.append(form_df)
            
            if form_data:
                all_form_df = pd.concat(form_data, ignore_index=True)
                all_form_df.to_csv(os.path.join(Config.DATA_PROCESSED_PATH, "team_form.csv"), index=False)
        
        print(f"Processed data saved to {Config.DATA_PROCESSED_PATH}")
        print(f"Total matches processed: {len(cleaned_df)}")


if __name__ == "__main__":
    processor = MatchDataProcessor()
    processor.process_and_save("historical_matches.csv", "processed_matches.csv")