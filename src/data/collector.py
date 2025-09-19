"""Football data collection module with Supabase integration."""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import os
from src.utils.config import Config
from src.data.database import db_manager


class FootballDataCollector:
    """Collects football data from football-data.org API with Supabase storage."""
    
    def __init__(self):
        """Initialize the Football Data Collector"""
        self.api_key = Config.FOOTBALL_DATA_API_KEY
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": self.api_key}
        
        # Initialize database manager
        try:
            self.db_manager = db_manager
        except Exception as e:
            print(f"⚠️  Database manager not available: {e}")
            self.db_manager = None
        
        # Competition mappings
        self.competitions = {
            "premier_league": {"id": 2021, "name": "Premier League"},
            "champions_league": {"id": 2001, "name": "UEFA Champions League"},
            "bundesliga": {"id": 2002, "name": "Bundesliga"},
            "serie_a": {"id": 2019, "name": "Serie A"},
            "ligue_1": {"id": 2015, "name": "Ligue 1"},
            "la_liga": {"id": 2014, "name": "Primera División"},
            "eredivisie": {"id": 2003, "name": "Eredivisie"}
        }
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            # Handle rate limiting
            if response.status_code == 429:
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, headers=self.headers, params=params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}
    
    def get_competition_matches(self, competition_key: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get matches for a specific competition with Supabase integration.
        
        Args:
            competition_key: Key from COMPETITIONS dict (e.g., 'premier_league')
            days_back: Number of days to look back for matches
            
        Returns:
            DataFrame with match data
        """
        if competition_key not in self.competitions:
            available = list(self.competitions.keys())
            print(f"❌ Competition '{competition_key}' not found. Available: {available}")
            return pd.DataFrame()
        
        competition_id = self.competitions[competition_key]["id"]
        competition_name = self.competitions[competition_key]["name"]
        
        print(f"🔄 Collecting {competition_name} matches...")
        
        # First, try to get recent data from Supabase
        try:
            existing_matches = db_manager.get_matches(competition=competition_key, limit=100)
            if not existing_matches.empty:
                print(f"📊 Found {len(existing_matches)} existing matches in database")
        except Exception as e:
            print(f"⚠️  Could not retrieve existing matches: {e}")
            existing_matches = pd.DataFrame()
        
        # Check for existing matches to avoid duplicates
        existing_api_ids = set()
        if self.db_manager:
            try:
                existing_matches = self.db_manager.get_matches(limit=1000)
                if existing_matches:
                    existing_api_ids = {match['api_id'] for match in existing_matches}
                    print(f"📊 Found {len(existing_api_ids)} existing matches in database")
                else:
                    print("📊 No existing matches found in database")
            except Exception as e:
                print(f"⚠️  Could not retrieve existing matches: {e}")
                existing_api_ids = set()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/competitions/{competition_id}/matches"
        params = {
            "dateFrom": date_from,
            "dateTo": date_to,
            "status": "FINISHED"  # Only get completed matches
        }
        
        try:
            print(f"🌐 Fetching matches from {date_from} to {date_to}...")
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get("matches", [])
                
                if matches:
                    # Process matches data
                    processed_matches = []
                    for match in matches:
                        processed_match = self._process_match_data(match, competition_key)
                        if processed_match and processed_match['api_id'] not in existing_api_ids:
                            processed_matches.append(processed_match)
                    
                    if processed_matches:
                        df = pd.DataFrame(processed_matches)
                        print(f"✅ Processed {len(processed_matches)} new matches (filtered {len(matches) - len(processed_matches)} duplicates)")
                        
                        # Save to Supabase if available
                        if self.db_manager:
                            try:
                                success = self.db_manager.save_matches(df)
                                if success:
                                    print(f"✅ Saved {len(df)} matches to Supabase")
                                else:
                                    print("⚠️  Could not save to Supabase")
                            except Exception as e:
                                print(f"⚠️  Could not save to Supabase: {e}")
                        else:
                            print("⚠️  Database manager not available")
                        
                        # Save backup locally
                        self._save_matches_locally(df, competition_key)
                        
                        return df
                    else:
                        print("⚠️  No new matches to process")
                        return pd.DataFrame()
                else:
                    print("⚠️  No matches found in API response")
                    return pd.DataFrame()
            else:
                print(f"❌ API request failed: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error fetching matches: {e}")
            return pd.DataFrame()
    
    def get_team_matches(self, team_id: int, limit: int = 50) -> pd.DataFrame:
        """Get recent matches for a specific team."""
        endpoint = f"teams/{team_id}/matches"
        params = {"limit": limit}
        
        data = self._make_request(endpoint, params)
        
        if not data or "matches" not in data:
            return pd.DataFrame()
        
        matches = []
        for match in data["matches"]:
            # Determine if team was home or away
            is_home = match["homeTeam"]["id"] == team_id
            
            match_data = {
                "id": match["id"],
                "date": match["utcDate"],
                "team_id": team_id,
                "is_home": is_home,
                "opponent_id": match["awayTeam"]["id"] if is_home else match["homeTeam"]["id"],
                "opponent_name": match["awayTeam"]["name"] if is_home else match["homeTeam"]["name"],
                "goals_for": match["score"]["fullTime"]["home"] if is_home else match["score"]["fullTime"]["away"],
                "goals_against": match["score"]["fullTime"]["away"] if is_home else match["score"]["fullTime"]["home"],
                "result": self._get_result(match, team_id),
                "competition": match["competition"]["name"],
                "status": match["status"]
            }
            matches.append(match_data)
        
        return pd.DataFrame(matches)
    
    def _get_result(self, match: Dict, team_id: int) -> str:
        """Determine match result for a specific team."""
        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]
        
        if home_goals is None or away_goals is None:
            return "N/A"
        
        is_home = match["homeTeam"]["id"] == team_id
        
        if home_goals > away_goals:
            return "W" if is_home else "L"
        elif home_goals < away_goals:
            return "L" if is_home else "W"
        else:
            return "D"
    
    def _process_match_data(self, match: Dict, competition_key: str) -> Optional[Dict]:
        """Process raw match data from API into Supabase schema format."""
        try:
            # Only process finished matches with valid scores
            if match["status"] != "FINISHED":
                return None
            
            home_goals = match["score"]["fullTime"]["home"]
            away_goals = match["score"]["fullTime"]["away"]
            
            if home_goals is None or away_goals is None:
                return None
            
            # Map to Supabase schema columns
            processed_match = {
                "api_id": match["id"],  # Use api_id instead of match_id
                "match_date": match["utcDate"],  # Use match_date instead of date
                "home_team_id": match["homeTeam"]["id"],
                "away_team_id": match["awayTeam"]["id"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "status": match["status"],
                "stats": {
                    "matchday": match.get("matchday"),
                    "stage": match.get("stage", "REGULAR_SEASON"),
                    "home_team_name": match["homeTeam"]["name"],
                    "away_team_name": match["awayTeam"]["name"],
                    "home_goals_ht": match["score"]["halfTime"]["home"],
                    "away_goals_ht": match["score"]["halfTime"]["away"],
                    "competition_key": competition_key,
                    "competition_name": self.competitions[competition_key]["name"],
                    "season": match.get("season", {}).get("startYear"),
                    "result": "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D"),
                    "venue": match.get("venue"),
                    "referee": match.get("referees", [{}])[0].get("name") if match.get("referees") else None,
                    "attendance": match.get("attendance")
                },
                "conditions": {}  # For weather, pitch conditions, etc.
            }
            
            return processed_match
            
        except Exception as e:
            print(f"⚠️  Error processing match {match.get('id', 'unknown')}: {e}")
            return None
    
    def _save_matches_locally(self, df: pd.DataFrame, competition_key: str):
        """Save matches data locally as backup."""
        try:
            os.makedirs(Config.DATA_RAW_PATH, exist_ok=True)
            filename = f"{competition_key}_matches_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(Config.DATA_RAW_PATH, filename)
            df.to_csv(filepath, index=False)
            print(f"💾 Backup saved locally: {filepath}")
        except Exception as e:
            print(f"⚠️  Could not save local backup: {e}")
    
    def get_upcoming_matches(self, competition_id: int, days_ahead: int = 7) -> pd.DataFrame:
        """Get upcoming matches for a competition."""
        endpoint = f"competitions/{competition_id}/matches"
        
        # Calculate date range
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)
        
        params = {
            "dateFrom": today.strftime("%Y-%m-%d"),
            "dateTo": end_date.strftime("%Y-%m-%d")
        }
        
        data = self._make_request(endpoint, params)
        
        if not data or "matches" not in data:
            return pd.DataFrame()
        
        matches = []
        for match in data["matches"]:
            if match["status"] in ["SCHEDULED", "TIMED"]:
                match_data = {
                    "id": match["id"],
                    "date": match["utcDate"],
                    "matchday": match.get("matchday"),
                    "home_team": match["homeTeam"]["name"],
                    "away_team": match["awayTeam"]["name"],
                    "home_team_id": match["homeTeam"]["id"],
                    "away_team_id": match["awayTeam"]["id"],
                    "competition": match["competition"]["name"],
                    "competition_id": competition_id
                }
                matches.append(match_data)
        
        return pd.DataFrame(matches)
    
    def collect_historical_data(self, seasons: List[int] = None) -> pd.DataFrame:
        """Collect historical data for all supported competitions."""
        if seasons is None:
            current_year = datetime.now().year
            seasons = [current_year - i for i in range(Config.HISTORICAL_SEASONS)]
        
        all_matches = []
        
        for comp_key, comp_info in self.competitions.items():
            print(f"Collecting data for {comp_info['name']}...")
            
            for season in seasons:
                print(f"  Season {season}...")
                matches_df = self.get_competition_matches(comp_info["id"], season)
                
                if not matches_df.empty:
                    matches_df["competition_name"] = comp_info["name"]
                    all_matches.append(matches_df)
                
                # Rate limiting
                time.sleep(1)
        
        if all_matches:
            return pd.concat(all_matches, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data to CSV file."""
        os.makedirs(Config.DATA_RAW_PATH, exist_ok=True)
        filepath = os.path.join(Config.DATA_RAW_PATH, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    collector = FootballDataCollector()
    
    # Collect historical data
    historical_data = collector.collect_historical_data()
    if not historical_data.empty:
        collector.save_data(historical_data, "historical_matches.csv")
    
    # Collect upcoming matches
    for comp_key, comp_info in collector.competitions.items():
        upcoming = collector.get_upcoming_matches(comp_info["id"])
        if not upcoming.empty:
            collector.save_data(upcoming, f"upcoming_{comp_key}.csv")