"""Database module for Supabase integration."""

import os
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()


class SupabaseManager:
    """Manages Supabase database operations for football data."""
    
    def __init__(self):
        # Load environment variables with fallback
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip()
        self.supabase_key = os.getenv("SUPABASE_KEY", "").strip()
        
        if not self.supabase_url or not self.supabase_key:
            print("⚠️  Warning: Supabase credentials not found. Database features will be disabled.")
            self.client = None
            return
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            print("✅ Supabase client initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize Supabase client: {e}")
            self.client = None
        
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        # This would typically be done via Supabase dashboard or migrations
        # For now, we'll assume tables exist or create them via SQL
        pass
    
    def save_matches(self, matches_df: pd.DataFrame) -> bool:
        """Save matches data to Supabase"""
        if not self.client:
            print("⚠️ Supabase client not available. Skipping database save.")
            return False
            
        try:
            # Convert DataFrame to list of dictionaries
            matches_data = matches_df.to_dict('records')
            
            # Clean the data for Supabase
            cleaned_matches = []
            for match in matches_data:
                # Handle NaN values
                cleaned_match = {}
                for key, value in match.items():
                    if pd.isna(value):
                        cleaned_match[key] = None
                    elif isinstance(value, (int, float)) and pd.isna(value):
                        cleaned_match[key] = None
                    else:
                        cleaned_match[key] = value
                
                # Convert team IDs to None since they expect UUID format
                # We'll store team info in stats instead
                if 'home_team_id' in cleaned_match:
                    cleaned_match['home_team_id'] = None
                if 'away_team_id' in cleaned_match:
                    cleaned_match['away_team_id'] = None
                
                # Ensure required fields are present
                if 'api_id' in cleaned_match and 'status' in cleaned_match:
                    cleaned_matches.append(cleaned_match)
            
            if not cleaned_matches:
                print("⚠️ No valid matches to save")
                return False
            
            # Insert data in batches
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(cleaned_matches), batch_size):
                batch = cleaned_matches[i:i + batch_size]
                try:
                    result = self.client.table('matches').upsert(batch).execute()
                    total_inserted += len(result.data)
                    print(f"✅ Inserted batch {i//batch_size + 1}: {len(result.data)} matches")
                except Exception as batch_error:
                    print(f"❌ Error inserting batch {i//batch_size + 1}: {batch_error}")
                    continue
            
            print(f"✅ Successfully saved {total_inserted} matches to Supabase")
            return total_inserted > 0
            
        except Exception as e:
            print(f"❌ Error saving matches to Supabase: {e}")
            return False
    
    def get_matches(self, limit: int = 100, competition: str = None) -> List[Dict]:
        """Get matches from Supabase"""
        if not self.client:
            return []
            
        try:
            query = self.client.table('matches').select('*')
            # Note: competition column doesn't exist in current schema
            # Filter by stats->competition_key if needed
            
            result = query.limit(limit).execute()
            print(f"✅ Retrieved {len(result.data)} matches from Supabase")
            return result.data
        except Exception as e:
            print(f"❌ Error retrieving matches from Supabase: {e}")
            return []
    
    def save_predictions(self, predictions: List[Dict[str, Any]]) -> bool:
        """Save prediction results to Supabase."""
        if not self.client:
            print("⚠️  Supabase not available, skipping prediction save")
            return False
            
        try:
            # Add metadata
            for pred in predictions:
                pred['created_at'] = datetime.utcnow().isoformat()
            
            result = self.client.table('predictions').insert(predictions).execute()
            
            print(f"✅ Saved {len(predictions)} predictions to Supabase")
            return True
            
        except Exception as e:
            print(f"❌ Error saving predictions to Supabase: {e}")
            return False
    
    def get_predictions(self, match_id: Optional[int] = None, 
                       limit: int = 100) -> pd.DataFrame:
        """Retrieve predictions from Supabase."""
        if not self.client:
            return pd.DataFrame()
            
        try:
            query = self.client.table('predictions').select('*')
            
            if match_id:
                query = query.eq('match_id', match_id)
            
            result = query.limit(limit).order('created_at', desc=True).execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                print(f"✅ Retrieved {len(df)} predictions from Supabase")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving predictions from Supabase: {e}")
            return pd.DataFrame()
    
    def save_model_performance(self, performance_data: Dict[str, Any]) -> bool:
        """Save model performance metrics to Supabase."""
        if not self.client:
            return False
            
        try:
            performance_data['created_at'] = datetime.utcnow().isoformat()
            
            result = self.client.table('model_performance').insert(performance_data).execute()
            
            print("✅ Saved model performance to Supabase")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model performance to Supabase: {e}")
            return False
    
    def get_latest_model_performance(self) -> Dict[str, Any]:
        """Get the latest model performance metrics."""
        if not self.client:
            return {}
            
        try:
            result = self.client.table('model_performance')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if result.data:
                return result.data[0]
            else:
                return {}
                
        except Exception as e:
            print(f"❌ Error retrieving model performance from Supabase: {e}")
            return {}
    
    def save_team_stats(self, team_stats: List[Dict[str, Any]]) -> bool:
        """Save team statistics to Supabase."""
        if not self.client:
            return False
            
        try:
            for stats in team_stats:
                stats['updated_at'] = datetime.utcnow().isoformat()
            
            result = self.client.table('team_stats').upsert(team_stats).execute()
            
            print(f"✅ Saved {len(team_stats)} team stats to Supabase")
            return True
            
        except Exception as e:
            print(f"❌ Error saving team stats to Supabase: {e}")
            return False
    
    def get_team_stats(self, team_id: Optional[int] = None) -> pd.DataFrame:
        """Retrieve team statistics from Supabase."""
        if not self.client:
            return pd.DataFrame()
            
        try:
            query = self.client.table('team_stats').select('*')
            
            if team_id:
                query = query.eq('team_id', team_id)
            
            result = query.execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving team stats from Supabase: {e}")
            return pd.DataFrame()
    
    def health_check(self) -> bool:
        """Check if Supabase connection is working."""
        if not self.client:
            print("❌ Supabase client not initialized")
            return False
            
        try:
            # Simple query to test connection
            result = self.client.table('matches').select('id').limit(1).execute()
            print("✅ Supabase connection healthy")
            return True
            
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            return False


# Global instance
db_manager = SupabaseManager()