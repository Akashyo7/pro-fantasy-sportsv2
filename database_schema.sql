-- Supabase Database Schema for Fantasy Sports Predictions

-- Matches table
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    match_id INTEGER UNIQUE NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    matchday INTEGER,
    stage TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    home_goals_ht INTEGER,
    away_goals_ht INTEGER,
    status TEXT NOT NULL,
    competition TEXT NOT NULL,
    competition_key TEXT NOT NULL,
    competition_name TEXT,
    season TEXT,
    result TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL,
    home_win_prob DECIMAL(5,4),
    draw_prob DECIMAL(5,4),
    away_win_prob DECIMAL(5,4),
    predicted_home_goals DECIMAL(3,2),
    predicted_away_goals DECIMAL(3,2),
    confidence_score DECIMAL(5,4),
    model_version TEXT,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- Team statistics table
CREATE TABLE IF NOT EXISTS team_stats (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    competition TEXT NOT NULL,
    season TEXT,
    matches_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    goals_for INTEGER DEFAULT 0,
    goals_against INTEGER DEFAULT 0,
    goal_difference INTEGER DEFAULT 0,
    points INTEGER DEFAULT 0,
    home_wins INTEGER DEFAULT 0,
    home_draws INTEGER DEFAULT 0,
    home_losses INTEGER DEFAULT 0,
    away_wins INTEGER DEFAULT 0,
    away_draws INTEGER DEFAULT 0,
    away_losses INTEGER DEFAULT 0,
    form_last_5 TEXT,
    avg_goals_scored DECIMAL(3,2) DEFAULT 0,
    avg_goals_conceded DECIMAL(3,2) DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(team_id, competition, season)
);

-- Model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    accuracy DECIMAL(5,4),
    precision_home DECIMAL(5,4),
    precision_draw DECIMAL(5,4),
    precision_away DECIMAL(5,4),
    recall_home DECIMAL(5,4),
    recall_draw DECIMAL(5,4),
    recall_away DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    log_loss DECIMAL(8,6),
    training_samples INTEGER,
    test_samples INTEGER,
    training_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_competition ON matches(competition);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_team_stats_team_id ON team_stats(team_id);
CREATE INDEX IF NOT EXISTS idx_team_stats_competition ON team_stats(competition);

-- Enable Row Level Security (RLS)
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_performance ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (adjust as needed for production)
CREATE POLICY "Allow public read access on matches" ON matches FOR SELECT USING (true);
CREATE POLICY "Allow public insert access on matches" ON matches FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access on matches" ON matches FOR UPDATE USING (true);

CREATE POLICY "Allow public read access on predictions" ON predictions FOR SELECT USING (true);
CREATE POLICY "Allow public insert access on predictions" ON predictions FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read access on team_stats" ON team_stats FOR SELECT USING (true);
CREATE POLICY "Allow public insert access on team_stats" ON team_stats FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access on team_stats" ON team_stats FOR UPDATE USING (true);

CREATE POLICY "Allow public read access on model_performance" ON model_performance FOR SELECT USING (true);
CREATE POLICY "Allow public insert access on model_performance" ON model_performance FOR INSERT WITH CHECK (true);