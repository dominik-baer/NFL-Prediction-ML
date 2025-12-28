"""
MEGA FEATURE ENGINEERING + AUTOMATED FEATURE SELECTION - ULTRA ROBUST
Alle NaN Issues gefixt!

IMPROVEMENTS:
- Configurable default fillna values
- Better documentation
- Cleaner structure
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
import warnings
warnings.filterwarnings('ignore')

class MegaFeatureEngineer:
    """Erstellt ALLE möglichen Features - ULTRA ROBUST"""
    
    def __init__(self, default_fills=None):
        """
        Initialize Feature Engineer
        
        Args:
            default_fills (dict): Custom default values for filling NaN
                Example: {'elo': 1500, 'temp': 70, 'wind': 5}
        """
        self.df = None
        self.pbp_data = None
        self.feature_groups = {}
        
        # Configurable default values for fillna
        self.default_fills = default_fills or {
            'elo': 1500,              # Average Elo rating
            'temp': 65,               # Comfortable temperature
            'wind': 8,                # Average wind speed
            'week': 9,                # Mid-season
            'rolling_stats': 0,       # For missing rolling stats
            'rest_advantage': 0,      # No rest advantage
            'is_division_game': 0     # Not a division game
        }
        
    def load_data(self, filepath='nfl_training_data.csv'):
        """Lädt Basis-Daten"""
        print(f"📥 Lade Daten aus {filepath}...")
        self.df = pd.read_csv(filepath)
        self.df['gameday'] = pd.to_datetime(self.df['gameday'])
        print(f"✓ {len(self.df)} Spiele geladen")
        print(f"✓ Start Features: {len(self.df.columns)}\n")
        return self.df
    
    def track_features(self, group_name, start_cols):
        """Tracked welche Features zu welcher Gruppe gehören"""
        new_cols = [c for c in self.df.columns if c not in start_cols]
        self.feature_groups[group_name] = new_cols
        return len(new_cols)
    
    # ==================== FEATURE GROUP 1: ELO VARIATIONS ====================
    
    def create_elo_features(self):
        """Alle möglichen Elo-basierten Features"""
        print("🎯 Erstelle Elo Features...")
        start_cols = list(self.df.columns)
        
        avg_elo = self.default_fills['elo']
        
        # Fill NaN FIRST
        self.df['home_elo_before'] = self.df['home_elo_before'].fillna(avg_elo)
        self.df['away_elo_before'] = self.df['away_elo_before'].fillna(avg_elo)
        
        # Basic differences and ratios
        self.df['elo_diff'] = self.df['home_elo_before'] - self.df['away_elo_before']
        self.df['elo_ratio'] = self.df['home_elo_before'] / (self.df['away_elo_before'] + 1)
        self.df['elo_sum'] = self.df['home_elo_before'] + self.df['away_elo_before']
        self.df['elo_product'] = self.df['home_elo_before'] * self.df['away_elo_before'] / 1000
        
        # Distance from average
        self.df['home_elo_momentum'] = self.df['home_elo_before'] - avg_elo
        self.df['away_elo_momentum'] = self.df['away_elo_before'] - avg_elo
        
        # Absolute values
        self.df['elo_spread'] = np.abs(self.df['elo_diff'])
        self.df['elo_gap_squared'] = self.df['elo_diff'] ** 2
        self.df['elo_gap_log'] = np.log1p(np.abs(self.df['elo_diff']))
        
        # Categorical bins - ROBUST version without pd.cut
        self.df['elo_diff_bin'] = 0
        self.df.loc[self.df['elo_diff'] < -100, 'elo_diff_bin'] = 0
        self.df.loc[(self.df['elo_diff'] >= -100) & (self.df['elo_diff'] < -50), 'elo_diff_bin'] = 1
        self.df.loc[(self.df['elo_diff'] >= -50) & (self.df['elo_diff'] < 0), 'elo_diff_bin'] = 2
        self.df.loc[(self.df['elo_diff'] >= 0) & (self.df['elo_diff'] < 50), 'elo_diff_bin'] = 3
        self.df.loc[(self.df['elo_diff'] >= 50) & (self.df['elo_diff'] < 100), 'elo_diff_bin'] = 4
        self.df.loc[self.df['elo_diff'] >= 100, 'elo_diff_bin'] = 5
        
        # Flags
        self.df['home_is_underdog'] = (self.df['elo_diff'] < -50).astype(int)
        self.df['home_is_favorite'] = (self.df['elo_diff'] > 50).astype(int)
        self.df['close_matchup'] = (np.abs(self.df['elo_diff']) < 50).astype(int)
        self.df['blowout_expected'] = (np.abs(self.df['elo_diff']) > 150).astype(int)
        
        # Both teams quality
        self.df['both_teams_good'] = (
            (self.df['home_elo_before'] > 1550) & (self.df['away_elo_before'] > 1550)
        ).astype(int)
        self.df['both_teams_bad'] = (
            (self.df['home_elo_before'] < 1450) & (self.df['away_elo_before'] < 1450)
        ).astype(int)
        
        # Elite team indicators
        self.df['home_elite'] = (self.df['home_elo_before'] > 1600).astype(int)
        self.df['away_elite'] = (self.df['away_elo_before'] > 1600).astype(int)
        self.df['elite_matchup'] = (self.df['home_elite'] & self.df['away_elite']).astype(int)
        
        n = self.track_features('elo', start_cols)
        print(f"  ✓ {n} Elo features")
    
    # ==================== FEATURE GROUP 2: ROLLING STATS VARIATIONS ====================
    
    def create_rolling_stats_features(self):
        """Alle Rolling Stats Variationen"""
        print("📊 Erstelle Rolling Stats Features...")
        start_cols = list(self.df.columns)
        
        for team in ['home', 'away']:
            # Fill NaN FIRST
            for col in [f'points_scored_L3_{team}', f'points_scored_L5_{team}',
                       f'points_allowed_L3_{team}', f'points_allowed_L5_{team}',
                       f'win_rate_L3_{team}', f'win_rate_L5_{team}']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(self.default_fills['rolling_stats'])
            
            # Points ratios
            self.df[f'scoring_efficiency_{team}'] = (
                self.df[f'points_scored_L5_{team}'] / 
                (self.df[f'points_allowed_L5_{team}'] + 1)
            )
            
            # Recent vs longer term
            self.df[f'recent_momentum_{team}'] = (
                self.df[f'win_rate_L3_{team}'] - self.df[f'win_rate_L5_{team}']
            )
            self.df[f'scoring_trend_{team}'] = (
                self.df[f'points_scored_L3_{team}'] - self.df[f'points_scored_L5_{team}']
            )
            self.df[f'defense_trend_{team}'] = (
                self.df[f'points_allowed_L3_{team}'] - self.df[f'points_allowed_L5_{team}']
            )
            
            # Weighted recent
            self.df[f'win_rate_weighted_{team}'] = (
                self.df[f'win_rate_L3_{team}'] * 0.6 + self.df[f'win_rate_L5_{team}'] * 0.4
            )
            
            # Variance/Consistency
            self.df[f'scoring_variance_{team}'] = np.abs(
                self.df[f'points_scored_L3_{team}'] - self.df[f'points_scored_L5_{team}']
            )
            self.df[f'defense_consistency_{team}'] = 1 / (
                np.abs(self.df[f'points_allowed_L3_{team}'] - self.df[f'points_allowed_L5_{team}']) + 1
            )
            
            # Form flags
            self.df[f'is_hot_{team}'] = (self.df[f'win_rate_L3_{team}'] >= 0.67).astype(int)
            self.df[f'is_cold_{team}'] = (self.df[f'win_rate_L3_{team}'] <= 0.33).astype(int)
            self.df[f'high_scoring_{team}'] = (self.df[f'points_scored_L5_{team}'] > 26).astype(int)
            self.df[f'low_scoring_{team}'] = (self.df[f'points_scored_L5_{team}'] < 20).astype(int)
            self.df[f'good_defense_{team}'] = (self.df[f'points_allowed_L5_{team}'] < 20).astype(int)
            self.df[f'bad_defense_{team}'] = (self.df[f'points_allowed_L5_{team}'] > 26).astype(int)
            
            # Combined scores
            self.df[f'net_points_L5_{team}'] = (
                self.df[f'points_scored_L5_{team}'] - self.df[f'points_allowed_L5_{team}']
            )
            self.df[f'total_points_L5_{team}'] = (
                self.df[f'points_scored_L5_{team}'] + self.df[f'points_allowed_L5_{team}']
            )
        
        # Head-to-head comparisons
        self.df['win_rate_diff'] = self.df['win_rate_L5_home'] - self.df['win_rate_L5_away']
        self.df['scoring_diff'] = self.df['points_scored_L5_home'] - self.df['points_scored_L5_away']
        self.df['defense_diff'] = self.df['points_allowed_L5_away'] - self.df['points_allowed_L5_home']
        
        # Matchup analysis
        self.df['home_off_vs_away_def'] = (
            self.df['points_scored_L5_home'] - self.df['points_allowed_L5_away']
        )
        self.df['away_off_vs_home_def'] = (
            self.df['points_scored_L5_away'] - self.df['points_allowed_L5_home']
        )
        self.df['net_matchup_edge'] = (
            self.df['home_off_vs_away_def'] - self.df['away_off_vs_home_def']
        )
        
        # Expected total
        self.df['expected_total'] = (
            self.df['points_scored_L5_home'] + self.df['points_scored_L5_away']
        )
        
        # Form matchups
        self.df['hot_vs_cold'] = (
            (self.df['is_hot_home'] & self.df['is_cold_away']) |
            (self.df['is_cold_home'] & self.df['is_hot_away'])
        ).astype(int)
        
        self.df['both_hot'] = (
            self.df['is_hot_home'] & self.df['is_hot_away']
        ).astype(int)
        
        self.df['potential_shootout'] = (
            self.df['high_scoring_home'] & self.df['high_scoring_away']
        ).astype(int)
        
        self.df['defensive_battle'] = (
            self.df['good_defense_home'] & self.df['good_defense_away']
        ).astype(int)
        
        n = self.track_features('rolling_stats', start_cols)
        print(f"  ✓ {n} Rolling Stats features")
    
    # ==================== FEATURE GROUP 3: INTERACTION FEATURES ====================
    
    def create_interaction_features(self):
        """Alle Interaktionen zwischen Features"""
        print("🔗 Erstelle Interaction Features...")
        start_cols = list(self.df.columns)
        
        # Elo × Form
        self.df['elo_times_form_home'] = (
            self.df['home_elo_before'] * self.df['win_rate_L5_home']
        )
        self.df['elo_times_form_away'] = (
            self.df['away_elo_before'] * self.df['win_rate_L5_away']
        )
        self.df['elo_form_diff'] = (
            self.df['elo_times_form_home'] - self.df['elo_times_form_away']
        )
        
        # Elo × Momentum
        self.df['elo_momentum_home'] = (
            self.df['elo_diff'] * self.df['recent_momentum_home']
        )
        self.df['elo_momentum_away'] = (
            self.df['elo_diff'] * self.df['recent_momentum_away']
        )
        
        # Elo × Scoring
        self.df['elo_scoring_home'] = (
            self.df['home_elo_before'] * self.df['points_scored_L5_home'] / 100
        )
        self.df['elo_scoring_away'] = (
            self.df['away_elo_before'] * self.df['points_scored_L5_away'] / 100
        )
        
        # Form × Matchup
        self.df['form_matchup_score'] = (
            self.df['win_rate_diff'] * self.df['net_matchup_edge']
        )
        
        # Squared terms
        self.df['win_rate_home_squared'] = self.df['win_rate_L5_home'] ** 2
        self.df['win_rate_away_squared'] = self.df['win_rate_L5_away'] ** 2
        
        n = self.track_features('interactions', start_cols)
        print(f"  ✓ {n} Interaction features")
    
    # ==================== FEATURE GROUP 4: SITUATIONAL FEATURES ====================
    
    def create_situational_features(self):
        """Situationale Features"""
        print("🎯 Erstelle Situational Features...")
        start_cols = list(self.df.columns)
        
        if 'week' in self.df.columns:
            # Fill NaN FIRST
            self.df['week'] = self.df['week'].fillna(self.default_fills['week'])
            
            # Season phases
            self.df['early_season'] = (self.df['week'] <= 4).astype(int)
            self.df['mid_season'] = ((self.df['week'] > 4) & (self.df['week'] <= 12)).astype(int)
            self.df['late_season'] = (self.df['week'] > 12).astype(int)
            self.df['playoff_race'] = (self.df['week'] >= 14).astype(int)
            self.df['week_normalized'] = self.df['week'] / 18
            self.df['week_squared'] = (self.df['week'] / 18) ** 2
            
            # Week bins - ROBUST version
            self.df['week_bin'] = 0
            self.df.loc[self.df['week'] <= 4, 'week_bin'] = 0
            self.df.loc[(self.df['week'] > 4) & (self.df['week'] <= 8), 'week_bin'] = 1
            self.df.loc[(self.df['week'] > 8) & (self.df['week'] <= 12), 'week_bin'] = 2
            self.df.loc[self.df['week'] > 12, 'week_bin'] = 3
        
        if 'is_division_game' in self.df.columns:
            # Fill NaN FIRST
            self.df['is_division_game'] = self.df['is_division_game'].fillna(
                self.default_fills['is_division_game']
            )
            self.df['division_factor'] = self.df['is_division_game'] * 2.0
            self.df['division_rivalry'] = self.df['is_division_game'].astype(int)
        
        if 'rest_advantage' in self.df.columns:
            # Fill NaN FIRST
            self.df['rest_advantage'] = self.df['rest_advantage'].fillna(
                self.default_fills['rest_advantage']
            )
            
            # Rest categories
            self.df['home_well_rested'] = (self.df['rest_advantage'] >= 3).astype(int)
            self.df['away_well_rested'] = (self.df['rest_advantage'] <= -3).astype(int)
            self.df['rest_advantage_abs'] = np.abs(self.df['rest_advantage'])
            self.df['rest_advantage_squared'] = self.df['rest_advantage'] ** 2
            
            # Special situations
            self.df['thursday_night'] = (self.df['rest_advantage'].abs() <= 4).astype(int)
            self.df['post_bye'] = (self.df['rest_advantage'].abs() >= 7).astype(int)
            self.df['home_post_bye'] = (self.df['rest_advantage'] >= 7).astype(int)
            self.df['away_post_bye'] = (self.df['rest_advantage'] <= -7).astype(int)
        
        n = self.track_features('situational', start_cols)
        print(f"  ✓ {n} Situational features")
    
    # ==================== FEATURE GROUP 5: WEATHER FEATURES ====================
    
    def create_weather_features(self):
        """Weather Features"""
        print("🌦️ Erstelle Weather Features...")
        start_cols = list(self.df.columns)
        
        if 'temp' in self.df.columns:
            # Fill NaN FIRST
            self.df['temp'] = self.df['temp'].fillna(self.default_fills['temp'])
            
            # Temperature bins - ROBUST version
            self.df['temp_bin'] = 2  # default middle
            self.df.loc[self.df['temp'] < 32, 'temp_bin'] = 0
            self.df.loc[(self.df['temp'] >= 32) & (self.df['temp'] < 50), 'temp_bin'] = 1
            self.df.loc[(self.df['temp'] >= 50) & (self.df['temp'] < 70), 'temp_bin'] = 2
            self.df.loc[(self.df['temp'] >= 70) & (self.df['temp'] < 85), 'temp_bin'] = 3
            self.df.loc[self.df['temp'] >= 85, 'temp_bin'] = 4
            
            # Temperature flags
            self.df['freezing'] = (self.df['temp'] < 32).astype(int)
            self.df['cold'] = (self.df['temp'] < 50).astype(int)
            self.df['hot'] = (self.df['temp'] > 85).astype(int)
            self.df['perfect_temp'] = (
                (self.df['temp'] >= 60) & (self.df['temp'] <= 75)
            ).astype(int)
        
        if 'wind' in self.df.columns:
            # Fill NaN FIRST
            self.df['wind'] = self.df['wind'].fillna(self.default_fills['wind'])
            
            # Wind bins - ROBUST version
            self.df['wind_bin'] = 1  # default calm
            self.df.loc[self.df['wind'] < 5, 'wind_bin'] = 0
            self.df.loc[(self.df['wind'] >= 5) & (self.df['wind'] < 10), 'wind_bin'] = 1
            self.df.loc[(self.df['wind'] >= 10) & (self.df['wind'] < 15), 'wind_bin'] = 2
            self.df.loc[self.df['wind'] >= 15, 'wind_bin'] = 3
            
            # Wind flags
            self.df['windy'] = (self.df['wind'] > 15).astype(int)
            self.df['very_windy'] = (self.df['wind'] > 20).astype(int)
            self.df['calm'] = (self.df['wind'] < 5).astype(int)
        
        if 'temp' in self.df.columns and 'wind' in self.df.columns:
            # Wind chill
            self.df['wind_chill'] = np.where(
                self.df['temp'] < 50,
                self.df['temp'] - (self.df['wind'] * 0.5),
                self.df['temp']
            )
            
            # Combined bad weather
            self.df['bad_weather'] = (
                ((self.df['temp'] < 32) | (self.df['temp'] > 85)) |
                (self.df['wind'] > 15)
            ).astype(int)
            
            self.df['extreme_weather'] = (
                ((self.df['temp'] < 20) | (self.df['temp'] > 95)) |
                (self.df['wind'] > 25)
            ).astype(int)
            
            self.df['perfect_weather'] = (
                (self.df['temp'] >= 60) & (self.df['temp'] <= 75) &
                (self.df['wind'] < 10)
            ).astype(int)
        
        if 'roof' in self.df.columns:
            self.df['outdoor'] = (~self.df['roof'].isin(['dome', 'closed'])).astype(int)
            self.df['dome_game'] = (self.df['roof'] == 'dome').astype(int)
        
        n = self.track_features('weather', start_cols)
        print(f"  ✓ {n} Weather features")
    
    # ==================== FEATURE GROUP 6: POLYNOMIAL FEATURES ====================
    
    def create_polynomial_features(self):
        """Polynomial Transformationen der wichtigsten Features"""
        print("📐 Erstelle Polynomial Features...")
        start_cols = list(self.df.columns)
        
        key_features = [
            'elo_diff', 'win_rate_L5_home', 'win_rate_L5_away',
            'points_scored_L5_home', 'points_scored_L5_away'
        ]
        
        for feat in key_features:
            if feat in self.df.columns:
                self.df[f'{feat}_squared'] = self.df[feat] ** 2
                self.df[f'{feat}_cubed'] = self.df[feat] ** 3
                self.df[f'{feat}_sqrt'] = np.sqrt(np.abs(self.df[feat]))
                self.df[f'{feat}_log'] = np.log1p(np.abs(self.df[feat]))
        
        n = self.track_features('polynomial', start_cols)
        print(f"  ✓ {n} Polynomial features")
    
    # ==================== FEATURE GROUP 7: RATIO FEATURES ====================
    
    def create_ratio_features(self):
        """Ratio Features"""
        print("➗ Erstelle Ratio Features...")
        start_cols = list(self.df.columns)
        
        # Scoring ratios
        self.df['home_scoring_ratio'] = (
            self.df['points_scored_L5_home'] / (self.df['points_allowed_L5_home'] + 1)
        )
        self.df['away_scoring_ratio'] = (
            self.df['points_scored_L5_away'] / (self.df['points_allowed_L5_away'] + 1)
        )
        self.df['scoring_ratio_diff'] = (
            self.df['home_scoring_ratio'] - self.df['away_scoring_ratio']
        )
        
        # Win rate ratios
        self.df['win_rate_ratio'] = (
            self.df['win_rate_L5_home'] / (self.df['win_rate_L5_away'] + 0.01)
        )
        
        # Offense/Defense balance
        self.df['home_balance'] = (
            self.df['points_scored_L5_home'] / (self.df['points_scored_L5_home'] + self.df['points_allowed_L5_home'] + 1)
        )
        self.df['away_balance'] = (
            self.df['points_scored_L5_away'] / (self.df['points_scored_L5_away'] + self.df['points_allowed_L5_away'] + 1)
        )
        
        n = self.track_features('ratios', start_cols)
        print(f"  ✓ {n} Ratio features")
    
    # ==================== MAIN EXECUTION ====================
    
    def create_all_features(self):
        """Erstellt ALLE Features"""
        
        print("\n" + "="*70)
        print("🚀 MEGA FEATURE ENGINEERING")
        print("="*70 + "\n")
        
        initial_features = len(self.df.columns)
        
        self.create_elo_features()
        self.create_rolling_stats_features()
        self.create_interaction_features()
        self.create_situational_features()
        self.create_weather_features()
        self.create_polynomial_features()
        self.create_ratio_features()
        
        final_features = len(self.df.columns)
        new_features = final_features - initial_features
        
        print(f"\n✅ Feature Engineering Complete!")
        print(f"   Features vorher: {initial_features}")
        print(f"   Features nachher: {final_features}")
        print(f"   Neue Features: {new_features}")
        
        print(f"\n📊 Feature Groups:")
        for group, features in self.feature_groups.items():
            print(f"   {group}: {len(features)} features")
        
        return self.df


def automated_feature_selection(df, target_col='home_win', test_season=2025, 
                                k_ranges=None):
    """
    Automatische Feature Selection mit mehreren Methoden
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        test_season: Season to use for testing
        k_ranges: Custom K ranges for testing
            Example: {'rf': range(5, 101, 5), 'mi': range(5, 151, 10)}
    """
    
    print("\n" + "="*70)
    print("🎯 AUTOMATED FEATURE SELECTION")
    print("="*70 + "\n")
    
    # Default K ranges if not provided
    if k_ranges is None:
        k_ranges = {
            'rf': list(range(5, 51, 5)) + list(range(55, 101, 5)) + list(range(110, 201, 10)),
            'mi': list(range(5, 51, 5)) + list(range(55, 101, 5)) + list(range(110, 151, 10))
        }
    
    # Prepare data
    train_df = df[df['season'] < test_season].copy()
    test_df = df[df['season'] >= test_season].copy()
    
    # CRITICAL: Remove rows where target is NaN (games not yet played)
    train_df = train_df[train_df[target_col].notna()].copy()
    test_df = test_df[test_df[target_col].notna()].copy()
    
    print(f"   Removed games without results")
    print(f"   Train games: {len(train_df)}, Test games: {len(test_df)}")
    
    # Get feature columns (exclude target and metadata)
    exclude_cols = [
        target_col, 'season', 'week', 'gameday', 'home_team', 'away_team',
        'home_score', 'away_score', 'total_score', 'score_diff',
        'spread_line', 'total_line', 'over_line', 'under_line'  # VEGAS LINES = LEAKAGE!
    ]
    
    # Get all columns that are NOT excluded
    potential_features = [c for c in train_df.columns if c not in exclude_cols]
    
    # Keep only NUMERIC columns and exclude any potential leakage
    feature_cols = []
    for col in potential_features:
        # Skip non-numeric
        if train_df[col].dtype not in ['int64', 'float64', 'int32', 'float32', 'bool']:
            print(f"   Skipping non-numeric column: {col} (dtype: {train_df[col].dtype})")
            continue
        
        # Skip potential leakage columns
        col_lower = col.lower()
        
        if any(word in col_lower for word in ['result', 'winner']):
            print(f"   🚨 LEAKAGE WARNING: Skipping {col}")
            continue
        
        # Block vegas lines
        if any(word in col_lower for word in ['line', 'over', 'under']):
            print(f"   🚨 LEAKAGE WARNING: Skipping {col}")
            continue
        
        # Block current game scores (but allow rolling averages)
        if col in ['home_score', 'away_score', 'total_score', 'score_diff']:
            print(f"   🚨 LEAKAGE WARNING: Skipping {col}")
            continue
            
        feature_cols.append(col)
    
    print(f"   Kept {len(feature_cols)} numeric features\n")
    
    # Handle NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"📊 Starting Features: {len(feature_cols)}")
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}\n")
    
    results = {}
    
    # ==================== METHOD 1: Random Forest Feature Importance ====================
    
    print("1️⃣ Random Forest Feature Importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get importances
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Test different K values with multiple metrics
    print("   Testing top K features...")
    best_k = 0
    best_auc = 0
    
    for k in k_ranges['rf']:
        if k > len(feature_cols):
            break
        
        top_features = importances.head(k)['feature'].tolist()
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train[top_features], y_train)
        
        acc = rf_temp.score(X_test[top_features], y_test)
        y_pred_proba = rf_temp.predict_proba(X_test[top_features])[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        print(f"   Top {k:3d}: Acc={acc:.4f}, AUC={auc:.4f}, LogLoss={logloss:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_k = k
            best_acc = acc
    
    results['rf_importance'] = {
        'best_k': best_k,
        'accuracy': best_acc,
        'auc': best_auc,
        'features': importances.head(best_k)['feature'].tolist()
    }
    
    print(f"   ✓ Best: Top {best_k} features → Acc={best_acc:.4f}, AUC={best_auc:.4f}\n")
    
    # ==================== METHOD 2: Mutual Information ====================
    
    print("2️⃣ Mutual Information...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_importances = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    best_k = 0
    best_auc = 0
    
    for k in k_ranges['mi']:
        if k > len(feature_cols):
            break
        
        top_features = mi_importances.head(k)['feature'].tolist()
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train[top_features], y_train)
        
        acc = rf_temp.score(X_test[top_features], y_test)
        y_pred_proba = rf_temp.predict_proba(X_test[top_features])[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   Top {k:3d}: Acc={acc:.4f}, AUC={auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_k = k
            best_acc = acc
    
    results['mutual_info'] = {
        'best_k': best_k,
        'accuracy': best_acc,
        'auc': best_auc,
        'features': mi_importances.head(best_k)['feature'].tolist()
    }
    
    print(f"   ✓ Best: Top {best_k} features → Acc={best_acc:.4f}, AUC={best_auc:.4f}\n")
    
    # ==================== METHOD 3: Recursive Feature Elimination ====================
    
    print("3️⃣ Recursive Feature Elimination (may take time)...")
    
    # Start with top 100 from RF to speed up
    top_100 = importances.head(min(100, len(feature_cols)))['feature'].tolist()
    
    rfe = RFECV(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        step=5,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    rfe.fit(X_train[top_100], y_train)
    selected_features = [f for f, s in zip(top_100, rfe.support_) if s]
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train[selected_features], y_train)
    acc = rf_temp.score(X_test[selected_features], y_test)
    y_pred_proba = rf_temp.predict_proba(X_test[selected_features])[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results['rfe'] = {
        'best_k': len(selected_features),
        'accuracy': acc,
        'auc': auc,
        'features': selected_features
    }
    
    print(f"   ✓ Selected {len(selected_features)} features → Acc={acc:.4f}, AUC={auc:.4f}\n")
    
    # ==================== SUMMARY ====================
    
    print("="*70)
    print("🏆 RESULTS SUMMARY")
    print("="*70 + "\n")
    
    for method, result in results.items():
        print(f"{method.upper()}:")
        print(f"  Features: {result['best_k']}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC-ROC:  {result['auc']:.4f}")
        print()
    
    # Find best method by AUC
    best_method = max(results.items(), key=lambda x: x[1]['auc'])
    
    print(f"🎯 BEST METHOD: {best_method[0].upper()}")
    print(f"   Features: {best_method[1]['best_k']}")
    print(f"   Accuracy: {best_method[1]['accuracy']:.4f}")
    print(f"   AUC-ROC:  {best_method[1]['auc']:.4f}")
    
    return results, best_method


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("🏈 MEGA FEATURE ENGINEERING + AUTO SELECTION")
    print("="*70 + "\n")
    
    # Step 1: Create ALL features (with custom defaults if needed)
    custom_fills = {
        'elo': 1500,
        'temp': 65,
        'wind': 8,
        'week': 9,
        'rolling_stats': 0,
        'rest_advantage': 0,
        'is_division_game': 0
    }
    
    engineer = MegaFeatureEngineer(default_fills=custom_fills)
    df = engineer.load_data('nfl_training_data.csv')
    df_mega = engineer.create_all_features()
    
    # Save mega dataset
    df_mega.to_csv('nfl_training_data_MEGA.csv', index=False)
    print(f"\n💾 Saved: nfl_training_data_MEGA.csv")
    
    # Step 2: Automated feature selection
    results, best_method = automated_feature_selection(df_mega)
    
    # Step 3: Create optimized dataset with best features
    best_features = best_method[1]['features']
    
    # Meta cols - only keep what exists
    meta_cols = ['season', 'week', 'gameday', 'home_team', 'away_team', 'home_win']
    # Add score columns if they exist (but these shouldn't be used for training!)
    if 'home_score' in df_mega.columns:
        meta_cols.append('home_score')
    if 'away_score' in df_mega.columns:
        meta_cols.append('away_score')
    
    df_optimized = df_mega[meta_cols + best_features]
    df_optimized.to_csv('nfl_training_data_OPTIMIZED.csv', index=False)
    
    print(f"\n💾 Saved: nfl_training_data_OPTIMIZED.csv")
    print(f"   Features: {len(best_features)}")
    print(f"   Accuracy: {best_method[1]['accuracy']:.4f}")
    print(f"   AUC-ROC:  {best_method[1]['auc']:.4f}")
    
    # Save feature list
    with open('optimal_features.txt', 'w') as f:
        f.write(f"Method: {best_method[0]}\n")
        f.write(f"Accuracy: {best_method[1]['accuracy']:.4f}\n")
        f.write(f"AUC-ROC: {best_method[1]['auc']:.4f}\n")
        f.write(f"Features: {len(best_features)}\n\n")
        for feat in best_features:
            f.write(f"{feat}\n")
    
    print(f"💾 Saved: optimal_features.txt")
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print("\n🎯 Next Steps:")
    print("   1. Use nfl_training_data_OPTIMIZED.csv for training")
    print(f"   2. Accuracy: {best_method[1]['accuracy']:.1%}")
    print(f"   3. AUC-ROC:  {best_method[1]['auc']:.3f}")
    print("   4. Train final model with optimal features\n")


if __name__ == "__main__":
    main()