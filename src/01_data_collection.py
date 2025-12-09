"""
NFL Prediction Project - Data Collection (nflverse Edition)
Lädt Daten direkt von nflverse - bessere Qualität, mehr Features!
"""

import pandas as pd
import numpy as np
from datetime import datetime

class NFLDataCollector:
    """Sammelt NFL-Daten von nflverse"""
    
    def __init__(self, start_year=2015, end_year=None):
        self.start_year = start_year
        # Automatisch aktuelles Jahr verwenden wenn nicht angegeben
        self.end_year = end_year if end_year is not None else datetime.now().year
        self.games_df = None
        self.team_stats = None
        
    def load_nflverse_data(self):
        """Lädt games.csv direkt von nflverse GitHub"""
        
        print(f"🏈 Lade nflverse Daten von {self.start_year} bis {self.end_year}...")
        
        # Direct URL to nflverse games data
        url = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
        
        try:
            self.games_df = pd.read_csv(url, low_memory=False)
            print(f"✓ {len(self.games_df)} Spiele aus nflverse geladen")
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            print("Verwende Fallback-Methode...")
            # Fallback to alternative URL
            url_alt = "http://www.habitatring.com/games.csv"
            self.games_df = pd.read_csv(url_alt, low_memory=False)
            print(f"✓ {len(self.games_df)} Spiele geladen (Fallback)")
        
        # Filter für gewünschte Jahre
        self.games_df = self.games_df[
            (self.games_df['season'] >= self.start_year) & 
            (self.games_df['season'] <= self.end_year)
        ].copy()
        
        # Nur Regular Season + Playoffs
        if 'game_type' in self.games_df.columns:
            self.games_df = self.games_df[
                self.games_df['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])
            ]
        
        # WICHTIG: Behalte ALLE Spiele (auch kommende ohne Scores!)
        # Für kommende Spiele: result, home_score, away_score sind NaN
        
        # Gameday als datetime
        self.games_df['gameday'] = pd.to_datetime(self.games_df['gameday'])
        
        # Sortieren
        self.games_df = self.games_df.sort_values('gameday').reset_index(drop=True)
        
        print(f"✓ {len(self.games_df)} Spiele nach Filterung ({self.start_year}-{self.end_year})")
        print(f"   Spalten verfügbar: {len(self.games_df.columns)}")
        
        return self.games_df
    
    def create_basic_features(self):
        """Erstellt grundlegende Features"""
        
        print("\n📊 Erstelle Basic Features...")
        
        # Home Win (Target) - nur für gespielte Games
        self.games_df['home_win'] = np.where(
            self.games_df['home_score'].notna(),
            (self.games_df['home_score'] > self.games_df['away_score']).astype(int),
            np.nan
        )
        
        # Point Differential - nur für gespielte Games
        self.games_df['point_diff'] = np.where(
            self.games_df['home_score'].notna(),
            self.games_df['home_score'] - self.games_df['away_score'],
            np.nan
        )
        
        # Total Points - nur für gespielte Games
        self.games_df['total_points'] = np.where(
            self.games_df['home_score'].notna(),
            self.games_df['home_score'] + self.games_df['away_score'],
            np.nan
        )
        
        # Rest Days (falls vorhanden)
        if 'home_rest' in self.games_df.columns and 'away_rest' in self.games_df.columns:
            self.games_df['rest_advantage'] = self.games_df['home_rest'] - self.games_df['away_rest']
        
        # Division Game (falls vorhanden)
        if 'div_game' in self.games_df.columns:
            self.games_df['is_division_game'] = self.games_df['div_game'].fillna(0).astype(int)
        
        print("✓ Basic Features erstellt")
        
    def calculate_team_rolling_stats(self, windows=[3, 5]):
        """Berechnet Rolling Statistics für jedes Team"""
        
        print(f"\n📈 Berechne Rolling Stats (Windows: {windows})...")
        
        # Team-level stats erstellen
        home_games = self.games_df[['season', 'week', 'gameday', 'home_team', 
                                     'home_score', 'away_score']].copy()
        home_games.columns = ['season', 'week', 'gameday', 'team', 
                              'points_scored', 'points_allowed']
        home_games['is_home'] = 1
        home_games['won'] = (home_games['points_scored'] > home_games['points_allowed']).astype(int)
        
        away_games = self.games_df[['season', 'week', 'gameday', 'away_team', 
                                     'away_score', 'home_score']].copy()
        away_games.columns = ['season', 'week', 'gameday', 'team', 
                              'points_scored', 'points_allowed']
        away_games['is_home'] = 0
        away_games['won'] = (away_games['points_scored'] > away_games['points_allowed']).astype(int)
        
        # Kombinieren
        self.team_stats = pd.concat([home_games, away_games], ignore_index=True)
        self.team_stats = self.team_stats.sort_values(['team', 'gameday']).reset_index(drop=True)
        
        # Rolling Features
        for window in windows:
            # Points Scored
            self.team_stats[f'points_scored_L{window}'] = self.team_stats.groupby('team')['points_scored'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Points Allowed
            self.team_stats[f'points_allowed_L{window}'] = self.team_stats.groupby('team')['points_allowed'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Win Rate
            self.team_stats[f'win_rate_L{window}'] = self.team_stats.groupby('team')['won'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Point Differential
            self.team_stats[f'point_diff_L{window}'] = (
                self.team_stats[f'points_scored_L{window}'] - 
                self.team_stats[f'points_allowed_L{window}']
            )
        
        print(f"✓ Rolling Stats für {self.team_stats['team'].nunique()} Teams erstellt")
        
        return self.team_stats
    
    def calculate_elo_ratings(self, k_factor=20, home_advantage=55, initial_elo=1500):
        """Berechnet Elo Ratings"""
        
        print(f"\n🎯 Berechne Elo Ratings (K={k_factor}, HA={home_advantage})...")
        
        # Initialize Elo
        elo_ratings = {}
        teams = set(self.games_df['home_team'].unique()) | set(self.games_df['away_team'].unique())
        
        for team in teams:
            elo_ratings[team] = initial_elo
        
        # Track Elo before each game
        home_elo_list = []
        away_elo_list = []
        
        for _, game in self.games_df.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Store Elo BEFORE game
            home_elo_list.append(elo_ratings[home_team])
            away_elo_list.append(elo_ratings[away_team])
            
            # Calculate expected win probability
            home_elo_adj = elo_ratings[home_team] + home_advantage
            expected_home = 1 / (1 + 10**((elo_ratings[away_team] - home_elo_adj) / 400))
            
            # Actual result
            if game['home_score'] > game['away_score']:
                actual_home = 1
            elif game['home_score'] < game['away_score']:
                actual_home = 0
            else:
                actual_home = 0.5
            
            # Update Elo
            elo_ratings[home_team] += k_factor * (actual_home - expected_home)
            elo_ratings[away_team] += k_factor * ((1 - actual_home) - (1 - expected_home))
        
        # Add to dataframe
        self.games_df['home_elo_before'] = home_elo_list
        self.games_df['away_elo_before'] = away_elo_list
        self.games_df['elo_diff'] = self.games_df['home_elo_before'] - self.games_df['away_elo_before']
        
        print("✓ Elo Ratings berechnet")
        
        return self.games_df
    
    def merge_all_features(self):
        """Mergt alle Features zusammen"""
        
        print("\n🔗 Merge alle Features...")
        
        # Home Team Stats
        home_stats = self.team_stats.copy()
        home_cols_rename = {col: f'{col}_home' for col in home_stats.columns 
                           if col not in ['gameday', 'team']}
        home_stats = home_stats.rename(columns=home_cols_rename)
        
        merged = self.games_df.merge(
            home_stats,
            left_on=['gameday', 'home_team'],
            right_on=['gameday', 'team'],
            how='left'
        ).drop(columns=['team'])
        
        # Away Team Stats
        away_stats = self.team_stats.copy()
        away_cols_rename = {col: f'{col}_away' for col in away_stats.columns 
                           if col not in ['gameday', 'team']}
        away_stats = away_stats.rename(columns=away_cols_rename)
        
        merged = merged.merge(
            away_stats,
            left_on=['gameday', 'away_team'],
            right_on=['gameday', 'team'],
            how='left'
        ).drop(columns=['team'])
        
        print(f"✓ Features gemergt: {merged.shape[1]} Spalten")
        
        return merged
    
    def get_final_training_data(self):
        """Erstellt finales Training Dataset"""
        
        print("\n📦 Erstelle finales Training Dataset...")
        
        # Alle Schritte durchführen
        if self.games_df is None:
            self.load_nflverse_data()
        
        self.create_basic_features()
        self.calculate_team_rolling_stats(windows=[3, 5])
        self.calculate_elo_ratings()
        
        # Merge
        final_df = self.merge_all_features()
        
        # Nur Spiele mit vollständigen Features
        # Wir brauchen mindestens die Rolling Stats
        required_cols = [
            'elo_diff',
            'points_scored_L3_home', 'points_allowed_L3_home', 'win_rate_L3_home',
            'points_scored_L3_away', 'points_allowed_L3_away', 'win_rate_L3_away',
        ]
        
        # Check welche Spalten existieren
        available_cols = [col for col in required_cols if col in final_df.columns]
        
        # Filter nur Zeilen mit vollständigen Daten
        final_df = final_df.dropna(subset=available_cols)
        
        print(f"✓ {len(final_df)} Spiele mit vollständigen Features")
        print(f"   Shape: {final_df.shape}")
        
        # Wichtige Features für ML auswählen
        feature_cols = [
            # Target
            'home_win',
            
            # IDs
            'season', 'week', 'gameday', 'home_team', 'away_team',
            
            # Scores (für Analyse, nicht für Training!)
            'home_score', 'away_score', 'result',
            
            # Elo
            'elo_diff', 'home_elo_before', 'away_elo_before',
            
            # Rolling Stats Home (L3)
            'points_scored_L3_home', 'points_allowed_L3_home', 
            'win_rate_L3_home', 'point_diff_L3_home',
            
            # Rolling Stats Away (L3)
            'points_scored_L3_away', 'points_allowed_L3_away', 
            'win_rate_L3_away', 'point_diff_L3_away',
            
            # Rolling Stats Home (L5)
            'points_scored_L5_home', 'points_allowed_L5_home', 
            'win_rate_L5_home', 'point_diff_L5_home',
            
            # Rolling Stats Away (L5)
            'points_scored_L5_away', 'points_allowed_L5_away', 
            'win_rate_L5_away', 'point_diff_L5_away',
        ]
        
        # Optional features (falls vorhanden)
        optional_cols = ['spread_line', 'total_line', 'rest_advantage', 
                        'is_division_game', 'roof', 'surface', 'temp', 'wind']
        
        for col in optional_cols:
            if col in final_df.columns:
                feature_cols.append(col)
        
        # Nur verfügbare Spalten
        feature_cols = [col for col in feature_cols if col in final_df.columns]
        
        final_df = final_df[feature_cols].copy()
        
        print(f"\n✅ Finale Features: {len(feature_cols)} Spalten")
        print(f"   Erste Features: {feature_cols[:10]}")
        
        return final_df


def main():
    """Hauptfunktion"""
    
    print("\n" + "="*70)
    print("🏈 NFL PREDICTION - DATA COLLECTION (nflverse Edition)")
    print("="*70 + "\n")
    
    # Collector (automatisch bis aktuelles Jahr)
    collector = NFLDataCollector(start_year=2015)  # end_year wird automatisch gesetzt
    
    # Daten laden
    games = collector.load_nflverse_data()
    
    print("\n📊 Sample Games:")
    print(games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].head(10))
    
    # Finales Dataset
    training_data = collector.get_final_training_data()
    
    print("\n📈 Training Data Info:")
    print(f"   Seasons: {training_data['season'].min()} - {training_data['season'].max()}")
    print(f"   Total Games: {len(training_data)}")
    print(f"   Home Win Rate: {training_data['home_win'].mean():.1%}")
    
    # Speichern
    output_file = 'nfl_training_data.csv'
    training_data.to_csv(output_file, index=False)
    print(f"\n💾 Daten gespeichert: {output_file}")
    
    # Stats anzeigen
    print("\n" + "="*70)
    print("📊 DATEN ÜBERSICHT")
    print("="*70)
    print(f"\nGames pro Season:")
    print(training_data.groupby('season').size())
    
    print(f"\nFeature Columns ({len(training_data.columns)}):")
    for col in training_data.columns:
        print(f"  - {col}")
    
    print("\n✅ DATA COLLECTION KOMPLETT!\n")


if __name__ == "__main__":
    main()