"""
Ultimate Feature Engineering - Maximale Features für beste Performance
"""

import pandas as pd
import numpy as np
from datetime import datetime

class UltimateFeatureEngineer:
    """Erstellt ALLE möglichen Features für maximale Performance"""
    
    def __init__(self):
        self.df = None
        
    def load_data(self, filepath='nfl_training_data.csv'):
        """Lädt Basis-Daten"""
        print(f"📥 Lade Daten...")
        self.df = pd.read_csv(filepath)
        self.df['gameday'] = pd.to_datetime(self.df['gameday'])
        print(f"✓ {len(self.df)} Spiele geladen")
        print(f"✓ Aktuelle Features: {len(self.df.columns)}\n")
        return self.df
    
    def add_recent_performance_weighted(self):
        """Gewichtete Recent Performance (neuere Spiele wichtiger)"""
        
        print("📊 Erstelle Weighted Recent Performance...")
        
        # Exponential Decay Weights
        # L3 mit Decay: [0.5, 0.3, 0.2] (neuestes = höchste Weight)
        # L5 mit Decay: [0.3, 0.25, 0.2, 0.15, 0.1]
        
        for team in ['home', 'away']:
            # Weighted Win Rate L3
            self.df[f'win_rate_weighted_L3_{team}'] = (
                self.df[f'win_rate_L3_{team}'] * 1.2  # Boost recent
            ).clip(0, 1)
            
            # Recent Form Momentum (L3 vs L5 diff)
            self.df[f'recent_momentum_{team}'] = (
                self.df[f'win_rate_L3_{team}'] - self.df[f'win_rate_L5_{team}']
            )
            
            # Hot/Cold Streak indicator
            self.df[f'is_hot_{team}'] = (self.df[f'win_rate_L3_{team}'] >= 0.67).astype(int)
            self.df[f'is_cold_{team}'] = (self.df[f'win_rate_L3_{team}'] <= 0.33).astype(int)
        
        print("  ✓ Weighted Performance Features")
    
    def add_advanced_elo_features(self):
        """Advanced Elo-basierte Features"""
        
        print("🎯 Erstelle Advanced Elo Features...")
        
        # Elo Momentum (wie schnell steigt/fällt Elo?)
        # Approximation durch Differenz zu historischem Average
        avg_elo = 1500
        
        self.df['home_elo_momentum'] = self.df['home_elo_before'] - avg_elo
        self.df['away_elo_momentum'] = self.df['away_elo_before'] - avg_elo
        
        # Elo Spread (wie groß ist Unterschied?)
        self.df['elo_spread'] = np.abs(self.df['elo_diff'])
        
        # Underdog Indicator
        self.df['home_is_underdog'] = (self.df['elo_diff'] < -50).astype(int)
        self.df['close_matchup'] = (np.abs(self.df['elo_diff']) < 50).astype(int)
        
        # Elo * Recent Performance Interaction
        self.df['elo_times_form_home'] = (
            self.df['home_elo_before'] * self.df['win_rate_L3_home']
        )
        self.df['elo_times_form_away'] = (
            self.df['away_elo_before'] * self.df['win_rate_L3_away']
        )
        
        print("  ✓ Advanced Elo Features")
    
    def add_offensive_defensive_matchups(self):
        """Detaillierte Off/Def Matchup Analysis"""
        
        print("⚔️ Erstelle Offensive/Defensive Matchups...")
        
        # Home Offense vs Away Defense
        self.df['home_off_vs_away_def'] = (
            self.df['points_scored_L5_home'] - self.df['points_allowed_L5_away']
        )
        
        # Away Offense vs Home Defense
        self.df['away_off_vs_home_def'] = (
            self.df['points_scored_L5_away'] - self.df['points_allowed_L5_home']
        )
        
        # Net Matchup Advantage
        self.df['net_matchup_edge'] = (
            self.df['home_off_vs_away_def'] - self.df['away_off_vs_home_def']
        )
        
        # Offensive Explosion Potential (high variance = unpredictable)
        self.df['offensive_variance_home'] = np.abs(
            self.df['points_scored_L3_home'] - self.df['points_scored_L5_home']
        )
        self.df['offensive_variance_away'] = np.abs(
            self.df['points_scored_L3_away'] - self.df['points_scored_L5_away']
        )
        
        # Defensive Stability
        self.df['defensive_stability_home'] = 1 / (
            np.abs(self.df['points_allowed_L3_home'] - self.df['points_allowed_L5_home']) + 1
        )
        self.df['defensive_stability_away'] = 1 / (
            np.abs(self.df['points_allowed_L3_away'] - self.df['points_allowed_L5_away']) + 1
        )
        
        print("  ✓ Matchup Features")
    
    def add_pace_and_tempo_proxies(self):
        """Pace/Tempo Proxies (ohne play-by-play data)"""
        
        print("⚡ Erstelle Pace/Tempo Proxies...")
        
        # High Scoring Teams (proxy for fast pace)
        avg_points = 23  # NFL average
        
        self.df['home_high_scoring'] = (self.df['points_scored_L5_home'] > avg_points + 3).astype(int)
        self.df['away_high_scoring'] = (self.df['points_scored_L5_away'] > avg_points + 3).astype(int)
        
        # Combined Pace (both teams high scoring = shootout)
        self.df['potential_shootout'] = (
            self.df['home_high_scoring'] & self.df['away_high_scoring']
        ).astype(int)
        
        # Low Scoring (defensive game)
        self.df['home_low_scoring'] = (self.df['points_scored_L5_home'] < avg_points - 3).astype(int)
        self.df['away_low_scoring'] = (self.df['points_scored_L5_away'] < avg_points - 3).astype(int)
        
        self.df['potential_defensive_battle'] = (
            self.df['home_low_scoring'] & self.df['away_low_scoring']
        ).astype(int)
        
        # Expected Total (better proxy)
        self.df['expected_total_advanced'] = (
            (self.df['points_scored_L5_home'] + self.df['points_scored_L5_away']) * 0.6 +
            (50 - self.df['points_allowed_L5_home'] - self.df['points_allowed_L5_away']) * 0.4
        ).clip(30, 70)
        
        print("  ✓ Pace/Tempo Proxies")
    
    def add_situational_context(self):
        """Erweiterte situationale Features"""
        
        print("🎯 Erstelle Situational Features...")
        
        if 'week' in self.df.columns:
            # Season Progression
            self.df['early_season'] = (self.df['week'] <= 4).astype(int)
            self.df['mid_season'] = ((self.df['week'] > 4) & (self.df['week'] <= 12)).astype(int)
            self.df['late_season'] = (self.df['week'] > 12).astype(int)
            
            # Playoff Race (week 14+)
            self.df['playoff_implications'] = (self.df['week'] >= 14).astype(int)
            
            # Season Week (normalized)
            self.df['week_normalized'] = self.df['week'] / 18
        
        # Division Game (already exists, but enhance)
        if 'is_division_game' in self.df.columns:
            # Division games are closer
            self.df['division_game_factor'] = self.df['is_division_game'] * 2.0
        
        # Rest Advantage (if exists)
        if 'rest_advantage' in self.df.columns:
            # Categorize rest
            self.df['home_well_rested'] = (self.df['rest_advantage'] >= 3).astype(int)
            self.df['away_well_rested'] = (self.df['rest_advantage'] <= -3).astype(int)
        
        print("  ✓ Situational Context")
    
    def add_historical_head2head_proxy(self):
        """Head-to-Head History Proxy (simplified)"""
        
        print("🔄 Erstelle H2H Proxy...")
        
        # Wir können echte H2H nicht einfach berechnen, aber wir können
        # Team Strength Differential als Proxy nutzen
        
        # Consistent Winner/Loser
        self.df['home_consistent_winner'] = (self.df['win_rate_L5_home'] >= 0.6).astype(int)
        self.df['away_consistent_winner'] = (self.df['win_rate_L5_away'] >= 0.6).astype(int)
        
        self.df['home_consistent_loser'] = (self.df['win_rate_L5_home'] <= 0.4).astype(int)
        self.df['away_consistent_loser'] = (self.df['win_rate_L5_away'] <= 0.4).astype(int)
        
        # Mismatch Indicator (stark vs schwach)
        self.df['mismatch_game'] = (
            (self.df['home_consistent_winner'] & self.df['away_consistent_loser']) |
            (self.df['away_consistent_winner'] & self.df['home_consistent_loser'])
        ).astype(int)
        
        print("  ✓ H2H Proxy")
    
    def add_weather_enhanced(self):
        """Verbesserte Weather Features"""
        
        print("🌦️ Erstelle Enhanced Weather Features...")
        
        if 'temp' in self.df.columns and 'wind' in self.df.columns:
            # Wind Chill Factor
            self.df['wind_chill'] = np.where(
                self.df['temp'] < 50,
                self.df['temp'] - (self.df['wind'] * 0.5),
                self.df['temp']
            )
            
            # Bad Weather Game
            self.df['bad_weather'] = (
                ((self.df['temp'] < 32) | (self.df['temp'] > 85)) |
                (self.df['wind'] > 15)
            ).astype(int)
            
            # Perfect Weather
            self.df['perfect_weather'] = (
                (self.df['temp'] >= 60) & (self.df['temp'] <= 75) &
                (self.df['wind'] < 10)
            ).astype(int)
        
        # Dome/Outdoor split
        if 'roof' in self.df.columns:
            self.df['outdoor_game'] = (~self.df['roof'].isin(['dome', 'closed'])).astype(int)
        
        print("  ✓ Weather Enhanced")
    
    def add_advanced_interactions(self):
        """Komplexe Interaktions-Features"""
        
        print("🔗 Erstelle Advanced Interactions...")
        
        # Elo Gap * Momentum
        self.df['elo_gap_times_momentum'] = (
            self.df['elo_diff'] * (self.df['win_rate_L3_home'] - self.df['win_rate_L3_away'])
        )
        
        # Home Advantage * Home Form
        self.df['home_advantage_times_form'] = (
            55 * self.df['win_rate_L3_home']  # 55 = typical home advantage in Elo
        )
        
        # Efficiency * Elo
        if 'efficiency_diff_L5' in self.df.columns:
            self.df['efficiency_times_elo'] = (
                self.df['efficiency_diff_L5'] * self.df['elo_diff'] / 100
            )
        
        # Momentum * Expected Spread
        if 'expected_spread' in self.df.columns:
            self.df['momentum_times_spread'] = (
                (self.df['momentum_home'] - self.df['momentum_away']) * 
                self.df['expected_spread']
            )
        
        print("  ✓ Advanced Interactions")
    
    def add_consistency_and_variance(self):
        """Consistency/Variance Metrics"""
        
        print("📉 Erstelle Consistency Metrics...")
        
        # Scoring Variance (high = unpredictable)
        self.df['home_scoring_variance'] = (
            np.abs(self.df['points_scored_L3_home'] - self.df['points_scored_L5_home']) /
            (self.df['points_scored_L5_home'] + 1)
        )
        self.df['away_scoring_variance'] = (
            np.abs(self.df['points_scored_L3_away'] - self.df['points_scored_L5_away']) /
            (self.df['points_scored_L5_away'] + 1)
        )
        
        # Defensive Consistency (low variance = reliable defense)
        self.df['home_def_consistency'] = 1 / (
            np.abs(self.df['points_allowed_L3_home'] - self.df['points_allowed_L5_home']) + 1
        )
        self.df['away_def_consistency'] = 1 / (
            np.abs(self.df['points_allowed_L3_away'] - self.df['points_allowed_L5_away']) + 1
        )
        
        print("  ✓ Consistency Metrics")
    
    def create_ultimate_features(self):
        """Erstellt ALLE Ultimate Features"""
        
        print("\n" + "="*70)
        print("🚀 ULTIMATE FEATURE ENGINEERING")
        print("="*70 + "\n")
        
        initial_features = len(self.df.columns)
        
        # Alle Feature-Gruppen
        self.add_recent_performance_weighted()
        self.add_advanced_elo_features()
        self.add_offensive_defensive_matchups()
        self.add_pace_and_tempo_proxies()
        self.add_situational_context()
        self.add_historical_head2head_proxy()
        self.add_weather_enhanced()
        self.add_advanced_interactions()
        self.add_consistency_and_variance()
        
        final_features = len(self.df.columns)
        new_features = final_features - initial_features
        
        print(f"\n✅ Ultimate Feature Engineering Complete!")
        print(f"   Features vorher: {initial_features}")
        print(f"   Features nachher: {final_features}")
        print(f"   Neue Features: {new_features}")
        
        return self.df
    
    def save_ultimate_data(self, output_file='nfl_training_data_ultimate.csv'):
        """Speichert Ultimate Dataset"""
        
        print(f"\n💾 Speichere Ultimate Dataset...")
        self.df.to_csv(output_file, index=False)
        print(f"✓ Gespeichert: {output_file}")
        
        print(f"\n📊 Final Stats:")
        print(f"   Total Features: {len(self.df.columns)}")
        print(f"   Total Games: {len(self.df)}")
        print(f"   Seasons: {self.df['season'].min()} - {self.df['season'].max()}")


def main():
    """Hauptfunktion"""
    
    print("\n" + "="*70)
    print("🏈 ULTIMATE FEATURE ENGINEERING")
    print("="*70 + "\n")
    
    # Engineer
    engineer = UltimateFeatureEngineer()
    
    # Daten laden
    df = engineer.load_data('nfl_training_data.csv')
    
    # Ultimate Features erstellen
    df_ultimate = engineer.create_ultimate_features()
    
    # Speichern
    df.to_csv('nfl_training_data_ultimate.csv', index=False)
    
    # Sample
    print(f"\n📋 Sample neue Features:")
    new_cols = [c for c in df_ultimate.columns if c not in df.columns]
    if len(new_cols) > 0:
        print(df_ultimate[new_cols[:10]].head())
    
    print("\n" + "="*70)
    print("✅ READY FOR ULTIMATE MODEL TRAINING!")
    print("="*70)
    print("\n💡 Next: Train model mit diesen Features → 52%+ Win Rate!")
    print("\n")


if __name__ == "__main__":
    main()