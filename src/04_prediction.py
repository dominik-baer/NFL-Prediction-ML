"""
NFL ULTIMATE PREDICTOR - Clean Layout
Übersicht → Details
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

class NFLPredictor:
    """Clean NFL Prediction Interface"""
    
    def __init__(self):
        self.models_loaded = False
        self.data_loaded = False
        
    def select_model(self):
        """Model wählen"""
        
        print("\n" + "="*70)
        print("🏈 NFL PREDICTOR")
        print("="*70)
        
        print("\n📊 Wähle Model:")
        print("\n1️⃣  Academic (ohne Vegas)")
        print("2️⃣  Betting (mit Vegas)")
        
        while True:
            choice = input("\n👉 Eingabe (1 oder 2): ").strip()
            
            if choice == '1':
                self.model_type = 'academic'
                print("\n✅ Academic Model\n")
                break
            elif choice == '2':
                self.model_type = 'betting'
                print("\n✅ Betting Model\n")
                break
            else:
                print("❌ Bitte 1 oder 2!")
    
    def load_models(self):
        """Lädt Models"""
        
        try:
            if self.model_type == 'academic':
                with open('models/academic_win_rf.pkl', 'rb') as f:
                    self.win_model = pickle.load(f)
                with open('models/academic_spread_rf.pkl', 'rb') as f:
                    self.spread_model = pickle.load(f)
                with open('models/academic_total_rf.pkl', 'rb') as f:
                    self.total_model = pickle.load(f)
                with open('models/academic_features.pkl', 'rb') as f:
                    self.features = pickle.load(f)
            else:
                with open('models/betting_win.pkl', 'rb') as f:
                    self.win_model = pickle.load(f)
                with open('models/betting_spread.pkl', 'rb') as f:
                    self.spread_model = pickle.load(f)
                with open('models/betting_total.pkl', 'rb') as f:
                    self.total_model = pickle.load(f)
                with open('models/betting_features.pkl', 'rb') as f:
                    self.features = pickle.load(f)
            
            self.models_loaded = True
            return True
            
        except FileNotFoundError:
            print(f"❌ Models nicht gefunden! Führe 03_model_betting.py aus")
            return False
    
    def load_data(self):
        """Lädt Daten"""
        
        try:
            self.df = pd.read_csv('nfl_training_data_ultimate.csv')
            self.df['gameday'] = pd.to_datetime(self.df['gameday'])
            self.data_loaded = True
            return True
        except:
            print("❌ Daten nicht gefunden!")
            return False
    
    def predict_game(self, game_row):
        """Macht Prediction"""
        
        available_features = [f for f in self.features if f in game_row.index]
        X = game_row[available_features].values.reshape(1, -1)
        X = pd.DataFrame(X, columns=available_features)
        X = X.fillna(X.mean())
        
        win_proba = self.win_model.predict_proba(X)[0]
        spread_pred = self.spread_model.predict(X)[0]
        total_pred = self.total_model.predict(X)[0]
        
        return {
            'home_team': game_row.get('home_team', 'Home'),
            'away_team': game_row.get('away_team', 'Away'),
            'week': game_row.get('week', '?'),
            'win_prob_home': win_proba[1],
            'win_prob_away': 1 - win_proba[1],
            'predicted_spread': spread_pred,
            'predicted_total': total_pred,
            'vegas_spread': game_row.get('spread_line', None),
            'vegas_total': game_row.get('total_line', None)
        }
    
    def should_bet(self, prediction):
        """Check ob Ultra Conservative Bet"""
        
        win_prob_home = prediction['win_prob_home']
        win_prob_away = prediction['win_prob_away']
        vegas_spread = prediction['vegas_spread']
        our_spread = prediction['predicted_spread']
        
        if pd.isna(vegas_spread):
            return False, None
        
        spread_agreement = abs(our_spread - vegas_spread)
        
        # OPTIMIZED Ultra Conservative (from grid search):
        # Win Prob ≥ 80% AND Spread Agreement ≤ 4.5 pts
        # → 84.3% Win Rate, +40.7% ROI
        
        # Check HOME team
        if win_prob_home >= 0.80 and spread_agreement <= 4.5:
            return True, prediction['home_team']
        
        # Check AWAY team
        if win_prob_away >= 0.80 and spread_agreement <= 4.5:
            return True, prediction['away_team']
        
        return False, None
    
    def show_overview(self, predictions):
        """Zeigt Clean Overview"""
        
        print("\n" + "="*70)
        print("📊 WEEK OVERVIEW")
        print("="*70)
        print(f"\n{'Matchup':<25} {'Win Prob':<20} {'Betting':<15}")
        print("─"*70)
        
        for i, pred in enumerate(predictions, 1):
            away = pred['away_team']
            home = pred['home_team']
            prob_away = pred['win_prob_away']
            prob_home = pred['win_prob_home']
            
            matchup = f"{away} @ {home}"
            probs = f"{prob_away:.0%} - {prob_home:.0%}"
            
            should_bet, bet_team = self.should_bet(pred)
            
            if should_bet:
                bet_str = f"✅ BET {bet_team}"
            else:
                bet_str = "❌ SKIP"
            
            print(f"{i:2d}. {matchup:<22} {probs:<20} {bet_str:<15}")
        
        print("─"*70)
    
    def show_details(self, predictions):
        """Zeigt Details für alle Spiele"""
        
        for i, pred in enumerate(predictions, 1):
            print("\n" + "="*70)
            print(f"🏈 GAME {i}/{len(predictions)}: {pred['away_team']} @ {pred['home_team']}")
            print("="*70)
            
            # Win Probability
            print(f"\n🎯 WIN PROBABILITY:")
            print(f"   {pred['home_team']:<20} {pred['win_prob_home']:>6.1%}")
            print(f"   {pred['away_team']:<20} {pred['win_prob_away']:>6.1%}")
            
            # Spread
            print(f"\n📏 SPREAD:")
            print(f"   Our Prediction:  {pred['predicted_spread']:>+7.1f}")
            if not pd.isna(pred['vegas_spread']):
                print(f"   Vegas Line:      {pred['vegas_spread']:>+7.1f}")
                diff = abs(pred['predicted_spread'] - pred['vegas_spread'])
                print(f"   Difference:      {diff:>7.1f} pts")
            
            # Total
            print(f"\n🎲 OVER/UNDER:")
            print(f"   Our Prediction:  {pred['predicted_total']:>7.1f}")
            if not pd.isna(pred['vegas_total']):
                print(f"   Vegas Line:      {pred['vegas_total']:>7.1f}")
                diff = abs(pred['predicted_total'] - pred['vegas_total'])
                print(f"   Difference:      {diff:>7.1f} pts")
            
            # Betting Recommendation
            should_bet, bet_team = self.should_bet(pred)
            
            print(f"\n💰 BETTING:")
            if should_bet:
                print(f"   ✅ ULTRA CONSERVATIVE BET")
                print(f"   🎯 Bet: MONEYLINE {bet_team}")
                prob = pred['win_prob_home'] if bet_team == pred['home_team'] else pred['win_prob_away']
                print(f"   📊 Confidence: {prob:.1%}")
                print(f"   📈 Historical: 81% Win Rate, +35% ROI")
            else:
                print(f"   ❌ NO HIGH PRIORITY BET")
                print(f"   → Skip this game")
            
            # Pause
            if i < len(predictions):
                input("\n[Enter für nächstes Spiel...]")
    
    def run(self):
        """Hauptfunktion"""
        
        # Model wählen
        self.select_model()
        
        # Laden
        if not self.load_models():
            return
        
        if not self.load_data():
            return
        
        # Get upcoming week
        current_season = self.df['season'].max()
        season_games = self.df[self.df['season'] == current_season].copy()
        
        # Find upcoming games
        if 'home_score' in season_games.columns:
            upcoming = season_games[season_games['home_score'].isna()]
            if len(upcoming) > 0:
                week = upcoming['week'].min()
            else:
                week = season_games['week'].max()
        else:
            week = season_games['week'].max()
        
        week_games = season_games[season_games['week'] == week]
        
        print(f"📅 Season {current_season} - Week {week}")
        print(f"📊 {len(week_games)} Games\n")
        
        # Make predictions
        predictions = []
        for _, game in week_games.iterrows():
            pred = self.predict_game(game)
            predictions.append(pred)
        
        # Main loop
        while True:
            # Show overview
            self.show_overview(predictions)
            
            # Count bets
            bets = sum(1 for p in predictions if self.should_bet(p)[0])
            print(f"\n💪 Ultra Conservative Bets: {bets}")
            
            # Menu
            print("\n" + "="*70)
            print("\n1️⃣  Details anzeigen")
            print("2️⃣  Exit")
            
            choice = input("\n👉 Eingabe (1 oder 2): ").strip()
            
            if choice == '1':
                self.show_details(predictions)
            elif choice == '2':
                print("\n👋 Bye! Good luck! 🍀\n")
                break
            else:
                print("❌ Bitte 1 oder 2!")


def main():
    predictor = NFLPredictor()
    predictor.run()


if __name__ == "__main__":
    main()