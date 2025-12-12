"""
NFL GAME PREDICTOR - Academic Model
Two Strategies: High Confidence (≥75%) + Multi Strategy (ML:7%/60% + Spr:5/60%)
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime


class NFLPredictor:
    """NFL Game Prediction System"""
    
    def __init__(self):
        self.models_loaded = False
        self.data_loaded = False
        
    def load_models(self):
        """Load Academic Models"""
        
        try:
            with open('models/academic_win_rf.pkl', 'rb') as f:
                self.win_model = pickle.load(f)
            with open('models/academic_spread_rf.pkl', 'rb') as f:
                self.spread_model = pickle.load(f)
            with open('models/academic_total_rf.pkl', 'rb') as f:
                self.total_model = pickle.load(f)
            with open('models/academic_features.pkl', 'rb') as f:
                self.features = pickle.load(f)
            
            self.models_loaded = True
            print("✓ Academic Model loaded (Pure ML, no Vegas)")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Models not found! Run model_academic.py first")
            print(f"   Missing: {e}")
            return False
    
    def load_data(self):
        """Load data"""
        
        try:
            self.df = pd.read_csv('nfl_training_data_ultimate.csv')
            self.df['gameday'] = pd.to_datetime(self.df['gameday'])
            self.data_loaded = True
            return True
        except:
            print("❌ Data not found!")
            return False
    
    def calculate_implied_probability(self, spread_line):
        """Convert spread to implied win probability"""
        if pd.isna(spread_line):
            return 0.5
        prob = 0.5 + (spread_line / 28)
        return max(0.1, min(0.9, prob))
    
    def predict_game(self, game_row):
        """Make prediction for a game"""
        
        available_features = [f for f in self.features if f in game_row.index]
        X = game_row[available_features].values.reshape(1, -1)
        X = pd.DataFrame(X, columns=available_features)
        X = X.fillna(X.mean())
        
        # Predictions
        win_proba = self.win_model.predict_proba(X)[0]
        spread_pred = self.spread_model.predict(X)[0]
        total_pred = self.total_model.predict(X)[0]
        
        # Derive scores from spread + total
        home_score_pred = (total_pred + spread_pred) / 2
        away_score_pred = (total_pred - spread_pred) / 2
        
        # Get Vegas lines for edge calculation
        vegas_spread = game_row.get('spread_line', 0)
        vegas_total = game_row.get('total_line', 0)
        
        # Calculate edges
        vegas_prob_home = self.calculate_implied_probability(vegas_spread)
        edge_home = win_proba[1] - vegas_prob_home
        edge_away = (1 - win_proba[1]) - (1 - vegas_prob_home)
        spread_diff = spread_pred - vegas_spread
        
        return {
            'home_team': game_row.get('home_team', 'Home'),
            'away_team': game_row.get('away_team', 'Away'),
            'week': game_row.get('week', '?'),
            'win_prob_home': win_proba[1],
            'win_prob_away': 1 - win_proba[1],
            'predicted_spread': spread_pred,
            'predicted_total': total_pred,
            'predicted_home_score': home_score_pred,
            'predicted_away_score': away_score_pred,
            'actual_home_score': game_row.get('home_score', None),
            'actual_away_score': game_row.get('away_score', None),
            'vegas_spread': vegas_spread,
            'vegas_total': vegas_total,
            'edge_home': edge_home,
            'edge_away': edge_away,
            'spread_diff': spread_diff
        }
    
    def is_high_confidence(self, prediction):
        """Check if game meets High Confidence criteria (≥75% win probability)"""
        
        win_prob_home = prediction['win_prob_home']
        win_prob_away = prediction['win_prob_away']
        
        # HOME high confidence
        if win_prob_home >= 0.75:
            return True, prediction['home_team'], win_prob_home
        
        # AWAY high confidence
        if win_prob_away >= 0.75:
            return True, prediction['away_team'], win_prob_away
        
        return False, None, None
    
    def get_multi_strategy_bet(self, prediction):
        """
        Multi-Strategy Betting (ML:7%/60% + Spr:5/60%)
        Returns betting recommendation string
        """
        
        home_prob = prediction['win_prob_home']
        away_prob = prediction['win_prob_away']
        edge_home = prediction['edge_home']
        edge_away = prediction['edge_away']
        spread_diff = prediction['spread_diff']
        home_team = prediction['home_team']
        away_team = prediction['away_team']
        
        # Moneyline: Edge ≥7% + Confidence ≥60%
        if home_prob >= 0.60 and edge_home >= 0.07:
            return f"✅ ML: {home_team}"
        
        if away_prob >= 0.60 and edge_away >= 0.07:
            return f"✅ ML: {away_team}"
        
        # Spread: Diff ≥5pts + Confidence ≥60%
        if home_prob >= 0.60 and spread_diff >= 5:
            return f"✅ Spr: {home_team}"
        
        if away_prob >= 0.60 and spread_diff <= -5:
            return f"✅ Spr: {away_team}"
        
        return "⏸️  Skip"
    
    def show_overview(self, predictions):
        """Show clean overview of all games"""
        
        print("\n" + "="*116)
        print("📊 WEEK OVERVIEW")
        print("="*116)
        print(f"\n{'#':<4} {'Matchup':<25} {'Score':<12} {'Spread':<10} {'Total':<8} {'Win Prob':<18} {'High Conf':<14} {'Multi Strategy':<15}")
        print("─"*116)
        
        for i, pred in enumerate(predictions, 1):
            away = pred['away_team']
            home = pred['home_team']
            prob_away = pred['win_prob_away']
            prob_home = pred['win_prob_home']
            
            matchup = f"{away} @ {home}"
            probs = f"{prob_away:.0%}-{prob_home:.0%}"
            
            # Score - show actual if available, predicted otherwise
            actual_home = pred.get('actual_home_score')
            actual_away = pred.get('actual_away_score')
            
            if pd.notna(actual_home) and pd.notna(actual_away):
                score = f"{actual_away:.0f}-{actual_home:.0f}"
            else:
                pred_home = pred['predicted_home_score']
                pred_away = pred['predicted_away_score']
                score = f"~{pred_away:.0f}-{pred_home:.0f}"
            
            # Spread
            our_spread = pred['predicted_spread']
            spread = f"{our_spread:+.1f}"
            
            # Total
            our_total = pred['predicted_total']
            total = f"{our_total:.0f}"
            
            # High Confidence check
            is_hc, bet_team, confidence = self.is_high_confidence(pred)
            
            if is_hc:
                rec = f"✅ BET {bet_team}"
            else:
                rec = "❌ SKIP"
            
            # Multi Strategy check
            multi_bet = self.get_multi_strategy_bet(pred)
            
            print(f"{i:<4} {matchup:<25} {score:<12} {spread:<10} {total:<8} {probs:<18} {rec:<14} {multi_bet:<15}")
        
        print("─"*116)
    
    def show_details(self, predictions, strategy='high_conf'):
        """Show detailed analysis for bets"""
        
        if strategy == 'high_conf':
            # Filter for high confidence bets
            filtered_predictions = []
            for pred in predictions:
                is_hc, bet_team, confidence = self.is_high_confidence(pred)
                if is_hc:
                    filtered_predictions.append((pred, bet_team, confidence, 'High Confidence'))
        
        else:  # multi strategy
            # Filter for multi strategy bets
            filtered_predictions = []
            for pred in predictions:
                multi_bet = self.get_multi_strategy_bet(pred)
                if '✅' in multi_bet:
                    # Extract team and bet type
                    parts = multi_bet.split(': ')
                    bet_type = parts[0].replace('✅ ', '')
                    bet_team = parts[1]
                    confidence = max(pred['win_prob_home'], pred['win_prob_away'])
                    filtered_predictions.append((pred, bet_team, confidence, f'Multi Strategy ({bet_type})'))
        
        if len(filtered_predictions) == 0:
            print(f"\n⚠️  No {strategy.replace('_', ' ')} bets this week")
            return
        
        for i, (pred, bet_team, confidence, strategy_name) in enumerate(filtered_predictions, 1):
            print("\n" + "="*70)
            print(f"🎯 BET {i}/{len(filtered_predictions)}: {pred['away_team']} @ {pred['home_team']}")
            print("="*70)
            
            # Recommendation
            print(f"\n💰 RECOMMENDATION:")
            print(f"   ✅ BET: {bet_team}")
            print(f"   📊 Confidence: {confidence:.1%}")
            print(f"   🎯 Strategy: {strategy_name}")
            
            if 'Multi' in strategy_name:
                if 'ML' in strategy_name:
                    edge = pred['edge_home'] if bet_team == pred['home_team'] else pred['edge_away']
                    print(f"   📈 Edge vs Market: {edge:+.1%}")
                else:  # Spread
                    print(f"   📏 Spread Advantage: {abs(pred['spread_diff']):.1f} points")
            
            # Score Prediction
            actual_home = pred.get('actual_home_score')
            actual_away = pred.get('actual_away_score')
            
            print(f"\n🎯 SCORE PREDICTION:")
            print(f"   {pred['away_team']:<20} {pred['predicted_away_score']:>6.1f}")
            print(f"   {pred['home_team']:<20} {pred['predicted_home_score']:>6.1f}")
            
            if pd.notna(actual_home) and pd.notna(actual_away):
                print(f"\n   ACTUAL SCORE:")
                print(f"   {pred['away_team']:<20} {actual_away:>6.0f}")
                print(f"   {pred['home_team']:<20} {actual_home:>6.0f}")
                
                # Check if bet won
                home_won = actual_home > actual_away
                away_won = actual_away > actual_home
                
                bet_won = (bet_team == pred['home_team'] and home_won) or \
                         (bet_team == pred['away_team'] and away_won)
                
                if bet_won:
                    print(f"\n   ✅ BET WON!")
                else:
                    print(f"\n   ❌ BET LOST")
            
            # Win Probability
            print(f"\n📈 WIN PROBABILITY:")
            print(f"   {pred['home_team']:<20} {pred['win_prob_home']:>6.1%}")
            print(f"   {pred['away_team']:<20} {pred['win_prob_away']:>6.1%}")
            
            # Spread & Total
            print(f"\n📏 SPREAD & TOTAL:")
            print(f"   Predicted Spread:  {pred['predicted_spread']:>+7.1f}")
            print(f"   Predicted Total:   {pred['predicted_total']:>7.1f}")
            
            # Pause between bets
            if i < len(filtered_predictions):
                input("\n[Press Enter for next bet...]")
    
    def run(self):
        """Main function"""
        
        print("\n" + "="*70)
        print("🏈 NFL GAME PREDICTOR")
        print("="*70)
        print("\nAcademic Model - Pure Machine Learning")
        print("Two Strategies Available:")
        print("  1. High Confidence (≥75% Win Probability)")
        print("  2. Multi Strategy (ML:7%/60% + Spr:5/60%)\n")
        
        # Load
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
                # All games completed, show last week
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
            
            # Count bets for each strategy
            hc_count = sum(1 for p in predictions if self.is_high_confidence(p)[0])
            multi_count = sum(1 for p in predictions if '✅' in self.get_multi_strategy_bet(p))
            
            print(f"\n💪 High Confidence Bets (≥75%): {hc_count}")
            print(f"🎯 Multi Strategy Bets: {multi_count}")
            
            # Historical stats
            print(f"\n📊 Strategy Performance (Walk-Forward Validated 2020-2025):")
            print(f"\n   High Confidence (≥75%):")
            print(f"     Win Rate: 80.4%")
            print(f"     ROI: +9.4%")
            print(f"     Volume: 37 bets/year")
            print(f"     Consistency: 5/6 years profitable")
            
            print(f"\n   Multi Strategy (ML:7%/60% + Spr:5/60%):")
            print(f"     Win Rate: 57.3%")
            print(f"     ROI: +7.9%")
            print(f"     Volume: 70 bets/year")
            print(f"     Profit/Year: $550 (at $100/bet)")
            print(f"     Consistency: 5/6 years profitable")
            
            # Menu
            print("\n" + "="*70)
            print("\n1️⃣  Show High Confidence Bet Details")
            print("2️⃣  Show Multi Strategy Bet Details")
            print("3️⃣  Exit")
            
            choice = input("\n👉 Choice (1-3): ").strip()
            
            if choice == '1':
                self.show_details(predictions, strategy='high_conf')
            elif choice == '2':
                self.show_details(predictions, strategy='multi')
            elif choice == '3':
                print("\n👋 Good luck! 🍀\n")
                break
            else:
                print("❌ Please enter 1, 2, or 3")


def main():
    predictor = NFLPredictor()
    predictor.run()


if __name__ == "__main__":
    main()