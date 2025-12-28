"""
2025 SEASON PERFORMANCE TRACKER - FIXED VERSION
Shows all games with predictions vs actual + MAE tracking
Uses optimized K values (40/65/20/135/135) loaded dynamically
Includes Tiered Confidence Strategy with realistic odds
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)


class Season2025Tracker:
    """Track 2025 season prediction performance with MAE"""
    
    def __init__(self):
        self.load_models()
        self.load_data()
    
    def load_models(self):
        """Load trained models with dynamic K configuration"""
        print("Loading models...")
        
        with open('models/win_loss_model.pkl', 'rb') as f:
            self.win_model = pickle.load(f)
        with open('models/spread_model.pkl', 'rb') as f:
            self.spread_model = pickle.load(f)
        with open('models/total_model.pkl', 'rb') as f:
            self.total_model = pickle.load(f)
        with open('models/home_score_model.pkl', 'rb') as f:
            self.home_score_model = pickle.load(f)
        with open('models/away_score_model.pkl', 'rb') as f:
            self.away_score_model = pickle.load(f)
        with open('models/feature_sets.pkl', 'rb') as f:
            self.feature_sets = pickle.load(f)
        
        # Load K configuration dynamically
        try:
            with open('models/optimal_k_config.json', 'r') as f:
                self.k_config = json.load(f)
        except FileNotFoundError:
            # Fallback: infer from feature sets
            self.k_config = {name: len(features) for name, features in self.feature_sets.items()}
        
        print("✓ Models loaded")
        print(f"  - Win/Loss: K={self.k_config.get('win_loss', 'N/A')}")
        print(f"  - Spread: K={self.k_config.get('spread', 'N/A')}")
        print(f"  - Total: K={self.k_config.get('total', 'N/A')}")
        print(f"  - Home Score: K={self.k_config.get('home_score', 'N/A')}")
        print(f"  - Away Score: K={self.k_config.get('away_score', 'N/A')}\n")
    
    def load_data(self):
        """Load 2025 season data"""
        df = pd.read_csv('nfl_training_data_MEGA.csv')
        df['gameday'] = pd.to_datetime(df['gameday'])
        
        self.df = df[df['season'] == 2025].copy()
        self.completed = self.df[self.df['home_score'].notna()].copy()
        
        print(f"📅 2025 Season: {len(self.completed)} games completed\n")
    
    def calculate_american_odds(self, win_rate):
        """Convert win rate to American odds"""
        if win_rate >= 0.5:
            # Favorite (negative odds)
            if win_rate >= 0.99:
                return -10000
            odds = -(win_rate / (1 - win_rate)) * 100
        else:
            # Underdog (positive odds)
            if win_rate <= 0.01:
                return 10000
            odds = ((1 - win_rate) / win_rate) * 100
        return odds
    
    def calculate_realistic_profit(self, win_prob, bet_size, won):
        """Calculate profit with realistic odds based on win probability"""
        odds = self.calculate_american_odds(win_prob)
        
        if won:
            if odds < 0:
                # Favorite: risk |odds| to win 100
                profit = bet_size * (100 / abs(odds))
            else:
                # Underdog: risk 100 to win odds
                profit = bet_size * (odds / 100)
            return profit
        else:
            # Lost bet
            return -bet_size
    
    def calculate_implied_probability(self, spread_line):
        """Convert spread to implied win probability"""
        if pd.isna(spread_line):
            return 0.5
        prob = 0.5 + (spread_line / 28)
        return max(0.1, min(0.9, prob))
    
    def make_predictions(self, game_row):
        """Generate predictions using all 5 models"""
        
        # Win/Loss
        win_features = [f for f in self.feature_sets['win_loss'] if f in game_row.index]
        X_win = game_row[win_features].values.reshape(1, -1)
        X_win = pd.DataFrame(X_win, columns=win_features).fillna(0).infer_objects(copy=False)
        win_proba = self.win_model.predict_proba(X_win)[0]
        
        # Spread
        spread_features = [f for f in self.feature_sets['spread'] if f in game_row.index]
        X_spread = game_row[spread_features].values.reshape(1, -1)
        X_spread = pd.DataFrame(X_spread, columns=spread_features).fillna(0).infer_objects(copy=False)
        spread_pred = self.spread_model.predict(X_spread)[0]
        
        # Total
        total_features = [f for f in self.feature_sets['total'] if f in game_row.index]
        X_total = game_row[total_features].values.reshape(1, -1)
        X_total = pd.DataFrame(X_total, columns=total_features).fillna(0).infer_objects(copy=False)
        total_pred = self.total_model.predict(X_total)[0]
        
        # Home Score
        home_features = [f for f in self.feature_sets['home_score'] if f in game_row.index]
        X_home = game_row[home_features].values.reshape(1, -1)
        X_home = pd.DataFrame(X_home, columns=home_features).fillna(0).infer_objects(copy=False)
        home_pred = self.home_score_model.predict(X_home)[0]
        
        # Away Score
        away_features = [f for f in self.feature_sets['away_score'] if f in game_row.index]
        X_away = game_row[away_features].values.reshape(1, -1)
        X_away = pd.DataFrame(X_away, columns=away_features).fillna(0).infer_objects(copy=False)
        away_pred = self.away_score_model.predict(X_away)[0]
        
        home_prob = win_proba[1]
        away_prob = win_proba[0]
        
        vegas_spread = game_row.get('spread_line', 0)
        vegas_prob_home = self.calculate_implied_probability(vegas_spread)
        edge_home = home_prob - vegas_prob_home
        edge_away = away_prob - (1 - vegas_prob_home)
        spread_diff = spread_pred - vegas_spread
        
        return {
            'home_prob': home_prob,
            'away_prob': away_prob,
            'predicted_spread': spread_pred,
            'predicted_total': total_pred,
            'predicted_home_score': home_pred,
            'predicted_away_score': away_pred,
            'edge_home': edge_home,
            'edge_away': edge_away,
            'spread_diff': spread_diff
        }
    
    def check_tiered_confidence(self, pred, game_row):
        """
        Check Tiered Confidence Strategy (Base ≥70%)
        Returns: (should_bet, team, confidence, bet_size, tier)
        """
        max_prob = max(pred['home_prob'], pred['away_prob'])
        
        # Determine tier and bet size
        if max_prob >= 0.85:
            bet_size = 30
            tier = "T4"
        elif max_prob >= 0.80:
            bet_size = 20
            tier = "T3"
        elif max_prob >= 0.75:
            bet_size = 15
            tier = "T2"
        elif max_prob >= 0.70:
            bet_size = 10
            tier = "T1"
        else:
            return False, None, None, 0, None
        
        # Determine which team
        if pred['home_prob'] > pred['away_prob']:
            team = game_row['home_team']
            confidence = pred['home_prob']
        else:
            team = game_row['away_team']
            confidence = pred['away_prob']
        
        return True, team, confidence, bet_size, tier
    
    def check_result(self, game_row, predicted_team):
        """Check if prediction correct"""
        home_score = game_row['home_score']
        away_score = game_row['away_score']
        actual_winner = game_row['home_team'] if home_score > away_score else game_row['away_team']
        return actual_winner == predicted_team
    
    def print_week(self, week):
        """Print week results with MAE tracking"""
        
        week_games = self.completed[self.completed['week'] == week].copy()
        
        if len(week_games) == 0:
            return None
        
        week_games = week_games.sort_values('gameday')
        
        print("\n" + "="*130)
        print(f"📊 WEEK {week}")
        print("="*130)
        
        # Header
        print(f"\n{'#':<4} {'Matchup':<25} {'Score P/A':<15} {'Spread P/A':<15} {'Total P/A':<15} {'WinP':<12} {'Tiered':<10}")
        print("─"*130)
        
        week_stats = {
            'total': 0,
            'correct': 0,
            'tiered_bets': 0,
            'tiered_correct': 0,
            'tiered_profit': 0,
            'spread_errors': [],
            'total_errors': [],
            'home_score_errors': [],
            'away_score_errors': []
        }
        
        for i, (idx, game) in enumerate(week_games.iterrows(), 1):
            pred = self.make_predictions(game)
            
            matchup = f"{game['away_team']} @ {game['home_team']}"
            
            # Scores
            pred_home = pred['predicted_home_score']
            pred_away = pred['predicted_away_score']
            actual_home = game['home_score']
            actual_away = game['away_score']
            
            score_str = f"~{pred_away:.0f}-{pred_home:.0f}/{actual_away:.0f}-{actual_home:.0f}"
            
            # Spread
            actual_spread = actual_home - actual_away
            spread_pred_str = f"{pred['predicted_spread']:+.1f}/{actual_spread:+.1f}"
            
            # Total
            actual_total = actual_home + actual_away
            total_str = f"{pred['predicted_total']:.1f}/{actual_total:.1f}"
            
            # Track errors
            week_stats['spread_errors'].append(abs(pred['predicted_spread'] - actual_spread))
            week_stats['total_errors'].append(abs(pred['predicted_total'] - actual_total))
            week_stats['home_score_errors'].append(abs(pred_home - actual_home))
            week_stats['away_score_errors'].append(abs(pred_away - actual_away))
            
            # Win prob
            prob = f"{pred['away_prob']:.0%}-{pred['home_prob']:.0%}"
            
            # Prediction correct?
            pred_winner = game['home_team'] if pred['home_prob'] > 0.5 else game['away_team']
            actual_winner = game['home_team'] if actual_home > actual_away else game['away_team']
            
            week_stats['total'] += 1
            if pred_winner == actual_winner:
                week_stats['correct'] += 1
            
            # Tiered Confidence
            is_tiered, tiered_team, confidence, bet_size, tier = self.check_tiered_confidence(pred, game)
            tiered_str = "⏸️"
            if is_tiered:
                tiered_correct = self.check_result(game, tiered_team)
                tiered_str = f"✅{tier}" if tiered_correct else f"❌{tier}"
                week_stats['tiered_bets'] += 1
                
                # Calculate profit with realistic odds
                profit = self.calculate_realistic_profit(confidence, bet_size, tiered_correct)
                week_stats['tiered_profit'] += profit
                
                if tiered_correct:
                    week_stats['tiered_correct'] += 1
            
            print(f"{i:<4} {matchup:<25} {score_str:<15} {spread_pred_str:<15} {total_str:<15} {prob:<12} {tiered_str:<10}")
        
        print("─"*130)
        
        # MAEs
        spread_mae = np.mean(week_stats['spread_errors'])
        total_mae = np.mean(week_stats['total_errors'])
        home_mae = np.mean(week_stats['home_score_errors'])
        away_mae = np.mean(week_stats['away_score_errors'])
        
        # Summary
        acc = week_stats['correct'] / week_stats['total'] * 100
        print(f"\n📈 OVERALL: {week_stats['correct']}/{week_stats['total']} ({acc:.1f}%)")
        print(f"📏 MAE: Spread {spread_mae:.2f}pts | Total {total_mae:.2f}pts | Home {home_mae:.2f}pts | Away {away_mae:.2f}pts")
        
        if week_stats['tiered_bets'] > 0:
            tiered_wr = week_stats['tiered_correct'] / week_stats['tiered_bets'] * 100
            print(f"💰 TIERED CONFIDENCE (≥70%): {week_stats['tiered_correct']}/{week_stats['tiered_bets']} ({tiered_wr:.1f}%)")
            print(f"   Profit: CHF {week_stats['tiered_profit']:.2f}")
        else:
            print(f"💰 TIERED CONFIDENCE: No bets")
        
        return week_stats
    
    def run(self):
        """Main function"""
        
        print("\n" + "="*130)
        print("🏈 2025 NFL SEASON PERFORMANCE TRACKER - FIXED VERSION")
        print("="*130)
        
        weeks = sorted(self.completed['week'].unique())
        
        season_totals = {
            'total': 0,
            'correct': 0,
            'tiered_bets': 0,
            'tiered_correct': 0,
            'tiered_profit': 0,
            'spread_errors': [],
            'total_errors': [],
            'home_score_errors': [],
            'away_score_errors': []
        }
        
        # Print weeks
        for week in weeks:
            week_stats = self.print_week(week)
            
            if week_stats:
                season_totals['total'] += week_stats['total']
                season_totals['correct'] += week_stats['correct']
                season_totals['tiered_bets'] += week_stats['tiered_bets']
                season_totals['tiered_correct'] += week_stats['tiered_correct']
                season_totals['tiered_profit'] += week_stats['tiered_profit']
                season_totals['spread_errors'].extend(week_stats['spread_errors'])
                season_totals['total_errors'].extend(week_stats['total_errors'])
                season_totals['home_score_errors'].extend(week_stats['home_score_errors'])
                season_totals['away_score_errors'].extend(week_stats['away_score_errors'])
        
        # Season Summary
        print("\n" + "="*130)
        print("🏆 2025 SEASON SUMMARY")
        print("="*130)
        
        print(f"\n📊 GAMES: {season_totals['total']} ({len(weeks)} weeks)")
        
        # Overall
        if season_totals['total'] > 0:
            acc = season_totals['correct'] / season_totals['total'] * 100
            print(f"\n📈 OVERALL PREDICTIONS:")
            print(f"   Accuracy: {season_totals['correct']}/{season_totals['total']} ({acc:.1f}%)")
            print(f"   Expected: 67.92% (from test set 2024-2025)")
            print(f"   {'✅ ON TARGET' if acc >= 65 else '⚠️ BELOW' if acc >= 60 else '❌ POOR'}")
        
        # MAE Summary
        spread_mae = np.mean(season_totals['spread_errors'])
        total_mae = np.mean(season_totals['total_errors'])
        home_mae = np.mean(season_totals['home_score_errors'])
        away_mae = np.mean(season_totals['away_score_errors'])
        
        print(f"\n📏 MEAN ABSOLUTE ERROR (MAE):")
        print(f"   Spread: {spread_mae:.2f} pts (Expected: 10.21)")
        print(f"   Total: {total_mae:.2f} pts (Expected: 10.10)")
        print(f"   Home Score: {home_mae:.2f} pts (Expected: 7.35)")
        print(f"   Away Score: {away_mae:.2f} pts (Expected: 7.39)")
        
        # Tiered Confidence
        if season_totals['tiered_bets'] > 0:
            tiered_wr = season_totals['tiered_correct'] / season_totals['tiered_bets'] * 100
            print(f"\n💰 TIERED CONFIDENCE STRATEGY (Base ≥70%):")
            print(f"   Win Rate: {season_totals['tiered_correct']}/{season_totals['tiered_bets']} ({tiered_wr:.1f}%)")
            print(f"   Expected: 74.7% WR, -295 avg odds (from realistic backtest)")
            print(f"   Profit: CHF {season_totals['tiered_profit']:.2f} (variable bet sizing CHF 10-30)")
            print(f"   ROI: {(season_totals['tiered_profit'] / (season_totals['tiered_bets'] * 11) * 100):.1f}% (avg bet ~CHF 11)")
            print(f"   {'✅ ON TARGET' if tiered_wr >= 72 else '⚠️ BELOW' if tiered_wr >= 67 else '❌ POOR'}")
        else:
            print(f"\n💰 TIERED CONFIDENCE: No bets yet")
        
        print("\n")


def main():
    tracker = Season2025Tracker()
    tracker.run()


if __name__ == "__main__":
    main()