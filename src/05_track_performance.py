"""
PERFORMANCE TRACKER
Tracks predictions vs actual results - Essential for monitoring system performance!
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle


class PerformanceTracker:
    """Tracks and analyzes prediction performance"""
    
    def __init__(self):
        self.results_file = 'performance_results.csv'
        self.summary_file = 'performance_summary.csv'
        
    def load_data(self):
        """Load training data with actual results"""
        try:
            self.df = pd.read_csv('nfl_training_data_ultimate.csv')
            self.df['gameday'] = pd.to_datetime(self.df['gameday'])
            print("✓ Data loaded")
            return True
        except:
            print("❌ Failed to load data")
            return False
    
    def load_models(self, model_type='betting'):
        """Load models"""
        try:
            with open(f'models/{model_type}_win.pkl', 'rb') as f:
                self.win_model = pickle.load(f)
            with open(f'models/{model_type}_spread.pkl', 'rb') as f:
                self.spread_model = pickle.load(f)
            with open(f'models/{model_type}_total.pkl', 'rb') as f:
                self.total_model = pickle.load(f)
            with open(f'models/{model_type}_home_score.pkl', 'rb') as f:
                self.home_score_model = pickle.load(f)
            with open(f'models/{model_type}_away_score.pkl', 'rb') as f:
                self.away_score_model = pickle.load(f)
            with open(f'models/{model_type}_features.pkl', 'rb') as f:
                self.features = pickle.load(f)
            
            self.model_type = model_type
            print(f"✓ {model_type.upper()} models loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
    
    def predict_game(self, game_row):
        """Make prediction for a game"""
        available_features = [f for f in self.features if f in game_row.index]
        X = game_row[available_features].values.reshape(1, -1)
        X = pd.DataFrame(X, columns=available_features)
        X = X.fillna(X.mean())
        
        win_proba = self.win_model.predict_proba(X)[0]
        spread_pred = self.spread_model.predict(X)[0]
        total_pred = self.total_model.predict(X)[0]
        home_score_pred = self.home_score_model.predict(X)[0]
        away_score_pred = self.away_score_model.predict(X)[0]
        
        return {
            'win_prob_home': win_proba[1],
            'win_prob_away': 1 - win_proba[1],
            'predicted_spread': spread_pred,
            'predicted_total': total_pred,
            'predicted_home_score': home_score_pred,
            'predicted_away_score': away_score_pred
        }
    
    def is_ultra_conservative_bet(self, prediction, vegas_spread):
        """Check if qualifies as Ultra Conservative bet"""
        if pd.isna(vegas_spread):
            return False, None
        
        spread_agreement = abs(prediction['predicted_spread'] - vegas_spread)
        
        # Optimized thresholds
        if prediction['win_prob_home'] >= 0.80 and spread_agreement <= 4.5:
            return True, 'home'
        
        if prediction['win_prob_away'] >= 0.80 and spread_agreement <= 4.5:
            return True, 'away'
        
        return False, None
    
    def analyze_season(self, season, week_start=1, week_end=18):
        """Analyze performance for a season"""
        
        print(f"\n{'='*70}")
        print(f"📊 ANALYZING {season} SEASON (Weeks {week_start}-{week_end})")
        print(f"{'='*70}\n")
        
        # Get completed games for this season
        season_games = self.df[
            (self.df['season'] == season) & 
            (self.df['week'] >= week_start) &
            (self.df['week'] <= week_end) &
            (self.df['home_score'].notna())
        ].copy()
        
        if len(season_games) == 0:
            print(f"⚠️  No completed games found for {season}")
            return None
        
        print(f"Found {len(season_games)} completed games\n")
        
        results = []
        
        for _, game in season_games.iterrows():
            # Make prediction
            pred = self.predict_game(game)
            
            # Check if bet
            is_bet, bet_side = self.is_ultra_conservative_bet(pred, game.get('spread_line'))
            
            if not is_bet:
                continue
            
            # Actual result
            actual_home_win = game['home_win']
            bet_won = (bet_side == 'home' and actual_home_win == 1) or \
                      (bet_side == 'away' and actual_home_win == 0)
            
            # Calculate errors
            spread_error = abs(pred['predicted_spread'] - game['result'])
            total_error = abs(pred['predicted_total'] - (game['home_score'] + game['away_score']))
            home_score_error = abs(pred['predicted_home_score'] - game['home_score'])
            away_score_error = abs(pred['predicted_away_score'] - game['away_score'])
            
            results.append({
                'season': season,
                'week': game['week'],
                'date': game['gameday'],
                'away_team': game['away_team'],
                'home_team': game['home_team'],
                'bet_team': game['home_team'] if bet_side == 'home' else game['away_team'],
                'win_prob': pred['win_prob_home'] if bet_side == 'home' else pred['win_prob_away'],
                'predicted_spread': pred['predicted_spread'],
                'actual_spread': game['result'],
                'spread_error': spread_error,
                'predicted_total': pred['predicted_total'],
                'actual_total': game['home_score'] + game['away_score'],
                'total_error': total_error,
                'predicted_home_score': pred['predicted_home_score'],
                'actual_home_score': game['home_score'],
                'home_score_error': home_score_error,
                'predicted_away_score': pred['predicted_away_score'],
                'actual_away_score': game['away_score'],
                'away_score_error': away_score_error,
                'vegas_spread': game.get('spread_line'),
                'spread_agreement': abs(pred['predicted_spread'] - game.get('spread_line', 0)),
                'bet_won': bet_won
            })
        
        if len(results) == 0:
            print("⚠️  No Ultra Conservative bets found")
            return None
        
        results_df = pd.DataFrame(results)
        
        # Summary stats
        total_bets = len(results_df)
        wins = results_df['bet_won'].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets
        
        # ROI calculation (assuming -150 favorite odds)
        profit = (wins * 0.6667) - losses
        roi = (profit / total_bets) * 100
        
        # Accuracy metrics
        avg_spread_error = results_df['spread_error'].mean()
        avg_total_error = results_df['total_error'].mean()
        avg_home_score_error = results_df['home_score_error'].mean()
        avg_away_score_error = results_df['away_score_error'].mean()
        
        print(f"{'='*70}")
        print(f"🎯 ULTRA CONSERVATIVE BETS")
        print(f"{'='*70}")
        print(f"Total Bets: {total_bets}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"ROI: {roi:+.1f}%")
        
        print(f"\n{'='*70}")
        print(f"📊 PREDICTION ACCURACY")
        print(f"{'='*70}")
        print(f"Spread MAE: {avg_spread_error:.2f} points")
        print(f"Total MAE: {avg_total_error:.2f} points")
        print(f"Home Score MAE: {avg_home_score_error:.2f} points")
        print(f"Away Score MAE: {avg_away_score_error:.2f} points")
        
        # Weekly breakdown
        print(f"\n{'='*70}")
        print(f"📅 WEEKLY BREAKDOWN")
        print(f"{'='*70}")
        
        weekly = results_df.groupby('week').agg({
            'bet_won': ['count', 'sum', 'mean']
        }).round(3)
        
        for week in sorted(results_df['week'].unique()):
            week_data = results_df[results_df['week'] == week]
            week_bets = len(week_data)
            week_wins = week_data['bet_won'].sum()
            week_wr = week_wins / week_bets
            
            print(f"Week {week:2d}: {week_bets} bets, {week_wins} wins ({week_wr:.0%})")
        
        # Show losses
        if losses > 0:
            print(f"\n{'='*70}")
            print(f"❌ LOSSES ANALYSIS")
            print(f"{'='*70}")
            
            losses_df = results_df[~results_df['bet_won']]
            
            for _, loss in losses_df.iterrows():
                print(f"\nWeek {loss['week']}: {loss['away_team']} @ {loss['home_team']}")
                print(f"   Bet: {loss['bet_team']} ({loss['win_prob']:.0%} confidence)")
                print(f"   Spread Agreement: {loss['spread_agreement']:.1f} pts")
                print(f"   Actual Score: {loss['actual_away_score']:.0f}-{loss['actual_home_score']:.0f}")
        
        return {
            'season': season,
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'roi': roi,
            'spread_mae': avg_spread_error,
            'total_mae': avg_total_error,
            'home_score_mae': avg_home_score_error,
            'away_score_mae': avg_away_score_error,
            'results': results_df
        }
    
    def save_results(self, season_results):
        """Save results to CSV"""
        if season_results is None:
            return
        
        # Save detailed results
        results_df = season_results['results']
        
        if os.path.exists(self.results_file):
            existing = pd.read_csv(self.results_file)
            # Remove old results for this season
            existing = existing[existing['season'] != season_results['season']]
            results_df = pd.concat([existing, results_df], ignore_index=True)
        
        results_df.to_csv(self.results_file, index=False)
        print(f"\n💾 Saved: {self.results_file}")
        
        # Save summary
        summary = {
            'season': season_results['season'],
            'total_bets': season_results['total_bets'],
            'wins': season_results['wins'],
            'losses': season_results['losses'],
            'win_rate': season_results['win_rate'],
            'roi': season_results['roi'],
            'spread_mae': season_results['spread_mae'],
            'total_mae': season_results['total_mae'],
            'home_score_mae': season_results['home_score_mae'],
            'away_score_mae': season_results['away_score_mae'],
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_df = pd.DataFrame([summary])
        
        if os.path.exists(self.summary_file):
            existing = pd.read_csv(self.summary_file)
            existing = existing[existing['season'] != season_results['season']]
            summary_df = pd.concat([existing, summary_df], ignore_index=True)
        
        summary_df = summary_df.sort_values('season')
        summary_df.to_csv(self.summary_file, index=False)
        print(f"💾 Saved: {self.summary_file}")
    
    def run(self):
        """Main function"""
        
        print("\n" + "="*70)
        print("📊 PERFORMANCE TRACKER")
        print("="*70)
        
        # Load
        if not self.load_data():
            return
        
        if not self.load_models('betting'):
            return
        
        # Get available seasons
        seasons = sorted(self.df['season'].unique())
        current_season = seasons[-1]
        
        print(f"\n📅 Available seasons: {seasons[0]}-{current_season}")
        
        # Menu
        print("\n1️⃣  Analyze current season ({})".format(current_season))
        print("2️⃣  Analyze specific season")
        print("3️⃣  Analyze all seasons")
        
        choice = input("\n👉 Choice (1-3): ").strip()
        
        if choice == '1':
            results = self.analyze_season(current_season)
            if results:
                self.save_results(results)
        
        elif choice == '2':
            season = int(input("Season: "))
            results = self.analyze_season(season)
            if results:
                self.save_results(results)
        
        elif choice == '3':
            # Analyze last 5 seasons
            for season in seasons[-5:]:
                results = self.analyze_season(season)
                if results:
                    self.save_results(results)
                print()
            
            # Overall summary
            if os.path.exists(self.summary_file):
                print("\n" + "="*70)
                print("🏆 OVERALL PERFORMANCE")
                print("="*70 + "\n")
                
                summary = pd.read_csv(self.summary_file)
                
                for _, row in summary.iterrows():
                    print(f"{row['season']}: {row['wins']}-{row['losses']} ({row['win_rate']:.1%}), ROI: {row['roi']:+.1f}%")
                
                # Totals
                total_bets = summary['total_bets'].sum()
                total_wins = summary['wins'].sum()
                overall_wr = total_wins / total_bets
                total_profit = (total_wins * 0.6667) - (total_bets - total_wins)
                overall_roi = (total_profit / total_bets) * 100
                
                print(f"\n{'='*70}")
                print(f"TOTAL: {total_wins}-{total_bets - total_wins} ({overall_wr:.1%}), ROI: {overall_roi:+.1f}%")
                print(f"{'='*70}")
        
        print("\n✅ Done!\n")


def main():
    tracker = PerformanceTracker()
    tracker.run()


if __name__ == "__main__":
    main()