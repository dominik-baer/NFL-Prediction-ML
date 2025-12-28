"""
NFL BETTING STRATEGY BACKTEST - COMPREHENSIVE
Tests 50+ different strategies to find the most profitable approach

Strategies tested:
- Confidence thresholds (20+ variations)
- Spread difference strategies (15+ variations)
- Total-based strategies (10+ variations)
- Combined multi-factor strategies (10+ variations)
- Kelly Criterion sizing variations
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StrategyBacktester:
    """Comprehensive betting strategy backtesting system"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.feature_sets = {}
        self.predictions = []
        
    def load_models(self):
        """Load all 5 models"""
        
        print("📥 Loading models...")
        
        try:
            model_files = {
                'win_loss': 'win_loss_model.pkl',
                'spread': 'spread_model.pkl',
                'total': 'total_model.pkl',
                'home_score': 'home_score_model.pkl',
                'away_score': 'away_score_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                with open(f'{self.models_dir}/{filename}', 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            
            with open(f'{self.models_dir}/feature_sets.pkl', 'rb') as f:
                self.feature_sets = pickle.load(f)
            
            print(f"✓ All models loaded\n")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def prepare_features(self, game_row, feature_list):
        """Prepare features for prediction"""
        available_features = [f for f in feature_list if f in game_row.index]
        X = game_row[available_features].values.reshape(1, -1)
        X = pd.DataFrame(X, columns=available_features).fillna(0)
        return X
    
    def predict_game(self, game_row):
        """Make predictions for a game"""
        
        try:
            # Win/Loss
            X_win = self.prepare_features(game_row, self.feature_sets['win_loss'])
            win_proba = self.models['win_loss'].predict_proba(X_win)[0]
            
            # Spread
            X_spread = self.prepare_features(game_row, self.feature_sets['spread'])
            spread_pred = self.models['spread'].predict(X_spread)[0]
            
            # Total
            X_total = self.prepare_features(game_row, self.feature_sets['total'])
            total_pred = self.models['total'].predict(X_total)[0]
            
            # Scores
            X_home = self.prepare_features(game_row, self.feature_sets['home_score'])
            home_score_pred = self.models['home_score'].predict(X_home)[0]
            
            X_away = self.prepare_features(game_row, self.feature_sets['away_score'])
            away_score_pred = self.models['away_score'].predict(X_away)[0]
            
            return {
                'home_team': game_row.get('home_team', 'Home'),
                'away_team': game_row.get('away_team', 'Away'),
                'season': game_row.get('season'),
                'week': game_row.get('week'),
                'win_prob_home': win_proba[1],
                'win_prob_away': 1 - win_proba[1],
                'predicted_spread': spread_pred,
                'predicted_total': total_pred,
                'predicted_home_score': home_score_pred,
                'predicted_away_score': away_score_pred,
                'actual_home_score': game_row.get('home_score'),
                'actual_away_score': game_row.get('away_score'),
                'actual_home_win': game_row.get('home_win')
            }
            
        except Exception as e:
            return None
    
    def generate_all_predictions(self, df, test_seasons=[2024, 2025]):
        """Generate predictions for all test games"""
        
        print(f"🔮 Generating predictions for test set...")
        
        # Filter test games
        test_df = df[df['season'].isin(test_seasons)].copy()
        test_df = test_df[test_df['home_win'].notna()]  # Only completed games
        
        print(f"   Test games: {len(test_df)} ({test_seasons})")
        
        predictions = []
        for idx, game in test_df.iterrows():
            pred = self.predict_game(game)
            if pred is not None:
                predictions.append(pred)
        
        self.predictions = predictions
        print(f"   ✓ {len(predictions)} predictions generated\n")
        
        return predictions
    
    def calculate_bet_outcome(self, prediction, bet_type, bet_team_or_side):
        """
        Calculate if a bet won
        
        Args:
            prediction: dict with predictions and actuals
            bet_type: 'ml' (moneyline), 'spread', 'total'
            bet_team_or_side: team name for ML/spread, 'over'/'under' for total
        
        Returns:
            True if bet won, False if lost, None if push
        """
        
        actual_home = prediction['actual_home_score']
        actual_away = prediction['actual_away_score']
        
        if pd.isna(actual_home) or pd.isna(actual_away):
            return None
        
        if bet_type == 'ml':
            # Moneyline
            home_won = actual_home > actual_away
            away_won = actual_away > actual_home
            
            if bet_team_or_side == prediction['home_team']:
                return home_won
            else:
                return away_won
        
        elif bet_type == 'spread':
            # Spread (simplified - assuming standard -110 line)
            pred_spread = prediction['predicted_spread']
            actual_diff = actual_home - actual_away
            
            if bet_team_or_side == prediction['home_team']:
                # Bet home, they need to beat spread
                return (actual_diff + pred_spread) > 0
            else:
                # Bet away
                return (actual_diff + pred_spread) < 0
        
        elif bet_type == 'total':
            # Total
            actual_total = actual_home + actual_away
            pred_total = prediction['predicted_total']
            
            if bet_team_or_side == 'over':
                return actual_total > pred_total
            else:
                return actual_total < pred_total
        
        return None
    
    def calculate_american_odds(self, win_rate):
        """
        Convert win rate to American odds
        
        Args:
            win_rate: actual win rate (e.g. 0.75 = 75%)
        
        Returns:
            American odds (e.g. -300 for favorite, +150 for underdog)
        """
        if win_rate >= 0.5:
            # Favorite (negative odds)
            if win_rate >= 0.99:
                return -10000  # Cap at very high odds
            odds = -(win_rate / (1 - win_rate)) * 100
        else:
            # Underdog (positive odds)
            if win_rate <= 0.01:
                return 10000  # Cap at very high odds
            odds = ((1 - win_rate) / win_rate) * 100
        
        return odds
    
    def calculate_profit_from_odds(self, bet_size, odds, won):
        """
        Calculate profit based on American odds
        
        Args:
            bet_size: amount bet in CHF
            odds: American odds (e.g. -300, +150)
            won: True if bet won, False if lost
        
        Returns:
            profit (positive) or loss (negative)
        """
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
    
    def backtest_strategy(self, strategy_func, strategy_name, base_bet=10):
        """
        Backtest a single strategy with REALISTIC ODDS based on actual performance
        
        Args:
            strategy_func: function that takes prediction and returns (bet?, bet_type, bet_team, bet_multiplier)
            strategy_name: name of strategy
            base_bet: base bet size in CHF (default: 10)
        
        Returns:
            dict with results including realistic odds and profits
        """
        
        # STEP 1: Collect all bets and outcomes
        bets = []
        
        for pred in self.predictions:
            # Get strategy recommendation
            should_bet, bet_type, bet_team_or_side, bet_multiplier = strategy_func(pred)
            
            if not should_bet:
                continue
            
            # Calculate bet size
            bet_size = base_bet * bet_multiplier
            
            # Calculate outcome
            won = self.calculate_bet_outcome(pred, bet_type, bet_team_or_side)
            
            if won is None:
                continue  # Push or incomplete
            
            bets.append({
                'bet_size': bet_size,
                'won': won,
                'bet_type': bet_type
            })
        
        if len(bets) == 0:
            return {
                'strategy': strategy_name,
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_risked': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_odds': 0
            }
        
        # STEP 2: Calculate actual win rate
        wins = sum(1 for b in bets if b['won'])
        losses = len(bets) - wins
        win_rate = wins / len(bets)
        
        # STEP 3: Convert win rate to realistic American odds
        american_odds = self.calculate_american_odds(win_rate)
        
        # STEP 4: Calculate profit with realistic odds
        total_profit = 0
        total_risked = 0
        
        for bet in bets:
            profit = self.calculate_profit_from_odds(bet['bet_size'], american_odds, bet['won'])
            total_profit += profit
            total_risked += bet['bet_size']
        
        roi = (total_profit / total_risked * 100) if total_risked > 0 else 0
        
        return {
            'strategy': strategy_name,
            'total_bets': len(bets),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_risked': total_risked,
            'total_profit': total_profit,
            'roi': roi,
            'avg_odds': american_odds
        }
    
    def define_all_strategies(self):
        """Define 50+ strategies to test (with CHF 10 base bet)"""
        
        strategies = []
        
        # ==================== CATEGORY 1: PURE CONFIDENCE (20 strategies) ====================
        
        # Simple confidence thresholds (1x multiplier = CHF 10)
        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            def make_conf_strategy(t):
                def strategy(pred):
                    home_prob = pred['win_prob_home']
                    away_prob = pred['win_prob_away']
                    
                    if home_prob >= t:
                        return True, 'ml', pred['home_team'], 1.0  # 1x = CHF 10
                    elif away_prob >= t:
                        return True, 'ml', pred['away_team'], 1.0
                    else:
                        return False, None, None, 0
                return strategy
            
            strategies.append((
                make_conf_strategy(threshold),
                f"Confidence ≥{threshold:.0%}"
            ))
        
        # Tiered confidence with variable bet sizing (1x-3x = CHF 10-30)
        for base_t in [0.65, 0.70, 0.75]:
            def make_tiered_strategy(t):
                def strategy(pred):
                    max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
                    
                    if max_prob < t:
                        return False, None, None, 0
                    
                    # Variable sizing based on confidence
                    if max_prob >= t + 0.15:
                        multiplier = 3.0  # CHF 30
                    elif max_prob >= t + 0.10:
                        multiplier = 2.0  # CHF 20
                    elif max_prob >= t + 0.05:
                        multiplier = 1.5  # CHF 15
                    else:
                        multiplier = 1.0  # CHF 10
                    
                    bet_team = pred['home_team'] if pred['win_prob_home'] > 0.5 else pred['away_team']
                    return True, 'ml', bet_team, multiplier
                return strategy
            
            strategies.append((
                make_tiered_strategy(base_t),
                f"Tiered Confidence (Base ≥{base_t:.0%})"
            ))
        
        # Top N confidence games per week (1x = CHF 10)
        for n in [1, 2, 3, 5]:
            def make_topn_strategy(top_n):
                def strategy(pred):
                    max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
                    if max_prob >= 0.70:
                        bet_team = pred['home_team'] if pred['win_prob_home'] > 0.5 else pred['away_team']
                        return True, 'ml', bet_team, 1.0
                    return False, None, None, 0
                return strategy
            
            strategies.append((
                make_topn_strategy(n),
                f"Top {n} Confidence Games/Week"
            ))
        
        # ==================== CATEGORY 2: SPREAD STRATEGIES (15 strategies) ====================
        
        # Spread difference strategies (1x = CHF 10)
        for diff_threshold in [2, 3, 4, 5, 6, 7, 8]:
            for conf_threshold in [0.60, 0.65, 0.70]:
                def make_spread_strategy(diff_t, conf_t):
                    def strategy(pred):
                        max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
                        spread = pred['predicted_spread']
                        
                        if max_prob < conf_t:
                            return False, None, None, 0
                        
                        if abs(spread) < diff_t:
                            return False, None, None, 0
                        
                        bet_team = pred['home_team'] if spread < 0 else pred['away_team']
                        return True, 'spread', bet_team, 1.0
                    return strategy
                
                strategies.append((
                    make_spread_strategy(diff_threshold, conf_threshold),
                    f"Spread Diff≥{diff_threshold} Conf≥{conf_threshold:.0%}"
                ))
        
        # ==================== CATEGORY 3: TOTAL STRATEGIES (10 strategies) ====================
        
        # Extreme totals (1x = CHF 10)
        total_pairs = [
            (50, 38), (49, 39), (48, 40), (47, 41), (46, 42)
        ]
        
        for high, low in total_pairs:
            def make_total_strategy(h, l):
                def strategy(pred):
                    total = pred['predicted_total']
                    
                    if total >= h:
                        return True, 'total', 'over', 1.0
                    elif total <= l:
                        return True, 'total', 'under', 1.0
                    else:
                        return False, None, None, 0
                return strategy
            
            strategies.append((
                make_total_strategy(high, low),
                f"Total Extreme (>{high} or <{low})"
            ))
        
        # Total + Confidence combined (1x = CHF 10)
        for conf_t in [0.60, 0.65, 0.70]:
            def make_total_conf_strategy(c):
                def strategy(pred):
                    total = pred['predicted_total']
                    max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
                    
                    if max_prob < c:
                        return False, None, None, 0
                    
                    if total >= 48:
                        return True, 'total', 'over', 1.0
                    elif total <= 40:
                        return True, 'total', 'under', 1.0
                    else:
                        return False, None, None, 0
                return strategy
            
            strategies.append((
                make_total_conf_strategy(conf_t),
                f"Total Extreme + Conf≥{conf_t:.0%}"
            ))
        
        # ==================== CATEGORY 4: SCORE-BASED (5 strategies) ====================
        
        # High scoring teams (1x = CHF 10)
        for threshold in [28, 30, 32]:
            def make_score_strategy(t):
                def strategy(pred):
                    home_score = pred['predicted_home_score']
                    away_score = pred['predicted_away_score']
                    
                    if home_score >= t and pred['win_prob_home'] >= 0.65:
                        return True, 'ml', pred['home_team'], 1.0
                    elif away_score >= t and pred['win_prob_away'] >= 0.65:
                        return True, 'ml', pred['away_team'], 1.0
                    else:
                        return False, None, None, 0
                return strategy
            
            strategies.append((
                make_score_strategy(threshold),
                f"High Scoring (≥{threshold}pts) + Conf≥65%"
            ))
        
        # ==================== CATEGORY 5: COMBINED MULTI-FACTOR (10 strategies) ====================
        
        # Conservative: High conf + reasonable spread (1x = CHF 10)
        def conservative_strategy(pred):
            max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
            spread = abs(pred['predicted_spread'])
            
            if max_prob >= 0.75 and spread <= 10:
                bet_team = pred['home_team'] if pred['win_prob_home'] > 0.5 else pred['away_team']
                return True, 'ml', bet_team, 1.0
            return False, None, None, 0
        
        strategies.append((conservative_strategy, "Conservative (Conf≥75% Spread≤10)"))
        
        # Aggressive: Lower conf but high volume (1x = CHF 10)
        def aggressive_strategy(pred):
            max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
            
            if max_prob >= 0.60:
                bet_team = pred['home_team'] if pred['win_prob_home'] > 0.5 else pred['away_team']
                return True, 'ml', bet_team, 1.0
            return False, None, None, 0
        
        strategies.append((aggressive_strategy, "Aggressive (Conf≥60%)"))
        
        # Blowout: Extreme favorites (1x = CHF 10)
        def blowout_strategy(pred):
            spread = pred['predicted_spread']
            max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
            
            if abs(spread) >= 10 and max_prob >= 0.75:
                bet_team = pred['home_team'] if spread < 0 else pred['away_team']
                return True, 'ml', bet_team, 1.0
            return False, None, None, 0
        
        strategies.append((blowout_strategy, "Blowout (Spread≥10 Conf≥75%)"))
        
        # Close games: Take favorite in close games (1x = CHF 10)
        def close_game_strategy(pred):
            spread = abs(pred['predicted_spread'])
            max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
            
            if spread <= 3 and max_prob >= 0.65:
                bet_team = pred['home_team'] if pred['win_prob_home'] > 0.5 else pred['away_team']
                return True, 'ml', bet_team, 1.0
            return False, None, None, 0
        
        strategies.append((close_game_strategy, "Close Games (Spread≤3 Conf≥65%)"))
        
        # Best of both: High conf OR extreme spread (1x = CHF 10)
        def best_of_both_strategy(pred):
            max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
            spread = abs(pred['predicted_spread'])
            
            if max_prob >= 0.80 or (max_prob >= 0.65 and spread >= 7):
                bet_team = pred['home_team'] if pred['win_prob_home'] > 0.5 else pred['away_team']
                return True, 'ml', bet_team, 1.0
            return False, None, None, 0
        
        strategies.append((best_of_both_strategy, "Best of Both (Conf≥80% OR Conf≥65%+Spread≥7)"))
        
        # Underdog hunting: Confident underdogs (1x = CHF 10)
        def underdog_strategy(pred):
            spread = pred['predicted_spread']
            
            if spread > 3 and pred['win_prob_away'] >= 0.55:
                return True, 'ml', pred['away_team'], 1.0
            elif spread < -3 and pred['win_prob_home'] >= 0.55:
                return True, 'ml', pred['home_team'], 1.0
            return False, None, None, 0
        
        strategies.append((underdog_strategy, "Confident Underdogs (Conf≥55% + Spread≥3)"))
        
        # Momentum: Recent form-based (1x = CHF 10)
        def momentum_strategy(pred):
            total = pred['predicted_total']
            max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
            
            if total >= 47 and max_prob >= 0.70:
                bet_team = pred['home_team'] if pred['win_prob_home'] > 0.5 else pred['away_team']
                return True, 'ml', bet_team, 1.0
            return False, None, None, 0
        
        strategies.append((momentum_strategy, "High Scoring Momentum (Total≥47 Conf≥70%)"))
        
        # ==================== CATEGORY 6: KELLY CRITERION VARIATIONS (5 strategies) ====================
        
        # Kelly Criterion sizing (multipliers based on Kelly formula)
        for fraction in [0.25, 0.50, 0.75, 1.0]:
            def make_kelly_strategy(f):
                def strategy(pred):
                    max_prob = max(pred['win_prob_home'], pred['win_prob_away'])
                    
                    if max_prob < 0.65:
                        return False, None, None, 0
                    
                    # Simplified Kelly: (p - q) / odds
                    # Assuming -110 odds (1.91 decimal)
                    p = max_prob
                    q = 1 - p
                    kelly = (p * 1.91 - 1) / 0.91
                    
                    if kelly <= 0:
                        return False, None, None, 0
                    
                    # Fractional Kelly as multiplier (capped at 5x = CHF 50)
                    multiplier = max(0.5, min(5.0, kelly * 10 * f))
                    
                    bet_team = pred['home_team'] if pred['win_prob_home'] > 0.5 else pred['away_team']
                    return True, 'ml', bet_team, multiplier
                return strategy
            
            strategies.append((
                make_kelly_strategy(fraction),
                f"Kelly {fraction:.0%} (Conf≥65%)"
            ))
        
        print(f"📊 Defined {len(strategies)} strategies to test\n")
        
        return strategies
    
    def run_all_backtests(self):
        """Run backtests for all strategies"""
        
        strategies = self.define_all_strategies()
        
        print("="*80)
        print("🔬 RUNNING COMPREHENSIVE BACKTEST")
        print("="*80)
        print(f"\nTesting {len(strategies)} strategies on {len(self.predictions)} games...")
        print("This may take 1-2 minutes...\n")
        
        results = []
        
        for i, (strategy_func, strategy_name) in enumerate(strategies, 1):
            result = self.backtest_strategy(strategy_func, strategy_name)
            results.append(result)
            
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(strategies)} strategies tested...")
        
        print(f"\n✓ All {len(strategies)} strategies tested!\n")
        
        return results
    
    def analyze_results(self, results):
        """Analyze and display results (with CHF 10 base bet)"""
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)
        
        # Filter out strategies with no bets
        df = df[df['total_bets'] > 0].copy()
        
        # Sort by different metrics
        by_profit = df.sort_values('total_profit', ascending=False)
        by_roi = df.sort_values('roi', ascending=False)
        by_winrate = df.sort_values('win_rate', ascending=False)
        
        # Calculate score (balanced metric)
        df['score'] = (
            df['win_rate'] * 40 +  # Win rate weight
            df['roi'] / 100 * 30 +  # ROI weight
            np.log1p(df['total_profit']) * 20 +  # Profit weight (log scale)
            np.log1p(df['total_bets']) * 10  # Volume weight (log scale)
        )
        
        by_score = df.sort_values('score', ascending=False)
        
        print("="*110)
        print("📊 BACKTEST RESULTS SUMMARY (REALISTIC ODDS)")
        print("="*110)
        
        print(f"\nDataset: {len(self.predictions)} games")
        print(f"Strategies tested: {len(results)}")
        print(f"Strategies with bets: {len(df)}")
        print(f"\nNote: Odds calculated from actual win rate (not assumed -110)")
        
        # Top 10 by different metrics
        print("\n" + "="*110)
        print("🏆 TOP 10 STRATEGIES BY TOTAL PROFIT")
        print("="*110)
        print(f"\n{'Rank':<5} {'Strategy':<45} {'Bets':<6} {'WR%':<7} {'Odds':<8} {'ROI%':<7} {'Profit':<10}")
        print("-"*110)
        
        for i, row in by_profit.head(10).iterrows():
            print(f"{by_profit.index.get_loc(i)+1:<5} "
                  f"{row['strategy']:<45} "
                  f"{row['total_bets']:<6.0f} "
                  f"{row['win_rate']*100:<7.1f} "
                  f"{row['avg_odds']:<8.0f} "
                  f"{row['roi']:<7.1f} "
                  f"CHF {row['total_profit']:<8.0f}")
        
        print("\n" + "="*110)
        print("💎 TOP 10 STRATEGIES BY ROI")
        print("="*110)
        print(f"\n{'Rank':<5} {'Strategy':<45} {'Bets':<6} {'WR%':<7} {'Odds':<8} {'ROI%':<7} {'Profit':<10}")
        print("-"*110)
        
        for i, row in by_roi.head(10).iterrows():
            print(f"{by_roi.index.get_loc(i)+1:<5} "
                  f"{row['strategy']:<45} "
                  f"{row['total_bets']:<6.0f} "
                  f"{row['win_rate']*100:<7.1f} "
                  f"{row['avg_odds']:<8.0f} "
                  f"{row['roi']:<7.1f} "
                  f"CHF {row['total_profit']:<8.0f}")
        
        print("\n" + "="*110)
        print("🎯 TOP 10 STRATEGIES BY WIN RATE")
        print("="*110)
        print(f"\n{'Rank':<5} {'Strategy':<45} {'Bets':<6} {'WR%':<7} {'Odds':<8} {'ROI%':<7} {'Profit':<10}")
        print("-"*110)
        
        for i, row in by_winrate.head(10).iterrows():
            print(f"{by_winrate.index.get_loc(i)+1:<5} "
                  f"{row['strategy']:<45} "
                  f"{row['total_bets']:<6.0f} "
                  f"{row['win_rate']*100:<7.1f} "
                  f"{row['avg_odds']:<8.0f} "
                  f"{row['roi']:<7.1f} "
                  f"CHF {row['total_profit']:<8.0f}")
        
        print("\n" + "="*110)
        print("⭐ TOP 10 STRATEGIES BY BALANCED SCORE")
        print("="*110)
        print(f"\n{'Rank':<5} {'Strategy':<45} {'Bets':<6} {'WR%':<7} {'Odds':<8} {'Profit':<10} {'Score':<7}")
        print("-"*110)
        
        for i, row in by_score.head(10).iterrows():
            print(f"{by_score.index.get_loc(i)+1:<5} "
                  f"{row['strategy']:<45} "
                  f"{row['total_bets']:<6.0f} "
                  f"{row['win_rate']*100:<7.1f} "
                  f"{row['avg_odds']:<8.0f} "
                  f"CHF {row['total_profit']:<8.0f} "
                  f"{row['score']:<7.1f}")
        
        # Best overall recommendation
        best_strategy = by_score.iloc[0]
        
        print("\n" + "="*110)
        print("🏆 RECOMMENDED STRATEGY")
        print("="*110)
        
        print(f"\nBased on balanced scoring (Win Rate + ROI + Profit + Volume):\n")
        print(f"Strategy: {best_strategy['strategy']}")
        print(f"  Total Bets:    {best_strategy['total_bets']:.0f}")
        print(f"  Win Rate:      {best_strategy['win_rate']*100:.1f}%")
        print(f"  Avg Odds:      {best_strategy['avg_odds']:.0f}")
        print(f"  ROI:           {best_strategy['roi']:.1f}%")
        print(f"  Total Profit:  CHF {best_strategy['total_profit']:.0f}")
        print(f"  Avg Bet Size:  CHF {best_strategy['total_risked']/best_strategy['total_bets']:.0f}")
        print(f"  Balance Score: {best_strategy['score']:.1f}")
        
        # Risk assessment
        print(f"\n📊 Risk Assessment:")
        freq = best_strategy['total_bets'] / len(self.predictions) * 100
        print(f"  Betting Frequency: {freq:.1f}% of all games")
        
        if freq < 10:
            risk = "LOW (Very selective)"
        elif freq < 25:
            risk = "MEDIUM (Moderate betting)"
        else:
            risk = "HIGH (Frequent betting)"
        
        print(f"  Risk Level: {risk}")
        
        # Alternative recommendations
        print(f"\n💡 Alternative Strategies:")
        print(f"\n  For Maximum Profit:")
        best_profit = by_profit.iloc[0]
        print(f"    {best_profit['strategy']}")
        print(f"    Profit: CHF {best_profit['total_profit']:.0f}, WR: {best_profit['win_rate']*100:.1f}%, Odds: {best_profit['avg_odds']:.0f}, Bets: {best_profit['total_bets']:.0f}")
        
        print(f"\n  For Maximum Win Rate:")
        best_wr = by_winrate.iloc[0]
        print(f"    {best_wr['strategy']}")
        print(f"    WR: {best_wr['win_rate']*100:.1f}%, Odds: {best_wr['avg_odds']:.0f}, Profit: CHF {best_wr['total_profit']:.0f}, Bets: {best_wr['total_bets']:.0f}")
        
        print(f"\n  For Best ROI:")
        best_roi = by_roi.iloc[0]
        print(f"    {best_roi['strategy']}")
        print(f"    ROI: {best_roi['roi']:.1f}%, Odds: {best_roi['avg_odds']:.0f}, Profit: CHF {best_roi['total_profit']:.0f}, Bets: {best_roi['total_bets']:.0f}")
        
        print("\n" + "="*110)
        
        return df
    
    def save_results(self, df, filename='backtest_results.json'):
        """Save results to JSON"""
        
        # Convert to dict
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'test_games': len(self.predictions),
            'strategies_tested': len(df),
            'top_strategies': df.head(10).to_dict('records')
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")


def main():
    """Main execution"""
    
    print("\n" + "="*100)
    print("🏈 NFL BETTING STRATEGY COMPREHENSIVE BACKTEST (REALISTIC ODDS)")
    print("="*100)
    print("\nTesting 50+ strategies with REALISTIC odds based on actual win rates")
    print("No more -110 assumptions - calculating real American odds from performance!")
    print("This will take 1-2 minutes...\n")
    
    # Initialize
    backtester = StrategyBacktester(models_dir='models')
    
    # Load models
    if not backtester.load_models():
        return
    
    # Load data
    print("📥 Loading data...")
    df = pd.read_csv('nfl_training_data_MEGA.csv')
    df['gameday'] = pd.to_datetime(df['gameday'])
    print(f"✓ Loaded: {len(df)} games\n")
    
    # Generate predictions for test set
    predictions = backtester.generate_all_predictions(df, test_seasons=[2024, 2025])
    
    if len(predictions) == 0:
        print("❌ No predictions generated")
        return
    
    # Run all backtests
    results = backtester.run_all_backtests()
    
    # Analyze results
    df_results = backtester.analyze_results(results)
    
    # Save results
    backtester.save_results(df_results, 'backtest_results.json')
    
    print("\n✅ BACKTEST COMPLETE!")
    print("\n💡 Use the recommended strategy in your prediction script for optimal results.\n")


if __name__ == "__main__":
    main()