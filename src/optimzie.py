"""
Threshold Optimizer - Findet besten Spread Agreement Wert
Testet verschiedene Thresholds für Ultra Conservative Strategy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🔬 ULTRA CONSERVATIVE OPTIMIZER")
print("="*70)
print("\nOptimiert BEIDE Parameter:")
print("  1. Win Probability Threshold")
print("  2. Spread Agreement Threshold")
print("\nZiel: Maximaler ROI\n")

# Lade Daten
print("📥 Lade Daten...")
df = pd.read_csv('nfl_training_data_ultimate.csv')
print(f"✓ {len(df)} Spiele geladen\n")

df['gameday'] = pd.to_datetime(df['gameday'])

# Features
exclude_cols = [
    'home_win', 'season', 'week', 'gameday', 'game_id',
    'home_team', 'away_team', 'home_score', 'away_score', 
    'result', 'total_points', 'point_diff',
    'season_home', 'week_home', 'season_away', 'week_away',
    'points_scored_home', 'points_allowed_home', 'won_home', 'is_home_home',
    'points_scored_away', 'points_allowed_away', 'won_away', 'is_home_away'
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

print(f"📊 Features: {len(feature_cols)}\n")

# Test years
test_years = [2020, 2021, 2022, 2023, 2024]

# Parameters to test
win_prob_thresholds = [0.70, 0.72, 0.75, 0.77, 0.80, 0.82, 0.85]
spread_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

print("="*70)
print("🧪 GRID SEARCH")
print("="*70)
print(f"\nWin Prob Thresholds: {win_prob_thresholds}")
print(f"Spread Thresholds: {spread_thresholds}")
print(f"Total Combinations: {len(win_prob_thresholds) * len(spread_thresholds)}\n")

results = []
total_combos = len(win_prob_thresholds) * len(spread_thresholds)
current = 0

for win_threshold in win_prob_thresholds:
    for spread_threshold in spread_thresholds:
        current += 1
        print(f"[{current}/{total_combos}] Testing: Win≥{win_threshold:.0%}, Spread≤{spread_threshold}pts...", end='')
        
        all_bets = []
        
        for test_year in test_years:
            # Skip incomplete years
            test_df_check = df[df['season'] == test_year]
            completed = test_df_check['home_score'].notna().sum()
            
            if completed < 100:
                continue
            
            train_df = df[(df['season'] < test_year) & (df['home_score'].notna())]
            test_df = df[(df['season'] == test_year) & (df['home_score'].notna())]
            
            if len(test_df) == 0:
                continue
            
            # Train
            X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())
            X_test = test_df[feature_cols].fillna(train_df[feature_cols].mean())
            
            y_win_train = train_df['home_win']
            y_spread_train = train_df['result']
            
            # Models
            win_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
            win_model.fit(X_train, y_win_train)
            
            spread_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
            spread_model.fit(X_train, y_spread_train)
            
            # Predict
            win_proba = win_model.predict_proba(X_test)[:, 1]
            spread_pred = spread_model.predict(X_test)
            
            # Actuals
            actual_home_win = test_df['home_win'].values
            vegas_spread = test_df['spread_line'].values
            
            # Find bets
            for i in range(len(test_df)):
                our_win_prob = win_proba[i]
                our_spread = spread_pred[i]
                vegas_line = vegas_spread[i]
                actual_win = actual_home_win[i]
                
                if pd.isna(vegas_line):
                    continue
                
                spread_agreement = abs(our_spread - vegas_line)
                
                # Test THIS combination
                if our_win_prob >= win_threshold and spread_agreement <= spread_threshold:
                    all_bets.append({
                        'year': test_year,
                        'won': actual_win == 1
                    })
                elif our_win_prob <= (1 - win_threshold) and spread_agreement <= spread_threshold:
                    all_bets.append({
                        'year': test_year,
                        'won': actual_win == 0
                    })
        
        # Calculate stats
        if len(all_bets) > 0:
            bets_df = pd.DataFrame(all_bets)
            
            total_bets = len(bets_df)
            wins = bets_df['won'].sum()
            win_rate = wins / total_bets
            
            # ROI (assuming -150 favorite odds)
            profit = (wins * 0.67) - (total_bets - wins)
            roi = (profit / total_bets) * 100
            
            results.append({
                'win_threshold': win_threshold,
                'spread_threshold': spread_threshold,
                'total_bets': total_bets,
                'wins': wins,
                'win_rate': win_rate,
                'roi': roi
            })
            
            print(f" {total_bets} bets, {win_rate:.1%} WR, {roi:+.1f}% ROI")
        else:
            print(f" No bets")

# Display results
print("\n" + "="*70)
print("📊 RESULTS - TOP 10 COMBINATIONS")
print("="*70 + "\n")

if len(results) > 0:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roi', ascending=False)
    
    print(f"{'Win≥':<8} {'Spread≤':<10} {'Bets':<8} {'Wins':<8} {'Win Rate':<12} {'ROI':<10}")
    print("─"*70)
    
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        win_t = row['win_threshold']
        spread_t = row['spread_threshold']
        bets = row['total_bets']
        wins = row['wins']
        wr = row['win_rate']
        roi = row['roi']
        
        marker = "🏆" if i == 1 else f"{i:2d}."
        
        print(f"{marker} {win_t:<6.0%} {spread_t:<9.1f}  {bets:<8} {wins:<8} {wr:<11.1%} {roi:>+9.1f}%")
    
    # Best combination
    best = results_df.iloc[0]
    
    print("\n" + "="*70)
    print("🏆 OPTIMAL COMBINATION")
    print("="*70)
    print(f"\n   Win Probability: ≥ {best['win_threshold']:.0%}")
    print(f"   Spread Agreement: ≤ {best['spread_threshold']} pts")
    print(f"\n   Win Rate: {best['win_rate']:.1%}")
    print(f"   ROI: {best['roi']:+.1f}%")
    print(f"   Total Bets: {int(best['total_bets'])}")
    print(f"   Bets per Year: ~{int(best['total_bets'] / len(test_years))}")
    
    # Current vs Optimal
    current_combo = results_df[
        (results_df['win_threshold'] == 0.75) & 
        (results_df['spread_threshold'] == 3.0)
    ]
    
    if len(current_combo) > 0:
        current = current_combo.iloc[0]
        improvement = best['roi'] - current['roi']
        
        print("\n💡 COMPARISON:")
        print(f"   Current (75%, 3.0pts): {current['roi']:+.1f}% ROI")
        print(f"   Optimal ({best['win_threshold']:.0%}, {best['spread_threshold']}pts): {best['roi']:+.1f}% ROI")
        
        if improvement > 0.5:
            print(f"\n   ✅ IMPROVEMENT: {improvement:+.1f}% ROI!")
            print(f"   → Update thresholds in prediction script")
        else:
            print(f"\n   ✅ Current settings already near-optimal!")
    
    # Save all results
    results_df.to_csv('threshold_optimization_full.csv', index=False)
    print(f"\n💾 Saved: threshold_optimization_full.csv")
    
    # Show distribution
    print("\n📊 ROI DISTRIBUTION:")
    print(f"   Best: {results_df['roi'].max():+.1f}%")
    print(f"   Median: {results_df['roi'].median():+.1f}%")
    print(f"   Worst: {results_df['roi'].min():+.1f}%")

else:
    print("❌ No results found!")

print("\n" + "="*70 + "\n")