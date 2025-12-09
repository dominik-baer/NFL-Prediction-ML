"""
NFL Prediction - Betting Model (WITH Vegas Lines)
Für Sportwetten: Nutzt alle verfügbaren Informationen inkl. Vegas
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle


class NFLBettingPredictor:
    """Betting Model - MIT Vegas Lines für maximale Accuracy"""
    
    def __init__(self):
        self.win_model = None
        self.spread_model = None
        self.total_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """Features MIT Vegas Lines"""
        
        # Alle Features inkl. Vegas
        exclude_cols = [
            'home_win', 'season', 'week', 'gameday', 'game_id',
            'home_team', 'away_team', 'home_score', 'away_score', 
            'result', 'total_points', 'point_diff',
            'season_home', 'week_home', 'season_away', 'week_away',
            'points_scored_home', 'points_allowed_home', 'won_home', 'is_home_home',
            'points_scored_away', 'points_allowed_away', 'won_away', 'is_home_away'
        ]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Sicherstellen dass Vegas Lines dabei sind
        vegas_features = [c for c in feature_cols if 'spread_line' in c or 'total_line' in c or 'vegas' in c]
        print(f"\n📊 Features MIT Vegas: {len(feature_cols)}")
        print(f"🎰 Vegas Features: {vegas_features}")
        
        self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        
        y_win = df['home_win'].copy()
        y_spread = df['result'].copy()
        y_total = (df['home_score'] + df['away_score']).copy()
        
        return X, y_win, y_spread, y_total
    
    def train_all_models(self, X_train, y_win, y_spread, y_total):
        """Trainiert Betting Models"""
        
        print("\n" + "="*70)
        print("🎰 TRAINING BETTING MODELS (With Vegas Lines)")
        print("="*70)
        
        # Win/Loss - Random Forest
        print("\n1️⃣ Win/Loss Model...")
        self.win_model = RandomForestClassifier(
            n_estimators=300, max_depth=15, 
            min_samples_split=15, random_state=42, n_jobs=-1
        )
        self.win_model.fit(X_train, y_win)
        
        # Spread - Random Forest
        print("2️⃣ Spread Model...")
        self.spread_model = RandomForestRegressor(
            n_estimators=300, max_depth=15,
            min_samples_split=15, random_state=42, n_jobs=-1
        )
        self.spread_model.fit(X_train, y_spread)
        
        # Over/Under - Random Forest
        print("3️⃣ Over/Under Model...")
        self.total_model = RandomForestRegressor(
            n_estimators=300, max_depth=15,
            min_samples_split=15, random_state=42, n_jobs=-1
        )
        self.total_model.fit(X_train, y_total)
        
        print("\n✅ Betting models trained!")
        
        # Feature Importance für Win/Loss
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.win_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 wichtigste Features:")
        print(importance_df.head(10).to_string(index=False))
    
    def save_models(self, prefix='betting'):
        """Speichert Betting Models"""
        
        import os
        os.makedirs('models', exist_ok=True)
        
        print(f"\n💾 Speichere Betting Modelle...")
        
        with open(f'models/{prefix}_win.pkl', 'wb') as f:
            pickle.dump(self.win_model, f)
        
        with open(f'models/{prefix}_spread.pkl', 'wb') as f:
            pickle.dump(self.spread_model, f)
        
        with open(f'models/{prefix}_total.pkl', 'wb') as f:
            pickle.dump(self.total_model, f)
        
        with open(f'models/{prefix}_features.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"✓ Modelle gespeichert!")


def main():
    """Hauptfunktion"""
    
    print("\n" + "="*70)
    print("🎰 NFL BETTING MODEL (With Vegas Lines)")
    print("="*70 + "\n")
    
    # Daten laden
    print("Lade Daten...")
    try:
        df = pd.read_csv('nfl_training_data_ultimate.csv')
        print("✓ Ultimate Data geladen")
    except:
        try:
            df = pd.read_csv('nfl_training_data_engineered.csv')
            print("✓ Engineered Data geladen")
        except:
            df = pd.read_csv('nfl_training_data.csv')
            print("✓ Base Data geladen")
    
    df['gameday'] = pd.to_datetime(df['gameday'])
    
    # Predictor
    predictor = NFLBettingPredictor()
    
    # Train/Test Split - NUR completed games!
    df = df.sort_values('gameday')
    unique_seasons = sorted(df['season'].unique())
    test_season = unique_seasons[-1]
    
    # Filter: Nur Spiele mit Scores (completed)
    train_df = df[(df['season'] < test_season) & (df['home_score'].notna())]
    test_df = df[(df['season'] == test_season) & (df['home_score'].notna())]
    
    print(f"Training: {train_df['season'].min()}-{train_df['season'].max()} ({len(train_df)} games)")
    print(f"Test: {test_season} ({len(test_df)} games - completed only)")
    
    # Features
    X_train, y_win_train, y_spread_train, y_total_train = predictor.prepare_features(train_df)
    X_test, y_win_test, y_spread_test, y_total_test = predictor.prepare_features(test_df)
    
    print(f"\nTraining Set: {X_train.shape}")
    print(f"Test Set: {X_test.shape}")
    
    # Training
    predictor.train_all_models(X_train, y_win_train, y_spread_train, y_total_train)
    
    # Quick Evaluation
    print("\n" + "="*70)
    print("📊 QUICK EVALUATION")
    print("="*70)
    
    from sklearn.metrics import accuracy_score, mean_absolute_error
    
    win_pred = predictor.win_model.predict(X_test)
    win_acc = accuracy_score(y_win_test, win_pred)
    
    spread_pred = predictor.spread_model.predict(X_test)
    spread_mae = mean_absolute_error(y_spread_test, spread_pred)
    
    total_pred = predictor.total_model.predict(X_test)
    total_mae = mean_absolute_error(y_total_test, total_pred)
    
    print(f"\n🎯 Win/Loss Accuracy: {win_acc:.1%}")
    print(f"📏 Spread MAE: {spread_mae:.2f} points")
    print(f"🎲 Total MAE: {total_mae:.2f} points")
    
    # Speichern
    predictor.save_models('betting')
    
    print("\n" + "="*70)
    print("🏆 BETTING MODEL READY!")
    print("="*70)
    print("\n💡 Next: python src/prediction.py")
    print("   → Make predictions for upcoming games!\n")


if __name__ == "__main__":
    main()