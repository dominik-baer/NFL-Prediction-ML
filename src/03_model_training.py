"""
NFL Prediction - Academic Model (WITHOUT Vegas Lines)
Für Uni-Projekt: Zeigt echtes ML-Learning ohne externe Predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, classification_report,
                              mean_absolute_error, mean_squared_error, r2_score)
import pickle


class NFLAcademicPredictor:
    """Academic ML Model - OHNE Vegas Lines"""
    
    def __init__(self):
        self.win_models = {}
        self.spread_models = {}
        self.total_models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """Features OHNE Vegas Lines"""
        
        # EXPLIZIT Vegas Lines ausschließen
        exclude_cols = [
            'home_win', 'season', 'week', 'gameday', 'game_id',
            'home_team', 'away_team', 'home_score', 'away_score', 
            'result', 'total_points', 'point_diff',
            'season_home', 'week_home', 'season_away', 'week_away',
            'points_scored_home', 'points_allowed_home', 'won_home', 'is_home_home',
            'points_scored_away', 'points_allowed_away', 'won_away', 'is_home_away',
            # WICHTIG: Vegas Lines raus!
            'spread_line', 'total_line', 
            'vegas_deviation_spread', 'vegas_deviation_total', 'vegas_confidence_home'
        ]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"\n📊 Features OHNE Vegas Lines: {len(feature_cols)}")
        print("Top Features:")
        for col in feature_cols[:15]:
            print(f"   - {col}")
        if len(feature_cols) > 15:
            print(f"   ... und {len(feature_cols) - 15} weitere")
        
        self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        
        # NaN handling
        if X.isnull().any().any():
            print(f"\n⚠️  Fülle {X.isnull().sum().sum()} NaN-Werte")
            X = X.fillna(X.mean())
        
        # Targets
        y_win = df['home_win'].copy()
        y_spread = df['result'].copy()
        y_total = (df['home_score'] + df['away_score']).copy()
        
        return X, y_win, y_spread, y_total
    
    def train_test_split_temporal(self, df, test_seasons=1):
        """Zeitbasierter Split"""
        df = df.sort_values('gameday').reset_index(drop=True)
        unique_seasons = sorted(df['season'].unique())
        test_seasons_list = unique_seasons[-test_seasons:]
        
        train_df = df[~df['season'].isin(test_seasons_list)]
        test_df = df[df['season'].isin(test_seasons_list)]
        
        print(f"Training: {train_df['season'].min()}-{train_df['season'].max()} ({len(train_df)} games)")
        print(f"Test: {test_df['season'].min()}-{test_df['season'].max()} ({len(test_df)} games)")
        
        return train_df, test_df
    
    def train_all_models(self, X_train, y_win, y_spread, y_total):
        """Trainiert alle Modelle"""
        
        print("\n" + "="*70)
        print("🎓 TRAINING ACADEMIC MODELS (No Vegas Lines)")
        print("="*70)
        
        # Win/Loss
        print("\n1️⃣ Win/Loss Models...")
        X_scaled = self.scaler.fit_transform(X_train)
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y_win)
        self.win_models['logistic'] = lr
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, 
                                     min_samples_split=20, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_win)
        self.win_models['random_forest'] = rf
        
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, 
                                       learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_win)
        self.win_models['xgboost'] = xgb_model
        
        # Spread
        print("2️⃣ Spread Models...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_scaled, y_spread)
        self.spread_models['ridge'] = ridge
        
        rf_spread = RandomForestRegressor(n_estimators=200, max_depth=12,
                                          min_samples_split=20, random_state=42, n_jobs=-1)
        rf_spread.fit(X_train, y_spread)
        self.spread_models['random_forest'] = rf_spread
        
        # Over/Under
        print("3️⃣ Over/Under Models...")
        rf_total = RandomForestRegressor(n_estimators=200, max_depth=12,
                                         min_samples_split=20, random_state=42, n_jobs=-1)
        rf_total.fit(X_train, y_total)
        self.total_models['random_forest'] = rf_total
        
        print("\n✅ All models trained!")
    
    def evaluate(self, X_test, y_win, y_spread, y_total):
        """Evaluiert alle Modelle"""
        
        print("\n" + "="*70)
        print("📊 EVALUATION")
        print("="*70)
        
        # Win/Loss
        print("\n🎯 WIN/LOSS:")
        rf = self.win_models['random_forest']
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_win, y_pred)
        roc = roc_auc_score(y_win, y_pred_proba)
        
        print(f"  Random Forest Accuracy: {acc:.1%}")
        print(f"  ROC-AUC: {roc:.4f}")
        
        # Spread
        print("\n📏 SPREAD:")
        rf_spread = self.spread_models['random_forest']
        y_pred_spread = rf_spread.predict(X_test)
        
        mae = mean_absolute_error(y_spread, y_pred_spread)
        rmse = np.sqrt(mean_squared_error(y_spread, y_pred_spread))
        
        print(f"  MAE: {mae:.2f} points")
        print(f"  RMSE: {rmse:.2f} points")
        
        # Over/Under
        print("\n🎲 OVER/UNDER:")
        rf_total = self.total_models['random_forest']
        y_pred_total = rf_total.predict(X_test)
        
        mae_total = mean_absolute_error(y_total, y_pred_total)
        
        print(f"  MAE: {mae_total:.2f} points")
        
        return {
            'win_accuracy': acc,
            'win_roc_auc': roc,
            'spread_mae': mae,
            'total_mae': mae_total
        }
    
    def save_models(self, prefix='academic'):
        """Speichert trainierte Modelle"""
        
        print(f"\n💾 Speichere Modelle...")
        
        with open(f'models/{prefix}_win_rf.pkl', 'wb') as f:
            pickle.dump(self.win_models['random_forest'], f)
        
        with open(f'models/{prefix}_spread_rf.pkl', 'wb') as f:
            pickle.dump(self.spread_models['random_forest'], f)
        
        with open(f'models/{prefix}_total_rf.pkl', 'wb') as f:
            pickle.dump(self.total_models['random_forest'], f)
        
        with open(f'models/{prefix}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'models/{prefix}_features.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"✓ Modelle gespeichert in models/")


def main():
    """Hauptfunktion"""
    
    print("\n" + "="*70)
    print("🎓 NFL ACADEMIC MODEL (No Vegas Lines)")
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
    
    # Filter nur completed games
    df = df[df['home_score'].notna()].copy()
    print(f"✓ {len(df)} completed games")
    
    # Predictor
    predictor = NFLAcademicPredictor()
    
    # Train/Test Split
    train_df, test_df = predictor.train_test_split_temporal(df, test_seasons=1)
    
    # Features vorbereiten (OHNE Vegas!)
    X_train, y_win_train, y_spread_train, y_total_train = predictor.prepare_features(train_df)
    X_test, y_win_test, y_spread_test, y_total_test = predictor.prepare_features(test_df)
    
    print(f"\nTraining Set: {X_train.shape}")
    print(f"Test Set: {X_test.shape}")
    
    # Training
    predictor.train_all_models(X_train, y_win_train, y_spread_train, y_total_train)
    
    # Evaluation
    results = predictor.evaluate(X_test, y_win_test, y_spread_test, y_total_test)
    
    # Speichern
    import os
    os.makedirs('models', exist_ok=True)
    predictor.save_models('academic')
    
    # Summary
    print("\n" + "="*70)
    print("🏆 ACADEMIC MODEL SUMMARY")
    print("="*70)
    print(f"\n✅ Win/Loss Accuracy: {results['win_accuracy']:.1%}")
    print(f"✅ Spread MAE: {results['spread_mae']:.2f} points")
    print(f"✅ Over/Under MAE: {results['total_mae']:.2f} points")
    print("\n💡 Model nutzt NUR statistische Features:")
    print("   - Elo Ratings")
    print("   - Rolling Statistics (L3, L5)")
    print("   - Offensive/Defensive Efficiency")
    print("   - Momentum Features")
    print("   - KEINE Vegas Lines!")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()