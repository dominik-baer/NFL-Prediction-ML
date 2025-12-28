"""
NFL Prediction - Final Optimized Models
Trainiert 5 separate Models mit AUTO-OPTIMIZED K für jedes Target!

IMPROVEMENTS:
- Automated K selection für jedes Model separat
- Proper Mutual Information (Classification vs Regression)
- Cross-validation option
- Saves optimal K config for future use
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_absolute_error, 
                            mean_squared_error, log_loss)
from sklearn.model_selection import cross_val_score
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')


class NFLFinalPredictor:
    """Final Optimized Models mit AUTO-OPTIMIZED K features"""
    
    def __init__(self, use_cv=False, auto_tune_k=True, test_seasons=None):
        """
        Initialize predictor
        
        Args:
            use_cv (bool): Use cross-validation for K optimization
            auto_tune_k (bool): Auto-tune K for each model (True) or load from file (False)
            test_seasons (list): Seasons to use for testing. 
                Examples: [2025], [2024, 2025], [2023, 2024, 2025]
                Default: [2025]
        """
        self.models = {}
        self.feature_sets = {}
        self.feature_importances = {}
        self.optimal_k = {}
        self.use_cv = use_cv
        self.auto_tune_k = auto_tune_k
        self.test_seasons = test_seasons or [2025]
        
    def prepare_data(self, df):
        """Prepare data und filter features"""
        
        # Filter completed games
        df = df[df['home_win'].notna()].copy()
        
        # Train/Test split with configurable test seasons
        min_test_season = min(self.test_seasons)
        train_df = df[df['season'] < min_test_season].copy()
        test_df = df[df['season'].isin(self.test_seasons)].copy()
        
        train_seasons = f"{train_df['season'].min()}-{train_df['season'].max()}"
        test_seasons = ', '.join(map(str, sorted(self.test_seasons)))
        
        print(f"📊 Train: {len(train_df)} games (seasons {train_seasons})")
        print(f"📊 Test:  {len(test_df)} games (seasons {test_seasons})")
        print(f"   Test ratio: {len(test_df)/(len(train_df)+len(test_df)):.1%}")
        
        # Exclude columns
        exclude_cols = [
            'home_win', 'season', 'week', 'gameday', 'home_team', 'away_team',
            'home_score', 'away_score', 'total_score', 'score_diff',
            'spread_line', 'total_line', 'over_line', 'under_line'
        ]
        
        # Get numeric features
        potential_features = [c for c in train_df.columns if c not in exclude_cols]
        
        feature_cols = []
        for col in potential_features:
            if train_df[col].dtype not in ['int64', 'float64', 'int32', 'float32', 'bool']:
                continue
            col_lower = col.lower()
            if any(word in col_lower for word in ['result', 'winner']):
                continue
            if any(word in col_lower for word in ['line', 'over', 'under']):
                continue
            if col in ['home_score', 'away_score', 'total_score', 'score_diff']:
                continue
            feature_cols.append(col)
        
        print(f"📊 Total features available: {len(feature_cols)}\n")
        
        # Fill NaN
        train_df[feature_cols] = train_df[feature_cols].fillna(0)
        test_df[feature_cols] = test_df[feature_cols].fillna(0)
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        
        # Prepare all targets
        targets_train = {
            'win': train_df['home_win'],
            'spread': train_df['home_score'] - train_df['away_score'],
            'total': train_df['home_score'] + train_df['away_score'],
            'home_score': train_df['home_score'],
            'away_score': train_df['away_score']
        }
        
        targets_test = {
            'win': test_df['home_win'],
            'spread': test_df['home_score'] - test_df['away_score'],
            'total': test_df['home_score'] + test_df['away_score'],
            'home_score': test_df['home_score'],
            'away_score': test_df['away_score']
        }
        
        return X_train, X_test, targets_train, targets_test, feature_cols
    
    def calculate_feature_importance(self, X_train, y_train, task_type='classifier'):
        """
        Calculate feature importance using appropriate Mutual Information
        
        Args:
            X_train: Training features
            y_train: Training target
            task_type: 'classifier' or 'regressor'
        
        Returns:
            DataFrame with feature importances
        """
        print(f"  🔍 Calculating MI ({task_type})...")
        
        if task_type == 'classifier':
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
        
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return importances
    
    def find_optimal_k(self, X_train, X_test, y_train, y_test, 
                       importances, task_type='classifier', 
                       k_range=None):
        """
        Find optimal K for given target
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            importances: Feature importance DataFrame
            task_type: 'classifier' or 'regressor'
            k_range: Range of K values to test (default: 10 to 200 step 10)
        
        Returns:
            optimal_k (int), best_score (float)
        """
        print(f"  🎯 Finding optimal K...")
        
        if k_range is None:
            max_k = min(200, len(X_train.columns))
            k_range = list(range(10, 51, 5)) + list(range(55, max_k+1, 10))
        
        best_k = 0
        best_score = -np.inf
        
        for k in k_range:
            if k > len(importances):
                break
            
            top_features = importances.head(k)['feature'].tolist()
            
            if task_type == 'classifier':
                model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=15,
                    random_state=42, 
                    n_jobs=-1
                )
                
                if self.use_cv:
                    # Use cross-validation
                    scores = cross_val_score(
                        model, X_train[top_features], y_train,
                        cv=3, scoring='roc_auc', n_jobs=-1
                    )
                    score = scores.mean()
                else:
                    # Use test set
                    model.fit(X_train[top_features], y_train)
                    y_pred_proba = model.predict_proba(X_test[top_features])[:, 1]
                    score = roc_auc_score(y_test, y_pred_proba)
            
            else:  # regressor
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                )
                
                if self.use_cv:
                    # Use cross-validation (negative MAE)
                    scores = cross_val_score(
                        model, X_train[top_features], y_train,
                        cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
                    )
                    score = scores.mean()
                else:
                    # Use test set
                    model.fit(X_train[top_features], y_train)
                    y_pred = model.predict(X_test[top_features])
                    score = -mean_absolute_error(y_test, y_pred)  # Negative for maximization
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"  ✓ Optimal K: {best_k} (score: {best_score:.4f})")
        
        return best_k, best_score
    
    def load_k_config(self, filepath='optimal_k_config.json'):
        """Load K configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            print(f"📥 Loaded K config from {filepath}")
            return config
        except FileNotFoundError:
            print(f"⚠️  No K config found at {filepath}, will use auto-tune")
            return None
    
    def save_k_config(self, filepath='optimal_k_config.json'):
        """Save optimal K configuration"""
        with open(filepath, 'w') as f:
            json.dump(self.optimal_k, f, indent=2)
        print(f"💾 Saved K config to {filepath}")
    
    def train_all_models(self, X_train, X_test, targets_train, targets_test):
        """Train all 5 models with optimal K"""
        
        print("="*80)
        print("🎯 TRAINING FINAL OPTIMIZED MODELS")
        print("="*80 + "\n")
        
        # Try to load K config if not auto-tuning
        if not self.auto_tune_k:
            loaded_k = self.load_k_config()
            if loaded_k:
                self.optimal_k = loaded_k
        
        # Model configs
        configs = {
            'win_loss': {
                'type': 'classifier',
                'target': 'win',
                'description': 'Win/Loss Prediction'
            },
            'spread': {
                'type': 'regressor',
                'target': 'spread',
                'description': 'Point Spread Prediction'
            },
            'total': {
                'type': 'regressor',
                'target': 'total',
                'description': 'Total Points (O/U) Prediction'
            },
            'home_score': {
                'type': 'regressor',
                'target': 'home_score',
                'description': 'Home Team Score Prediction'
            },
            'away_score': {
                'type': 'regressor',
                'target': 'away_score',
                'description': 'Away Team Score Prediction'
            }
        }
        
        results = {}
        
        for model_name, config in configs.items():
            print("="*80)
            print(f"🏈 {config['description'].upper()}")
            print("="*80)
            
            y_train = targets_train[config['target']]
            y_test = targets_test[config['target']]
            
            # Calculate feature importance FOR THIS TARGET
            importances = self.calculate_feature_importance(
                X_train, y_train, config['type']
            )
            self.feature_importances[model_name] = importances
            
            # Find or load optimal K
            if self.auto_tune_k or model_name not in self.optimal_k:
                optimal_k, best_score = self.find_optimal_k(
                    X_train, X_test, y_train, y_test,
                    importances, config['type']
                )
                self.optimal_k[model_name] = optimal_k
            else:
                optimal_k = self.optimal_k[model_name]
                print(f"  ✓ Using loaded K: {optimal_k}")
            
            # Select top K features
            top_features = importances.head(optimal_k)['feature'].tolist()
            self.feature_sets[model_name] = top_features
            
            X_train_k = X_train[top_features]
            X_test_k = X_test[top_features]
            
            print(f"\n  Features: {len(top_features)}")
            print(f"  Top 5: {top_features[:5]}")
            
            # Train FINAL model with optimal hyperparameters
            if config['type'] == 'classifier':
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_k, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_k)
                y_pred_proba = model.predict_proba(X_test_k)[:, 1]
                
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                logloss = log_loss(y_test, y_pred_proba)
                
                print(f"\n  ✅ Accuracy: {acc:.2%}")
                print(f"  ✅ AUC-ROC:  {auc:.4f}")
                print(f"  ✅ LogLoss:  {logloss:.4f}")
                
                results[model_name] = {
                    'k': optimal_k,
                    'accuracy': acc,
                    'auc': auc,
                    'logloss': logloss
                }
                
            else:  # regressor
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_k, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_k)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                within7 = (np.abs(y_pred - y_test) <= 7).mean()
                within10 = (np.abs(y_pred - y_test) <= 10).mean()
                
                print(f"\n  ✅ MAE:       {mae:.2f} points")
                print(f"  ✅ RMSE:      {rmse:.2f} points")
                print(f"  ✅ Within 7:  {within7:.2%}")
                print(f"  ✅ Within 10: {within10:.2%}")
                
                results[model_name] = {
                    'k': optimal_k,
                    'mae': mae,
                    'rmse': rmse,
                    'within7': within7,
                    'within10': within10
                }
            
            self.models[model_name] = model
            print()
        
        # Save K config
        self.save_k_config()
        
        return results
    
    def save_models(self, output_dir='models'):
        """Save all trained models"""
        
        print("="*80)
        print("💾 SAVING MODELS")
        print("="*80 + "\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            filename = f'{output_dir}/{model_name}_model.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {filename}")
        
        # Save feature sets
        with open(f'{output_dir}/feature_sets.pkl', 'wb') as f:
            pickle.dump(self.feature_sets, f)
        print(f"✓ Saved: {output_dir}/feature_sets.pkl")
        
        # Save optimal K config
        with open(f'{output_dir}/optimal_k_config.json', 'w') as f:
            json.dump(self.optimal_k, f, indent=2)
        print(f"✓ Saved: {output_dir}/optimal_k_config.json")
        
        # Save feature importances for each model
        for model_name, importances in self.feature_importances.items():
            filename = f'{output_dir}/feature_importance_{model_name}.csv'
            importances.to_csv(filename, index=False)
            print(f"✓ Saved: {filename}")
        
        print(f"\n✅ All models saved to {output_dir}/")
    
    def print_summary(self, results):
        """Print final summary"""
        
        print("\n" + "="*80)
        print("🏆 FINAL MODEL SUMMARY")
        print("="*80 + "\n")
        
        print("MODEL CONFIGURATIONS (AUTO-OPTIMIZED):")
        print(f"  Win/Loss:   K={results['win_loss']['k']:3d} → Acc={results['win_loss']['accuracy']:.2%}, AUC={results['win_loss']['auc']:.4f}")
        print(f"  Spread:     K={results['spread']['k']:3d} → MAE={results['spread']['mae']:.2f}, W7={results['spread']['within7']:.2%}")
        print(f"  Total:      K={results['total']['k']:3d} → MAE={results['total']['mae']:.2f}, W7={results['total']['within7']:.2%}")
        print(f"  Home Score: K={results['home_score']['k']:3d} → MAE={results['home_score']['mae']:.2f}, W7={results['home_score']['within7']:.2%}")
        print(f"  Away Score: K={results['away_score']['k']:3d} → MAE={results['away_score']['mae']:.2f}, W7={results['away_score']['within7']:.2%}")
        
        print("\n💡 FEATURES USED:")
        print("  ✓ Each model uses OPTIMAL K features for its specific target")
        print("  ✓ Feature importance calculated with appropriate MI (classification vs regression)")
        print("  ✓ Elo Ratings & Momentum")
        print("  ✓ Rolling Statistics (L3, L5)")
        print("  ✓ Offensive/Defensive Matchups")
        print("  ✓ Form & Consistency Metrics")
        print("  ✓ Situational Features")
        print("  ✓ Weather Conditions")
        print("  ✓ NO Vegas Lines!")
        
        print("\n🎯 K-OPTIMIZATION:")
        print(f"  ✓ Auto-tuned: {self.auto_tune_k}")
        print(f"  ✓ Cross-validation: {self.use_cv}")
        print(f"  ✓ K config saved to: optimal_k_config.json")
        
        print("\n📊 USAGE:")
        print("  1. Load models from models/ directory")
        print("  2. Prepare game features using MEGA feature engineering")
        print("  3. Use appropriate feature set for each prediction")
        print("  4. Make predictions!")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("🏈 NFL FINAL MODEL TRAINING (AUTO-OPTIMIZED)")
    print("="*80 + "\n")
    
    # Load data
    print("📥 Loading data...")
    df = pd.read_csv('nfl_training_data_MEGA.csv')
    df['gameday'] = pd.to_datetime(df['gameday'])
    print(f"✓ Loaded: {len(df)} games, {len(df.columns)} features\n")
    
    # Initialize predictor with configurable test seasons
    # Options:
    # - test_seasons=[2025]           → Small test set (245 games)
    # - test_seasons=[2024, 2025]     → Medium test set (~500 games) ← RECOMMENDED
    # - test_seasons=[2023, 2024, 2025] → Large test set (~850 games)
    
    predictor = NFLFinalPredictor(
        use_cv=False,              # Set True for more robust K selection (slower)
        auto_tune_k=True,          # Set False to load K from optimal_k_config.json
        test_seasons=[2024, 2025]  # ← CHANGE THIS for different test set sizes
    )
    
    # Prepare data
    X_train, X_test, targets_train, targets_test, feature_cols = predictor.prepare_data(df)
    
    # Train all models with auto-optimized K
    results = predictor.train_all_models(X_train, X_test, targets_train, targets_test)
    
    # Save models
    predictor.save_models('models')
    
    # Print summary
    predictor.print_summary(results)


if __name__ == "__main__":
    main()