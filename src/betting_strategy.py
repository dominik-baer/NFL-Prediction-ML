"""
NFL Betting Strategy - Findet profitable Wett-Opportunities
Vergleicht Model Predictions mit Vegas Lines
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime


class BettingStrategy:
    """Findet profitable Wetten durch Model vs Vegas Vergleich"""
    
    def __init__(self):
        self.betting_model = None
        self.academic_model = None
        self.load_models()
        
    def load_models(self):
        """Lädt trainierte Modelle"""
        
        print("📥 Lade Modelle...")
        
        try:
            with open('models/betting_win.pkl', 'rb') as f:
                self.betting_win = pickle.load(f)
            with open('models/betting_spread.pkl', 'rb') as f:
                self.betting_spread = pickle.load(f)
            with open('models/betting_total.pkl', 'rb') as f:
                self.betting_total = pickle.load(f)
            with open('models/betting_features.pkl', 'rb') as f:
                self.betting_features = pickle.load(f)
            
            print("✓ Betting Models geladen")
            self.has_betting = True
        except:
            print("⚠️  Betting Models nicht gefunden - führe model_betting.py aus!")
            self.has_betting = False
        
        try:
            with open('models/academic_win_rf.pkl', 'rb') as f:
                self.academic_win = pickle.load(f)
            with open('models/academic_spread_rf.pkl', 'rb') as f:
                self.academic_spread = pickle.load(f)
            with open('models/academic_features.pkl', 'rb') as f:
                self.academic_features = pickle.load(f)
            
            print("✓ Academic Models geladen")
            self.has_academic = True
        except:
            print("⚠️  Academic Models nicht gefunden - führe model_academic.py aus!")
            self.has_academic = False
    
    def prepare_features(self, df, feature_list):
        """Bereitet Features vor"""
        
        available_features = [f for f in feature_list if f in df.columns]
        X = df[available_features].copy()
        
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        
        return X
    
    def find_betting_opportunities(self, df, min_edge=0.10, min_confidence=0.60):
        """
        Findet profitable Wett-Opportunities
        
        min_edge: Minimum erforderliche Abweichung von Vegas (z.B. 0.10 = 10%)
        min_confidence: Minimum Model Confidence (z.B. 0.60 = 60%)
        """
        
        print("\n" + "="*70)
        print("🔍 SUCHE BETTING OPPORTUNITIES")
        print("="*70)
        print(f"\nParameter:")
        print(f"  Min Edge: {min_edge:.0%}")
        print(f"  Min Confidence: {min_confidence:.0%}")
        
        if not self.has_betting:
            print("\n❌ Betting Model nicht verfügbar!")
            return None
        
        # Features vorbereiten
        X = self.prepare_features(df, self.betting_features)
        
        # Predictions
        win_proba = self.betting_win.predict_proba(X)[:, 1]  # Probability Home wins
        spread_pred = self.betting_spread.predict(X)
        total_pred = self.betting_total.predict(X)
        
        # Results DataFrame
        opportunities = []
        
        for idx, row in df.iterrows():
            game_id = idx
            home = row.get('home_team', 'Home')
            away = row.get('away_team', 'Away')
            gameday = row.get('gameday', 'TBD')
            
            # Vegas Lines
            vegas_spread = row.get('spread_line', None)
            vegas_total = row.get('total_line', None)
            
            # Unsere Predictions
            our_win_prob = win_proba[idx] if idx < len(win_proba) else None
            our_spread = spread_pred[idx] if idx < len(spread_pred) else None
            our_total = total_pred[idx] if idx < len(total_pred) else None
            
            # === WIN/LOSS BET CHECK ===
            if our_win_prob is not None and vegas_spread is not None:
                # Vegas sagt: spread > 0 → Home favored
                vegas_home_prob = 0.5 + (vegas_spread / 28)  # Rough conversion (28 = max spread)
                vegas_home_prob = np.clip(vegas_home_prob, 0.1, 0.9)
                
                prob_diff = abs(our_win_prob - vegas_home_prob)
                
                if prob_diff >= min_edge and (our_win_prob >= min_confidence or our_win_prob <= (1-min_confidence)):
                    bet_on = home if our_win_prob > vegas_home_prob else away
                    confidence = our_win_prob if our_win_prob > 0.5 else (1 - our_win_prob)
                    
                    opportunities.append({
                        'game': f"{away} @ {home}",
                        'date': gameday,
                        'bet_type': 'MONEYLINE',
                        'bet_on': bet_on,
                        'confidence': confidence,
                        'our_prob': our_win_prob,
                        'vegas_prob': vegas_home_prob,
                        'edge': prob_diff,
                        'our_prediction': our_spread,
                        'vegas_line': vegas_spread
                    })
            
            # === SPREAD BET CHECK ===
            if our_spread is not None and vegas_spread is not None:
                spread_diff = abs(our_spread - vegas_spread)
                
                # Wenn unser Spread deutlich anders als Vegas
                if spread_diff >= 1.5:  # 1.5+ Punkte Unterschied (war 3.0)
                    # Wenn wir Home stärker sehen als Vegas
                    if our_spread > vegas_spread:
                        bet_on = f"{home} ({vegas_spread:+.1f})"
                        explanation = f"Wir sehen Home {spread_diff:.1f} pts stärker"
                    else:
                        bet_on = f"{away} ({-vegas_spread:+.1f})"
                        explanation = f"Wir sehen Away {spread_diff:.1f} pts stärker"
                    
                    opportunities.append({
                        'game': f"{away} @ {home}",
                        'date': gameday,
                        'bet_type': 'SPREAD',
                        'bet_on': bet_on,
                        'confidence': min(spread_diff / 7, 0.95),  # Max 95% confidence
                        'our_prediction': our_spread,
                        'vegas_line': vegas_spread,
                        'edge': spread_diff,
                        'explanation': explanation
                    })
            
            # === OVER/UNDER BET CHECK ===
            if our_total is not None and vegas_total is not None:
                total_diff = abs(our_total - vegas_total)
                
                if total_diff >= 3.0:  # 3+ Punkte Unterschied (war 5.0)
                    if our_total > vegas_total:
                        bet_on = f"OVER {vegas_total:.1f}"
                        explanation = f"Erwarten {total_diff:.1f} pts mehr"
                    else:
                        bet_on = f"UNDER {vegas_total:.1f}"
                        explanation = f"Erwarten {total_diff:.1f} pts weniger"
                    
                    opportunities.append({
                        'game': f"{away} @ {home}",
                        'date': gameday,
                        'bet_type': 'TOTAL',
                        'bet_on': bet_on,
                        'confidence': min(total_diff / 10, 0.90),
                        'our_prediction': our_total,
                        'vegas_line': vegas_total,
                        'edge': total_diff,
                        'explanation': explanation
                    })
        
        if len(opportunities) == 0:
            print("\n⚠️  Keine klaren Betting Opportunities gefunden")
            print("   → Parameter lockern (min_edge runter) oder mehr Spiele laden")
            return None
        
        opp_df = pd.DataFrame(opportunities)
        opp_df = opp_df.sort_values('confidence', ascending=False)
        
        print(f"\n✅ {len(opp_df)} Betting Opportunities gefunden!")
        
        return opp_df
    
    def display_recommendations(self, opportunities_df, top_n=10):
        """Zeigt Top Betting Recommendations"""
        
        if opportunities_df is None or len(opportunities_df) == 0:
            print("\n⚠️  Keine Recommendations")
            return
        
        print("\n" + "="*70)
        print(f"🎯 TOP {min(top_n, len(opportunities_df))} BETTING RECOMMENDATIONS")
        print("="*70 + "\n")
        
        for i, row in opportunities_df.head(top_n).iterrows():
            print(f"{'='*70}")
            print(f"🏈 {row['game']}")
            if 'date' in row:
                print(f"📅 {row['date']}")
            print(f"\n💰 BET TYPE: {row['bet_type']}")
            print(f"✅ BET ON: {row['bet_on']}")
            print(f"🎯 CONFIDENCE: {row['confidence']:.1%}")
            
            if 'our_prediction' in row and 'vegas_line' in row:
                print(f"\n📊 Analysis:")
                if row['bet_type'] == 'SPREAD':
                    print(f"   Our Spread: {row['our_prediction']:+.1f}")
                    print(f"   Vegas Spread: {row['vegas_line']:+.1f}")
                    print(f"   Edge: {row['edge']:.1f} points")
                elif row['bet_type'] == 'TOTAL':
                    print(f"   Our Total: {row['our_prediction']:.1f}")
                    print(f"   Vegas Total: {row['vegas_line']:.1f}")
                    print(f"   Edge: {row['edge']:.1f} points")
            
            if 'explanation' in row:
                print(f"\n💡 {row['explanation']}")
            
            print()
        
        print("="*70)
        print("\n⚠️  WICHTIG: Risikomanagement!")
        print("   → Niemals mehr als 1-2% der Bankroll pro Wette")
        print("   → Minimum 20-30 Wetten für statistische Relevanz")
        print("   → Track deine Ergebnisse!")
    
    def backtest_strategy(self, df):
        """Backtestet die Strategie auf historischen Daten"""
        
        print("\n" + "="*70)
        print("📈 BACKTEST BETTING STRATEGY")
        print("="*70)
        
        opportunities = self.find_betting_opportunities(df, min_edge=0.08, min_confidence=0.55)
        
        if opportunities is None or len(opportunities) == 0:
            print("\n⚠️  Keine Opportunities zum Backtesten")
            return
        
        # Simuliere Wetten
        # Hier würde man die tatsächlichen Ergebnisse mit den Predictions vergleichen
        # Für echten Backtest brauchst du die actual results im DataFrame
        
        print(f"\n📊 Backtest Summary:")
        print(f"   Total Opportunities: {len(opportunities)}")
        print(f"   Spread Bets: {len(opportunities[opportunities['bet_type']=='SPREAD'])}")
        print(f"   Total Bets: {len(opportunities[opportunities['bet_type']=='TOTAL'])}")
        print(f"   Moneyline Bets: {len(opportunities[opportunities['bet_type']=='MONEYLINE'])}")
        
        avg_confidence = opportunities['confidence'].mean()
        print(f"\n   Average Confidence: {avg_confidence:.1%}")
        
        high_conf = opportunities[opportunities['confidence'] >= 0.70]
        print(f"   High Confidence Bets (≥70%): {len(high_conf)}")


def main():
    """Hauptfunktion"""
    
    print("\n" + "="*70)
    print("🎰 NFL BETTING STRATEGY ANALYZER")
    print("="*70 + "\n")
    
    # Strategy initialisieren
    strategy = BettingStrategy()
    
    # Daten laden
    print("\n📥 Lade aktuelle Daten...")
    try:
        df = pd.read_csv('nfl_training_data_engineered.csv')
        df['gameday'] = pd.to_datetime(df['gameday'])
        print(f"✓ {len(df)} Spiele geladen")
    except:
        print("❌ Keine Daten gefunden!")
        return
    
    # Nur aktuelle/kommende Season
    current_season = df['season'].max()
    current_games = df[df['season'] == current_season].copy()
    
    print(f"\n🏈 Analysiere Season {current_season}: {len(current_games)} Spiele")
    
    # Finde Opportunities (gelockerte Parameter!)
    opportunities = strategy.find_betting_opportunities(
        current_games, 
        min_edge=0.03,      # 3% minimum edge (war 8%)
        min_confidence=0.50  # 50% minimum confidence (war 55%)
    )
    
    # Zeige Top Recommendations
    strategy.display_recommendations(opportunities, top_n=10)
    
    # Speichere Opportunities
    if opportunities is not None and len(opportunities) > 0:
        output_file = f'betting_opportunities_{current_season}.csv'
        opportunities.to_csv(output_file, index=False)
        print(f"\n💾 Opportunities gespeichert: {output_file}")
    
    # Backtest
    print("\n" + "="*70)
    print("📊 BACKTEST AUF VERGANGENE SAISONS")
    print("="*70)
    
    past_seasons = df[df['season'] < current_season]
    if len(past_seasons) > 500:
        past_seasons = past_seasons.tail(500)  # Letzte 500 Spiele
    
    strategy.backtest_strategy(past_seasons)
    
    # ALTERNATIVE: Zeige einfach alle Predictions
    print("\n" + "="*70)
    print("📊 ALL PREDICTIONS (Top 20 by Confidence)")
    print("="*70)
    
    all_predictions = strategy.find_betting_opportunities(
        current_games,
        min_edge=0.0,  # Keine Edge required
        min_confidence=0.0  # Keine Confidence required
    )
    
    if all_predictions is not None:
        strategy.display_recommendations(all_predictions, top_n=20)
    
    print("\n" + "="*70)
    print("✅ ANALYSE COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()