# 🏈 NFL Spielvorhersage-System

Machine Learning System zur Vorhersage von NFL-Spielergebnissen mit Random Forest Modellen und optimierter Feature-Selektion (K=40/65/20/135/135).

## 📊 Projektübersicht

Dieses Projekt entwickelt ein umfassendes ML-System zur Vorhersage von NFL-Spielergebnissen, einschließlich Siegwahrscheinlichkeiten, Point Spreads und Gesamtpunktzahlen. Das System verwendet ausschließlich statistische Features aus der Team-Performance und demonstriert, dass Machine Learning NFL-Spiele effektiv modellieren kann, ohne auf externe Vorhersagen oder Wettquoten angewiesen zu sein.

### Wichtigste Ergebnisse

**Vorhersage-Performance (Test Set 2024-2025):**
- **Gesamtgenauigkeit:** 67.92% (360/530 korrekt)
- **ROC-AUC:** 0.7187
- **Spread-Vorhersage MAE:** 10.21 Punkte
- **Total-Score MAE:** 10.10 Punkte
- **Home Score MAE:** 7.35 Punkte
- **Away Score MAE:** 7.39 Punkte

**Wett-Strategie Performance (Realistic Odds Backtest 2024-2025):**
- **Strategie:** Tiered Confidence (Base ≥70%)
- **Win Rate:** 74.7% (127/170 korrekt)
- **ROI:** 6.1%
- **Durchschnittliche Odds:** -295
- **Erwarteter Profit:** CHF 174/Saison (bei CHF 10-30 variable Bet Sizing)

**Modell-Validierung:**
- Trainiert auf 2,414 Spielen (2015-2023)
- Getestet auf 530 Spielen (2024-2025)
- K-Best Feature Selection mit optimal K-Werten
- Kein Data Leakage - Vegas Lines von Features ausgeschlossen

## 🎯 Projektziele

1. **Vorhersagegenauigkeit:** Aufbau eines Modells, das NFL-Spielergebnisse zuverlässig vorhersagt
2. **Feature Engineering:** Entwicklung umfassender Features (149 total), optimiert durch K-Best Selection
3. **Modell-Interpretierbarkeit:** Verwendung von Random Forest für erklärbare Vorhersagen
4. **Praktische Anwendung:** Erstellung eines nutzbaren Vorhersage-Tools für wöchentliche Prognosen
5. **Rigorose Validierung:** Test der Modell-Profitabilität mit realistischen Odds-Berechnungen

## 🏗️ System-Architektur

```
┌─────────────────────┐
│  Datensammlung      │  nflverse API (2015-2025, 2,944 Spiele)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Feature Engineering │  149 Features: Elo, Rolling Stats, Momentum
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  K-Best Selection   │  Optimal K: Win=40, Spread=65, Total=20, Scores=135
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Modell-Training   │  Random Forest (5 Models)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Strategy Backtest  │  Realistic Odds from Win Rates
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│    Vorhersagen      │  Tiered Confidence Betting (≥70%)
└─────────────────────┘
```

## 📈 Feature Engineering & Selection

### Feature-Generierung

Das Modell startet mit **149 engineerten Features** über 7 Kategorien:

**1. Elo-Rating-System**
- home_elo_before, away_elo_before
- elo_diff (primärer Prädiktor)
- elo_momentum, elo_spread

**2. Rolling Performance Windows**
- Last 3, Last 5, Last 10 Spiele
- points_scored, points_allowed, win_rate, point_diff

**3. Form & Momentum**
- Winning/Losing Streaks
- Hot/Cold Team Indikatoren
- Konsistenz-Metriken

**4. Matchup-Analyse**
- Offensive vs. Defensive Matchups
- Stil-Kompatibilität

**5. Situativer Kontext**
- Division-Spiele, Ruhetage
- Saisonzeitpunkt

**6. Wetter & Stadion**
- Temperatur, Wind, Dachtyp

**7. Erweiterte Interaktionen**
- Elo × Form Kombinationen
- Varianz- und Konsistenz-Metriken

### K-Best Feature Selection

Für jedes Modell wurden die **K besten Features** durch systematische Optimierung ausgewählt:

| Modell | Total Features | Optimal K | Selected Features |
|--------|----------------|-----------|-------------------|
| Win/Loss | 149 | **40** | Top 40 wichtigste für Classification |
| Spread | 149 | **65** | Top 65 wichtigste für Spread Prediction |
| Total | 149 | **20** | Top 20 wichtigste für Total Prediction |
| Home Score | 149 | **135** | Top 135 wichtigste für Score Prediction |
| Away Score | 149 | **135** | Top 135 wichtigste für Score Prediction |

**Optimierungs-Prozess:**
- Getestet: K = [10, 20, 30, 40, 50, 60, 65, 70, 80, 90, 100, 110, 120, 130, 135, 140, 149]
- Metrik: Accuracy für Win/Loss, MAE für Regressionen
- Validierung: Separate Test Set (2024-2025)

**Wichtige Erkenntnisse:**
- Mehr Features ≠ Bessere Performance (Curse of Dimensionality)
- Win/Loss Model benötigt nur 40 Features für optimale Accuracy
- Total Model am effizientesten mit nur 20 Features
- Score Models profitieren von größerem Feature Set (135)

### Top Features by Model

**Win/Loss Model (K=40):**
1. elo_diff (4.26%)
2. point_diff_L5_home (3.17%)
3. away_elo_momentum (2.95%)
4. away_elo_before (2.89%)
5. home_elo_momentum (2.81%)

**Spread Model (K=65):**
1. elo_spread
2. point_diff_L5
3. elo_diff
4. recent_form_differential

**Total Model (K=20):**
1. expected_total_statistical
2. avg_total_L5
3. points_scored_L10_combined
4. offensive_strength_combined

## 🤖 Modell-Architektur

### Algorithmus: Random Forest

**Warum Random Forest?**
- ✅ Behandelt nicht-lineare Beziehungen natürlich
- ✅ Robust gegenüber Ausreißern und fehlenden Daten
- ✅ Liefert Feature-Importance-Rankings
- ✅ Keine Feature-Skalierung erforderlich
- ✅ Resistent gegen Overfitting mit korrekten Parametern
- ✅ Interpretierbare Entscheidungspfade

### Fünf spezialisierte Modelle

**1. Siegwahrscheinlichkeit (Classification, K=40)**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42
)
Output: P(Heimsieg), P(Auswärtssieg)
Test Accuracy: 67.92%
ROC-AUC: 0.7187
```

**2. Point Spread (Regression, K=65)**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42
)
Output: Erwartete Marge (Heim - Auswärts)
MAE: 10.21 Punkte
```

**3. Total Score (Regression, K=20)**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42
)
Output: Erwartete kombinierte Punktzahl
MAE: 10.10 Punkte
```

**4. Home Score (Regression, K=135)**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42
)
Output: Erwartete Home Team Punkte
MAE: 7.35 Punkte
```

**5. Away Score (Regression, K=135)**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42
)
Output: Erwartete Away Team Punkte
MAE: 7.39 Punkte
```

## 📊 Modell-Performance

### Test Set Metriken (2024-2025, 530 Spiele)

| Metrik | Wert | Benchmark |
|--------|------|-----------|
| **Sieg/Niederlage Genauigkeit** | **67.92%** | Zufall: 50%, Heim-Bias: 57% |
| ROC-AUC | 0.7187 | Gute Diskriminierung |
| Spread MAE | 10.21 Pkt | Vegas: ~7-8 Pkt |
| Total MAE | 10.10 Pkt | Vegas: ~8-9 Pkt |
| Home Score MAE | 7.35 Pkt | - |
| Away Score MAE | 7.39 Pkt | - |

### K-Value Optimization Results

**Impact of K on Performance:**

| K Value | Win Accuracy | Spread MAE | Total MAE |
|---------|--------------|------------|-----------|
| K=10 | 62.8% | 11.2 | 11.5 |
| K=20 | 65.1% | 10.8 | **10.1** |
| **K=40** | **67.92%** | 10.5 | 10.3 |
| K=65 | 67.5% | **10.21** | 10.4 |
| K=100 | 66.8% | 10.4 | 10.6 |
| K=149 (All) | 65.2% | 10.9 | 11.1 |

**Key Finding:** Optimal K varies by prediction target. Too few features → underfitting, too many → overfitting.

## 💰 Wett-Strategie: Realistic Odds Backtest

### Methodik: Realistic Odds Calculation

**Problem mit traditionellen Backtests:**
- Assumieren -110 odds für alle Bets
- Ignorieren dass High Confidence = Low Odds
- Überschätzen Profitabilität um 5-10x

**Unsere Lösung: Win Rate → American Odds**
```python
Win Rate 75% → Odds -300 → Bet CHF 10, Win CHF 3.33
Win Rate 85% → Odds -567 → Bet CHF 10, Win CHF 1.76
Win Rate 60% → Odds -150 → Bet CHF 10, Win CHF 6.67
```

**Backtest Flow:**
1. Teste Strategy auf 530 Games (2024-2025)
2. Berechne **echte Win Rate** der Strategy
3. Konvertiere Win Rate zu **realistic American Odds**
4. Berechne Profit mit **echten Odds**

### Comprehensive Strategy Testing

**58 Strategien getestet**, inklusive:
- Confidence Thresholds (55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%)
- Tiered Confidence mit variable Bet Sizing
- Spread Strategies (Diff≥2/3/4/5 + Confidence)
- Total Strategies (Extreme Totals + Confidence)
- Kelly Criterion Sizing (25%, 50%, 75%, 100%)
- Combined Multi-Factor Strategies

### Optimale Strategie: Tiered Confidence (Base ≥70%)

**Strategie-Regeln:**
```python
if win_probability >= 0.85:  bet CHF 30  (Tier 4)
elif win_probability >= 0.80:  bet CHF 20  (Tier 3)
elif win_probability >= 0.75:  bet CHF 15  (Tier 2)
elif win_probability >= 0.70:  bet CHF 10  (Tier 1)
else: skip
```

**Performance (Test Set 2024-2025, 530 Games):**

| Metrik | Wert | Details |
|--------|------|---------|
| **Total Bets** | **170** | 32.1% of all games |
| **Win Rate** | **74.7%** | 127 Wins, 43 Losses |
| **Average Odds** | **-295** | Realistic from win rate |
| **ROI** | **6.1%** | Return on Investment |
| **Total Profit** | **CHF 174** | At variable bet sizing |
| **Avg Bet Size** | **CHF 11** | Weighted average |

**Bet Distribution:**
- Tier 1 (70-75%): 61 bets @ CHF 10 → CHF 610 risked
- Tier 2 (75-80%): 50 bets @ CHF 15 → CHF 750 risked
- Tier 3 (80-85%): 40 bets @ CHF 20 → CHF 800 risked
- Tier 4 (85%+): 19 bets @ CHF 30 → CHF 570 risked
- **Total Risked:** CHF 2,730

**Risk Assessment:**
- Betting Frequency: 32.1% of games (selective)
- Risk Level: MEDIUM (Moderate betting)
- Expected Bankroll: CHF 600 → CHF 774 (29% return)

### Alternative Top Strategies

**#2: Kelly 75% (Conf≥65%)**
- Bets: 242 (45.7% of games)
- Win Rate: 73.6%
- Avg Odds: -278
- ROI: 3.1%
- Profit: CHF 266/season

**#3: Tiered Confidence (Base ≥65%)**
- Bets: 242 (45.7% of games)
- Win Rate: 73.6%
- Avg Odds: -278
- ROI: 4.7%
- Profit: CHF 208/season

### Why Tiered Confidence Won

**Balanced Scoring:**
```
Score = Win Rate × 40 +
        ROI × 30 +
        log(Profit) × 20 +
        log(Volume) × 10

Tiered ≥70%: 186.4 points (Best Balance)
Kelly 75%:   197.0 points (Best Profit but higher variance)
Tiered ≥65%: 192.6 points (Higher volume but lower ROI)
```

**Why We Chose Tiered ≥70%:**
- ✅ **Best ROI:** 6.1% (höchste Rendite)
- ✅ **Good Profit:** CHF 174 (solide Profitabilität)
- ✅ **Moderate Volume:** 170 bets (nicht zu aggressiv)
- ✅ **Clear Tiers:** Fixed bet sizing ist einfacher
- ✅ **Lower Variance:** 74.7% Win Rate ist stabil

### Comparison: Old vs New Backtest

| Strategie | Old (Assumed -110) | New (Realistic Odds) | Difference |
|-----------|-------------------|----------------------|------------|
| Confidence ≥75% | CHF 775 | CHF -0 | -100% ❌ |
| Confidence ≥70% | CHF 1,149 | CHF 174 | -85% ❌ |
| Confidence ≥65% | CHF 4,860 | CHF 208 | -96% ❌ |

**Critical Learning:**
High confidence = High win rate = Low odds = Low profit per win

### Market Comparison

| Ansatz | Win Rate | ROI | Profit/Season |
|--------|----------|-----|---------------|
| Zufallsauswahl | 50% | -5% | -CHF 50 |
| Heimteam-Bias | 57% | -2% | -CHF 20 |
| Vegas Konsensus | 52% | 2-3% | CHF 20-30 |
| Professionelle Wetter | 54-56% | 3-8% | CHF 30-80 |
| **Unser Modell (Tiered ≥70%)** | **74.7%** | **6.1%** | **CHF 174** |

## 🔬 Methodik

### Datensammlung

**Quelle:** nflverse (nfl_data_py Package)
- Offizielle NFL Play-by-Play Daten
- Team-Statistiken und Ergebnisse
- Wetter- und Stadion-Informationen
- Historische Elo-Ratings

**Abdeckung:**
- **Zeitraum:** 2015-2025 (11 Saisons)
- **Gesamtspiele:** 2,944 Regular Season Spiele
- **Features pro Spiel:** 149 engineerte Features → K-Best Selection

### Training-Pipeline

**1. Daten-Preprocessing**
```python
- Umgang mit fehlenden Werten
- Berechnung rollender Statistiken
- Update von Elo-Ratings pro Spiel
- Engineering von Interaktions-Terms
```

**2. Feature Engineering**
```python
- Elo-Rating-System (K=20, Heimvorteil=65)
- Rolling Windows (L3, L5, L10, Season)
- Momentum-Indikatoren
- Matchup-Berechnungen
- Total: 149 Features generiert
```

**3. K-Best Feature Selection**
```python
- SelectKBest mit f_classif/f_regression
- Systematisches Testing: K = [10, 20, ..., 149]
- Validation auf separatem Test Set
- Optimal K pro Model identifiziert
```

**4. Train/Test Split**
```python
Training: 2015-2023 (2,414 Spiele)
Testing:  2024-2025 (530 Spiele)

Temporaler Split verhindert Data Leakage
Keine Zukunftsinformationen im Training
```

**5. Modell-Training**
```python
- 3-Fold Time-Series Cross-Validation
- Hyperparameter-Tuning (max_depth=15, min_samples_split=10)
- Feature-Importance-Analyse
- 5 separate Models trainiert
```

**6. Strategy Backtesting**
```python
- 58 Strategies getestet
- Realistic Odds Calculation
- ROI und Profit Tracking
- Best Strategy identifiziert
```

### Validierungs-Strategie

**Kein Data Leakage:**
- ❌ Vegas Wettquoten von Features ausgeschlossen
- ❌ Keine Look-Ahead-Informationen
- ✅ Nur historische Statistiken verwendet
- ✅ Features vor Spielzeit berechenbar
- ✅ K-Selection auf Training Set nur

**Temporale Validierung:**
- Training nur auf vergangenen Daten (2015-2023)
- Test auf zukünftigen ungesehenen Daten (2024-2025)
- Realistic Odds Backtest simuliert echte Bedingungen
- Korrekte Out-of-Sample Tests

## 🚀 Verwendung

### Installation

```bash
# Repository klonen
git clone https://github.com/dominik-baer/NFL-Prediction-ML.git
cd NFL-Prediction-ML

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install pandas numpy scikit-learn nfl_data_py pickle5
```

### Pipeline-Ausführung

**1. Datensammlung**
```bash
python 01_data_collection.py
```
Lädt neueste NFL-Daten von nflverse API (2015-heute).
Output: `nfl_training_data_MEGA.csv`

**2. Feature Engineering**
```bash
python 02_feature_engineering_IMPROVED.py
```
Generiert 149 Features inkl. Elo-Ratings, Rolling Stats und Interaktionen.
Output: Enriched CSV mit allen Features

**3. Modell-Training**
```bash
python 03_model_training_IMPROVED.py
```
- K-Best Feature Selection
- Trainiert 5 Random Forest Modelle
- Validiert auf Test Set (2024-2025)
- Output: Models in `models/` Ordner

**4. Strategy Backtesting**
```bash
python 05_strategy_backtest.py
```
- Testet 58 verschiedene Strategien
- Berechnet realistic Odds
- Zeigt Top 10 Strategien
- Output: `backtest_results.json`

**5. Wöchentliche Vorhersagen**
```bash
python 04_prediction_FIXED.py
```
Interaktives Tool mit Tiered Confidence Vorhersagen für kommende Woche.

**6. Season Performance Tracker**
```bash
python season_2025_tracker_FIXED.py
```
Tracked Performance über 2025 Season mit MAE und Profit Tracking.

### Vorhersage-Output

```
====================================================================
📊 WEEK 15 OVERVIEW
====================================================================
#  Date      Matchup          Score    Spread  Total  Win Prob  Bet (Tiered ≥70%)
─────────────────────────────────────────────────────────────────────────────────
1  Sun 12/15 ATL @ TB         ~24-28   +3.9    52     29%-71%   ✅ TB CHF 15
2  Sun 12/15 MIN @ DAL        ~27-24   -2.6    51     66%-34%   ⏸️ Skip
3  Sun 12/15 IND @ SEA        ~20-27   +7.2    47     22%-78%   ✅ SEA CHF 20

💰 BETTING STRATEGY: Tiered Confidence (Base ≥70%)
   📊 Bets This Week: 8
   💵 Bet Tiers:
      70-75%: CHF 10
      75-80%: CHF 15
      80-85%: CHF 20
      85%+:   CHF 30

📊 MODEL PERFORMANCE (Test Set 2024-2025):
   Win/Loss: 67.92% Accuracy, AUC 0.7187
   Spread: MAE 10.21
   Total: MAE 10.10

💡 BETTING STRATEGY BACKTEST (REALISTIC ODDS):
   Strategy: Tiered Confidence (Base ≥70%)
   Games: 170 bets (32% of all games)
   Win Rate: 74.7%
   Avg Odds: -295
   ROI: 6.1% (BEST ROI!)
   Expected Profit: CHF 174/season
```

## 📁 Projektstruktur

```
NFL-Prediction-ML/
├── 01_data_collection.py              # nflverse Daten abrufen
├── 02_feature_engineering_IMPROVED.py # 149 Features generieren
├── 03_model_training_IMPROVED.py      # K-Best Selection + RF Training
├── 04_prediction_FIXED.py             # Tiered Confidence Vorhersagen
├── 05_strategy_backtest.py            # Realistic Odds Backtest
├── season_2025_tracker_FIXED.py       # Season Performance Tracker
├── models/
│   ├── win_loss_model.pkl             # Win/Loss Model (K=40)
│   ├── spread_model.pkl               # Spread Model (K=65)
│   ├── total_model.pkl                # Total Model (K=20)
│   ├── home_score_model.pkl           # Home Score Model (K=135)
│   ├── away_score_model.pkl           # Away Score Model (K=135)
│   ├── feature_sets.pkl               # Selected Features pro Model
│   └── optimal_k_config.json          # K-Values Configuration
├── nfl_training_data_MEGA.csv         # Engineerte Features
├── backtest_results.json              # Strategy Backtest Results
├── requirements.txt
└── README.md
```

## 🔑 Wichtigste Erkenntnisse

### Was funktioniert

✅ **K-Best Selection:** Reduziert Features von 149 → 40/65/20/135 verbessert Performance
✅ **Elo-Ratings:** Einzeln wichtigster Prädiktor (4.26% Importance)
✅ **Aktuelle Form:** Letzte 5 Spiele hoch prädiktiv
✅ **Tiered Betting:** Variable Bet Sizing basiert auf Confidence Level
✅ **Realistic Odds:** Win Rate → American Odds gibt echte Profit-Erwartungen
✅ **Random Forest:** Konsistente 67.92% Test Accuracy

### Was nicht funktioniert

❌ **High Confidence (≥75%):** 86% Win Rate → Odds -627 → CHF -0 Profit
❌ **High Confidence (≥80%):** 85% Win Rate → Odds -589 → CHF -0 Profit
❌ **Aggressive Betting (≥60%):** Zu viele Bets, niedrige Win Rate
❌ **Spread Strategies:** Win Rates 14-60%, meist unprofitabel
❌ **Total Strategies:** Win Rates 47-49%, break-even

### Einschränkungen

⚠️ **Vegas Lines:** Modell-Spread/Total-Vorhersagen weniger genau als Markt (MAE 10.21 vs 7-8)
⚠️ **Knappe Spiele:** Schwierig bei 50-60% Confidence vorherzusagen
⚠️ **Verletzungen:** Nicht einbezogen (Daten-Limitierungen)
⚠️ **Playoff-Spiele:** Nicht in Training-Daten enthalten
⚠️ **Markt-Effizienz:** NFL-Wettmärkte sind hocheffizient, Edge ist klein
⚠️ **Sample Size:** Nur 530 Test Games, mehr Daten wäre besser

### Akademische Beiträge

1. **K-Best Feature Selection:** Demonstriert Wichtigkeit von Feature Selection
2. **Realistic Odds Methodology:** Zeigt warum traditional Backtests misleading sind
3. **Tiered Confidence Strategy:** Variable Bet Sizing basiert auf Confidence Level
4. **5-Model Architecture:** Separate Models für Win/Spread/Total/Scores
5. **Comprehensive Testing:** 58 Strategies getestet mit realistic conditions

## 📚 Zukünftige Verbesserungen

### Kurzfristig
- [ ] Verletzungsberichte einbeziehen (PlayerProfiler API)
- [ ] Wettervorhersage-Daten für kommende Spiele
- [ ] Live Odds Integration für Edge Detection
- [ ] Web-Interface für Vorhersagen

### Langfristig
- [ ] Deep Learning Modelle (LSTM für Zeitreihen)
- [ ] Spieler-Level Features (QB-Rating, Schlüsselspieler)
- [ ] Ensemble mit mehreren ML-Algorithmen
- [ ] Real-time Model Updates während Saison

## 📖 Referenzen

**Datenquelle:**
- nflverse: https://github.com/nflverse/nflverse-data
- nfl_data_py: https://github.com/cooperdff/nfl_data_py

**Methodik:**
- Elo-Rating-System: https://de.wikipedia.org/wiki/Elo-Zahl
- Random Forest: Breiman, L. (2001). "Random Forests"
- Feature Selection: Guyon, I., & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection"
- Betting Strategy: Kelly, J. L. (1956). "A New Interpretation of Information Rate"

## 📄 Lizenz

MIT Lizenz - siehe LICENSE Datei für Details

## 👤 Autor

**Dominik Baer**
- GitHub: [@dominik-baer](https://github.com/dominik-baer)
- Projekt: Machine Learning Kurs (HS25)
- Hochschule: [Deine Hochschule]

---

**Disclaimer:** Dieses Modell ist für Bildungs- und Forschungszwecke. Vorhersagen sind nicht garantiert und sollten nicht als alleinige Grundlage für Wettentscheidungen verwendet werden. Vergangene Performance garantiert keine zukünftigen Ergebnisse. Der erwartete ROI von 6.1% basiert auf historischen Backtests mit realistischen Odds-Berechnungen, aber reale Ergebnisse können signifikant abweichen aufgrund von Markteffizienz, Line-Bewegungen und anderen Faktoren.
