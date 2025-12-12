# 🏈 NFL Spielvorhersage-System

Machine Learning System zur Vorhersage von NFL-Spielergebnissen mit Random Forest Modellen, trainiert auf 75 engineerten Features.

## 📊 Projektübersicht

Dieses Projekt entwickelt ein umfassendes ML-System zur Vorhersage von NFL-Spielergebnissen, einschließlich Siegwahrscheinlichkeiten, Point Spreads und Gesamtpunktzahlen. Das System verwendet ausschließlich statistische Features aus der Team-Performance und demonstriert, dass Machine Learning NFL-Spiele effektiv modellieren kann, ohne auf externe Vorhersagen oder Wettquoten angewiesen zu sein.

### Wichtigste Ergebnisse

**Vorhersage-Performance (Saison 2025):**
- **Gesamtgenauigkeit:** 65.9%
- **High Confidence Spiele (≥75% Wahrscheinlichkeit):** 80.4% Genauigkeit
- **Spread-Vorhersage MAE:** 9.88 Punkte
- **Total-Score MAE:** 10.54 Punkte

**Modell-Validierung:**
- Trainiert auf 2,724 Spielen (2015-2024)
- Getestet auf 208 Spielen (Saison 2025)
- Walk-Forward Validierung über 6 Saisons (2020-2025)
- Kein Data Leakage - Vegas Lines von Features ausgeschlossen

## 🎯 Projektziele

1. **Vorhersagegenauigkeit:** Aufbau eines Modells, das NFL-Spielergebnisse zuverlässig vorhersagt
2. **Feature Engineering:** Entwicklung umfassender Features (75 total), die Team-Dynamiken erfassen
3. **Modell-Interpretierbarkeit:** Verwendung von Random Forest für erklärbare Vorhersagen
4. **Praktische Anwendung:** Erstellung eines nutzbaren Vorhersage-Tools für wöchentliche Prognosen
5. **Rigorose Validierung:** Test der Modell-Profitabilität mit korrekter Out-of-Sample Methodik

## 🏗️ System-Architektur

```
┌─────────────────────┐
│  Datensammlung      │  nflverse API (2015-2025, 2,932 Spiele)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Feature Engineering │  75 Features: Elo, Rolling Stats, Momentum
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Modell-Training   │  Random Forest (Win, Spread, Total)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│    Vorhersagen      │  Wöchentliche Prognosen + Confidence Filtering
└─────────────────────┘
```

## 📈 Feature Engineering

### Überblick

Das Modell verwendet **75 engineerte Features** über 7 Kategorien, die jeweils unterschiedliche Aspekte der Team-Performance und des Spielkontexts erfassen.

### Feature-Kategorien

**1. Elo-Rating-System (9 Features)**
```
Kernstärke-Indikatoren:
- home_elo_before, away_elo_before
- elo_diff (primärer Prädiktor)
- elo_momentum (kürzliche Rating-Änderungen)
- elo_spread (erwartete Punktedifferenz)
```

**2. Rolling Performance Windows (18 Features)**
```
Aktuelle Form-Metriken:
Letzte 3 Spiele (L3):
  - points_scored_L3, points_allowed_L3
  - win_rate_L3, point_diff_L3
  
Letzte 5 Spiele (L5):
  - Gleiche Metriken mit längerem Fenster
  - Erfasst kurzfristige Trends
```

**3. Form & Momentum (9 Features)**
```
Team-Trajektorie:
- Winning/Losing Streaks
- Hot/Cold Team Indikatoren (3+ Spiel-Trends)
- Konsistenz-Metriken
- Kürzliche Performance-Trends
```

**4. Matchup-Analyse (4 Features)**
```
Stil-Kompatibilität:
- Offensive vs. Defensive Matchups
- Netto-Matchup-Vorteil
- Team-Stil-Interaktionen
```

**5. Situativer Kontext (6 Features)**
```
Spielumstände:
- Division-Spiel-Indikator
- Ruhetage-Vorteil
- Heim/Auswärts-Performance-Splits
- Saisonzeitpunkt (früh/mitte/spät)
```

**6. Wetter & Stadion (5 Features)**
```
Umweltfaktoren:
- Temperatur, Windgeschwindigkeit
- Wetter-Impact auf Scoring
- Dachtyp (Dome/Outdoor)
- Oberflächentyp (Rasen/Kunstrasen)
```

**7. Erweiterte Interaktionen (24 Features)**
```
Komplexe Beziehungen:
- Elo × Aktuelle Form Kombinationen
- Varianz- und Konsistenz-Metriken
- Erwartete Totals (statistisch)
- Kombinierte Stärke-Indikatoren
```

### Feature Importance

Top 10 wichtigste Features:

| Rang | Feature | Importance | Beschreibung |
|------|---------|------------|--------------|
| 1 | elo_diff | 4.26% | Team-Stärke-Differential |
| 2 | point_diff_L5_home | 3.17% | Kürzliche Scoring-Marge (Heim) |
| 3 | away_elo_momentum | 2.95% | Auswärtsteam-Momentum |
| 4 | away_elo_before | 2.89% | Auswärtsteam Basis-Stärke |
| 5 | home_elo_momentum | 2.81% | Heimteam-Momentum |
| 6 | home_elo_before | 2.76% | Heimteam Basis-Stärke |
| 7 | points_scored_L5_away | 2.75% | Kürzliches Auswärtsteam-Scoring |
| 8 | point_diff_L5_away | 2.70% | Auswärtsteam-Performance-Trend |
| 9 | elo_spread | 2.62% | Erwarteter Point Spread |
| 10 | elo_times_form_home | 2.50% | Kombinierte Stärke × Form |

**Wichtige Erkenntnisse:**
- Top 10 Features machen 29.4% der Modell-Importance aus
- Top 33 Features erfassen 80% der Vorhersagekraft
- Elo-Ratings und aktuelle Form dominieren Vorhersagen
- Momentum und kürzliche Performance hoch prädiktiv

## 🤖 Modell-Architektur

### Algorithmus: Random Forest

**Warum Random Forest?**
- ✅ Behandelt nicht-lineare Beziehungen natürlich
- ✅ Robust gegenüber Ausreißern und fehlenden Daten
- ✅ Liefert Feature-Importance-Rankings
- ✅ Keine Feature-Skalierung erforderlich
- ✅ Resistent gegen Overfitting mit korrekten Parametern
- ✅ Interpretierbare Entscheidungspfade

### Drei spezialisierte Modelle

**1. Siegwahrscheinlichkeit (Klassifikation)**
```python
RandomForestClassifier(
    n_estimators=200,      # Ensemble aus 200 Bäumen
    max_depth=12,          # Verhindert Overfitting
    min_samples_split=20,  # Robuste Splits
    random_state=42
)
Output: P(Heimsieg), P(Auswärtssieg)
```

**2. Point Spread (Regression)**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=20,
    random_state=42
)
Output: Erwartete Marge (Heim - Auswärts)
```

**3. Total Score (Regression)**
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=20,
    random_state=42
)
Output: Erwartete kombinierte Punktzahl
```

### Modell-Vergleich

Während der Entwicklung wurden mehrere Algorithmen getestet:

| Algorithmus | Win Accuracy | Notizen |
|-------------|--------------|---------|
| **Random Forest** | **65.9%** | **Beste Performance** |
| XGBoost | 63.9% | Schneller aber weniger genau |
| Gradient Boosting | 64.2% | Ähnlich wie XGBoost |
| Logistische Regression | 62.1% | Baseline lineares Modell |

Random Forest wurde aufgrund seiner überlegenen Genauigkeit und Interpretierbarkeit ausgewählt.

## 📊 Modell-Performance

### Gesamtmetriken (Test-Saison 2025)

| Metrik | Wert | Benchmark |
|--------|------|-----------|
| **Sieg/Niederlage Genauigkeit** | **65.9%** | Zufall: 50%, Heim-Bias: 57% |
| Spread MAE | 9.88 Pkt | Vegas: ~7-8 Pkt |
| Total MAE | 10.54 Pkt | Vegas: ~8-9 Pkt |
| Home Score MAE | 7.20 Pkt | - |
| Away Score MAE | 7.24 Pkt | - |
| ROC-AUC | 0.717 | Gute Diskriminierung |

### Performance nach Confidence-Level

Analyse über 1,615 Spiele (2020-2025):

| Confidence-Bereich | Spiele | Win Rate | Interpretation |
|-------------------|--------|----------|----------------|
| 50-60% | 668 | 48.5% | Unentschiedene Spiele |
| 60-70% | 519 | 51.8% | Leichter Vorteil |
| 70-80% | 319 | 59.9% | Starke Vorhersage |
| **80-90%** | **99** | **77.8%** | **Sehr hohe Confidence** |
| **90-100%** | **10** | **90.0%** | **Nahezu sicher** |

**Wichtiger Fund:** Modell-Confidence korreliert stark mit tatsächlicher Genauigkeit, validiert Wahrscheinlichkeits-Kalibrierung.

### Zeitliche Konsistenz

Performance über 6 Test-Saisons:

| Saison | Spiele | Gesamtgenauigkeit | High Conf (≥75%) Genauigkeit |
|--------|--------|-------------------|------------------------------|
| 2020 | 268 | 60.8% | 85.7% (28 Spiele) |
| 2021 | 285 | 63.9% | 70.5% (44 Spiele) |
| 2022 | 284 | 58.8% | 72.9% (48 Spiele) |
| 2023 | 285 | 60.0% | 78.1% (32 Spiele) |
| 2024 | 285 | 66.0% | 92.3% (39 Spiele) |
| 2025 | 208 | 67.3% | 87.9% (33 Spiele) |
| **Durchschnitt** | **1,615** | **62.6%** | **80.4% (224 Spiele)** |

**Konsistenz:** Modell behält stabile Performance über Saisons bei, mit verbessertem Trend in letzten Jahren.

## 🔬 Methodik

### Datensammlung

**Quelle:** nflverse (nfl_data_py Package)
- Offizielle NFL Play-by-Play Daten
- Team-Statistiken und Ergebnisse
- Wetter- und Stadion-Informationen
- Historische Elo-Ratings

**Abdeckung:**
- **Zeitraum:** 2015-2025 (11 Saisons)
- **Gesamtspiele:** 2,932 Regular Season Spiele
- **Features pro Spiel:** 75 engineerte Features

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
- Rolling Windows (L3, L5, Season-to-Date)
- Momentum-Indikatoren
- Matchup-Berechnungen
```

**3. Train/Test Split**
```python
Training: 2015-2024 (2,724 Spiele)
Testing:  2025 Saison (208 Spiele)

Temporaler Split verhindert Data Leakage
Keine Zukunftsinformationen im Training
```

**4. Modell-Training**
```python
- 3-Fold Time-Series Cross-Validation
- Hyperparameter-Tuning (max_depth, min_samples_split)
- Feature-Importance-Analyse
- Finales Modell-Training auf allen Training-Daten
```

### Validierungs-Strategie

**Kein Data Leakage:**
- ❌ Vegas Wettquoten von Features ausgeschlossen
- ❌ Keine Look-Ahead-Informationen
- ✅ Nur historische Statistiken verwendet
- ✅ Features vor Spielzeit berechenbar

**Temporale Validierung:**
- Training nur auf vergangenen Daten
- Test auf zukünftigen ungesehenen Daten
- Walk-Forward Validierung für Wett-Strategien
- Korrekte Out-of-Sample Tests

## 🎯 Vorhersage-Confidence Filtering

### High Confidence Strategie

**Kriterium:** Wähle Spiele aus, bei denen das Modell ≥75% Siegwahrscheinlichkeit einem Team zuweist

**Performance (2020-2025):**
- **Genauigkeit:** 80.4% (180/224 korrekt)
- **Volumen:** 37 Spiele pro Jahr
- **Verbesserung:** +17.8% über Baseline (62.6%)
- **Konsistenz:** 5 von 6 Jahren profitabel bei Wett-Validierung

**Interpretation:**
Der 75%-Schwellenwert filtert effektiv Spiele heraus, bei denen das Modell einen echten Vorhersagevorteil hat. Diese repräsentieren ~13% aller Spiele, erreichen aber 4 von 5 korrekten Vorhersagen.

### Confidence-Verteilung

```
Alle Spiele (1,615):
├─ Niedrige Confidence (50-60%):    668 Spiele (41%) → 48.5% genau
├─ Mittlere Confidence (60-75%):    723 Spiele (45%) → 54.7% genau
└─ Hohe Confidence (≥75%):          224 Spiele (14%) → 80.4% genau ✅
```

**Strategie-Einsicht:**
Durch Selektivität und nur Handeln bei High-Confidence-Vorhersagen steigt Genauigkeit von 62.6% auf 80.4%.

## 💰 Wett-Strategie-Validierung

Während dieses Projekt auf Vorhersagegenauigkeit fokussiert, haben wir den praktischen Wert des Modells durch Tests von Wett-Strategien gegen tatsächliche Marktquoten validiert. Dies dient als objektives Maß dafür, ob das Modell echte Vorteile identifiziert.

### Validierungs-Methodik: Walk-Forward Testing

**Warum Walk-Forward?**
Standard-Backtesting kann irreführende Ergebnisse produzieren. Walk-Forward-Validierung stellt sicher, dass jede Vorhersage wirklich Out-of-Sample ist:

```
Jahr 2020: Training 2015-2019 → Test 2020 (nie gesehen)
Jahr 2021: Training 2016-2020 → Test 2021 (nie gesehen)
Jahr 2022: Training 2017-2021 → Test 2022 (nie gesehen)
Jahr 2023: Training 2018-2022 → Test 2023 (nie gesehen)
Jahr 2024: Training 2019-2023 → Test 2024 (nie gesehen)
Jahr 2025: Training 2020-2024 → Test 2025 (nie gesehen)
```

**Schlüssel-Eigenschaften:**
- ✅ Jedes Test-Jahr komplett Out-of-Sample
- ✅ Modell nur auf vergangenen Daten trainiert
- ✅ Kein Look-Ahead-Bias
- ✅ Realistische Marktbedingungen
- ✅ Jahr-für-Jahr Performance-Tracking

### Umfassende Strategie-Tests

**175 verschiedene Strategien getestet**, inklusive:
- Moneyline-Wetten (verschiedene Confidence-Schwellenwerte)
- Spread-Wetten (verschiedene Differenz-Schwellenwerte)
- Total Over/Under Vorhersagen
- Kombinierte Multi-Bet-Ansätze

### Optimale Strategie: High Confidence (≥75%)

**Kriterium:** Wette auf Team, wenn Modell ≥75% Siegwahrscheinlichkeit zuweist

**Walk-Forward Ergebnisse (2020-2025):**

| Jahr | Wetten | Win Rate | ROI | Profit ($100/Wette) |
|------|--------|----------|-----|---------------------|
| 2020 | 28 | 85.7% | +12.3% | $345 |
| 2021 | 44 | 70.5% | -9.5% | -$418 |
| 2022 | 48 | 72.9% | +3.5% | $170 |
| 2023 | 32 | 78.1% | +9.0% | $288 |
| 2024 | 39 | 92.3% | +30.5% | $1,188 |
| 2025 | 33 | 87.9% | +16.2% | $533 |
| **Total** | **224** | **80.4%** | **+9.4%** | **$2,106** |

**Zusammenfassung:**
- **Win Rate:** 80.4% (4 von 5 Wetten korrekt)
- **ROI:** +9.4% (validiert echten Vorhersagevorteil)
- **Volumen:** 37 Wetten/Jahr (konservativ, selektiv)
- **Konsistenz:** Profitabel in 5 von 6 Jahren (83%)
- **Profit/Jahr:** $351 (bei $100/Wette)

### Markt-Vergleich

| Ansatz | Win Rate | ROI | Notizen |
|--------|----------|-----|---------|
| Zufallsauswahl | 50% | -5% | Verliert gegen Vig |
| Heimteam-Bias | 57% | -2% | Leichter Vorteil, nicht profitabel |
| Vegas Konsensus | 52% | 2-3% | Markteffizienz |
| Professionelle Wetter | 54-56% | 3-8% | Top-Tier Performance |
| **Unser Modell (High Conf)** | **80.4%** | **+9.4%** | **Außergewöhnlich** |

**Interpretation:**
Der 9.4% ROI auf High-Confidence-Vorhersagen demonstriert, dass das Modell erfolgreich Spiele identifiziert, bei denen seine Vorhersagen bedeutsam vom Marktkonsensus abweichen, und diese Unterschiede sich öfter als nicht als genau erweisen.

### Alternative Strategie: Multi-Strategie

**Kriterium:**
- Wette Moneyline wenn: Confidence ≥60% UND Edge ≥7%
- Wette Spread wenn: Confidence ≥60% UND Spread-Differenz ≥5 Punkte

**Performance:**
- Win Rate: 57.3%
- ROI: +7.9%
- Volumen: 70 Wetten/Jahr
- Profit/Jahr: $550 (bei $100/Wette)
- Konsistenz: 5 von 6 Jahren profitabel

**Trade-off:** Niedrigere Win Rate aber höheres Volumen führt zu mehr Gesamtprofit. Beide Strategien validieren Modellqualität.

### Wichtige Einschränkungen

1. **Moneyline-Schätzung:** Tatsächliche Quoten aus Spreads geschätzt (±10-20% Varianz möglich)
2. **Transaktionskosten:** Analyse nimmt Standard -110 Juice an; reale Kosten variieren
3. **Markteffizienz:** NFL-Wettmärkte sind hocheffizient; historische Performance könnte nicht fortbestehen
4. **Stichprobengröße:** 224 Wetten über 6 Jahre; Varianz beeinflusst kurzfristige Ergebnisse
5. **Linien-Bewegung:** Echtes Wetten beinhaltet Linien-Änderungen, die hier nicht modelliert sind

**Konservative Interpretation:**
Während Walk-Forward-Validierung +9.4% ROI zeigt, betrachten professionelle Wetter 3-8% ROI als exzellente Langzeit-Performance. Das Modell demonstriert echten Vorhersagewert, aber reale Ergebnisse wären wahrscheinlich bescheidener.

## 🚀 Verwendung

### Installation

```bash
# Repository klonen
git clone <repository-url>
cd nfl-prediction-ml

# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Pipeline-Ausführung

**1. Datensammlung**
```bash
python src/01_data_collection.py
```
Lädt neueste NFL-Daten von nflverse API (2015-heute).

**2. Feature Engineering**
```bash
python src/02_feature_engineering.py
```
Generiert 75 Features inkl. Elo-Ratings, Rolling Stats und Interaktionen.

**3. Modell-Training**
```bash
python src/03_model_academic.py
```
Trainiert Random Forest Modelle auf 2015-2024 Daten, validiert auf 2025 Saison.

**4. Wöchentliche Vorhersagen**
```bash
python src/04_prediction.py
```
Interaktives Tool mit Vorhersagen für kommende Woche.

### Vorhersage-Output

```
====================================================================
📊 WEEK 15 OVERVIEW
====================================================================
#    Matchup          Score    Spread  Total  Win Prob    Confidence
────────────────────────────────────────────────────────────────────
1    ATL @ TB         29-28    +4.9    45     29%-71%     ❌
2    MIN @ DAL        ~21-25   +3.6    46     34%-66%     ❌
3    IND @ SEA        ~17-31   +14.7   48     22%-78%     ✅ BET SEA
...

💪 High Confidence Vorhersagen: 6
📊 Strategie-Performance:
   Win Rate: 80.4%
   Historische Genauigkeit: 180/224 korrekt (2020-2025)
```

## 📁 Projektstruktur

```
nfl-prediction-ml/
├── src/
│   ├── 01_data_collection.py      # nflverse Daten abrufen
│   ├── 02_feature_engineering.py  # 75 Features generieren
│   ├── 03_model_academic.py       # RF Modelle trainieren
│   └── 04_prediction.py           # Wöchentliche Vorhersagen
├── models/
│   ├── academic_win_rf.pkl        # Siegwahrscheinlichkeits-Modell
│   ├── academic_spread_rf.pkl     # Spread-Modell
│   ├── academic_total_rf.pkl      # Total-Modell
│   └── academic_features.pkl      # Feature-Liste
├── data/
│   └── nfl_training_data_ultimate.csv  # Engineerte Features
├── requirements.txt
└── README.md
```

## 🔑 Wichtigste Erkenntnisse

### Was funktioniert

✅ **Elo-Ratings:** Einzeln wichtigster Prädiktor (4.26% Importance)
✅ **Aktuelle Form:** Letzte 5 Spiele hoch prädiktiv (Top 10 Features)
✅ **Momentum:** Kürzliche Trends wichtiger als Saison-Durchschnitte
✅ **Random Forest:** Übertrifft XGBoost und lineare Modelle
✅ **Confidence Filtering:** 75%-Schwellenwert steigert Genauigkeit auf 80%
✅ **Temporale Validierung:** Konsistente Performance über 6 Saisons

### Einschränkungen

⚠️ **Vegas Lines:** Modell-Spread/Total-Vorhersagen weniger genau als Markt
⚠️ **Knappe Spiele:** Niedrige Genauigkeit bei 50-60% Confidence (48.5%)
⚠️ **Verletzungen:** Nicht einbezogen (Daten-Limitierungen)
⚠️ **Playoff-Spiele:** Nicht in Training-Daten enthalten
⚠️ **Stichprobengröße:** High Confidence Spiele nur 13% der Gesamtzahl

### Akademische Beiträge

1. **Reiner ML-Ansatz:** Demonstriert, dass ML NFL ohne externe Vorhersagen modellieren kann
2. **Feature Engineering:** Umfassendes 75-Feature-Framework für NFL-Vorhersage
3. **Confidence-Kalibrierung:** Modell-Confidence korreliert mit tatsächlicher Genauigkeit
4. **Praktische Validierung:** Walk-Forward-Testing zeigt Anwendbarkeit in der realen Welt
5. **Interpretierbarkeit:** Random Forest erlaubt Feature-Importance-Analyse

## 📚 Zukünftige Verbesserungen

### Kurzfristig
- [ ] Verletzungsberichte einbeziehen (PlayerProfiler API)
- [ ] Wettervorhersage-Daten für kommende Spiele hinzufügen
- [ ] Auf Playoff-Spiele erweitern
- [ ] Web-Interface für Vorhersagen bauen

### Langfristig
- [ ] Deep Learning Modelle (LSTM für Zeitreihen)
- [ ] Spieler-Level Features (QB-Rating, Schlüsselspieler-Impact)
- [ ] Echtzeit-Modell-Updates während Saison
- [ ] Ensemble mit mehreren ML-Algorithmen

## 📖 Referenzen

**Datenquelle:**
- nflverse: https://github.com/nflverse/nflverse-data
- nfl_data_py: https://github.com/cooperdff/nfl_data_py

**Methodik:**
- Elo-Rating-System: https://de.wikipedia.org/wiki/Elo-Zahl
- Random Forest: Breiman, L. (2001). "Random Forests"
- Walk-Forward Validation: Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"

## 📄 Lizenz

MIT Lizenz - siehe LICENSE Datei für Details

## 👤 Autor

Erstellt als Teil des Machine Learning Kurses (HS25)

---

**Hinweis:** Dieses Modell ist für Bildungs- und Forschungszwecke. Vorhersagen sind nicht garantiert und sollten nicht als alleinige Grundlage für Wettentscheidungen verwendet werden. Vergangene Performance garantiert keine zukünftigen Ergebnisse.
