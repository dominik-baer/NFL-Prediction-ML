# 🏈 NFL Game Prediction - Machine Learning Projekt

Ein Machine Learning Projekt zur Vorhersage von NFL-Spielergebnissen.

## 📋 Setup

1. Virtual Environment aktivieren:
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

2. Dependencies installieren:
```bash
pip install -r requirements.txt
```

3. Daten sammeln:
```bash
python src/data_collection.py
```

4. Modelle trainieren:
```bash
python src/model_training.py
```

## 📁 Projektstruktur

```
nfl-prediction-ml/
├── data/               # Daten
│   ├── raw/           # Rohdaten
│   ├── processed/     # Verarbeitete Daten
│   └── features/      # Feature-Sets
├── src/               # Source Code
├── notebooks/         # Jupyter Notebooks
├── models/            # Gespeicherte Modelle
└── results/           # Ergebnisse & Plots
```

## 🎯 Ziel

Vorhersage von NFL-Spielergebnissen mit ~60-65% Accuracy.

## 📊 Features

- Rolling Statistics (3, 5, 10 games)
- Elo Rating System
- Home Field Advantage
- Team Performance Metrics

## 🤖 Modelle

- Logistic Regression (Baseline)
- Random Forest
- XGBoost

## 📈 Erwartete Performance

- Baseline: ~57% (Home Team gewinnt immer)
- Gutes Modell: 60-63%
- Sehr gutes Modell: 63-65%

---

**Erstellt für ML Uni-Projekt HS25**
