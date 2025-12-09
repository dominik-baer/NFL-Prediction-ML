# 🏈 NFL Betting ML System

Machine Learning System für profitable NFL-Sportwetten mit optimierter Ultra Conservative Strategy.

[![Win Rate](https://img.shields.io/badge/Win%20Rate-84.3%25-success)]()
[![ROI](https://img.shields.io/badge/ROI-+40.7%25-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()

## 🎯 Key Results

**Ultra Conservative Strategy (Optimized):**
- **Win Rate:** 84.3%
- **ROI:** +40.7%
- **Thresholds:** Win Probability ≥ 80%, Spread Agreement ≤ 4.5 pts
- **Bets per Year:** ~17 (quality over quantity)

## 📊 System Overview

Machine Learning System das profitable Wett-Opportunities im NFL identifiziert durch:

1. **Data Collection:** Automatisches Laden von nflverse
2. **Feature Engineering:** 127 Features (Elo, Form, Matchups)
3. **Model Training:** Random Forest Models
4. **Predictions:** Weekly Betting Recommendations

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/dominik-baer/NFL-Prediction-ML.git
cd NFL-Prediction-ML
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Weekly Workflow

```bash
# Update (jeden Dienstag)
python src/01_data_collection.py
python src/02_feature_engineering.py
python src/03_model_betting.py

# Predictions
python src/04_prediction.py
```

## 💰 Ultra Conservative Strategy

**Kriterien:**
- Model Win Probability ≥ 80%
- Spread Agreement ≤ 4.5 pts
- Bet Moneyline auf Favorite

**Performance (2020-2024):**
- 89 Bets, 75 Wins
- 84.3% Win Rate
- +40.7% ROI

**Expected Profit:**
```
$100/bet: ~$633/year
$500/bet: ~$3,165/year
```

## 📁 Structure

```
src/
├── 01_data_collection.py      # Data loading
├── 02_feature_engineering.py  # Feature creation
├── 03_model_betting.py        # Model training
├── 04_prediction.py           # Predictions
└── betting_strategy.py        # Backtests
```

## 📚 Documentation

- [Methodology](docs/METHODOLOGY.md)
- [Results](docs/RESULTS.md)
- [Uni Projekt](docs/UNI_PROJEKT.md)

## ⚠️ Disclaimer

Für Bildungszwecke. Sportwetten bergen Risiken!

---

**Built with:** Python, scikit-learn, pandas, nflverse  
**Last Updated:** December 2024
