# 🌍 Exoplanet Hunter - NASA Space Apps Challenge 2025

**AI-Powered Exoplanet Detection Using NASA Data**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://exoplanet-hunters.streamlit.app)

---

## 🎯 Project Overview

Exoplanet Hunter is a comprehensive machine learning web application that automatically detects exoplanets from NASA Kepler and TESS mission data with **93.03% accuracy**. Using XGBoost with 22 physics-based features, we reduce manual review time from hours to seconds while maintaining high precision.

**Key Achievement**: Scientifically optimized through grid search of 1,728 hyperparameter combinations, validated with cross-validation.

---

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 93.03% |
| **Cross-Validation** | 91.65% ± 4.93% |
| **Precision** | 93.5% |
| **Recall** | 90.9% |
| **False Positive Rate** | 3.5% |
| **Training Speed** | 0.30 seconds |

---

## 🚀 Features

### 🔍 Star Lookup
- Search **7,300+ Kepler and TESS stars**
- Comprehensive star profiles with AI predictions
- Compare AI results with NASA confirmations
- Mission filter (All/Kepler/TESS)

### 📊 Batch Analysis
- **Multi-file upload** (200MB total limit)
- Auto-detects Kepler or TESS format
- **Hyperparameter tuning** for advanced users
- Interactive visualizations (4 Plotly charts)
- CSV download with star names

### 🔮 Single Star Prediction
- Manual measurement entry (9 NASA parameters)
- Instant predictions
- Planet type classification
- Habitable zone calculation

### 📚 Educational
- Hyperparameter tuning with tooltips
- Physics-based feature explanations
- Interactive learning

---

## 🧬 Technical Details

### Model
- **Algorithm**: XGBoost (Gradient Boosting)
- **Features**: 22 (9 original + 13 engineered)
- **Optimization**: Grid search (1,728 combinations)
- **Training**: 8,849 stars (Kepler + TESS)
- **Split**: 80/20 train-test, stratified

### Feature Engineering
**Original (9 NASA measurements)**:
- Orbital period, transit depth, duration
- Planet radius, temperature, insolation
- Stellar temperature, gravity, radius

**Engineered (13 features)**:
- Signal-to-noise ratios
- Transit physics validation
- Orbital mechanics (Kepler's laws)
- Habitable zone calculations
- Planet type classification

### Hyperparameters (Optimized via Grid Search)
```python
n_estimators: 300
max_depth: 5
learning_rate: 0.12
min_child_weight: 1
subsample: 0.8
colsample_bytree: 0.9
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3.13
- **ML Libraries**: XGBoost, scikit-learn
- **Web Framework**: Streamlit
- **Visualizations**: Plotly
- **Data**: NASA Exoplanet Archive (Kepler + TESS)
- **Deployment**: Streamlit Community Cloud

---

## 📁 Project Structure

```
exoplanet-hunters/
├── app.py                    # Main Streamlit application
├── page_components/          # 5 pages
│   ├── home.py              # Homepage
│   ├── batch_analysis.py    # Multi-file upload & analysis
│   ├── star_lookup.py       # Search 7,300+ stars
│   ├── single_prediction.py # Manual entry
│   └── about.py             # Project info + team
├── utils/
│   ├── data_processor.py    # Kepler/TESS handler
│   └── visualizations.py    # Plotly charts
├── data/
│   ├── cumulative.csv       # Kepler data
│   └── tess_data.csv        # TESS data
├── results/
│   └── xgboost_model.pkl    # Trained model
└── requirements.txt         # Dependencies
```

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run app.py
```

**Access at**: http://localhost:8501

---

## 🌐 Live Demo

**Try it now**: [https://exoplanet-hunter.streamlit.app](https://exoplanet-hunters.streamlit.app)

---

## 👥 Team 2muchcoffe

**First-Year Students, University of Kerala**

- **Munjid V H** - Team Lead & ML Engineer (BBA)
- **Nazeeh Nabhan V** - Science Advisor & Presentation Lead (CS)
- **Abhishek M Raj** - Web Application Developer (CS)

---

## 📚 Data Sources

- [NASA Exoplanet Archive - Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [NASA Exoplanet Archive - TESS Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)

---

## 🏆 NASA Space Apps Challenge 2025

Built with ❤️ for the NASA Space Apps Challenge 2025

**Helping humanity discover new worlds** 🌟

<<<<<<< HEAD
---

=======
>>>>>>> 7247b7da0231a9c0d544f8bdac8d040ad83fe338
## 📝 License

This project uses public NASA data and is created for the NASA Space Apps Challenge.
