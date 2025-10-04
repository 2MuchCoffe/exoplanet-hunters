# 🌍 Exoplanet Hunter - NASA Space Apps Challenge

An AI-powered exoplanet detection system using Machine Learning to analyze NASA Kepler mission data.

## 🎯 Project Overview

This project uses a Random Forest machine learning model to automatically detect exoplanets from star brightness measurements (transit photometry). The AI was trained on 5,612 stars from NASA's Kepler mission and achieved **88.75% accuracy** in identifying planets.

## 📊 Key Results

- **Accuracy**: 88.75%
- **Training Time**: 0.27 seconds
- **Dataset**: 7,016 stars from NASA Kepler mission
- **Confirmed Planets Detected**: 377 out of 459 (82% recall)
- **False Positives**: Only 76 (8% false positive rate)

## 🚀 How It Works

1. **Data Collection**: Uses NASA Kepler exoplanet search results (transit method)
2. **Feature Selection**: 9 key stellar measurements:
   - Orbital period
   - Transit depth (brightness dip)
   - Transit duration
   - Planet radius
   - Equilibrium temperature
   - Insolation flux
   - Stellar temperature
   - Stellar surface gravity
   - Stellar radius

3. **Machine Learning**: Random Forest classifier with 100 decision trees
4. **Prediction**: Instant classification of new star data

## 📁 Project Structure

```
exoplanet-hunter/
├── data/
│   └── cumulative.csv           # NASA Kepler dataset
├── results/
│   ├── confusion_matrix.png     # Visual performance metrics
│   ├── feature_importance.png   # Most important features
│   ├── performance_metrics.png  # Accuracy breakdown
│   ├── exoplanet_model.pkl      # Trained model
│   └── model_report.txt         # Summary report
├── exoplanet_detector.py        # Main program
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🔧 Installation & Usage

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the program
python exoplanet_detector.py
```

### Making Predictions
The program includes a `predict_exoplanet()` function that can analyze new star data:

```python
# Example: Predict if a star has a planet
prediction, probability = predict_exoplanet(
    period=10.5,      # days
    depth=200,        # ppm
    duration=3.2,     # hours
    prad=1.5,         # Earth radii
    teq=400,          # Kelvin
    insol=1.5,        # Earth flux
    steff=5500,       # Kelvin
    slogg=4.5,        # log10(cm/s^2)
    srad=1.0          # Solar radii
)
```

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| Overall Accuracy | 88.75% |
| Precision (Planets) | 83.19% |
| Recall (Planets) | 82.14% |
| F1-Score | 82.66% |

## 🎓 What Makes This Special

1. **Fast Training**: Trains in under 1 second vs hours for neural networks
2. **High Accuracy**: 88.75% accuracy competitive with complex deep learning models
3. **Interpretable**: Feature importance shows which measurements matter most
4. **Practical**: Can process new star data instantly
5. **Robust**: Handles real-world NASA data with missing values

## 🔬 Scientific Approach

**Transit Method**: When a planet passes in front of its star (transit), it blocks a tiny amount of starlight. Our AI learned to recognize these patterns and distinguish real planets from false positives (eclipsing binaries, instrument noise, etc.).

**Key Insight**: The model learned that transit depth and planet radius are the most important features for detection, which aligns with astrophysical theory!

## 🎤 Demo for Judges

1. **Show the visualizations** in `results/` folder
2. **Run the program** - demonstrates quick training
3. **Explain the accuracy** - 88.75% with low false positives
4. **Live prediction** - feed new star data for instant results
5. **Feature importance** - show what the AI learned

## 📚 Data Source

- NASA Exoplanet Archive: Kepler Objects of Interest
- Dataset: cumulative.csv (9,564 stars)
- Confirmed Planets: 2,292
- False Positives: 4,724

## 🛠️ Technologies Used

- **Python 3**: Programming language
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning (Random Forest)
- **matplotlib & seaborn**: Visualizations
- **NumPy**: Numerical computing

## 🎯 Future Improvements

1. Add neural network option for comparison
2. Real-time data fetching from NASA API
3. Web interface for easier demonstration
4. Support for TESS mission data
5. Ensemble of multiple models

## 👨‍💻 Hackathon Presentation Tips

1. Start with the problem: Manual exoplanet detection is slow
2. Explain your solution: AI can automate this
3. Show the results: 88.75% accuracy, fast training
4. Live demo: Predict on new data
5. Discuss impact: Could help NASA process thousands of stars quickly

## 📝 License

This project uses public NASA data and is created for the NASA Space Apps Challenge.

---

**Built with ❤️ for NASA Space Apps Challenge 2024**
