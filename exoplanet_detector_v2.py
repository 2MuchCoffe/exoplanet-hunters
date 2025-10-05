# This script is only used to train our AI Model :)

import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom data processor
from utils.data_processor import load_and_prepare_data, prepare_for_training

print("=" * 70)
print("🌍 EXOPLANET HUNTER - NASA SPACE APPS CHALLENGE")
print("=" * 70)
print("✓ All libraries imported successfully!")
print()

# ============================================================
# SECTION 1: LOAD AND PREPARE DATA (KEPLER + TESS)
# ============================================================

print("=" * 70)
print("SECTION 1: LOADING NASA DATA")
print("=" * 70)

# Load both Kepler and TESS data (if available)
combined_data = load_and_prepare_data(
    kepler_path='data/cumulative.csv',
    tess_path='data/tess_data.csv'  # Will gracefully skip if not found
)

# Prepare for training (filter, clean, engineer features)
print("\n🧹 PREPARING DATA FOR TRAINING...")
X, y, feature_names, mission_info = prepare_for_training(combined_data)

print(f"\n📊 FINAL DATASET:")
print(f"   Total stars: {len(X)}")
print(f"   Confirmed planets: {y.sum()}")
print(f"   Non-planets: {len(y) - y.sum()}")
print(f"   Features: {len(feature_names)}")

if mission_info is not None:
    mission_counts = mission_info.value_counts()
    print(f"\n📡 MISSION BREAKDOWN:")
    for mission, count in mission_counts.items():
        print(f"   {mission}: {count} stars ({count/len(X)*100:.1f}%)")

print("\n✓ Data loading complete!")

# ============================================================
# SECTION 2: SPLIT DATA (80% TRAIN, 20% TEST)
# ============================================================

print("\n" + "=" * 70)
print("SECTION 2: SPLITTING DATA (80% TRAIN / 20% TEST)")
print("=" * 70)

# Stratified split to maintain planet ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n📊 TRAINING SET (80%):")
print(f"   Total samples: {len(X_train)}")
print(f"   Planets: {y_train.sum()}")
print(f"   Non-planets: {len(y_train) - y_train.sum()}")

print(f"\n📊 TEST SET (20%):")
print(f"   Total samples: {len(X_test)}")
print(f"   Planets: {y_test.sum()}")
print(f"   Non-planets: {len(y_test) - y_test.sum()}")

# Save test set for demo
print("\n💾 SAVING TEST SET FOR DEMO...")
test_data_folder = Path('data/test_data_for_demo')
test_data_folder.mkdir(exist_ok=True)

# Combine X_test and y_test for saving
test_df = X_test.copy()
test_df['target'] = y_test.values
test_df.to_csv(test_data_folder / 'test_set_20percent.csv', index=False)
print(f"   ✓ Saved: {len(test_df)} stars to test_data_for_demo/test_set_20percent.csv")

print("\n✓ Data split complete!")

# ============================================================
# SECTION 3: TRAIN XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("SECTION 3: TRAINING XGBOOST MODEL")
print("=" * 70)

# Create scientifically optimized XGBoost model (grid search validated)
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.12,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

print("\n🤖 XGBoost Configuration (Optimized via Grid Search):")
print(f"   • n_estimators: 300 trees")
print(f"   • max_depth: 5 levels")
print(f"   • learning_rate: 0.12")
print(f"   • min_child_weight: 1")
print(f"   • subsample: 0.8")
print(f"   • colsample_bytree: 0.9")

print("\n🎓 TRAINING MODEL...")
start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"✓ Training complete in {training_time:.2f} seconds!")

# ============================================================
# SECTION 4: EVALUATE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SECTION 4: MODEL EVALUATION")
print("=" * 70)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 TEST SET PERFORMANCE:")
print(f"   Accuracy: {accuracy * 100:.2f}%")
print(f"   Correct predictions: {int(accuracy * len(y_test))} out of {len(y_test)}")

print("\n📈 DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['No Planet', 'Planet']))

print("\n🔢 CONFUSION MATRIX:")
print(f"   True Negatives:  {cm[0][0]} (Correctly identified non-planets)")
print(f"   False Positives: {cm[0][1]} (Wrongly predicted planet)")
print(f"   False Negatives: {cm[1][0]} (Missed real planets)")
print(f"   True Positives:  {cm[1][1]} (Correctly identified planets)")

# Cross-validation
print("\n🔬 CROSS-VALIDATION (5-Fold):")
print("-" * 70)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print("   Results:")
for i, score in enumerate(cv_scores, 1):
    print(f"   • Fold {i}: {score*100:.2f}%")
print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"   📊 Mean: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")

print("\n✓ Evaluation complete!")

# ============================================================
# SECTION 5: CREATE VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("SECTION 5: CREATING VISUALIZATIONS")
print("=" * 70)

sns.set_style("whitegrid")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Planet', 'Planet'],
            yticklabels=['No Planet', 'Planet'])
plt.title('XGBoost Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# 2. Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# 3. Performance Metrics
precision = cm[1][1] / (cm[1][1] + cm[0][1])
recall = cm[1][1] / (cm[1][1] + cm[1][0])
f1 = 2 * precision * recall / (precision + recall)

metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [accuracy, precision, recall, f1]
}

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_data['Metric'], [s * 100 for s in metrics_data['Score']],
               color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
plt.ylabel('Score (%)', fontsize=12)
plt.title('XGBoost Performance Metrics', fontsize=14, fontweight='bold')
plt.ylim(0, 100)

for bar, score in zip(bars, metrics_data['Score']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/performance_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_metrics.png")
plt.close()

print("\n✓ All visualizations saved!")

# ============================================================
# SECTION 6: SAVE MODEL AND REPORT
# ============================================================

print("\n" + "=" * 70)
print("SECTION 6: SAVING MODEL AND REPORT")
print("=" * 70)

# Save XGBoost model
with open('results/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved: xgboost_model.pkl")

# Save detailed report
with open('results/model_report.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("EXOPLANET HUNTER - MODEL REPORT\n")
    f.write("NASA Space Apps Challenge - XGBoost Model\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("DATASET:\n")
    f.write(f"  Total Stars: {len(X)}\n")
    f.write(f"  Training Stars: {len(X_train)}\n")
    f.write(f"  Test Stars: {len(X_test)}\n")
    f.write(f"  Features: {len(feature_names)}\n\n")
    
    f.write("MODEL CONFIGURATION:\n")
    f.write(f"  Algorithm: XGBoost (Gradient Boosting)\n")
    f.write(f"  n_estimators: 200\n")
    f.write(f"  max_depth: 6\n")
    f.write(f"  learning_rate: 0.1\n")
    f.write(f"  Training Time: {training_time:.2f} seconds\n\n")
    
    f.write("PERFORMANCE METRICS:\n")
    f.write(f"  Test Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"  Cross-Val Accuracy: {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%\n")
    f.write(f"  Precision: {precision * 100:.2f}%\n")
    f.write(f"  Recall: {recall * 100:.2f}%\n")
    f.write(f"  F1-Score: {f1 * 100:.2f}%\n\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write(f"  True Negatives:  {cm[0][0]}\n")
    f.write(f"  False Positives: {cm[0][1]}\n")
    f.write(f"  False Negatives: {cm[1][0]}\n")
    f.write(f"  True Positives:  {cm[1][1]}\n\n")
    
    f.write("TOP 5 MOST IMPORTANT FEATURES:\n")
    for i, (idx, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        f.write(f"  {i}. {row['feature']}: {row['importance']:.4f}\n")

print("✓ Report saved: model_report.txt")

# ============================================================
# SECTION 7: PREDICTION FUNCTION
# ============================================================

def predict_exoplanet(period, depth, duration, prad, teq, insol, steff, slogg, srad,
                     period_err=0.01, depth_err=1.0):
    """
    Predict if a star has an exoplanet using XGBoost model
    
    Parameters (9 NASA measurements + optional errors):
    - period: Orbital period (days)
    - depth: Transit depth (ppm)
    - duration: Transit duration (hours)
    - prad: Planet radius (Earth radii)
    - teq: Equilibrium temperature (K)
    - insol: Insolation flux (Earth flux)
    - steff: Stellar temperature (K)
    - slogg: Stellar gravity (log10)
    - srad: Stellar radius (Solar radii)
    - period_err, depth_err: Optional error estimates
    
    Returns: prediction (0/1), confidence (%)
    """
    # Engineer all 13 features
    depth_per_duration = depth / (duration + 1e-10)
    planet_star_ratio = prad / (srad + 1e-10)
    temp_insol_ratio = teq / (insol + 1e-10)
    
    period_snr = period / (period_err + 1e-10)
    depth_snr = depth / (depth_err + 1e-10)
    expected_depth = (prad / (srad + 1e-10)) ** 2
    depth_consistency = depth / (expected_depth + 1e-10)
    orbital_speed = (2 * np.pi * srad) / (period + 1e-10)
    semi_major_axis = period ** (2/3)
    stellar_density = slogg / (srad ** 2 + 1e-10)
    log_period = np.log10(period + 1)
    
    stellar_luminosity = (srad ** 2) * ((steff / 5778) ** 4)
    hz_inner = np.sqrt(stellar_luminosity / 1.1)
    hz_outer = np.sqrt(stellar_luminosity / 0.53)
    in_habitable_zone = 1 if (semi_major_axis > hz_inner and semi_major_axis < hz_outer) else 0
    
    if prad < 1.25:
        planet_type = 1
    elif prad < 2.0:
        planet_type = 2
    elif prad < 6.0:
        planet_type = 3
    else:
        planet_type = 4
    
    period_depth_interaction = period * depth
    
    # Create feature array (22 features)
    features = np.array([[
        period, depth, duration, prad, teq, insol, steff, slogg, srad,
        depth_per_duration, planet_star_ratio, temp_insol_ratio,
        period_snr, depth_snr, depth_consistency, orbital_speed,
        semi_major_axis, stellar_density, log_period, in_habitable_zone,
        planet_type, period_depth_interaction
    ]])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return prediction, probabilities[prediction] * 100

# Test prediction
print("\n🔮 EXAMPLE PREDICTION:")
print("-" * 70)
sample_idx = y_test[y_test == 1].index[0]
sample = X_test.loc[sample_idx]
pred, conf = predict_exoplanet(*sample.values[:9])

print(f"Sample star from test set:")
print(f"  Prediction: {'PLANET DETECTED! 🌍' if pred == 1 else 'No planet'}")
print(f"  Confidence: {conf:.2f}%")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✅ EXOPLANET HUNTER IS READY!")
print("=" * 70)
print(f"\n🏆 Model Performance:")
print(f"   • Accuracy: {accuracy * 100:.2f}%")
print(f"   • Cross-Validation: {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%")
print(f"   • Training Time: {training_time:.2f}s")
print(f"   • False Positive Rate: {(cm[0][1] / (cm[0][0] + cm[0][1])) * 100:.2f}%")
print(f"\n📁 All results saved in: results/")
print(f"📁 Test data for demo: data/test_data_for_demo/")
print("=" * 70)
