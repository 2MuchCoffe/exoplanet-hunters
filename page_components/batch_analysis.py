"""
Batch Analysis Page - Upload CSV and analyze multiple stars
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import smart_csv_reader, detect_mission_type, standardize_columns, engineer_features

def show(model):
    """Display batch analysis page"""
    
    st.title("📊 Batch Analysis")
    st.markdown("Upload NASA Kepler or TESS data for comprehensive exoplanet detection")
    
    # FILE UPLOAD (Always visible)
    uploaded_file = st.file_uploader(
        "Choose a CSV file (any filename - we auto-detect format!)",
        type=['csv'],
        help="Accepts NASA Kepler or TESS exoplanet data. Filename doesn't matter - we detect the format automatically."
    )
    
    st.markdown("---")
    
    # HYPERPARAMETERS (Always visible)
    st.markdown("### ⚙️ Model Settings")
    st.info("💡 Adjust hyperparameters before analyzing (optional). Defaults work best for most cases.")
    
    # Initialize reset counter
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    
    with st.expander("🔧 Customize Hyperparameters", expanded=False):
        st.warning("⚠️ For advanced users - changing these affects model training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider(
                "Number of Trees",
                min_value=50,
                max_value=500,
                value=300,
                step=50,
                key=f'n_est_{st.session_state.reset_counter}',
                help="More trees = more accurate but slower"
            )
        
        with col2:
            max_depth = st.slider(
                "Max Depth",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                key=f'depth_{st.session_state.reset_counter}',
                help="Controls complexity. Optimal: 5 (grid search validated)"
            )
        
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.12,
            step=0.01,
            key=f'lr_{st.session_state.reset_counter}',
            help="Optimal: 0.12 (grid search validated)"
        )
        
        if st.button("🔄 Reset to Defaults"):
            st.session_state.reset_counter += 1
            st.rerun()
        
        # Comparison table
        st.markdown("**Current Settings vs Optimal (Grid Search):**")
        comparison_df = pd.DataFrame({
            'Parameter': ['Number of Trees', 'Max Depth', 'Learning Rate'],
            'Your Settings': [n_estimators, max_depth, learning_rate],
            'Optimal': [300, 5, 0.12],
            'Impact': [
                '=' if n_estimators == 300 else ('↑ Accuracy' if n_estimators > 300 else '↓ Speed'),
                '=' if max_depth == 5 else ('↑ Complexity' if max_depth > 5 else '↓ Simpler'),
                '=' if abs(learning_rate - 0.12) < 0.001 else ('↑ Aggressive' if learning_rate > 0.12 else '↓ Careful')
            ]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ANALYZE BUTTON (Always visible)
    if st.button("🚀 Analyze Data", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("⚠️ Please upload a CSV file first!")
        else:
            with st.spinner("🔍 Analyzing your data..."):
                try:
                    # Read and detect mission type
                    uploaded_file.seek(0)
                    lines = uploaded_file.readlines()
                    
                    data_start = 0
                    for i, line in enumerate(lines):
                        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                        if not line_str.strip().startswith('#'):
                            data_start = i
                            break
                    
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, skiprows=data_start, low_memory=False)
                    
                    mission = detect_mission_type(df.columns)
                    
                    if mission is None:
                        st.error("❌ Could not detect Kepler or TESS format.")
                        return
                    
                    st.success(f"✅ Detected: **{mission.upper()} Mission Data**")
                    
                    # Standardize and prepare
                    std_df, missing = standardize_columns(df, mission)
                    
                    if std_df is None or len(missing) > 5:
                        st.error(f"❌ Missing critical columns.")
                        return
                    
                    base_features = ['period', 'depth', 'duration', 'planet_radius', 'equilibrium_temp',
                                   'insolation_flux', 'stellar_temp', 'stellar_gravity', 'stellar_radius']
                    
                    std_df = std_df.dropna(subset=base_features)
                    engineered_df = engineer_features(std_df)
                    
                    feature_names = [
                        'period', 'depth', 'duration', 'planet_radius', 'equilibrium_temp',
                        'insolation_flux', 'stellar_temp', 'stellar_gravity', 'stellar_radius',
                        'depth_per_duration', 'planet_star_ratio', 'temp_insol_ratio',
                        'period_snr', 'depth_snr', 'depth_consistency', 'orbital_speed',
                        'semi_major_axis', 'stellar_density', 'log_period', 'in_habitable_zone',
                        'planet_type', 'period_depth_interaction'
                    ]
                    
                    engineered_df = engineered_df.dropna(subset=feature_names)
                    X = engineered_df[feature_names]
                    
                    # Create CUSTOM model with slider values
                    from xgboost import XGBClassifier
                    from sklearn.model_selection import train_test_split
                    
                    analysis_model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric='logloss'
                    )
                    
                    # Train on uploaded data
                    if 'disposition' in engineered_df.columns:
                        y_labels = (engineered_df['disposition'].str.contains('CONFIRMED|CP', case=False, na=False)).astype(int)
                        
                        if len(y_labels[y_labels == 1]) > 10 and len(y_labels[y_labels == 0]) > 10:
                            X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
                            analysis_model.fit(X_train, y_train)
                            predictions = analysis_model.predict(X_test)
                            probabilities = analysis_model.predict_proba(X_test)
                            X_results = X_test
                            engineered_results = engineered_df.loc[X_test.index]
                        else:
                            # Not enough labels, use pre-trained model
                            predictions = model.predict(X)
                            probabilities = model.predict_proba(X)
                            X_results = X
                            engineered_results = engineered_df
                    else:
                        # No labels, use pre-trained model
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)
                        X_results = X
                        engineered_results = engineered_df
                    
                    star_names = engineered_results['star_name'].values if 'star_name' in engineered_results.columns else [f"Star_{i}" for i in range(len(predictions))]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📈 Analysis Results")
                    st.caption(f"Analyzed with: {n_estimators} trees, depth {max_depth}, learning rate {learning_rate}")
                    
                    # Summary
                    total_stars = len(predictions)
                    planets_found = predictions.sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Stars", f"{total_stars:,}")
                    col2.metric("Planets Detected", f"{planets_found:,}")
                    col3.metric("Non-Planets", f"{total_stars - planets_found:,}")
                    col4.metric("Avg Confidence", f"{(probabilities.max(axis=1) * 100).mean():.1f}%")
                    
                    # Planets table
                    st.markdown("### 🌍 Detected Planets")
                    
                    planet_indices = predictions == 1
                    if planet_indices.sum() > 0:
                        planet_df = pd.DataFrame({
                            'Star Name': star_names[planet_indices],
                            'Confidence': (probabilities[planet_indices, 1] * 100).round(2),
                            'Type': engineered_results.loc[engineered_results.index[planet_indices], 'planet_type'].map({
                                1: 'Earth-like', 2: 'Super-Earth', 3: 'Neptune', 4: 'Jupiter'
                            }).values,
                            'In HZ': engineered_results.loc[engineered_results.index[planet_indices], 'in_habitable_zone'].map({1: 'Yes', 0: 'No'}).values
                        })
                        st.dataframe(planet_df, use_container_width=True, height=300)
                    else:
                        st.info("No planets detected in this dataset.")
                    
                    # Visualizations
                    st.markdown("### 📊 Visualizations")
                    
                    from utils.visualizations import (
                        create_confusion_matrix, 
                        create_prediction_distribution,
                        create_confidence_histogram,
                        create_feature_importance
                    )
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "Distribution", "Confidence", "Features"])
                    
                    with tab1:
                        st.plotly_chart(create_confusion_matrix(predictions, planets_found, total_stars), use_container_width=True)
                    
                    with tab2:
                        st.plotly_chart(create_prediction_distribution(predictions), use_container_width=True)
                    
                    with tab3:
                        st.plotly_chart(create_confidence_histogram(probabilities.max(axis=1) * 100), use_container_width=True)
                    
                    with tab4:
                        st.plotly_chart(create_feature_importance(analysis_model if 'analysis_model' in locals() else model, feature_names), use_container_width=True)
                    
                    # Download
                    st.markdown("### 💾 Download Results")
                    
                    results_df = pd.DataFrame({
                        'Star_Name': star_names,
                        'Prediction': ['Planet' if p == 1 else 'No Planet' for p in predictions],
                        'Confidence': (probabilities.max(axis=1) * 100).round(2)
                    })
                    
                    st.download_button(
                        "📥 Download Results as CSV",
                        results_df.to_csv(index=False),
                        "exoplanet_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
