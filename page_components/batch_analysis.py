"""
Batch Analysis Page - Upload multiple CSV files and analyze
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import smart_csv_reader, detect_mission_type, standardize_columns, engineer_features

def show(model):
    """Display batch analysis page"""
    
    st.title("📊 Batch Analysis")
    st.markdown("Upload NASA Kepler or TESS data for comprehensive exoplanet detection")
    
    # MULTI-FILE UPLOAD
    uploaded_files = st.file_uploader(
        "Choose CSV files (Kepler, TESS, or both)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload multiple files. Total limit: 200MB for all files combined. We auto-detect each file's format."
    )
    
    # Show upload info
    if uploaded_files:
        total_size = sum(file.size for file in uploaded_files) / (1024*1024)
        if total_size > 200:
            st.error(f"⚠️ Total size ({total_size:.1f}MB) exceeds 200MB limit.")
            uploaded_files = None
        else:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded ({total_size:.1f}MB total)")
    
    st.markdown("---")
    
    # HYPERPARAMETERS
    st.markdown("### ⚙️ Model Settings")
    st.info("💡 Adjust before analyzing (optional).")
    
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    
    with st.expander("🔧 Customize Hyperparameters", expanded=False):
        st.warning("⚠️ For advanced users")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("Trees", 50, 500, 300, 50, key=f'n_est_{st.session_state.reset_counter}')
        
        with col2:
            max_depth = st.slider("Depth", 3, 10, 5, 1, key=f'depth_{st.session_state.reset_counter}')
        
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.12, 0.01, key=f'lr_{st.session_state.reset_counter}')
        
        if st.button("🔄 Reset"):
            st.session_state.reset_counter += 1
            st.rerun()
        
        comparison_df = pd.DataFrame({
            'Parameter': ['Trees', 'Depth', 'LR'],
            'Yours': [n_estimators, max_depth, learning_rate],
            'Optimal': [300, 5, 0.12],
            'Impact': [
                '=' if n_estimators == 300 else ('↑' if n_estimators > 300 else '↓'),
                '=' if max_depth == 5 else ('↑' if max_depth > 5 else '↓'),
                '=' if abs(learning_rate - 0.12) < 0.001 else ('↑' if learning_rate > 0.12 else '↓')
            ]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ANALYZE BUTTON
    if st.button("🚀 Analyze Data", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("⚠️ Upload files first!")
        else:
            with st.spinner(f"🔍 Processing {len(uploaded_files)} file(s)..."):
                try:
                    # Process all files
                    all_dfs = []
                    file_info = []
                    
                    for file in uploaded_files:
                        file.seek(0)
                        lines = file.readlines()
                        
                        data_start = 0
                        for i, line in enumerate(lines):
                            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                            if not line_str.strip().startswith('#'):
                                data_start = i
                                break
                        
                        file.seek(0)
                        df = pd.read_csv(file, skiprows=data_start, low_memory=False)
                        
                        mission = detect_mission_type(df.columns)
                        if mission:
                            std_df, _ = standardize_columns(df, mission)
                            if std_df is not None:
                                all_dfs.append(std_df)
                                file_info.append({'name': file.name, 'mission': mission, 'stars': len(std_df)})
                    
                    if not all_dfs:
                        st.error("❌ No valid files detected.")
                        return
                    
                    # Show file breakdown
                    st.success(f"✅ Processed {len(all_dfs)} file(s)")
                    for info in file_info:
                        st.caption(f"  • {info['name']}: {info['mission'].upper()}, {info['stars']:,} stars")
                    
                    # Combine all
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    
                    base_features = ['period', 'depth', 'duration', 'planet_radius', 'equilibrium_temp',
                                   'insolation_flux', 'stellar_temp', 'stellar_gravity', 'stellar_radius']
                    
                    combined_df = combined_df.dropna(subset=base_features)
                    engineered_df = engineer_features(combined_df)
                    
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
                    
                    # Predict
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)
                    star_names = engineered_df['star_name'].values if 'star_name' in engineered_df.columns else [f"Star_{i}" for i in range(len(X))]
                    
                    # Results
                    st.markdown("---")
                    st.markdown("## 📈 Analysis Results")
                    st.caption(f"Combined {len(uploaded_files)} file(s) | Settings: {n_estimators} trees, depth {max_depth}, LR {learning_rate}")
                    
                    total_stars = len(predictions)
                    planets_found = predictions.sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Stars", f"{total_stars:,}")
                    col2.metric("Planets", f"{planets_found:,}")
                    col3.metric("Non-Planets", f"{total_stars - planets_found:,}")
                    col4.metric("Avg Confidence", f"{(probabilities.max(axis=1) * 100).mean():.1f}%")
                    
                    # Planets table
                    st.markdown("### 🌍 Detected Planets")
                    
                    planet_indices = predictions == 1
                    if planet_indices.sum() > 0:
                        planet_df = pd.DataFrame({
                            'Star': star_names[planet_indices],
                            'Confidence': (probabilities[planet_indices, 1] * 100).round(2),
                            'Type': engineered_df.loc[engineered_df.index[planet_indices], 'planet_type'].map({
                                1: 'Earth-like', 2: 'Super-Earth', 3: 'Neptune', 4: 'Jupiter'
                            }).values
                        })
                        st.dataframe(planet_df, use_container_width=True, height=300)
                    else:
                        st.info("No planets detected.")
                    
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
                        st.plotly_chart(create_feature_importance(model, feature_names), use_container_width=True)
                    
                    # Download
                    st.markdown("### 💾 Download")
                    
                    results_df = pd.DataFrame({
                        'Star': star_names,
                        'Prediction': ['Planet' if p == 1 else 'No Planet' for p in predictions],
                        'Confidence': (probabilities.max(axis=1) * 100).round(2)
                    })
                    
                    st.download_button(
                        "📥 Download Results",
                        results_df.to_csv(index=False),
                        "exoplanet_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
