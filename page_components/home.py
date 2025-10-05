"""
Homepage for Exoplanet Hunter
"""

import streamlit as st

def show():
    """Display the homepage"""
    
    # Hero section
    st.markdown("<h1 style='text-align: center;'>🌍 Exoplanet Hunter</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #e0e6ed;'>Discover Exoplanets with AI-Powered Detection</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #95a5a6; font-size: 18px;'>Analyze NASA telescope data instantly with 93% accuracy</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature cards
    st.markdown("### Why Exoplanet Hunter?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### ⚡ Fast\n\nAnalyze thousands of stars in seconds. Our XGBoost model trains in 0.30 seconds and provides instant predictions.")
    
    with col2:
        st.success("### 🎯 Accurate\n\n93.03% accuracy on NASA data. Validated with 5-fold cross-validation (91.35% ± 4.93%).")
    
    with col3:
        st.warning("### 📚 Educational\n\nLearn about exoplanet detection. Explore features, adjust hyperparameters, see what-if scenarios.")
    
    st.markdown("---")
    
    # How it works
    st.markdown("### How It Works")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **1. Upload NASA Data**
        - Accepts Kepler or TESS mission CSVs
        - Auto-detects data format
        - No file naming rules required
        
        **2. AI Analysis**
        - 22 physics-based features
        - XGBoost classification
        - Instant predictions
        
        **3. Results & Insights**
        - Comprehensive dashboard
        - Star names and confidence scores
        - Interactive visualizations
        - Download reports
        """)
    
    with col2:
        st.markdown("""
        **Features:**
        - 🔬 Batch analysis (upload CSV)
        - 🔮 Single star prediction
        - 🎛️ Hyperparameter tuning
        - 🧪 What-if scenarios
        - 📊 4 interactive charts
        - 💾 Download results (CSV + PDF)
        
        **Supports:**
        - NASA Kepler mission data
        - NASA TESS mission data
        - Combined datasets
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Test Accuracy", "93.03%", "+4.28%")
    col2.metric("Cross-Validation", "91.35%", "±4.93%")
    col3.metric("False Positives", "3.5%", "-44% from baseline")
    col4.metric("Training Time", "0.30s", "38% faster")
    
    st.markdown("---")
    
    # Call to action
    st.markdown("### Get Started")
    st.markdown("Choose an option from the sidebar to begin:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**📊 Batch Analysis**\n\nUpload a CSV file with multiple stars for comprehensive analysis.")
    
    with col2:
        st.success("**🔮 Single Prediction**\n\nEnter measurements for one star to get instant prediction.")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #95a5a6; padding: 20px;'>
    <p>Built for NASA Space Apps Challenge 2025</p>
    <p>Using official NASA Kepler and TESS mission data</p>
    </div>
    """, unsafe_allow_html=True)
