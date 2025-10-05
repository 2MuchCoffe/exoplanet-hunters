"""
Exoplanet Hunter - Streamlit Web Application
NASA Space Apps Challenge

Dark-themed web app for exoplanet detection using XGBoost
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Exoplanet Hunter 🌍",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for dark theme
def load_custom_css():
    st.markdown("""
    <style>
    /* Dark theme enhancements */
    .stApp {
        background-color: #0a0e27;
    }
    
    .metric-card {
        background-color: #1a1f3a;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4a90e2;
    }
    
    h1, h2, h3 {
        color: #4a90e2 !important;
    }
    
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #357abd;
        transform: scale(1.02);
    }
    
    .planet-detected {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    
    .no-planet {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# Load XGBoost model
@st.cache_resource
def load_model():
    """Load the trained XGBoost model with retry logic"""
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with open('results/xgboost_model.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retry
            else:
                return None  # Give up after 3 tries

model = load_model()

# Sidebar navigation
with st.sidebar:
    st.markdown("# 🌍 Exoplanet Hunter")
    st.markdown("---")
    
    # Custom CSS to hide radio circles ONLY in sidebar navigation
    st.markdown("""
    <style>
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 8px 12px;
        border-radius: 5px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(74, 144, 226, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Batch Analysis", "🔍 Star Lookup", "🔮 Single Prediction", "📚 About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Model Stats")
    st.metric("Accuracy", "93.03%", "+4.28%")
    st.metric("Speed", "0.30s", "-38%")
    st.metric("Features", "22")
    
    st.markdown("---")
    st.markdown("**NASA Space Apps 2025**")

# Route to pages
if page == "🏠 Home":
    from page_components import home
    home.show()
elif page == "📊 Batch Analysis":
    from page_components import batch_analysis
    batch_analysis.show(model)
elif page == "🔍 Star Lookup":
    from page_components import star_lookup
    star_lookup.show(model)
elif page == "🔮 Single Prediction":
    from page_components import single_prediction
    single_prediction.show(model)
elif page == "📚 About":
    from page_components import about
    about.show()
