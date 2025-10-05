"""
About Page - Information about the project
"""

import streamlit as st

def show():
    """Display about page"""
    
    st.title("📚 About Exoplanet Hunter")
    
    st.markdown("""
    ## What is This?
    
    Exoplanet Hunter is an AI-powered system that automatically detects exoplanets from NASA telescope data. 
    Using machine learning trained on thousands of confirmed discoveries, our tool can instantly analyze star 
    brightness measurements and identify potential planetary systems with 93% accuracy.
    
    ---
    
    ## How It Works: The Transit Method
    
    When a planet passes in front of its star (a "transit"), it blocks a tiny amount of starlight. Our AI 
    learned to recognize these patterns and distinguish real planets from false positives like:
    - Eclipsing binary stars
    - Instrumental noise
    - Stellar variability
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### The Science
        
        **Transit Detection:**
        - Planet blocks ~0.01-1% of starlight
        - Creates periodic dips in brightness
        - Depth reveals planet size
        - Duration reveals orbit
        
        **Our Approach:**
        - 9 core NASA measurements
        - 13 engineered features based on physics
        - XGBoost machine learning
        - 93.03% detection accuracy
        """)
    
    with col2:
        st.markdown("""
        ### The Features (22 Total)
        
        **Original (9):**
        - Orbital period, transit depth/duration
        - Planet radius, equilibrium temperature
        - Stellar properties (temp, gravity, radius)
        
        **Engineered (13):**
        - Signal-to-noise ratios
        - Transit physics validation
        - Orbital mechanics
        - Habitable zone calculations
        - Planet type classification
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## Model Performance
    
    Our XGBoost model achieves exceptional accuracy through systematic optimization:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Accuracy", "93.03%")
        st.metric("Cross-Validation", "91.35% ± 4.93%")
    
    with col2:
        st.metric("Precision", "93.5%")
        st.metric("Recall", "90.9%")
    
    with col3:
        st.metric("False Positive Rate", "3.5%")
        st.metric("Training Time", "0.30 seconds")
    
    st.markdown("---")
    
    st.markdown("""
    ## Who Can Use This?
    
    **🔬 Researchers**
    - Analyze new telescope observations
    - Process large datasets quickly
    - Export results for publications
    
    **🚀 NASA Scientists**
    - Prioritize follow-up observations
    - Process mission catalogs automatically
    - Reduce manual review time by 90%
    
    **🎓 Students & Educators**
    - Learn about exoplanet detection
    - Experiment with real NASA data
    - Understand machine learning applications
    
    **🌌 Space Enthusiasts**
    - Explore the cosmos
    - Discover patterns in real data
    - Contribute to citizen science
    
    ---
    
    ## Data Sources
    
    This tool works with official NASA data from:
    
    **Kepler Mission** (2009-2018)
    - 9,564 stars analyzed
    - 2,700+ exoplanets confirmed
    - Deep survey of Cygnus-Lyra region
    
    **TESS Mission** (2018-present)
    - All-sky survey (ongoing)
    - 400+ exoplanets confirmed
    - Brighter, closer stars
    
    Our model is trained on data from both missions, making it versatile and robust across different 
    stellar types and observing conditions.
    
    ---
    
    ## Technical Details
    
    **Algorithm:** XGBoost (Extreme Gradient Boosting)
    - 200 decision trees
    - Maximum depth: 6 levels
    - Learning rate: 0.1
    
    **Validation:**
    - 5-fold cross-validation
    - 80/20 train-test split
    - Stratified sampling (maintains planet ratio)
    
    **Feature Engineering:**
    - Physics-based calculations (Kepler's laws, transit photometry)
    - Signal-to-noise ratios (using NASA error columns)
    - Habitable zone determination (stellar luminosity-based)
    - Planet type classification (radius-based)
    
    ---
    
    ## Limitations
    
    **What we can detect:**
    - ✅ Transiting exoplanets (planets that cross in front of their star)
    - ✅ From Kepler or TESS mission data
    - ✅ With sufficient signal-to-noise ratio
    
    **What we cannot detect:**
    - ❌ Non-transiting planets (wrong orbital orientation)
    - ❌ Planets found via other methods (radial velocity, direct imaging)
    - ❌ Planets around very faint or very bright stars (data quality issues)
    
    **Accuracy considerations:**
    - 93% accuracy means ~7% error rate
    - False positives: 3.5% (some non-planets flagged as planets)
    - False negatives: 9.1% (some planets missed)
    - Always verify results with follow-up observations!
    
    ---
    
    ## Built For
    
    **NASA Space Apps Challenge 2025**
    
    This project demonstrates how machine learning can accelerate exoplanet discovery, making the search 
    for Earth-like worlds faster, more efficient, and more accessible to researchers worldwide.
    
    ---
    
    ## Tech Stack
    
    - **Backend:** Python with XGBoost
    - **Frontend:** Streamlit
    - **Visualizations:** Plotly (interactive charts)
    - **Data:** NASA Exoplanet Archive
    - **Deployment:** Streamlit Community Cloud
    
    ---
    
    ## 👥 Meet Team 2muchcoffe
    
    We're three first-year students from University of Kerala who came together for NASA Space Apps Challenge 2025, 
    combining diverse academic backgrounds with a shared passion for space exploration and technology.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Munjid V H**  
        *Team Lead & ML Engineer*
        
        - BBA Student, University of Kerala
        - Self-taught in AI/ML and Python
        - Used AI to learn exoplanet detection science
        - Developed the complete ML pipeline and web application
        """)
    
    with col2:
        st.markdown("""
        **Nazeeh Nabhan V**  
        *Presentation Lead*
        
        - Computer Science Engineering
        - UC Engineering Kariavattom
        - Crafted presentation materials and visual storytelling
        """)
    
    with col3:
        st.markdown("""
        **Abhishek M Raj**  
        *Repository Manager*
        
        - Computer Science Engineering
        - UC Engineering Kariavattom
        - Django/Flask expert
        - Managed GitHub repository and deployment
        """)
    
    st.markdown("""
    ---
    
    ## 💡 Our Approach
    
    As first-year students tackling a complex astrophysics challenge, we took an unconventional approach. 
    Coming from a commerce background, Munjid used AI as a teacher—leveraging prompt engineering and machine 
    learning not just to build the solution, but to understand the underlying science of exoplanet detection. 
    This meta approach—using AI to learn AI for space science—represents how accessible advanced technology 
    has become for motivated learners from any background.
    
    ---
    
    <div style='text-align: center; padding: 20px;'>
    <p style='color: #95a5a6;'>Built with ❤️ for NASA Space Apps Challenge 2025</p>
    <p style='color: #95a5a6;'>Helping humanity discover new worlds 🌟</p>
    <p style='color: #95a5a6; font-style: italic;'>*Fueled by curiosity and too much coffee ☕*</p>
    </div>
    """, unsafe_allow_html=True)
