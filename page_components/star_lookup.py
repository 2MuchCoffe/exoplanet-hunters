"""
Star Lookup Page - Search and analyze specific stars from NASA database
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

@st.cache_data
def load_star_database():
    """Load star databases from Kepler and TESS"""
    try:
        # Load Kepler data using smart reader
        from utils.data_processor import smart_csv_reader
        kepler_df, skip_lines = smart_csv_reader('data/cumulative.csv')
        kepler_df = kepler_df[kepler_df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
        kepler_df = kepler_df.dropna(subset=['koi_period', 'koi_depth', 'koi_duration', 'koi_prad'])
        
        # Add star names
        kepler_df['display_name'] = kepler_df['kepler_name'].fillna(kepler_df['kepoi_name'])
        kepler_df['mission'] = 'Kepler'
        
        # Load TESS data using smart reader
        try:
            tess_df, tess_skip = smart_csv_reader('data/tess_data.csv')
            tess_df = tess_df[tess_df['tfopwg_disp'].isin(['PC', 'FP', 'CP'])]
            tess_df['display_name'] = 'TOI-' + tess_df['toi'].astype(str)
            tess_df['mission'] = 'TESS'
        except:
            tess_df = pd.DataFrame()
        
        # Combine
        return kepler_df, tess_df
    except Exception as e:
        st.error(f"Error loading star database: {e}")
        return pd.DataFrame(), pd.DataFrame()

def show(model):
    """Display star lookup page"""
    
    st.title("🔍 Star Lookup & Analysis")
    st.markdown("Search for specific stars from NASA's Kepler and TESS missions")
    
    # Load database
    kepler_df, tess_df = load_star_database()
    
    if len(kepler_df) == 0:
        st.error("⚠️ Star database not loaded. Please ensure data files are available.")
        return
    
    # Create star list
    kepler_stars = kepler_df['display_name'].dropna().unique()[:1000]  # Limit for performance
    all_stars = sorted([str(s) for s in kepler_stars if str(s) != 'nan'])
    
    # Search box
    st.markdown("### Search for a Star")
    selected_star = st.selectbox(
        "Type to search (showing first 1,000 Kepler stars)",
        options=[''] + all_stars,
        help="Select a star to see its complete analysis"
    )
    
    if selected_star:
        # Find star data
        star_data = kepler_df[kepler_df['display_name'] == selected_star].iloc[0]
        
        st.markdown("---")
        
        # Star Profile Header
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"### 🌟 {selected_star}")
        col2.metric("Mission", "Kepler")
        col3.metric("Status", star_data['koi_disposition'])
        
        st.markdown("---")
        
        # AI Prediction Section
        st.markdown("## 🤖 AI Analysis")
        
        try:
            # Extract measurements
            measurements = {
                'period': star_data['koi_period'],
                'depth': star_data['koi_depth'],
                'duration': star_data['koi_duration'],
                'prad': star_data['koi_prad'],
                'teq': star_data['koi_teq'],
                'insol': star_data['koi_insol'],
                'steff': star_data['koi_steff'],
                'slogg': star_data['koi_slogg'],
                'srad': star_data['koi_srad']
            }
            
            # Check for NaN
            if any(pd.isna(v) for v in measurements.values()):
                st.warning("⚠️ Some measurements missing for this star. Showing available data only.")
            else:
                # Engineer features (simplified)
                depth_per_duration = measurements['depth'] / (measurements['duration'] + 1e-10)
                planet_star_ratio = measurements['prad'] / (measurements['srad'] + 1e-10)
                temp_insol_ratio = measurements['teq'] / (measurements['insol'] + 1e-10)
                
                features = np.array([[
                    measurements['period'], measurements['depth'], measurements['duration'],
                    measurements['prad'], measurements['teq'], measurements['insol'],
                    measurements['steff'], measurements['slogg'], measurements['srad'],
                    depth_per_duration, planet_star_ratio, temp_insol_ratio,
                    measurements['period']/0.01, measurements['depth']/1.0,
                    measurements['depth']/((measurements['prad']/measurements['srad'])**2+1e-10),
                    (2*np.pi*measurements['srad'])/measurements['period'],
                    measurements['period']**(2/3),
                    measurements['slogg']/(measurements['srad']**2+1e-10),
                    np.log10(measurements['period']+1),
                    1, 2,  # Simplified habitable zone and planet type
                    measurements['period']*measurements['depth']
                ]])
                
                # Make prediction
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                confidence = probabilities[prediction] * 100
                
                # Display prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.success("### AI Prediction\n**PLANET DETECTED** ✅")
                    else:
                        st.info("### AI Prediction\n**No Planet**")
                
                with col2:
                    st.metric("AI Confidence", f"{confidence:.2f}%")
                
                with col3:
                    nasa_status = star_data['koi_disposition']
                    match = (prediction == 1 and 'CONFIRMED' in nasa_status) or (prediction == 0 and 'FALSE' in nasa_status)
                    st.metric("Match with NASA", "✅" if match else "❌")
            
            st.markdown("---")
            
            # Planet Properties
            st.markdown("## 🌍 Planet Properties")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Radius", f"{measurements['prad']:.2f} R⊕")
            col2.metric("Temperature", f"{measurements['teq']:.0f} K")
            col3.metric("Orbital Period", f"{measurements['period']:.2f} days")
            col4.metric("Transit Depth", f"{measurements['depth']:.0f} ppm")
            
            st.markdown("---")
            
            # Measurements Table
            st.markdown("## 📊 All Measurements")
            
            measurements_df = pd.DataFrame({
                'Measurement': [
                    'Orbital Period', 'Transit Depth', 'Transit Duration',
                    'Planet Radius', 'Equilibrium Temp', 'Insolation Flux',
                    'Stellar Temp', 'Stellar Gravity', 'Stellar Radius'
                ],
                'Value': [
                    f"{measurements['period']:.4f} days",
                    f"{measurements['depth']:.2f} ppm",
                    f"{measurements['duration']:.4f} hours",
                    f"{measurements['prad']:.4f} R⊕",
                    f"{measurements['teq']:.2f} K",
                    f"{measurements['insol']:.4f} Earth flux",
                    f"{measurements['steff']:.2f} K",
                    f"{measurements['slogg']:.4f} log10",
                    f"{measurements['srad']:.4f} R☉"
                ]
            })
            
            st.dataframe(measurements_df, use_container_width=True, hide_index=True)
            
            # Visualization: Feature Radar
            st.markdown("---")
            st.markdown("## 📊 Feature Analysis")
            
            # Normalized values for radar
            fig = go.Figure()
            
            categories = ['Period', 'Depth', 'Radius', 'Temperature', 'Insolation']
            values = [
                min(measurements['period']/100, 1),
                min(measurements['depth']/1000, 1),
                min(measurements['prad']/10, 1),
                min(measurements['teq']/2000, 1),
                min(measurements['insol']/100, 1)
            ]
            
            values = values + values[:1]  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=selected_star,
                line_color='#4a90e2'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=True,
                title=f"{selected_star} - Feature Profile",
                template='plotly_dark',
                paper_bgcolor='#1a1f3a',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error analyzing star: {e}")
    
    else:
        st.info("👆 Select a star from the dropdown above to see its complete analysis")
        
        # Show sample stars
        st.markdown("---")
        st.markdown("### 🌟 Featured Stars")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Earth-like Planets**\nSearch for stars with small radii (<1.5 R⊕)")
        
        with col2:
            st.success("**Hot Jupiters**\nSearch for stars with large radii (>10 R⊕)")
        
        with col3:
            st.warning("**Habitable Zone**\nSearch for stars with moderate temperatures")
