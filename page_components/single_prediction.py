"""
Single Star Prediction Page
"""

import streamlit as st
import numpy as np

def show(model):
    """Display single star prediction page"""
    
    st.title("🔮 Single Star Prediction")
    st.markdown("Enter custom measurements to test the AI model")
    
    st.markdown("### Star Measurements")
    st.markdown("Enter the 9 core measurements from NASA telescope data:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.number_input(
            "Orbital Period (days)",
            min_value=0.0,
            value=10.0,
            help="Time for planet to complete one orbit around its star"
        )
        
        depth = st.number_input(
            "Transit Depth (ppm)",
            min_value=0.0,
            value=200.0,
            help="How much the star dims when planet passes in front (parts per million)"
        )
        
        duration = st.number_input(
            "Transit Duration (hours)",
            min_value=0.0,
            value=3.0,
            help="How long the transit lasts"
        )
        
        prad = st.number_input(
            "Planet Radius (Earth radii)",
            min_value=0.0,
            value=1.5,
            help="Size of planet compared to Earth (1.0 = same as Earth)"
        )
        
        teq = st.number_input(
            "Equilibrium Temperature (K)",
            min_value=0.0,
            value=400.0,
            help="Planet's equilibrium temperature in Kelvin"
        )
    
    with col2:
        insol = st.number_input(
            "Insolation Flux (Earth flux)",
            min_value=0.0,
            value=1.5,
            help="Stellar radiation received (1.0 = same as Earth)"
        )
        
        steff = st.number_input(
            "Stellar Temperature (K)",
            min_value=0.0,
            value=5500.0,
            help="Temperature of the host star"
        )
        
        slogg = st.number_input(
            "Stellar Gravity (log10)",
            min_value=0.0,
            value=4.5,
            help="Surface gravity of the star (log scale)"
        )
        
        srad = st.number_input(
            "Stellar Radius (Solar radii)",
            min_value=0.0,
            value=1.0,
            help="Size of star compared to our Sun (1.0 = same as Sun)"
        )
    
    # Predict button
    if st.button("🚀 Analyze Star", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            # Engineer features
            depth_per_duration = depth / (duration + 1e-10)
            planet_star_ratio = prad / (srad + 1e-10)
            temp_insol_ratio = teq / (insol + 1e-10)
            
            period_snr = period / 0.01
            depth_snr = depth / 1.0
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
            
            # Create feature array
            features = np.array([[
                period, depth, duration, prad, teq, insol, steff, slogg, srad,
                depth_per_duration, planet_star_ratio, temp_insol_ratio,
                period_snr, depth_snr, depth_consistency, orbital_speed,
                semi_major_axis, stellar_density, log_period, in_habitable_zone,
                planet_type, period_depth_interaction
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities[prediction] * 100
            
            # Display result
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")
            
            if prediction == 1:
                st.balloons()
                st.markdown(
                    '<div class="planet-detected">🌍 PLANET DETECTED!</div>',
                    unsafe_allow_html=True
                )
                st.success(f"**Confidence: {confidence:.2f}%**")
                
                # Additional info
                col1, col2 = st.columns(2)
                with col1:
                    planet_types = {1: 'Earth-like', 2: 'Super-Earth', 3: 'Neptune-like', 4: 'Jupiter-like'}
                    st.info(f"**Planet Type:** {planet_types[planet_type]}")
                with col2:
                    hz_status = "Yes ✅" if in_habitable_zone == 1 else "No"
                    st.info(f"**In Habitable Zone:** {hz_status}")
                
            else:
                st.markdown(
                    '<div class="no-planet">❌ No Planet Detected</div>',
                    unsafe_allow_html=True
                )
                st.info(f"**Confidence: {confidence:.2f}%**")
            
            # Confidence visualization
            st.markdown("### Confidence Breakdown")
            col1, col2 = st.columns(2)
            col1.metric("Planet Probability", f"{probabilities[1]*100:.2f}%")
            col2.metric("Non-Planet Probability", f"{probabilities[0]*100:.2f}%")
            
            # Progress bar
            st.progress(float(confidence / 100))
