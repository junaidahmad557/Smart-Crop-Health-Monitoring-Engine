import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Web Page Title Configuration
st.set_page_config(page_title="Smart Crop Monitor App", layout="centered")
st.title("🌾 Smart Crop Health Monitoring Engine")
st.write("This live production application uses a Supervised Linear Regression model to calculate economic crop yield destruction scores based on real-time environmental metrics.")

# 2. Train the Background Engine (Project 2 Logic)
np.random.seed(42)
num_farms = 500
temperature = np.random.uniform(15.0, 42.0, num_farms)
humidity = np.random.uniform(30.0, 95.0, num_farms)
soil_moisture = np.random.uniform(10.0, 60.0, num_farms)
raw_damage = (0.6 * temperature) + (0.4 * humidity) - (0.5 * soil_moisture) + np.random.normal(0, 3, num_farms)
crop_damage_percent = np.clip(raw_damage, 0.0, 100.0)

X = pd.DataFrame({'Air_Temperature': temperature, 'Air_Humidity': humidity, 'Soil_Moisture_Level': soil_moisture})
y = crop_damage_percent
severity_model = LinearRegression().fit(X, y)

# 3. User Interface Controls (Sliders)
st.sidebar.header("Live Farm Telemetry Inputs")
user_temp = st.sidebar.slider("Air Temperature (°C)", 15.0, 42.0, 30.0)
user_humid = st.sidebar.slider("Air Humidity (%)", 30.0, 95.0, 60.0)
user_soil = st.sidebar.slider("Soil Moisture Level", 10.0, 60.0, 35.0)

# 4. Generate Real-Time Prediction Output
input_data = pd.DataFrame([[user_temp, user_humid, user_soil]], columns=['Air_Temperature', 'Air_Humidity', 'Soil_Moisture_Level'])
live_prediction = severity_model.predict(input_data)

# 5. Display Dynamic Results Metrics Grid
st.markdown("---")
st.metric(label="Calculated Yield Destruction Risk Score", value=f"{live_prediction:.2f}%")

if live_prediction > 50.0:
    st.error("⚠️ CRITICAL ALERT: Environmental parameters indicate severe disease propagation risks. High crop loss expected.")
elif live_prediction > 25.0:
    st.warning("⚠️ MODERATE WARNING: Disease vector development conditions detected. Monitor field metrics closely.")
else:
    st.success("✅ SYSTEM NORMAL: Crop conditions verified stable. Minimum structural risk detected.")
