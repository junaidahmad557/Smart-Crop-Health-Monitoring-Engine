import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image

# Page Settings
st.set_page_config(page_title="Smart Crop Health Hub", layout="centered")

st.title("🌾 Smart Crop Health Monitoring & Automation Engine")
st.write("Complete 3-Model AI Production Pipeline Dashboard")

# =====================================================================
# 1. MODELS KO LOAD KARNA
# =====================================================================
@st.cache_resource
def load_pipeline_models():
    regression = joblib.load("crop_severity_regression_model (1).pkl") 
    decision_tree = joblib.load("crop_automation_tree_model.pkl")
    return regression, decision_tree

try:
    reg_model, tree_model = load_pipeline_models()
    st.sidebar.success("✅ SYSTEM LOG: AI Models Connected!")
except Exception as e:
    st.sidebar.error("⚠️ File Missing Error! Check your .pkl files on GitHub.")

# =====================================================================
# SECTION 1: LIVE FARM TELEMETRY INPUTS (Model 2: Regression)
# =====================================================================
st.subheader("📊 Live Farm Telemetry Inputs")

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input("Air Temperature (°C)", min_value=15.0, max_value=42.0, value=23.71)
with col2:
    humidity = st.number_input("Air Humidity (%)", min_value=30.0, max_value=95.0, value=46.51)
with col3:
    moisture = st.number_input("Soil Moisture Level", min_value=10.0, max_value=60.0, value=24.92)

# --- [MODEL 2 EXECUTION] ---
telemetry_data = pd.DataFrame([[temp, humidity, moisture]], columns=['Temperature', 'Humidity', 'Moisture'])
raw_prediction = reg_model.predict(telemetry_data)
severity_score = float(raw_prediction) if hasattr(raw_prediction, "__len__") else float(raw_prediction)

st.markdown("### Calculated Yield Destruction Risk Score")
st.error(f"⚠️ **{severity_score:.2f}%**")

# =====================================================================
# SECTION 2: DISEASE SELECTION / IMAGE DIAGNOSIS (With Sample Photos)
# =====================================================================
st.subheader("📸 Crop Disease Diagnostic Input")

# Teeno options de diye hain testing kay liye
mode = st.radio("Choose Input Mode:", [
    "Manual Selection (Fast Testing)", 
    "Use a Sample Test Photo (Pre-loaded)",
    "Upload Your Own Leaf Image"
])

disease_idx = 2 # Default: Healthy
image_to_show = None

if mode == "Manual Selection (Fast Testing)":
    disease_choice = st.selectbox(
        "Select Simulated Disease Type:",
        ["Early Blight (Disease ID: 0)", "Late Blight (Disease ID: 1)", "Healthy (Disease ID: 2)"]
    )
    if "Early Blight" in disease_choice:
        disease_idx = 0
    elif "Late Blight" in disease_choice:
        disease_idx = 1
    else:
        disease_idx = 2

elif mode == "Use a Sample Test Photo (Pre-loaded)":
    # User sample select karay ga aur back-end par automatic uski Disease_ID lock ho jaye gi
    sample_choice = st.selectbox(
        "Choose a Sample Potato Leaf to Test:",
        ["Sample 1: Sick Leaf (Late Blight Case)", "Sample 2: Infected Leaf (Early Blight Case)", "Sample 3: Clean Leaf (Healthy Case)"]
    )
    
    if "Late Blight" in sample_choice:
        disease_idx = 1
        st.info("ℹ️ System simulation locked to **Late Blight (ID: 1)** via sample vector.")
    elif "Early Blight" in sample_choice:
        disease_idx = 0
        st.info("ℹ️ System simulation locked to **Early Blight (ID: 0)** via sample vector.")
    else:
        disease_idx = 2
        st.info("ℹ️ System simulation locked to **Healthy Leaf (ID: 2)** via sample vector.")

else:
    uploaded_file = st.file_uploader("Upload Potato Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_show = Image.open(uploaded_file)
        st.image(image_to_show, caption="Uploaded Leaf Preview", width=250)
        st.warning("⚠️ Note: Connect your CNN .h5 model file to extract Disease ID automatically from image.")
        disease_idx = 2 

# =====================================================================
# SECTION 3: AUTOMATED BOTANICAL PRESCRIPTION (Model 3: Decision Tree)
# =====================================================================
st.subheader("🤖 Automated Production Pipeline Prescription")
st.write("Combining Disease ID and Telemetry Severity Score via Decision Tree Engine...")

# --- [MODEL 3 EXECUTION] ---
pipeline_vector = np.array([[disease_idx, severity_score]])
predicted_action_code = tree_model.predict(pipeline_vector)

actions_dictionary = {
    0: "🔴 CRITICAL ALERT (CODE 0): Apply Industrial Copper Fungicide within 24 hours.",
    1: "🟡 WARNING UPDATE (CODE 1): Apply Organic Neem Spray and decrease irrigation frequency.",
    2: "🟢 SYSTEM NORMAL (CODE 2): Crop matrix is verified healthy. Continuous automated monitoring active."
}

st.markdown("#### Execution Decision:")
if predicted_action_code == 0:
    st.error(actions_dictionary[predicted_action_code])
elif predicted_action_code == 1:
    st.warning(actions_dictionary[predicted_action_code])
else:
    st.success(actions_dictionary[predicted_action_code])
