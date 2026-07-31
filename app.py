import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import tensorflow as tf
import os
import gdown

# Custom Styling inject karne ka safe tareeqa bina Python crash kiye
st.html("<style> .main-title { font-size: 38px !important; font-weight: 800 !important; color: #1E3A8A; margin-bottom: 5px; text-align: center; } .sub-title { font-size: 16px !important; color: #6B7280; text-align: center; margin-bottom: 30px; } .section-header { font-size: 22px !important; font-weight: 700 !important; color: #1F2937; border-left: 5px solid #3B82F6; padding-left: 12px; margin-top: 35px; margin-bottom: 15px; } .metric-card-container { display: flex; justify-content: space-between; gap: 15px; margin-bottom: 20px; } .metric-card { flex: 1; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); } .metric-label { font-size: 12px; font-weight: 600; color: #4B5563; text-transform: uppercase; letter-spacing: 0.5px; } .metric-value { font-size: 24px; font-weight: 700; color: #111827; margin-top: 5px; } .risk-panel { background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%); border: 1px solid #FCA5A5; border-radius: 16px; padding: 25px; text-align: center; margin-top: 15px; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(239,68,68,0.1); } .risk-label { font-size: 14px; font-weight: 700; color: #991B1B; text-transform: uppercase; letter-spacing: 1px; } .risk-value { font-size: 42px; font-weight: 900; color: #DC2626; margin: 8px 0; } .risk-subtext { font-size: 13px; color: #7F1D1D; opacity: 0.8; } .prescription-box-0 { background: #FEF2F2; border-left: 6px solid #EF4444; padding: 20px; border-radius: 8px; color: #991B1B; } .prescription-box-1 { background: #FFFBEB; border-left: 6px solid #F59E0B; padding: 20px; border-radius: 8px; color: #92400E; } .prescription-box-2 { background: #ECFDF5; border-left: 6px solid #10B981; padding: 20px; border-radius: 8px; color: #065F46; } </style>")

# App branding top headers
st.markdown('<div class="main-title">🥔 PotatoGuard AI™ Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Industrial Production-Grade 3-Model Optimization Pipeline</div>', unsafe_allow_html=True)

# =====================================================================
# BACKEND INITIALIZATION (MODEL LOADING)
# =====================================================================
@st.cache_resource
def load_pipeline_models():
    regression = joblib.load("crop_severity_regression_model (1).pkl") 
    decision_tree = joblib.load("crop_automation_tree_model.pkl")
    return regression, decision_tree

try:
    reg_model, tree_model = load_pipeline_models()
    st.sidebar.success("💡 Ecosystem Online: 3-Models Operational")
except Exception as e:
    st.sidebar.error("⚠️ Infrastructure Offline! Check .pkl filenames on GitHub.")

# =====================================================================
# SECTION 1: LIVE FARM TELEMETRY METRICS (Model 2: Regression)
# =====================================================================
st.html('<div class="section-header">📊 Real-Time Environmental Telemetry</div>')

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input("Air Temp (°C)", min_value=15.0, max_value=42.0, value=23.71, step=0.1)
with col2:
    humidity = st.number_input("Air Humidity (%)", min_value=30.0, max_value=95.0, value=46.51, step=0.1)
with col3:
    moisture = st.number_input("Soil Moisture Level", min_value=10.0, max_value=60.0, value=24.92, step=0.1)

telemetry_data = pd.DataFrame([[temp, humidity, moisture]], columns=['Temperature', 'Humidity', 'Moisture'])
try:
    raw_prediction = reg_model.predict(telemetry_data)
except:
    telemetry_inputs = np.array([[temp, humidity, moisture]])
    raw_prediction = reg_model.predict(telemetry_inputs)

try:
    severity_score = float(raw_prediction.item())
except:
    severity_score = float(raw_prediction)

st.markdown(f'<div class="risk-panel"><div class="risk-label">⚠️ Calculated Yield Destruction Risk</div><div class="risk-value">{severity_score:.2f}%</div><div class="risk-subtext">Continuous real-time loss function update mapped from active telemetry weights.</div></div>', unsafe_allow_html=True)

# =====================================================================
# SECTION 2: DEEP LEARNING IMAGE DIAGNOSTICS (Model 1: CNN Powered)
# =====================================================================
@st.cache_resource
def download_and_load_cnn():
    # File ka naam badal kar 'potato_final_network.h5' rakhlein taake ziddi cache bypass ho jaye
    model_path = 'potato_final_network.h5'
    
    if os.path.exists(model_path):
        if os.path.getsize(model_path) < 10 * 1024 * 1024: 
            os.remove(model_path)
            
    if not os.path.exists(model_path):
        with st.spinner("Downloading trained CNN model layers via secure link... Please wait."):
            # 100% Correct and direct layout link parameters:
            file_id = '1kuB-PC2qg742LTTvdPmArZJ-HZgFQKm3'
            url = f'https://google.com{file_id}' 
            gdown.download(url, model_path, quiet=False)
                
    return tf.keras.models.load_model(model_path)

try:
    cnn_model = download_and_load_cnn()
    cnn_online = True
except Exception as e:
    cnn_online = False
    st.error(f"⚠️ CNN Model load nahi ho saka. Error detail: {e}")

def predict_crop_health(image):
    classes = ["Potato Late Blight", "Potato Early Blight", "Healthy"]
    img = image.resize((224, 224)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    predictions = cnn_model.predict(img_array)
    predicted_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    return classes[predicted_idx], predicted_idx, confidence

def get_prescription(class_idx):
    prescriptions = {
        0: {
            "title": "🚨 High Risk Treatment Plan (Late Blight Detected)",
            "box_class": "prescription-box-0",
            "details": "Immediate action required! Apply systemic fungicides containing Mancozeb or Metalaxyl. Remove infected plants instantly to stop fungal spore spread."
        },
        1: {
            "title": "⚠️ Medium Risk Management Plan (Early Blight Detected)",
            "box_class": "prescription-box-1",
            "details": "Apply Copper-based fungicides. Improve plant spacing for better air circulation and avoid overhead watering on leaves."
        },
        2: {
            "title": "🌿 Organic Preventive Treatment (Healthy Leaf)",
            "box_class": "prescription-box-2",
            "details": "Fasal bilkul theek hai! Spray **Neem Oil / Neem Extract** mixture (1-2 teaspoons per liter of water with liquid soap) every 14 days as a natural organic barrier against pests and fungi."
        }
    }
    return prescriptions[class_idx]

# --- UI Interface Render ---
st.markdown('<div class="section-header">📸 Image Diagnostic Engine (CNN Powered)</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a Potato Leaf Image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Specimen Target", use_container_width=True)
    
    if st.button("🔮 Predict Now", type="primary"):
        if cnn_online:
            with st.spinner("Processing deep layers through CNN Pipeline..."):
                condition_label, class_idx, confidence = predict_crop_health(image)
                st.success(f"**Diagnosis:** {condition_label} ({confidence:.2f}% Confidence)")
                
                prescription = get_prescription(class_idx)
                st.markdown(
                    f'<div class="{prescription["box_class"]}">'
                    f'<h3>{prescription["title"]}</h3>'
                    f'<p>{prescription["details"]}</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
        else:
            st.warning("Prediction activation locked due to server model offline parameters.")
else:
    st.info("💡 Please upload a potato leaf image above to execute the CNN diagnostic pipeline.")
