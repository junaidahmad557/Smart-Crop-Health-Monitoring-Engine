"""PotatoGuard AI Streamlit application.

The app uses three models:
1. A regression model for crop-severity estimation.
2. A decision-tree model for automation decisions.
3. A TensorFlow/Keras CNN for potato-leaf classification.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PotatoGuard AI",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# PATHS AND MODEL SETTINGS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / "potatoguard_ai_models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

REGRESSION_MODEL_CANDIDATES = (
    "crop_severity_regression_model (1).pkl",
    "crop_severity_regression_model.pkl",
)
TREE_MODEL_CANDIDATES = ("crop_automation_tree_model.pkl",)
CNN_MODEL_CANDIDATES = ("potato_final_network.h5",)

# The Google Drive file must be shared as: Anyone with the link -> Viewer.
CNN_GOOGLE_DRIVE_FILE_ID = "1kuB-PC2qg742LTTvdPmArZJ-HZgFQKm3"
CNN_CACHE_FILE = MODEL_CACHE_DIR / "potato_final_network.h5"

# IMPORTANT: Keep this order identical to the class order used during training.
CNN_CLASS_NAMES = (
    "Potato Late Blight",
    "Potato Early Blight",
    "Healthy",
)


# -----------------------------------------------------------------------------
# STYLING
# -----------------------------------------------------------------------------
APP_CSS = """
<style>
    .main-title {
        font-size: 38px !important;
        font-weight: 800 !important;
        color: #1E3A8A;
        margin-bottom: 5px;
        text-align: center;
    }
    .sub-title {
        font-size: 16px !important;
        color: #6B7280;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1F2937;
        border-left: 5px solid #3B82F6;
        padding-left: 12px;
        margin-top: 35px;
        margin-bottom: 15px;
    }
    .risk-panel {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #FCA5A5;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 22px;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.10);
    }
    .risk-label {
        font-size: 14px;
        font-weight: 700;
        color: #991B1B;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-value {
        font-size: 42px;
        font-weight: 900;
        color: #DC2626;
        margin: 8px 0;
    }
    .risk-subtext {
        font-size: 13px;
        color: #7F1D1D;
        opacity: 0.85;
    }
    .automation-panel {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #93C5FD;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 30px;
    }
    .automation-label {
        color: #1E3A8A;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .automation-value {
        color: #1D4ED8;
        font-size: 25px;
        font-weight: 800;
        margin-top: 6px;
    }
    .prescription-box-0 {
        background: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 20px;
        border-radius: 8px;
        color: #991B1B;
    }
    .prescription-box-1 {
        background: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 20px;
        border-radius: 8px;
        color: #92400E;
    }
    .prescription-box-2 {
        background: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 20px;
        border-radius: 8px;
        color: #065F46;
    }
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="main-title">🥔 PotatoGuard AI™ Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Three-Model Crop Monitoring and Diagnostic Pipeline</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# MODEL-LOADING HELPERS
# -----------------------------------------------------------------------------
def find_existing_file(candidates: Iterable[str]) -> Optional[Path]:
    """Return the first matching model file located beside app.py."""
    for file_name in candidates:
        candidate = BASE_DIR / file_name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def require_model_file(candidates: Iterable[str], model_label: str) -> Path:
    """Resolve a required local model file or raise a useful error."""
    model_path = find_existing_file(candidates)
    if model_path is None:
        expected = ", ".join(candidates)
        raise FileNotFoundError(
            f"{model_label} model not found beside app.py. Expected: {expected}"
        )
    return model_path


@st.cache_resource
def load_sklearn_models() -> Tuple[Any, Any]:
    """Load the regression and decision-tree models once per server process."""
    regression_path = require_model_file(
        REGRESSION_MODEL_CANDIDATES, "Regression"
    )
    tree_path = require_model_file(TREE_MODEL_CANDIDATES, "Decision-tree")

    regression_model = joblib.load(regression_path)
    decision_tree_model = joblib.load(tree_path)
    return regression_model, decision_tree_model


def download_cnn_model(destination: Path) -> Path:
    """Download the CNN atomically so an interrupted download is never loaded."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".download")

    if temporary_path.exists():
        temporary_path.unlink()

    download_url = (
        "https://drive.google.com/uc?id=" + CNN_GOOGLE_DRIVE_FILE_ID
    )
    downloaded_path = gdown.download(
        url=download_url,
        output=str(temporary_path),
        quiet=True,
    )

    if not downloaded_path or not temporary_path.is_file():
        raise RuntimeError(
            "CNN download failed. Make the Google Drive file public: "
            "Anyone with the link -> Viewer."
        )

    # A tiny file usually means Google returned an HTML error page.
    if temporary_path.stat().st_size < 100_000:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded CNN file is invalid or too small. Check the Drive "
            "file ID, sharing permissions, and download quota."
        )

    os.replace(temporary_path, destination)
    return destination


def keras_load_model(model_path: Path) -> tf.keras.Model:
    """Load an inference-only Keras model."""
    return tf.keras.models.load_model(str(model_path), compile=False)


@st.cache_resource
def load_cnn_model() -> tf.keras.Model:
    """Load the CNN from the repository or download it to a writable cache."""
    repository_model = find_existing_file(CNN_MODEL_CANDIDATES)
    if repository_model is not None:
        return keras_load_model(repository_model)

    # Reuse a valid cached model across Streamlit reruns.
    if CNN_CACHE_FILE.is_file():
        try:
            return keras_load_model(CNN_CACHE_FILE)
        except Exception:
            # Remove only our temporary cached copy, never a repository model.
            CNN_CACHE_FILE.unlink(missing_ok=True)

    downloaded_model = download_cnn_model(CNN_CACHE_FILE)
    return keras_load_model(downloaded_model)


# Initialize every model independently so one failure does not crash the app.
reg_model: Optional[Any] = None
tree_model: Optional[Any] = None
cnn_model: Optional[tf.keras.Model] = None

sklearn_error: Optional[str] = None
cnn_error: Optional[str] = None

try:
    reg_model, tree_model = load_sklearn_models()
except Exception as exc:  # The UI remains usable and displays the real cause.
    sklearn_error = str(exc)

try:
    with st.spinner("Loading the CNN diagnostic model..."):
        cnn_model = load_cnn_model()
except Exception as exc:
    cnn_error = str(exc)

online_model_count = sum(
    model is not None for model in (reg_model, tree_model, cnn_model)
)

if online_model_count == 3:
    st.sidebar.success("✅ Ecosystem Online: 3/3 Models Operational")
elif online_model_count > 0:
    st.sidebar.warning(
        f"⚠️ Partial Service: {online_model_count}/3 Models Operational"
    )
else:
    st.sidebar.error("❌ Infrastructure Offline: 0/3 Models Operational")

with st.sidebar.expander("Model loading details", expanded=False):
    st.write(f"App directory: `{BASE_DIR}`")
    if sklearn_error:
        st.error(f"Regression/tree models: {sklearn_error}")
    else:
        st.success("Regression and decision-tree models loaded.")

    if cnn_error:
        st.error(f"CNN model: {cnn_error}")
    else:
        st.success("CNN model loaded.")


# Results must survive Streamlit reruns while the user completes all three
# stages of the pipeline.
st.session_state.setdefault("severity_score", None)
st.session_state.setdefault("detected_disease_id", None)
st.session_state.setdefault("detected_condition", None)
st.session_state.setdefault("cnn_confidence", None)
st.session_state.setdefault("cnn_file_signature", None)
st.session_state.setdefault("automation_result", None)
st.session_state.setdefault("automation_confidence", None)


# -----------------------------------------------------------------------------
# TABULAR-MODEL PREDICTION HELPERS
# -----------------------------------------------------------------------------
def normalize_feature_name(name: str) -> str:
    """Normalize a feature name for safe alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def create_regression_input(
    model: Any,
    temperature: float,
    humidity: float,
    moisture: float,
) -> Any:
    """Build the telemetry input expected by the regression model."""
    values_by_alias: Dict[str, float] = {
        "temperature": float(temperature),
        "temp": float(temperature),
        "airtemperature": float(temperature),
        "humidity": float(humidity),
        "airhumidity": float(humidity),
        "moisture": float(moisture),
        "soilmoisture": float(moisture),
        "soilmoisturelevel": float(moisture),
    }

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        row: Dict[str, float] = {}
        unsupported_features = []

        for feature_name in feature_names:
            feature_text = str(feature_name)
            normalized_name = normalize_feature_name(feature_text)

            if normalized_name not in values_by_alias:
                unsupported_features.append(feature_text)
            else:
                row[feature_text] = values_by_alias[normalized_name]

        if unsupported_features:
            raise ValueError(
                "The model expects unsupported features: "
                + ", ".join(unsupported_features)
            )

        return pd.DataFrame([row], columns=[str(item) for item in feature_names])

    expected_count = getattr(model, "n_features_in_", 3)
    if int(expected_count) != 3:
        raise ValueError(
            f"The model expects {expected_count} features, but this app provides "
            "Temperature, Humidity, and Moisture."
        )

    return np.array(
        [[float(temperature), float(humidity), float(moisture)]],
        dtype=np.float64,
    )


def predict_regression(
    model: Any,
    temperature: float,
    humidity: float,
    moisture: float,
) -> float:
    model_input = create_regression_input(model, temperature, humidity, moisture)
    prediction = np.asarray(model.predict(model_input)).reshape(-1)

    if prediction.size == 0:
        raise ValueError("The regression model returned an empty prediction.")

    value = float(prediction[0])
    if not np.isfinite(value):
        raise ValueError("The regression model returned a non-finite value.")
    return value


def format_decision_label(value: Any) -> str:
    """Create a readable label without inventing a class mapping."""
    if isinstance(value, (np.integer, int)):
        return f"Decision Class {int(value)}"
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return f"Decision Class {int(value)}"

    label = str(value).replace("_", " ").strip()
    return label.title() if label else "Unknown Decision"


def create_tree_input(
    model: Any,
    detected_disease_id: int,
    severity: float,
    loss_percent: float,
) -> Any:
    """Build the exact feature set expected by the automation tree.

    The saved tree reports these inputs through feature_names_in_:
    Detected Disease ID, Severity, and Loss Percent. The aliases below also
    support common underscore/capitalization variations of those names.
    """
    values_by_alias: Dict[str, float] = {
        "detecteddiseaseid": float(detected_disease_id),
        "diseaseid": float(detected_disease_id),
        "detectedid": float(detected_disease_id),
        "severity": float(severity),
        "severityscore": float(severity),
        "losspercent": float(loss_percent),
        "losspercentage": float(loss_percent),
        "croplosspercent": float(loss_percent),
        "yieldlosspercent": float(loss_percent),
    }

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        row: Dict[str, float] = {}
        unsupported_features = []

        for feature_name in feature_names:
            feature_text = str(feature_name)
            normalized_name = normalize_feature_name(feature_text)

            if normalized_name not in values_by_alias:
                unsupported_features.append(feature_text)
            else:
                row[feature_text] = values_by_alias[normalized_name]

        if unsupported_features:
            raise ValueError(
                "The decision tree expects unknown features: "
                + ", ".join(unsupported_features)
            )

        return pd.DataFrame([row], columns=[str(item) for item in feature_names])

    expected_count = getattr(model, "n_features_in_", 3)
    if int(expected_count) != 3:
        raise ValueError(
            f"The decision tree expects {expected_count} features, but this "
            "pipeline provides Detected Disease ID, Severity, and Loss Percent."
        )

    return np.array(
        [[float(detected_disease_id), float(severity), float(loss_percent)]],
        dtype=np.float64,
    )


def predict_tree_decision(
    model: Any,
    detected_disease_id: int,
    severity: float,
    loss_percent: float,
) -> Tuple[str, Optional[float]]:
    model_input = create_tree_input(
        model,
        detected_disease_id,
        severity,
        loss_percent,
    )
    prediction = np.asarray(model.predict(model_input)).reshape(-1)

    if prediction.size == 0:
        raise ValueError("The decision-tree model returned an empty prediction.")

    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(model_input))
        if probabilities.size:
            confidence = float(np.max(probabilities[0]) * 100.0)

    return format_decision_label(prediction[0]), confidence


# -----------------------------------------------------------------------------
# SECTION 1: ENVIRONMENTAL TELEMETRY
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="section-header">📊 Real-Time Environmental Telemetry</div>',
    unsafe_allow_html=True,
)

with st.form("telemetry_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.number_input(
            "Air Temperature (°C)",
            min_value=15.0,
            max_value=42.0,
            value=23.71,
            step=0.1,
        )
    with col2:
        humidity = st.number_input(
            "Air Humidity (%)",
            min_value=30.0,
            max_value=95.0,
            value=46.51,
            step=0.1,
        )
    with col3:
        moisture = st.number_input(
            "Soil Moisture Level",
            min_value=10.0,
            max_value=60.0,
            value=24.92,
            step=0.1,
        )

    analyze_telemetry = st.form_submit_button(
        "Analyze Telemetry",
        type="primary",
        use_container_width=True,
    )

if analyze_telemetry:
    if reg_model is None:
        st.session_state.severity_score = None
        st.session_state.automation_result = None
        st.error(
            "Severity prediction is unavailable because the regression model "
            "did not load. Open 'Model loading details' in the sidebar."
        )
    else:
        try:
            severity_score = predict_regression(
                reg_model, temp, humidity, moisture
            )
            st.session_state.severity_score = severity_score
            st.session_state.loss_percent_input = float(
                np.clip(severity_score, 0.0, 100.0)
            )
            st.session_state.automation_result = None
            st.session_state.automation_confidence = None
        except Exception as exc:
            st.session_state.severity_score = None
            st.session_state.automation_result = None
            st.error(f"Regression prediction failed: {exc}")

if st.session_state.severity_score is not None:
    st.markdown(
        f"""
        <div class="risk-panel">
            <div class="risk-label">⚠️ Predicted Crop Severity</div>
            <div class="risk-value">{st.session_state.severity_score:.2f}</div>
            <div class="risk-subtext">
                Regression output produced from the current environmental telemetry.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Stage 1 complete: this severity value will be passed to the "
        "decision-tree automation model."
    )


# -----------------------------------------------------------------------------
# CNN PREDICTION HELPERS
# -----------------------------------------------------------------------------
def get_cnn_input_details(model: tf.keras.Model) -> Tuple[int, int, bool]:
    """Return image height, width, and whether the model is channels-first."""
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        if not input_shape:
            raise ValueError("The CNN has no declared input shape.")
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(f"Unsupported CNN input shape: {input_shape}")

    channels_first = input_shape[1] in (1, 3, 4)
    if channels_first:
        height, width = input_shape[2], input_shape[3]
    else:
        height, width = input_shape[1], input_shape[2]

    if height is None or width is None:
        return 224, 224, channels_first

    return int(height), int(width), channels_first


def to_probabilities(raw_scores: np.ndarray) -> np.ndarray:
    """Accept either probabilities or logits and return normalized scores."""
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)

    if scores.size != len(CNN_CLASS_NAMES):
        raise ValueError(
            f"CNN returned {scores.size} outputs; expected "
            f"{len(CNN_CLASS_NAMES)} classes."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("CNN returned non-finite prediction values.")

    looks_like_probabilities = (
        np.all(scores >= 0.0)
        and np.all(scores <= 1.0)
        and np.isclose(scores.sum(), 1.0, atol=0.05)
    )

    if looks_like_probabilities:
        total = scores.sum()
        return scores / total if total > 0 else np.full_like(scores, 1 / scores.size)

    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def predict_crop_health(
    model: tf.keras.Model,
    image: Image.Image,
) -> Tuple[str, int, float]:
    """Preprocess an uploaded image and run CNN inference safely."""
    height, width, channels_first = get_cnn_input_details(model)

    rgb_image = image.convert("RGB")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized_image = rgb_image.resize((width, height), resampling)

    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    if channels_first:
        image_array = np.transpose(image_array, (2, 0, 1))

    batch = np.expand_dims(image_array, axis=0)
    raw_prediction = model.predict(batch, verbose=0)

    if isinstance(raw_prediction, list):
        if len(raw_prediction) != 1:
            raise ValueError("Multiple CNN outputs are not supported.")
        raw_prediction = raw_prediction[0]

    raw_scores = np.asarray(raw_prediction)
    if raw_scores.ndim > 1:
        raw_scores = raw_scores[0]

    probabilities = to_probabilities(raw_scores)
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index] * 100.0)

    return CNN_CLASS_NAMES[predicted_index], predicted_index, confidence


def get_prescription(class_index: int) -> Dict[str, str]:
    prescriptions = {
        0: {
            "title": "🚨 High Risk Treatment Plan (Late Blight Detected)",
            "box_class": "prescription-box-0",
            "details": (
                "Immediate action is recommended. Isolate visibly affected "
                "plants and consult a local crop specialist regarding an "
                "appropriate registered fungicide and application schedule."
            ),
        },
        1: {
            "title": "⚠️ Medium Risk Management Plan (Early Blight Detected)",
            "box_class": "prescription-box-1",
            "details": (
                "Improve plant spacing and airflow, avoid overhead watering, "
                "remove affected foliage safely, and consult a local crop "
                "specialist about a suitable registered treatment."
            ),
        },
        2: {
            "title": "🌿 Preventive Care Plan (Healthy Leaf)",
            "box_class": "prescription-box-2",
            "details": (
                "No disease was detected by the model. Continue regular field "
                "inspection, balanced irrigation, good airflow, and preventive "
                "crop-hygiene practices."
            ),
        },
    }
    return prescriptions[class_index]


# -----------------------------------------------------------------------------
# SECTION 2: IMAGE DIAGNOSTICS
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="section-header">📸 Image Diagnostic Engine (CNN Powered)</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a potato leaf image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.session_state.detected_disease_id = None
    st.session_state.detected_condition = None
    st.session_state.cnn_confidence = None
    st.session_state.cnn_file_signature = None
    st.session_state.automation_result = None
    st.session_state.automation_confidence = None
    st.info("💡 Upload a potato leaf image to run the CNN diagnostic pipeline.")
else:
    file_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.cnn_file_signature != file_signature:
        st.session_state.detected_disease_id = None
        st.session_state.detected_condition = None
        st.session_state.cnn_confidence = None
        st.session_state.automation_result = None
        st.session_state.automation_confidence = None

    try:
        uploaded_image = Image.open(uploaded_file)
        uploaded_image.load()
    except Exception as exc:
        st.error(f"The uploaded file is not a valid image: {exc}")
    else:
        st.image(
            uploaded_image,
            caption="Uploaded Leaf Specimen",
            use_container_width=True,
        )

        if st.button("🔮 Predict Leaf Health", type="primary"):
            if cnn_model is None:
                st.error(
                    "CNN prediction is unavailable. Open 'Model loading details' "
                    "in the sidebar to see the deployment error."
                )
            else:
                try:
                    with st.spinner("Running CNN inference..."):
                        condition, class_index, confidence = predict_crop_health(
                            cnn_model, uploaded_image
                        )

                    st.session_state.detected_disease_id = class_index
                    st.session_state.detected_condition = condition
                    st.session_state.cnn_confidence = confidence
                    st.session_state.cnn_file_signature = file_signature
                    st.session_state.automation_result = None
                    st.session_state.automation_confidence = None
                except Exception as exc:
                    st.session_state.detected_disease_id = None
                    st.session_state.detected_condition = None
                    st.session_state.cnn_confidence = None
                    st.session_state.automation_result = None
                    st.error(f"CNN prediction failed: {exc}")

        if (
            st.session_state.detected_disease_id is not None
            and st.session_state.cnn_file_signature == file_signature
        ):
            st.success(
                f"Diagnosis: **{st.session_state.detected_condition}** "
                f"({st.session_state.cnn_confidence:.2f}% confidence)"
            )

            prescription = get_prescription(
                int(st.session_state.detected_disease_id)
            )
            st.markdown(
                f"""
                <div class="{prescription['box_class']}">
                    <h3>{prescription['title']}</h3>
                    <p>{prescription['details']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Stage 2 complete: disease ID "
                f"{st.session_state.detected_disease_id} will be passed to the "
                "decision tree. AI predictions should be confirmed by a "
                "qualified local agricultural specialist."
            )


# -----------------------------------------------------------------------------
# SECTION 3: INTEGRATED DECISION-TREE AUTOMATION
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="section-header">🌱 Integrated Automation Decision</div>',
    unsafe_allow_html=True,
)

if tree_model is None:
    st.error(
        "Decision-tree automation is unavailable because the model did not "
        "load. Open 'Model loading details' in the sidebar."
    )
elif st.session_state.severity_score is None:
    st.info(
        "Complete Stage 1 first: enter environmental telemetry and click "
        "'Analyze Telemetry'."
    )
elif st.session_state.detected_disease_id is None:
    st.info(
        "Complete Stage 2 first: upload a potato leaf and click "
        "'Predict Leaf Health'."
    )
else:
    severity_for_tree = float(st.session_state.severity_score)
    detected_id_for_tree = int(st.session_state.detected_disease_id)

    st.write(
        "The decision tree will receive its required features: "
        f"**Detected Disease ID = {detected_id_for_tree}**, "
        f"**Severity = {severity_for_tree:.2f}**, and the loss percentage below."
    )

    if "loss_percent_input" not in st.session_state:
        st.session_state.loss_percent_input = float(
            np.clip(severity_for_tree, 0.0, 100.0)
        )

    with st.form("automation_form"):
        loss_percent = st.number_input(
            "Estimated Crop Loss (%)",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key="loss_percent_input",
            help=(
                "The current project has one regression output for severity. "
                "Confirm or edit the separate Loss Percent value required by "
                "the saved decision-tree model."
            ),
        )
        run_automation = st.form_submit_button(
            "Run Decision-Tree Automation",
            type="primary",
            use_container_width=True,
        )

    if run_automation:
        try:
            decision_label, decision_confidence = predict_tree_decision(
                tree_model,
                detected_disease_id=detected_id_for_tree,
                severity=severity_for_tree,
                loss_percent=float(loss_percent),
            )
            st.session_state.automation_result = decision_label
            st.session_state.automation_confidence = decision_confidence
        except Exception as exc:
            st.session_state.automation_result = None
            st.session_state.automation_confidence = None
            st.error(f"Decision-tree prediction failed: {exc}")

    if st.session_state.automation_result is not None:
        confidence_text = (
            f" ({st.session_state.automation_confidence:.2f}% confidence)"
            if st.session_state.automation_confidence is not None
            else ""
        )
        st.markdown(
            f"""
            <div class="automation-panel">
                <div class="automation-label">Decision-Tree Output</div>
                <div class="automation-value">
                    {st.session_state.automation_result}{confidence_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

