"""
ECG Arrhythmia Classifier - Streamlit Demonstration App
Using NeuroKit2 for robust R-peak detection
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import tempfile
import os
from io import StringIO
import joblib
import json
import neurokit2 as nk  # Add this import

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Input,
    Add, Activation, Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# =============================================================================
# Constants (Aligned with training)
# =============================================================================
TARGET_FS = 125                    # Dataset sampling frequency
TARGET_LENGTH = 187                 # Final fixed length
N_CLASSES = 5
CLASS_NAMES = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 
               'F (Fusion)', 'Q (Unknown)']
CLASS_SHORT = ['N', 'S', 'V', 'F', 'Q']
CLASS_COLORS = ['#28a745', '#fd7e14', '#dc3545', '#6f42c1', '#17a2b8']

# NeuroKit2 parameters
NK_METHOD = 'neurokit'  # Can also use 'pantompkins', 'hamilton', 'elgendi', 'engzeemod'
# NK_THRESHOLD = 0.5      # Detection threshold (adjust if needed)

# Page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Detector",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (keep as before)
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4a6fa5; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #6c757d; margin-bottom: 1rem; }
    .info-box { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 0.5rem solid #4a6fa5; margin-bottom: 1rem; }
    .warning-box { background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 0.5rem solid #ffc107; margin-bottom: 1rem; }
    .success-box { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 0.5rem solid #28a745; margin-bottom: 1rem; }
    .metric-card { background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
    .class-label { font-size: 1.2rem; font-weight: bold; }
    .class-N { color: #28a745; } .class-S { color: #fd7e14; } .class-V { color: #dc3545; } .class-F { color: #6f42c1; } .class-Q { color: #17a2b8; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Custom Metrics (must match training definitions)
# =============================================================================
def sensitivity(y_true, y_pred):
    """Sensitivity (Recall) metric"""
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    """Specificity metric"""
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def f1_score(y_true, y_pred):
    """F1 Score metric"""
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# =============================================================================
# NeuroKit2-based R-peak Detection and Beat Extraction
# =============================================================================

def preprocess_ecg_signal_neurokit(raw_signal, original_fs):
    """
    Preprocess ECG signal using NeuroKit2 for robust R-peak detection.
    
    Steps:
    1. Resample to 125Hz
    2. Apply bandpass filter
    3. Use NeuroKit2 for R-peak detection
    4. Extract 187-sample beats around each R-peak
    5. Apply [0,1] normalization (matching training data preparation)
    
    Returns:
        resampled_signal: Full resampled signal for visualization
        heartbeats_0_1: List of beats normalized to [0,1]
        r_peaks: Array of R-peak indices
    """
    # Ensure signal is 1D
    if len(raw_signal.shape) > 1:
        raw_signal = raw_signal.flatten()
    
    # Apply bandpass filter (0.5-45 Hz)
    nyquist = 0.5 * original_fs
    low = 0.5 / nyquist
    high = 45.0 / nyquist
    if high >= 1.0:
        high = 0.99
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, raw_signal)
    
    # Resample to target frequency (125Hz)
    if original_fs != TARGET_FS:
        original_time = np.arange(len(filtered_signal)) / original_fs
        target_time = np.arange(0, original_time[-1], 1/TARGET_FS)
        interpolator = interp1d(original_time, filtered_signal, kind='cubic', fill_value='extrapolate')
        resampled_signal = interpolator(target_time)
    else:
        resampled_signal = filtered_signal
    
    # Step 1: Clean the signal using NeuroKit2
    # This applies additional preprocessing optimized for R-peak detection
    cleaned_signal = nk.ecg_clean(resampled_signal, sampling_rate=TARGET_FS, method='neurokit')
    
    # Step 2: Detect R-peaks using NeuroKit2
    try:
        # Method options:
        # - 'neurokit' (default): Adaptive thresholding
        # - 'pantompkins': Classic Pan-Tompkins algorithm
        # - 'hamilton': Hamilton's algorithm
        # - 'elgendi': Elgendi's algorithm
        # - 'engzeemod': Modified Engzee algorithm
        
        r_peaks_info = nk.ecg_findpeaks(
            cleaned_signal, 
            sampling_rate=TARGET_FS, 
            method=NK_METHOD,
        )
        
        # Extract R-peak indices
        r_peaks = r_peaks_info['ECG_R_Peaks']
        
        if len(r_peaks) == 0:
            # Fallback: if no peaks found, use simple peak detection
            st.warning("NeuroKit2 found no peaks. Falling back to simple peak detection.")
            min_distance = int(0.6 * TARGET_FS)
            peak_height = np.percentile(resampled_signal, 85)
            peaks, _ = signal.find_peaks(
                resampled_signal,
                distance=min_distance,
                height=peak_height
            )
            r_peaks = peaks
            
    except Exception as e:
        st.warning(f"NeuroKit2 detection failed: {str(e)}. Using fallback method.")
        # Fallback: simple peak detection
        min_distance = int(0.6 * TARGET_FS)
        peak_height = np.percentile(resampled_signal, 85)
        peaks, _ = signal.find_peaks(
            resampled_signal,
            distance=min_distance,
            height=peak_height
        )
        r_peaks = peaks
    
    # Step 3: Extract heartbeats (187 samples centered on R-peak)
    heartbeats = []
    half_length = TARGET_LENGTH // 2
    
    for peak in r_peaks:
        start = peak - half_length
        end = peak + half_length
        
        # Handle boundaries with padding
        if start < 0:
            pad_left = abs(start)
            beat = np.pad(resampled_signal[:end], (pad_left, 0), 'constant', constant_values=0)
        elif end > len(resampled_signal):
            pad_right = end - len(resampled_signal)
            beat = np.pad(resampled_signal[start:], (0, pad_right), 'constant', constant_values=0)
        else:
            beat = resampled_signal[start:end]
        
        # Ensure exact length
        if len(beat) < TARGET_LENGTH:
            beat = np.pad(beat, (0, TARGET_LENGTH - len(beat)), 'constant', constant_values=0)
        elif len(beat) > TARGET_LENGTH:
            beat = beat[:TARGET_LENGTH]
        
        heartbeats.append(beat)
    
    # Step 4: Normalize each beat to [0,1] (matching training data preparation)
    heartbeats_0_1 = []
    for beat in heartbeats:
        beat_min = np.min(beat)
        beat_max = np.max(beat)
        if beat_max - beat_min > 1e-8:
            beat_norm = (beat - beat_min) / (beat_max - beat_min)
        else:
            beat_norm = beat - beat_min
        heartbeats_0_1.append(beat_norm)
    
    # If very few beats detected, use overlapping windows as fallback
    if len(heartbeats_0_1) < 3:
        st.warning("Very few beats detected. Using overlapping windows as fallback.")
        step = TARGET_LENGTH // 2
        heartbeats_0_1 = []
        for i in range(0, len(resampled_signal) - TARGET_LENGTH + 1, step):
            beat = resampled_signal[i:i + TARGET_LENGTH]
            beat_min = np.min(beat)
            beat_max = np.max(beat)
            if beat_max - beat_min > 1e-8:
                beat_norm = (beat - beat_min) / (beat_max - beat_min)
            else:
                beat_norm = beat - beat_min
            heartbeats_0_1.append(beat_norm)
    
    return resampled_signal, heartbeats_0_1, r_peaks

# =============================================================================
# Feature Extraction (from [0,1] beats)
# =============================================================================
def extract_handcrafted_features(signal_segment):
    """
    Extract 6 handcrafted features from an ECG segment normalized to [0,1].
    """
    # Statistical features
    mean_val = np.mean(signal_segment)
    std_val = np.std(signal_segment)
    skewness_val = skew(signal_segment)
    kurtosis_val = kurtosis(signal_segment)
    
    # Morphological features
    min_distance = int(0.4 * TARGET_FS)
    peak_height = 0.75  # 75th percentile in [0,1] space
    
    peaks, _ = signal.find_peaks(
        signal_segment,
        height=peak_height,
        distance=min_distance
    )
    num_peaks = len(peaks)
    
    # Energy features
    energy = np.sum(signal_segment ** 2)
    
    return np.array([mean_val, std_val, skewness_val, kurtosis_val, num_peaks, energy])

# =============================================================================
# Z-score Normalization (Your training method)
# =============================================================================
def normalize_heartbeats_zscore(heartbeats_0_1):
    """
    Apply z-score normalization (YOUR training method)
    Input: beats in [0,1] range
    Output: beats with mean=0, std=1 (for model input)
    """
    heartbeats_array = np.array(heartbeats_0_1)
    mean = np.mean(heartbeats_array)
    std = np.std(heartbeats_array) + 1e-8
    normalized = (heartbeats_array - mean) / std
    return normalized, mean, std

def normalize_features_zscore(features_array):
    """Z-score normalization for features"""
    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0) + 1e-8
    normalized = (features_array - mean) / std
    return normalized, mean, std

# =============================================================================
# Complete Processing Pipeline
# =============================================================================
def process_and_classify(raw_signal, original_fs, model, 
                        signal_scaler=None, feature_scaler=None,
                        threshold=0.7):
    """
    Complete pipeline with NeuroKit2 R-peak detection.
    """
    
    # Step 1: NeuroKit2 preprocessing
    resampled_signal, heartbeats_0_1, r_peaks = preprocess_ecg_signal_neurokit(
        raw_signal, original_fs
    )
    
    if len(heartbeats_0_1) == 0:
        return None, "No heartbeats detected", None
    
    # Step 2: Z-score normalization for signals
    heartbeats_array = np.array(heartbeats_0_1)
    if signal_scaler is not None:
        heartbeats_flat = heartbeats_array.reshape(-1, TARGET_LENGTH)
        heartbeats_zscore = signal_scaler.transform(heartbeats_flat)
        X = heartbeats_zscore.reshape(-1, TARGET_LENGTH, 1)
    else:
        X, beat_mean, beat_std = normalize_heartbeats_zscore(heartbeats_0_1)
        X = X.reshape(-1, TARGET_LENGTH, 1)
    
    # Step 3: Extract features from [0,1] beats
    features_list = []
    for beat in heartbeats_0_1:
        feat = extract_handcrafted_features(beat)
        features_list.append(feat)
    features_array = np.array(features_list)
    
    # Step 4: Z-score normalize features
    if feature_scaler is not None:
        features = feature_scaler.transform(features_array)
    else:
        features, feat_mean, feat_std = normalize_features_zscore(features_array)
    
    # Step 5: Classify
    results = classify_heartbeats(model, X, features, threshold)
    st.session_state['sig_segments'] = heartbeats_0_1
    st.session_state['signal_resampled'] = resampled_signal
    
    # Add R-peak info to results for visualization
    results['r_peaks'] = r_peaks
    st.session_state['rpks'] = r_peaks
    
    return results, "Success", resampled_signal

# =============================================================================
# Classification Function
# =============================================================================
def classify_heartbeats(model, heartbeats, features, threshold=0.7):
    """
    Classify heartbeats and calculate prevalence.
    """
    X = heartbeats.reshape(-1, TARGET_LENGTH, 1).astype(np.float32)
    predictions = model.predict([X, features], verbose=0)
    
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    high_confidence = confidence_scores >= threshold
    
    total_beats = len(predicted_classes)
    class_counts = np.bincount(predicted_classes, minlength=N_CLASSES)
    prevalence = (class_counts / total_beats * 100).round(2)
    
    results = {
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'confidence_scores': confidence_scores,
        'high_confidence': high_confidence,
        'class_counts': class_counts,
        'prevalence': prevalence,
        'total_beats': total_beats,
        'class_names': CLASS_NAMES,
        'class_short': CLASS_SHORT
    }
    
    return results

# =============================================================================
# Model Loading
# =============================================================================
@st.cache_resource
def load_ecg_model(model_file):
    """Load the trained ECG model with custom objects."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(model_file.getvalue())
            model_path = tmp_file.name

        custom_objects = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1_score
        }

        model = load_model(model_path, custom_objects=custom_objects)
        os.unlink(model_path)
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# =============================================================================
# Dummy ECG Generator (for testing)
# =============================================================================
def generate_dummy_ecg(ecg_type="normal", duration_seconds=30, fs=TARGET_FS):
    """Generate dummy ECG signals for demonstration."""
    t = np.linspace(0, duration_seconds, int(duration_seconds * fs))
    ecg = np.zeros_like(t)

    if ecg_type == "normal":
        heart_rate = 70
        beat_interval = fs * 60 / heart_rate
        for i in range(0, len(t), int(beat_interval)):
            if i + 50 < len(t):
                qrs = 1.5 * np.exp(-np.linspace(0, 3, 10))
                end_idx = min(i + 10, len(t))
                qrs_len = end_idx - i
                ecg[i:end_idx] += qrs[:qrs_len]
                
                t_wave = 0.3 * np.sin(np.linspace(0, np.pi, 20))
                t_start = i + 30
                t_end = min(t_start + 20, len(t))
                t_len = t_end - t_start
                if t_len > 0:
                    ecg[t_start:t_end] += t_wave[:t_len]
        
        ecg += 0.1 * np.sin(2 * np.pi * 0.2 * t)
        ecg += 0.05 * np.random.randn(len(t))

    elif ecg_type == "pvc":
        ecg, t = generate_dummy_ecg("normal", duration_seconds, fs)
        beat_interval = fs * 60 / 70
        for i in range(int(beat_interval * 5), len(ecg), int(beat_interval * 8)):
            if i + 30 < len(ecg):
                pvc_complex = 2.2 * np.sin(np.linspace(0, 2*np.pi, 30)) * np.exp(-np.linspace(0, 2, 30))
                ecg[i:i+30] += pvc_complex

    elif ecg_type == "svt":
        heart_rate = 150
        beat_interval = fs * 60 / heart_rate
        for i in range(0, len(t), int(beat_interval)):
            if i + 40 < len(t):
                qrs = 1.2 * np.exp(-np.linspace(0, 2, 8))
                end_idx = min(i + 8, len(t))
                qrs_len = end_idx - i
                ecg[i:end_idx] += qrs[:qrs_len]
        ecg += 0.08 * np.random.randn(len(t))

    elif ecg_type == "mixed":
        ecg, t = generate_dummy_ecg("normal", duration_seconds, fs)
        beat_interval = fs * 60 / 70
        for i in range(int(beat_interval * 10), len(ecg), int(beat_interval * 12)):
            if i + 30 < len(ecg):
                pvc = 2.2 * np.sin(np.linspace(0, 2*np.pi, 30))
                ecg[i:i+30] += pvc
        for i in range(int(beat_interval * 20), len(ecg), int(beat_interval * 30)):
            if i + 50 < len(ecg):
                svt_run = 1.5 * np.sin(np.linspace(0, 4*np.pi, 50))
                ecg[i:i+50] += svt_run

    return ecg, t

# =============================================================================
# Main App Interface
# =============================================================================
def main():
    global NK_METHOD
    # Header
    st.markdown("<h1 class='main-header'>💓 Hybrid Morphological Deep Learning-Based ECG Arrhythmia Detector</h1>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='text-align: center;'>
        Single-lead ECG classification into 5 AAMI arrhythmia classes<br>
        <b>Model trained on MIT-BIH Arrhythmia Dataset</b> ({TARGET_FS} Hz, {TARGET_LENGTH} samples per heartbeat)<br>
        <i>Using NeuroKit2 for robust R-peak detection (method: {NK_METHOD})</i>
    </p>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("logo.png", use_container_width=True)
        st.markdown("<h3>⚙️ Configuration</h3>", unsafe_allow_html=True)

        # Model upload
        st.markdown("**1. Upload Model**")
        model_file = st.file_uploader(
            "Choose your trained .h5 model file",
            type=['h5', 'hdf5'],
            help="Upload your trained model file (ecg_model_final_125Hz.h5)"
        )

        # NeuroKit2 method selection
        st.markdown("**2. R-peak Detection Method**")
        nk_method = st.selectbox(
            "Select NeuroKit2 method",
            ['neurokit', 'pantompkins', 'hamilton', 'elgendi', 'engzeemod'],
            index=0,
            help="Different R-peak detection algorithms"
        )
        
        # Update global method
        # global NK_METHOD
        NK_METHOD = nk_method

        # Confidence threshold
        st.markdown("**3. Set Threshold**")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05
        )

        # Load scalers if available
        signal_scaler = None
        feature_scaler = None
        if os.path.exists('signal_scaler.pkl') and os.path.exists('feature_scaler.pkl'):
            signal_scaler = joblib.load('signal_scaler.pkl')
            feature_scaler = joblib.load('feature_scaler.pkl')
            st.session_state['signal_scaler'] = signal_scaler
            st.session_state['feature_scaler'] = feature_scaler
            st.sidebar.success("✅ Scalers loaded successfully!")
        else:
            st.sidebar.warning("⚠️ Scaler files not found. Using on-the-fly normalization.")

        # st.markdown("---")
        st.markdown("<h4>📊 Class Information</h4>", unsafe_allow_html=True)

        class_info = pd.DataFrame({
            'Class': CLASS_SHORT,
            'Description': [c.split('(')[1].rstrip(')') for c in CLASS_NAMES],
            'Color': ['🟢', '🟠', '🔴', '🟣', '🔵']
        })
        st.dataframe(class_info, use_container_width=True, hide_index=True)

        # st.markdown("---")
        st.markdown("**📚 Reference:** [MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")

    # Load model
    model = None
    if model_file is not None:
        with st.spinner("Loading model..."):
            model = load_ecg_model(model_file)
            if model is not None:
                st.sidebar.success("✅ Model loaded successfully!")
                st.session_state['model'] = model
    elif 'model' in st.session_state:
        model = st.session_state['model']
    else:
        st.sidebar.warning("⚠️ Please upload a trained model file to begin.")

    # Main content
    if model is None:
        st.warning("👆 Please upload a trained model file using the sidebar to continue.")
        
        # Show download buttons
        st.markdown("---")
        st.markdown("<h3>📥 Download Sample ECG Files</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        for col, ecg_type, label in zip(
            [col1, col2, col3, col4],
            ["normal", "pvc", "svt", "mixed"],
            ["Normal ECG", "PVC ECG", "SVT ECG", "Mixed ECG"]
        ):
            with col:
                ecg, _ = generate_dummy_ecg(ecg_type)
                df = pd.DataFrame({'ecg_signal': ecg})
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 {label}",
                    data=csv,
                    file_name=f"{ecg_type}_ecg_sample.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload ECG", "📊 Classification Results", "📈 Detailed Analysis", "ℹ️ Help & Examples"
    ])

    with tab1:
        st.markdown("<h2 class='sub-header'>Upload ECG Signal</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("**📁 Input Method**")

            upload_option = st.radio(
                "Select input method",
                ["Upload ECG file", "Use dummy ECG file", "Enter values"]
            )

            raw_signal = None
            signal_fs = None

            if upload_option == "Upload ECG file":
                uploaded_file = st.file_uploader(
                    "Choose ECG file (CSV format)",
                    type=['csv']
                )

                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if 'ecg_signal' in df.columns:
                            raw_signal = df['ecg_signal'].values
                        else:
                            raw_signal = df.iloc[:, 0].values

                        st.success(f"✅ Loaded {len(raw_signal)} samples")

                        signal_fs = st.number_input(
                            "Original Sampling Frequency (Hz)",
                            min_value=1, max_value=10000, value=360, step=1
                        )

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            elif upload_option == "Use dummy ECG file":
                dummy_type = st.selectbox("Select type", ["normal", "pvc", "svt", "mixed"])
                raw_signal, _ = generate_dummy_ecg(dummy_type, duration_seconds=30)
                signal_fs = TARGET_FS
                st.info(f"Generated {dummy_type} ECG ({len(raw_signal)} samples at {signal_fs}Hz)")

            else:  # Enter values
                sample_input = st.text_area(
                    "Enter ECG values (comma-separated)",
                    placeholder="-0.1, 0.2, 0.5, 0.3, -0.2, ..."
                )
                if sample_input:
                    try:
                        values = [float(x.strip()) for x in sample_input.split(',')]
                        raw_signal = np.array(values)
                        signal_fs = st.number_input("Sampling Frequency (Hz)", value=125)
                        st.success(f"Loaded {len(raw_signal)} samples")
                    except:
                        st.error("Invalid format")

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            if raw_signal is not None and signal_fs is not None:
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("**📊 Preview**")

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=raw_signal[:min(1000, len(raw_signal))],
                    mode='lines',
                    line=dict(color='#4a6fa5', width=2)
                ))
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

                # Stats
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Length", f"{len(raw_signal)} samples")
                col_b.metric("Duration", f"{len(raw_signal)/signal_fs:.2f} s")
                col_c.metric("Mean", f"{np.mean(raw_signal):.3f}")

                if st.button("🚀 Process ECG", type="primary", use_container_width=True):
                    with st.spinner(f"Processing using NeuroKit2 ({NK_METHOD})..."):
                        signal_scaler = st.session_state.get('signal_scaler', None)
                        feature_scaler = st.session_state.get('feature_scaler', None)
                        
                        results, status, resampled_signal = process_and_classify(
                            raw_signal, signal_fs, model,
                            signal_scaler, feature_scaler,
                            threshold
                        )
                        
                        if results is not None:
                            st.session_state['processed'] = True
                            st.session_state['results'] = results
                            st.session_state['resampled_signal'] = resampled_signal
                            st.session_state['num_beats'] = results['total_beats']
                            
                            st.success(f"✅ Processed {results['total_beats']} beats!")
                            st.balloons()
                        else:
                            st.error(f"Processing failed: {status}")

                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Classification Results</h2>", unsafe_allow_html=True)

        if 'processed' in st.session_state and 'results' in st.session_state:
            results = st.session_state['results']

            # Prevalence cards
            st.markdown("<h3>📊 Arrhythmia Prevalence</h3>", unsafe_allow_html=True)
            cols = st.columns(N_CLASSES)

            for i, (col, class_name) in enumerate(zip(cols, CLASS_NAMES)):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: {CLASS_COLORS[i]}; font-size: 2rem; font-weight: bold;'>
                            {results['prevalence'][i]}%
                        </div>
                        <div style='font-size: 0.9rem;'>{class_name}</div>
                        <div style='font-size: 0.8rem;'>({results['class_counts'][i]} beats)</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Distribution plots
            st.markdown("<h3>📈 Class Distribution</h3>", unsafe_allow_html=True)
            
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'bar'}]])
            
            fig.add_trace(go.Pie(
                labels=CLASS_SHORT, values=results['class_counts'],
                marker=dict(colors=CLASS_COLORS), textinfo='label+percent', hole=0.3
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=CLASS_SHORT, y=results['class_counts'],
                marker_color=CLASS_COLORS, text=results['class_counts'], textposition='outside'
            ), row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Summary
            st.markdown("<h3>📋 Summary</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Beats", results['total_beats'])
            col2.metric("Normal", f"{results['prevalence'][0]:.1f}%")
            col3.metric("Abnormal", f"{100-results['prevalence'][0]:.1f}%")
            col4.metric("High Confidence", f"{np.mean(results['high_confidence'])*100:.1f}%")

            # Clinical interpretation
            st.markdown("<h3>🏥 Clinical Interpretation</h3>", unsafe_allow_html=True)
            
            warnings = []
            if results['prevalence'][2] > 5:
                warnings.append("⚠️ **Elevated ventricular ectopy** (>5%) - May require follow-up")
            if results['prevalence'][1] > 10:
                warnings.append("⚠️ **Elevated supraventricular ectopy** (>10%) - Consider evaluation")
            if results['prevalence'][0] < 80:
                warnings.append("⚠️ **Low normal beats** (<80%) - Significant arrhythmia burden")
                
            if warnings:
                for w in warnings:
                    st.markdown(f"<div class='warning-box'>{w}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'>✅ Normal arrhythmia burden</div>", unsafe_allow_html=True)
        else:
            st.info("👈 Please process an ECG signal first")

    with tab3:
        st.markdown("<h2 class='sub-header'>Detailed Analysis</h2>", unsafe_allow_html=True)

        if 'processed' in st.session_state and 'results' in st.session_state:
            results = st.session_state['results']

            st.markdown("<h3>📋 Plotting first six signal segments</h3>", unsafe_allow_html=True)
            signal_resampled = st.session_state.get('resampled_signal', None)
            # plot full resampled signal with R-peaks for visualize for 10 seconds and show legend with class colors
            if signal_resampled is not None and 'r_peaks' in results:
                fig_full = go.Figure()
                fig_full.add_trace(go.Scatter(
                    x=np.arange(len(signal_resampled[:TARGET_FS*10])) / TARGET_FS,
                    y=signal_resampled[:TARGET_FS*10],
                    mode='lines',
                    name='ECG Signal',
                    line=dict(color='#4a6fa5', width=1)
                ))
                r_peaks_in_view = results['r_peaks'][results['r_peaks'] < TARGET_FS*10]
                fig_full.add_trace(go.Scatter(
                    x=r_peaks_in_view/TARGET_FS,
                    y=signal_resampled[r_peaks_in_view],
                    mode='markers',
                    name='R-peaks',
                    marker=dict(color='red', size=8, symbol='x')
                ))
                fig_full.update_layout(height=300, title="Resampled ECG Signal with R-peaks (first 10 seconds)", xaxis_title="Time (s)", yaxis_title="Amplitude", showlegend=True)
                st.plotly_chart(fig_full, use_container_width=True)

            sig_segments = st.session_state.get('sig_segments', None)
            rpks = st.session_state.get('rpks', None)
            if sig_segments is not None:
                fig = make_subplots(rows=1, cols=6, shared_yaxes=True, horizontal_spacing=0.05)
                for i in range(min(6, len(sig_segments))):
                    fig.add_trace(go.Scatter(
                        y=sig_segments[i].flatten(),
                        mode='lines',
                        line=dict(color=CLASS_COLORS[results['predicted_classes'][i]], width=2)
                    ), row=1, col=i+1)
                    fig.update_xaxes(title_text=f"Beat {i+1}", row=1, col=i+1)
                fig.update_layout(height=400, title="Normalized ECG Beats (first 6 segments)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<h4>📋 Per-Beat Classification</h4>", unsafe_allow_html=True)
            
            df = pd.DataFrame({
                'Beat': range(1, results['total_beats']+1),
                'Class': [CLASS_NAMES[c] for c in results['predicted_classes']],
                'Confidence': results['confidence_scores'].round(3),
                'High Conf': ['✅' if h else '❌' for h in results['high_confidence']]
            })
            st.dataframe(df, use_container_width=True, height=400, hide_index=True)

            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="ecg_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("👈 Please classify an ECG signal first")

    with tab4:
        st.markdown("<h2 class='sub-header'>Help & Examples</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <h4>📖 How to Use</h4>
            <ol>
                <li><b>Upload Model:</b> Upload your trained .h5 model file</li>
                <li><b>Upload ECG:</b> Provide ECG recording (CSV, single column)</li>
                <li><b>Set Frequency:</b> Enter original sampling frequency</li>
                <li><b>Select Method:</b> Choose NeuroKit2 R-peak detection algorithm</li>
                <li><b>Process:</b> Click to detect beats and classify</li>
            </ol>
        </div>
        
        <div class='success-box'>
            <h4>🎯 Use Case Example</h4>
            <p><b>24-hour Holter Monitor:</b> Export as CSV (125 Hz), upload, get prevalence:
            <br>N >90%: Normal | S >5-10%: Supraventricular | V >5%: Ventricular</p>
        </div>
        
        <div class='warning-box'>
            <h4>⚠️ Important</h4>
            <ul>
                <li>Research use only - consult cardiologist for clinical decisions</li>
                <li>Model expects {TARGET_FS} Hz, {TARGET_LENGTH}-sample beats</li>
                <li>Using NeuroKit2 for robust R-peak detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    st.markdown(
        """
        <hr style='margin-top:3em;margin-bottom:0.5em;border:0;border-top:1px solid #eee;'>
        <div style='text-align:center; color: #888; font-size: 0.95rem; margin-bottom: 1em;'>
        ©Tilendra Choudhary, Ph.D. - Hybrid Morphological Deep Learning for ECG Arrhythmia Detection
        </div>
        """,
        unsafe_allow_html=True
    )