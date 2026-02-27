"""
ECG Arrhythmia Classifier - Streamlit Demonstration App
Aligned with final training script (125 Hz, 187 samples per heartbeat)
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

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Input,
    Add, Activation, Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
# from tensorflow.keras.optimizers.legacy import Adam  # Use legacy for compatibility


# =============================================================================
# Constants (Aligned with training)
# =============================================================================
TARGET_FS = 125           # Dataset sampling frequency
TARGET_LENGTH = 187       # Actual samples per heartbeat
N_CLASSES = 5
CLASS_NAMES = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 
               'F (Fusion)', 'Q (Unknown)']
CLASS_SHORT = ['N', 'S', 'V', 'F', 'Q']
CLASS_COLORS = ['#28a745', '#fd7e14', '#dc3545', '#6f42c1', '#17a2b8']

# Page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Detector",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
# Feature Extraction Functions (Aligned with training)
# =============================================================================
def extract_handcrafted_features(signal_segment):
    """
    Extract 6 handcrafted features from an ECG segment of length 187.
    """
    # Statistical features
    mean_val = np.mean(signal_segment)
    std_val = np.std(signal_segment)
    skewness_val = skew(signal_segment)
    kurtosis_val = kurtosis(signal_segment)

    # Morphological features
    min_distance = int(0.4 * TARGET_FS)  # Minimum 400ms between peaks
    peak_height = np.percentile(signal_segment, 75)
    
    peaks, _ = signal.find_peaks(
        signal_segment,
        height=peak_height,
        distance=min_distance
    )
    num_peaks = len(peaks)

    # Energy features
    energy = np.sum(signal_segment ** 2)

    return np.array([mean_val, std_val, skewness_val, kurtosis_val, num_peaks, energy])

def preprocess_ecg_signal(raw_signal, original_fs):
    """
    Preprocess ECG signal to match model input requirements (125Hz, 187-length beats).
    """
    # Ensure signal is 1D
    if len(raw_signal.shape) > 1:
        raw_signal = raw_signal.flatten()

    # Apply bandpass filter (0.5 Hz - 45 Hz)
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

    # R-peak detection
    min_distance = int(0.6 * TARGET_FS)  # ~75 samples at 125Hz
    peak_height = np.percentile(resampled_signal, 85)

    peaks, _ = signal.find_peaks(
        resampled_signal,
        distance=min_distance,
        height=peak_height,
        prominence=0.1 * np.std(resampled_signal)
    )

    # Extract heartbeats (187 samples centered on R-peak)
    heartbeats = []
    half_length = TARGET_LENGTH // 2

    for peak in peaks:
        start = peak - half_length
        end = peak + half_length

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

    # Fallback: overlapping windows
    if len(heartbeats) < 3:
        step = TARGET_LENGTH // 2
        for i in range(0, len(resampled_signal) - TARGET_LENGTH + 1, step):
            beat = resampled_signal[i:i + TARGET_LENGTH]
            heartbeats.append(beat)

    return resampled_signal, heartbeats

def normalize_heartbeats(heartbeats):
    """Z-score normalization"""
    heartbeats_array = np.array(heartbeats)
    mean = np.mean(heartbeats_array)
    std = np.std(heartbeats_array) + 1e-8
    normalized = (heartbeats_array - mean) / std
    return normalized, mean, std

def normalize_features(features_list):
    """Z-score normalization for features"""
    features_array = np.array(features_list)
    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0) + 1e-8
    normalized = (features_array - mean) / std
    return normalized, mean, std

# =============================================================================
# Model Loading with Keras 3 Compatibility
# =============================================================================
@st.cache_resource
def load_ecg_model(model_file):
    """
    Load the trained ECG model with custom objects.
    Handles Keras 3 compatibility issues.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(model_file.getvalue())
            model_path = tmp_file.name

        # Define custom objects
        custom_objects = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1_score
        }

        # Load model with custom objects
        model = load_model(model_path, custom_objects=custom_objects)
        
        # Clean up
        os.unlink(model_path)
        
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("Please ensure you're uploading the correct .h5 model file trained on the MIT-BIH dataset (125 Hz, 187 samples).")
        return None





# =============================================================================
# Classification Function
# =============================================================================
def classify_heartbeats(model, heartbeats, features, threshold=0.7):
    """
    Classify heartbeats and calculate prevalence.
    """
    # Prepare inputs
    X = heartbeats.reshape(-1, TARGET_LENGTH, 1).astype(np.float32)

    # Get predictions
    predictions = model.predict([X, features], verbose=0)

    # Get class with highest probability
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)

    # Apply threshold
    high_confidence = confidence_scores >= threshold

    # Calculate prevalence
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
# Dummy ECG Data Generator (Fixed broadcasting error)
# =============================================================================
def generate_dummy_ecg(ecg_type="normal", duration_seconds=30, fs=TARGET_FS):
    """
    Generate dummy ECG signals for demonstration purposes.
    """
    t = np.linspace(0, duration_seconds, int(duration_seconds * fs))
    ecg = np.zeros_like(t)

    if ecg_type == "normal":
        # Normal sinus rhythm (70 BPM)
        heart_rate = 70
        beat_interval = fs * 60 / heart_rate
        
        for i in range(0, len(t), int(beat_interval)):
            if i + 50 < len(t):
                # QRS complex
                qrs = 1.5 * np.exp(-np.linspace(0, 3, 10))
                # Fix broadcasting: ensure same length
                end_idx = min(i + 10, len(t))
                qrs_len = end_idx - i
                ecg[i:end_idx] += qrs[:qrs_len]
                
                # T wave
                t_wave = 0.3 * np.sin(np.linspace(0, np.pi, 20))
                t_start = i + 30
                t_end = min(t_start + 20, len(t))
                t_len = t_end - t_start
                if t_len > 0:
                    ecg[t_start:t_end] += t_wave[:t_len]
        
        # Add baseline variation and noise
        ecg += 0.1 * np.sin(2 * np.pi * 0.2 * t)
        ecg += 0.05 * np.random.randn(len(t))

    elif ecg_type == "pvc":
        # Start with normal
        ecg, t = generate_dummy_ecg("normal", duration_seconds, fs)
        
        # Add PVCs (wide QRS complexes)
        beat_interval = fs * 60 / 70
        for i in range(int(beat_interval * 5), len(ecg), int(beat_interval * 8)):
            if i + 30 < len(ecg):
                # Wide, bizarre QRS
                pvc_complex = 2.2 * np.sin(np.linspace(0, 2*np.pi, 30)) * np.exp(-np.linspace(0, 2, 30))
                ecg[i:i+30] += pvc_complex

    elif ecg_type == "svt":
        # Supraventricular tachycardia (fast but narrow QRS)
        heart_rate = 150
        beat_interval = fs * 60 / heart_rate
        
        for i in range(0, len(t), int(beat_interval)):
            if i + 40 < len(t):
                # Narrow QRS
                qrs = 1.2 * np.exp(-np.linspace(0, 2, 8))
                end_idx = min(i + 8, len(t))
                qrs_len = end_idx - i
                ecg[i:end_idx] += qrs[:qrs_len]
        
        ecg += 0.08 * np.random.randn(len(t))

    elif ecg_type == "mixed":
        # Mixed arrhythmias
        ecg, t = generate_dummy_ecg("normal", duration_seconds, fs)
        
        beat_interval = fs * 60 / 70
        
        # Add PVCs
        for i in range(int(beat_interval * 10), len(ecg), int(beat_interval * 12)):
            if i + 30 < len(ecg):
                pvc = 2.2 * np.sin(np.linspace(0, 2*np.pi, 30))
                ecg[i:i+30] += pvc
        
        # Add SVT runs
        for i in range(int(beat_interval * 20), len(ecg), int(beat_interval * 30)):
            if i + 50 < len(ecg):
                svt_run = 1.5 * np.sin(np.linspace(0, 4*np.pi, 50))
                ecg[i:i+50] += svt_run

    return ecg, t

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_saliency_map(ecg_signal, saliency_map, class_name, confidence):
    """Plot ECG with saliency overlay"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    time = np.arange(len(ecg_signal)) / TARGET_FS
    
    ax1.plot(time, ecg_signal, 'k-', linewidth=1, alpha=0.7)
    ax1.fill_between(time, ecg_signal, where=(saliency_map > 0.5),
                     alpha=0.3, color='red', label='High saliency')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'ECG with Saliency - {class_name} (conf={confidence:.3f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.bar(time, saliency_map, width=1/TARGET_FS, color='red', alpha=0.6)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Saliency')
    ax2.set_title('Saliency Map')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# Main App Interface
# =============================================================================
def main():
    # Header
    st.markdown("<h1 class='main-header'>💓 Hybrid Morphological Deep Learning-Based ECG Arrhythmia Detector</h1>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='text-align: center;'>
        Single-lead ECG classification into 5 AAMI arrhythmia classes<br>
        <b>Model trained on MIT-BIH Arrhythmia Dataset</b> ({TARGET_FS} Hz, {TARGET_LENGTH} samples per heartbeat)
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

        # Confidence threshold
        st.markdown("**2. Set Threshold**")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05
        )

        # Try to load scalers if available
        scalers_loaded = False
        if os.path.exists('signal_scaler.pkl') and os.path.exists('feature_scaler.pkl'):
            signal_scaler = joblib.load('signal_scaler.pkl')
            feature_scaler = joblib.load('feature_scaler.pkl')
            scalers_loaded = True
            st.session_state['signal_scaler'] = signal_scaler
            st.session_state['feature_scaler'] = feature_scaler

        st.markdown("---")
        st.markdown("<h4>📊 Class Information</h4>", unsafe_allow_html=True)

        class_info = pd.DataFrame({
            'Class': CLASS_SHORT,
            'Description': [c.split('(')[1].rstrip(')') for c in CLASS_NAMES],
            'Color': ['🟢', '🟠', '🔴', '🟣', '🔵']
        })
        st.dataframe(class_info, use_container_width=True, hide_index=True)

        st.markdown("---")
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
                    with st.spinner("Processing..."):
                        # Preprocess
                        resampled_signal, heartbeats = preprocess_ecg_signal(raw_signal, signal_fs)

                        if len(heartbeats) == 0:
                            st.error("No heartbeats detected")
                        else:
                            # Normalize
                            norm_beats, beat_mean, beat_std = normalize_heartbeats(heartbeats)
                            
                            # Features
                            features = []
                            for beat in heartbeats:
                                features.append(extract_handcrafted_features(beat))
                            features = np.array(features)
                            
                            # Normalize features
                            if 'feature_scaler' in st.session_state:
                                features = st.session_state['feature_scaler'].transform(features)
                            else:
                                features, feat_mean, feat_std = normalize_features(features)

                            # Store
                            st.session_state['processed'] = True
                            st.session_state['heartbeats'] = heartbeats
                            st.session_state['norm_beats'] = norm_beats
                            st.session_state['features'] = features
                            st.session_state['resampled_signal'] = resampled_signal
                            st.session_state['num_beats'] = len(heartbeats)

                            st.success(f"✅ Processed {len(heartbeats)} beats!")
                            st.balloons()

                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>Classification Results</h2>", unsafe_allow_html=True)

        if 'processed' in st.session_state:
            # Classify
            with st.spinner("Classifying..."):
                results = classify_heartbeats(
                    model,
                    st.session_state['norm_beats'],
                    st.session_state['features'],
                    threshold=threshold
                )
                st.session_state['results'] = results

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
            
            # Pie
            fig.add_trace(go.Pie(
                labels=CLASS_SHORT, values=results['class_counts'],
                marker=dict(colors=CLASS_COLORS), textinfo='label+percent', hole=0.3
            ), row=1, col=1)
            
            # Bar
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
            if results['prevalence'][2] > 5:   # V class
                warnings.append("⚠️ **Elevated ventricular ectopy** (>5%) - May require follow-up")
            if results['prevalence'][1] > 10:  # S class
                warnings.append("⚠️ **Elevated supraventricular ectopy** (>10%) - Consider evaluation")
            if results['prevalence'][0] < 80:  # N class
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
            
            # Per-beat table
            st.markdown("<h3>📋 Per-Beat Classification</h3>", unsafe_allow_html=True)
            
            df = pd.DataFrame({
                'Beat': range(1, results['total_beats']+1),
                'Class': [CLASS_NAMES[c] for c in results['predicted_classes']],
                'Confidence': results['confidence_scores'].round(3),
                'High Conf': ['✅' if h else '❌' for h in results['high_confidence']]
            })
            st.dataframe(df, use_container_width=True, height=400, hide_index=True)

            # Download
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
            </ul>
        </div>
        """, unsafe_allow_html=True)

    

if __name__ == "__main__":
    main()
    # Footer (persistent on every page)
    st.markdown(
        """
        <hr style='margin-top:10em;margin-bottom:0.5em;border:0;border-top:1px solid #eee;'>
        <div style='text-align:center; color: #888; font-size: 0.95rem; margin-bottom: 1em;'>
        ©Tilendra Choudhary, Ph.D. - Hybrid Morphological Deep Learning for ECG Arrhythmia Detection
        </div>
        """,
        unsafe_allow_html=True
    )