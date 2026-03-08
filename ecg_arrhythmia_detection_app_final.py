"""
ECG Arrhythmia Classifier – Streamlit App  (Extended Edition)
=============================================================
Adds a new "📄 ECG from Image" tab that lets users:
  1. Upload a printed 12-lead ECG image (PNG / JPG / BMP / TIFF)
  2. Select a ROI rectangle over a single lead strip
  3. Auto-digitise the trace (grid detection, color-grid removal)
  4. Run the same NeuroKit2 + deep-learning classification pipeline

All original functionality is preserved unchanged.
New dependency (already in most scientific Python stacks):
  pip install opencv-python-headless

Integration note
----------------
Drop this file alongside the original app.py and the new
ecg_image_digitizer.py module.  Run with:
  streamlit run app_with_image_input.py
"""

# ── Standard imports (identical to original app.py) ──────────────────────────
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
import neurokit2 as nk

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Input,
    Add, Activation, Concatenate,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# ── New: image digitiser module ───────────────────────────────────────────────
from ecg_image_digitizer import render_ecg_image_tab

# =============================================================================
# Constants
# =============================================================================
TARGET_FS     = 125
TARGET_LENGTH = 187
N_CLASSES     = 5
CLASS_NAMES   = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)',
                 'F (Fusion)', 'Q (Unknown)']
CLASS_SHORT   = ['N', 'S', 'V', 'F', 'Q']
CLASS_COLORS  = ['#28a745', '#fd7e14', '#dc3545', '#6f42c1', '#17a2b8']
NK_METHOD     = 'neurokit'

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="ECG Arrhythmia Detector",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header{font-size:2.5rem;color:#4a6fa5;text-align:center;margin-bottom:1rem;}
    .sub-header{font-size:1.5rem;color:#6c757d;margin-bottom:1rem;}
    .info-box{background-color:#f8f9fa;padding:1rem;border-radius:.5rem;
              border-left:.5rem solid #4a6fa5;margin-bottom:1rem;}
    .warning-box{background-color:#fff3cd;padding:1rem;border-radius:.5rem;
                 border-left:.5rem solid #ffc107;margin-bottom:1rem;}
    .success-box{background-color:#d4edda;padding:1rem;border-radius:.5rem;
                 border-left:.5rem solid #28a745;margin-bottom:1rem;}
    .metric-card{background-color:white;padding:1rem;border-radius:.5rem;
                 box-shadow:0 2px 4px rgba(0,0,0,.1);text-align:center;}
    .class-N{color:#28a745;} .class-S{color:#fd7e14;} .class-V{color:#dc3545;}
    .class-F{color:#6f42c1;} .class-Q{color:#17a2b8;}
    .stButton button{width:100%;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Custom Keras metrics (must match training)
# =============================================================================
def sensitivity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32'); y_pred = K.cast(y_pred, 'float32')
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pp + K.epsilon())

def specificity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32'); y_pred = K.cast(y_pred, 'float32')
    tn = K.sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    pn = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return tn / (pn + K.epsilon())

def f1_score(y_true, y_pred):
    y_true = K.cast(y_true, 'float32'); y_pred = K.cast(y_pred, 'float32')
    tp  = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    pp  = K.sum(K.round(K.clip(y_true, 0, 1)))
    prec   = tp / (pp2 + K.epsilon())
    recall = tp / (pp  + K.epsilon())
    return 2*((prec*recall)/(prec+recall+K.epsilon()))

# =============================================================================
# NeuroKit2 preprocessing + beat extraction (unchanged from original)
# =============================================================================
def preprocess_ecg_signal_neurokit(raw_signal, original_fs):
    if len(raw_signal.shape) > 1:
        raw_signal = raw_signal.flatten()

    nyquist = 0.5 * original_fs
    low = 0.5 / nyquist
    high = min(45.0 / nyquist, 0.99)
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, raw_signal)

    if original_fs != TARGET_FS:
        orig_t   = np.arange(len(filtered)) / original_fs
        targ_t   = np.arange(0, orig_t[-1], 1/TARGET_FS)
        interp   = interp1d(orig_t, filtered, kind='cubic', fill_value='extrapolate')
        resampled = interp(targ_t)
    else:
        resampled = filtered

    cleaned = nk.ecg_clean(resampled, sampling_rate=TARGET_FS, method='neurokit')

    try:
        r_info  = nk.ecg_findpeaks(cleaned, sampling_rate=TARGET_FS, method=NK_METHOD)
        r_peaks = r_info['ECG_R_Peaks']
        if len(r_peaks) == 0:
            raise ValueError("No peaks")
    except Exception:
        d = int(0.6 * TARGET_FS)
        h = np.percentile(resampled, 85)
        r_peaks, _ = signal.find_peaks(resampled, distance=d, height=h)

    half = TARGET_LENGTH // 2
    beats = []
    for pk in r_peaks:
        s, e = pk - half, pk + half
        if s < 0:
            beat = np.pad(resampled[:e], (abs(s), 0), constant_values=0)
        elif e > len(resampled):
            beat = np.pad(resampled[s:], (0, e-len(resampled)), constant_values=0)
        else:
            beat = resampled[s:e]
        if len(beat) < TARGET_LENGTH:
            beat = np.pad(beat, (0, TARGET_LENGTH-len(beat)), constant_values=0)
        beat = beat[:TARGET_LENGTH]
        mn, mx = beat.min(), beat.max()
        beats.append((beat-mn)/(mx-mn+1e-8))

    if len(beats) < 3:
        step = TARGET_LENGTH//2
        beats = []
        for i in range(0, len(resampled)-TARGET_LENGTH+1, step):
            b = resampled[i:i+TARGET_LENGTH]
            mn, mx = b.min(), b.max()
            beats.append((b-mn)/(mx-mn+1e-8))

    return resampled, beats, r_peaks

# =============================================================================
# Feature extraction
# =============================================================================
def extract_handcrafted_features(seg):
    from scipy.stats import skew as sk, kurtosis as ku
    return np.array([
        np.mean(seg), np.std(seg), sk(seg), ku(seg),
        len(signal.find_peaks(seg, height=0.75, distance=int(0.4*TARGET_FS))[0]),
        np.sum(seg**2),
    ])

# =============================================================================
# Normalisation
# =============================================================================
def normalize_heartbeats_zscore(beats):
    arr  = np.array(beats)
    mean = arr.mean(); std = arr.std() + 1e-8
    return (arr-mean)/std, mean, std

def normalize_features_zscore(feats):
    mean = feats.mean(0); std = feats.std(0)+1e-8
    return (feats-mean)/std, mean, std

# =============================================================================
# Classification
# =============================================================================
def classify_heartbeats(model, heartbeats, features, threshold=0.7):
    X = heartbeats.reshape(-1, TARGET_LENGTH, 1).astype(np.float32)
    preds = model.predict([X, features], verbose=0)
    classes = np.argmax(preds, 1)
    confs   = np.max(preds, 1)
    counts  = np.bincount(classes, minlength=N_CLASSES)
    return dict(
        predictions=preds, predicted_classes=classes, confidence_scores=confs,
        high_confidence=confs>=threshold, class_counts=counts,
        prevalence=(counts/len(classes)*100).round(2),
        total_beats=len(classes), class_names=CLASS_NAMES, class_short=CLASS_SHORT,
    )

# =============================================================================
# Full process + classify pipeline
# =============================================================================
def process_and_classify(raw_signal, original_fs, model,
                         signal_scaler=None, feature_scaler=None,
                         threshold=0.7):
    resampled, beats_01, r_peaks = preprocess_ecg_signal_neurokit(raw_signal, original_fs)
    if not beats_01:
        return None, "No heartbeats detected", None

    arr = np.array(beats_01)
    if signal_scaler is not None:
        X = signal_scaler.transform(arr.reshape(-1, TARGET_LENGTH)).reshape(-1, TARGET_LENGTH, 1)
    else:
        Xn, *_ = normalize_heartbeats_zscore(beats_01)
        X = Xn.reshape(-1, TARGET_LENGTH, 1)

    feats = np.array([extract_handcrafted_features(b) for b in beats_01])
    if feature_scaler is not None:
        feats = feature_scaler.transform(feats)
    else:
        feats, *_ = normalize_features_zscore(feats)

    results = classify_heartbeats(model, X, feats, threshold)
    st.session_state['sig_segments'] = beats_01
    st.session_state['signal_resampled'] = resampled
    results['r_peaks'] = r_peaks
    st.session_state['rpks'] = r_peaks
    return results, "Success", resampled

# =============================================================================
# Model loading
# =============================================================================
@st.cache_resource
def load_ecg_model(model_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as f:
            f.write(model_file.getvalue())
            path = f.name
        m = load_model(path, custom_objects={'sensitivity':sensitivity,
                                              'specificity':specificity,
                                              'f1_score':f1_score})
        os.unlink(path)
        return m
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# =============================================================================
# Dummy ECG generator
# =============================================================================
def generate_dummy_ecg(kind="normal", dur=30, fs=TARGET_FS):
    t   = np.linspace(0, dur, int(dur*fs))
    ecg = np.zeros_like(t)
    if kind == "normal":
        bi = int(fs*60/70)
        for i in range(0, len(t), bi):
            if i+50 < len(t):
                q = 1.5*np.exp(-np.linspace(0,3,10)); ecg[i:i+10] += q
                tw= 0.3*np.sin(np.linspace(0,np.pi,20))
                ecg[i+30:i+50] += tw
        ecg += 0.05*np.random.randn(len(t))
    elif kind == "pvc":
        ecg, t = generate_dummy_ecg("normal", dur, fs)
        bi=int(fs*60/70)
        for i in range(bi*5, len(ecg), bi*8):
            if i+30<len(ecg):
                ecg[i:i+30]+=2.2*np.sin(np.linspace(0,2*np.pi,30))*np.exp(-np.linspace(0,2,30))
    elif kind == "svt":
        bi=int(fs*60/150)
        for i in range(0,len(t),bi):
            if i+40<len(t):
                ecg[i:i+8]+=1.2*np.exp(-np.linspace(0,2,8))
        ecg+=0.08*np.random.randn(len(t))
    elif kind == "mixed":
        ecg, t = generate_dummy_ecg("normal", dur, fs)
        bi=int(fs*60/70)
        for i in range(bi*10,len(ecg),bi*12):
            if i+30<len(ecg): ecg[i:i+30]+=2.2*np.sin(np.linspace(0,2*np.pi,30))
    return ecg, t

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    global NK_METHOD

    st.markdown(
        "<h1 class='main-header'>💓 Hybrid Morphological Deep Learning–Based "
        "ECG Arrhythmia Detector</h1>", unsafe_allow_html=True
    )
    st.markdown(f"""
    <p style='text-align:center;'>
    Single-lead ECG classification into 5 AAMI arrhythmia classes<br>
    <b>Trained on MIT-BIH Arrhythmia Dataset</b>
    ({TARGET_FS} Hz · {TARGET_LENGTH} samples/beat)<br>
    <i>NeuroKit2 R-peak detection · method: {NK_METHOD}</i>
    </p>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True)
        st.markdown("<h3>⚙️ Configuration</h3>", unsafe_allow_html=True)

        st.markdown("**1. Upload Model**")
        model_file = st.file_uploader("Choose .h5 model", type=['h5','hdf5'])

        st.markdown("**2. R-peak Detection Method**")
        NK_METHOD = st.selectbox(
            "NeuroKit2 method",
            ['neurokit','pantompkins','hamilton','elgendi','engzeemod'], index=0
        )

        st.markdown("**3. Confidence Threshold**")
        threshold = st.slider("Threshold", 0.0, 1.0, 0.7, 0.05)

        # Scalers
        signal_scaler = feature_scaler = None
        if os.path.exists('signal_scaler.pkl') and os.path.exists('feature_scaler.pkl'):
            signal_scaler  = joblib.load('signal_scaler.pkl')
            feature_scaler = joblib.load('feature_scaler.pkl')
            st.session_state['signal_scaler']  = signal_scaler
            st.session_state['feature_scaler'] = feature_scaler
            st.sidebar.success("✅ Scalers loaded!")
        else:
            st.sidebar.warning("⚠️ Scalers not found – using on-the-fly normalisation.")

        st.markdown("<h4>📊 Class Info</h4>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Class': CLASS_SHORT,
            'Type' : [c.split('(')[1].rstrip(')') for c in CLASS_NAMES],
            '': ['🟢','🟠','🔴','🟣','🔵']
        }), use_container_width=True, hide_index=True)

        st.markdown(
            "**📚 Reference:** [MIT-BIH on Kaggle]"
            "(https://www.kaggle.com/datasets/shayanfazeli/heartbeat)"
        )

    # Load model
    model = None
    if model_file:
        with st.spinner("Loading model…"):
            model = load_ecg_model(model_file)
            if model: st.sidebar.success("✅ Model loaded!")
            st.session_state['model'] = model
    elif 'model' in st.session_state:
        model = st.session_state['model']
    else:
        st.sidebar.warning("⚠️ Upload a trained model to begin.")

    if model is None:
        st.warning("👆 Please upload a trained model file to continue.")
        st.markdown("---")
        st.markdown("### 📥 Download Sample ECG Files")
        c1,c2,c3,c4 = st.columns(4)
        for col, kind, label in zip([c1,c2,c3,c4],
                                    ["normal","pvc","svt","mixed"],
                                    ["Normal","PVC","SVT","Mixed"]):
            with col:
                ecg,_ = generate_dummy_ecg(kind)
                csv = pd.DataFrame({'ecg_signal':ecg}).to_csv(index=False)
                st.download_button(f"📥 {label}", csv, f"{kind}_ecg.csv", "text/csv",
                                   use_container_width=True)
        return

    signal_scaler  = st.session_state.get('signal_scaler',  None)
    feature_scaler = st.session_state.get('feature_scaler', None)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab_img, tab2, tab3, tab4 = st.tabs([
        "📤 Upload ECG (CSV)",
        "📄 ECG from Image",       # ← NEW TAB
        "📊 Classification Results",
        "📈 Detailed Analysis",
        "ℹ️ Help & Examples",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 – CSV upload (unchanged)
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("<h2 class='sub-header'>Upload ECG Signal (CSV)</h2>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            upload_option = st.radio("Input method",
                ["Upload ECG file (CSV)", "Use dummy ECG", "Enter values"])
            raw_signal = signal_fs = None

            if upload_option == "Upload ECG file (CSV)":
                uf = st.file_uploader("Choose CSV", type=['csv'])
                if uf:
                    df = pd.read_csv(uf)
                    raw_signal = df['ecg_signal'].values if 'ecg_signal' in df.columns \
                                 else df.iloc[:,0].values
                    st.success(f"✅ Loaded {len(raw_signal)} samples")
                    signal_fs = st.number_input("Original Sampling Frequency (Hz)",
                                                1, 10000, 360)
            elif upload_option == "Use dummy ECG":
                kind = st.selectbox("Type", ["normal","pvc","svt","mixed"])
                raw_signal, _ = generate_dummy_ecg(kind)
                signal_fs = TARGET_FS
                st.info(f"Generated {kind} ECG ({len(raw_signal)} samples @ {signal_fs} Hz)")
            else:
                txt = st.text_area("Values (comma-separated)")
                if txt:
                    try:
                        raw_signal = np.array([float(v.strip()) for v in txt.split(',')])
                        signal_fs  = st.number_input("Fs (Hz)", value=125)
                        st.success(f"Loaded {len(raw_signal)} samples")
                    except: st.error("Invalid format")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            if raw_signal is not None and signal_fs is not None:
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=raw_signal[:1000], mode='lines',
                                          line=dict(color='#4a6fa5', width=2)))
                fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
                ca,cb,cc = st.columns(3)
                ca.metric("Length", f"{len(raw_signal)}")
                cb.metric("Duration", f"{len(raw_signal)/signal_fs:.2f} s")
                cc.metric("Mean",  f"{np.mean(raw_signal):.3f}")

                if st.button("🚀 Process ECG", type="primary", use_container_width=True):
                    with st.spinner(f"Processing with NeuroKit2 ({NK_METHOD})…"):
                        results, status, rs = process_and_classify(
                            raw_signal, signal_fs, model,
                            signal_scaler, feature_scaler, threshold)
                    if results:
                        st.session_state.update(processed=True, results=results,
                                                resampled_signal=rs,
                                                num_beats=results['total_beats'])
                        st.success(f"✅ Processed {results['total_beats']} beats!")
                        st.balloons()
                    else:
                        st.error(f"Processing failed: {status}")
                st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB IMAGE – ECG from printed image  ← NEW
    # ════════════════════════════════════════════════════════════════════════
    with tab_img:
        render_ecg_image_tab(
            model, signal_scaler, feature_scaler, threshold,
            process_and_classify,
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 – Classification Results (unchanged)
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<h2 class='sub-header'>Classification Results</h2>",
                    unsafe_allow_html=True)
        if 'processed' in st.session_state and 'results' in st.session_state:
            results = st.session_state['results']
            st.markdown("<h3>📊 Arrhythmia Prevalence</h3>", unsafe_allow_html=True)
            cols = st.columns(N_CLASSES)
            for i,(col,cn) in enumerate(zip(cols, CLASS_NAMES)):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color:{CLASS_COLORS[i]};font-size:2rem;font-weight:bold;'>
                            {results['prevalence'][i]}%</div>
                        <div style='font-size:.9rem;'>{cn}</div>
                        <div style='font-size:.8rem;'>({results['class_counts'][i]} beats)</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<h3>📈 Distribution</h3>", unsafe_allow_html=True)
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'},{'type':'bar'}]])
            fig.add_trace(go.Pie(labels=CLASS_SHORT, values=results['class_counts'],
                                  marker=dict(colors=CLASS_COLORS),
                                  textinfo='label+percent', hole=.3), row=1, col=1)
            fig.add_trace(go.Bar(x=CLASS_SHORT, y=results['class_counts'],
                                  marker_color=CLASS_COLORS,
                                  text=results['class_counts'], textposition='outside'),
                          row=1, col=2)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Beats", results['total_beats'])
            c2.metric("Normal", f"{results['prevalence'][0]:.1f}%")
            c3.metric("Abnormal", f"{100-results['prevalence'][0]:.1f}%")
            c4.metric("High Confidence", f"{np.mean(results['high_confidence'])*100:.1f}%")

            st.markdown("<h3>🏥 Clinical Interpretation</h3>", unsafe_allow_html=True)
            warns = []
            if results['prevalence'][2] > 5:
                warns.append("⚠️ **Elevated ventricular ectopy** (>5%) – may require follow-up")
            if results['prevalence'][1] > 10:
                warns.append("⚠️ **Elevated supraventricular ectopy** (>10%) – consider evaluation")
            if results['prevalence'][0] < 80:
                warns.append("⚠️ **Low normal beat fraction** (<80%) – significant arrhythmia burden")
            for w in warns:
                st.markdown(f"<div class='warning-box'>{w}</div>", unsafe_allow_html=True)
            if not warns:
                st.markdown("<div class='success-box'>✅ Normal arrhythmia burden</div>",
                            unsafe_allow_html=True)
        else:
            st.info("👈 Process an ECG signal first (CSV tab or Image tab).")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 – Detailed Analysis (unchanged)
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<h2 class='sub-header'>Detailed Analysis</h2>", unsafe_allow_html=True)
        if 'processed' in st.session_state and 'results' in st.session_state:
            results = st.session_state['results']

            sig_rs = st.session_state.get('resampled_signal', None)
            if sig_rs is not None and 'r_peaks' in results:
                fig_full = go.Figure()
                fig_full.add_trace(go.Scatter(
                    x=np.arange(len(sig_rs[:TARGET_FS*10]))/TARGET_FS,
                    y=sig_rs[:TARGET_FS*10], mode='lines', name='ECG',
                    line=dict(color='#4a6fa5', width=1)))
                rp = results['r_peaks'][results['r_peaks'] < TARGET_FS*10]
                fig_full.add_trace(go.Scatter(
                    x=rp/TARGET_FS, y=sig_rs[rp], mode='markers',
                    name='R-peaks', marker=dict(color='red', size=8, symbol='x')))
                fig_full.update_layout(height=300,
                    title="Resampled ECG with R-peaks (first 10 s)",
                    xaxis_title="Time (s)", yaxis_title="Amplitude")
                st.plotly_chart(fig_full, use_container_width=True)

            segs = st.session_state.get('sig_segments', None)
            if segs is not None:
                fig = make_subplots(rows=1, cols=6, shared_yaxes=True,
                                    horizontal_spacing=.05)
                for i in range(min(6, len(segs))):
                    fig.add_trace(go.Scatter(
                        y=segs[i].flatten(), mode='lines',
                        line=dict(color=CLASS_COLORS[results['predicted_classes'][i]], width=2)),
                        row=1, col=i+1)
                    fig.update_xaxes(title_text=f"Beat {i+1}", row=1, col=i+1)
                fig.update_layout(height=400, title="First 6 ECG beats", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            df = pd.DataFrame({
                'Beat': range(1, results['total_beats']+1),
                'Class': [CLASS_NAMES[c] for c in results['predicted_classes']],
                'Confidence': results['confidence_scores'].round(3),
                'High Conf': ['✅' if h else '❌' for h in results['high_confidence']],
            })
            st.dataframe(df, use_container_width=True, height=400, hide_index=True)
            st.download_button("📥 Download Results (CSV)",
                               df.to_csv(index=False).encode(),
                               "ecg_results.csv", "text/csv", use_container_width=True)
        else:
            st.info("👈 Please classify an ECG signal first.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 – Help (unchanged + image digitiser notes)
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("<h2 class='sub-header'>Help & Examples</h2>", unsafe_allow_html=True)
        if os.path.exists("ECG_Arrhythmia_Classification_arch.png"):
            st.image("ECG_Arrhythmia_Classification_arch.png",
                     caption="Model Architecture", width=500)
        st.markdown("""
        <div class='info-box'>
            <h4>📖 How to Use</h4>
            <ol>
                <li><b>Upload Model:</b> Upload your trained .h5 model file</li>
                <li><b>CSV tab:</b> Provide raw ECG recording as single-column CSV</li>
                <li><b>Image tab (NEW):</b> Upload a printed ECG scan, select a lead strip ROI,
                    and let the system digitise & classify it automatically</li>
                <li><b>Set Frequency / Method:</b> Configure paper speed and NeuroKit2 algorithm</li>
            </ol>
        </div>

        <div class='info-box'>
            <h4>📄 ECG Image Digitisation Notes</h4>
            <ul>
                <li>Supports PNG, JPG, BMP, TIFF (scans, photos, screenshots)</li>
                <li>Select Lead II (long rhythm strip) for best classification results</li>
                <li>Grid spacing is detected automatically to estimate sampling frequency</li>
                <li>Red/orange ECG paper grids are detected and removed before trace extraction</li>
                <li>The extracted signal is resampled to 125 Hz before classification</li>
                <li>For noisy photographs, try increasing the ROI height slightly</li>
            </ul>
        </div>

        <div class='success-box'>
            <h4>🎯 Use Case Example</h4>
            <p>24-hour Holter Monitor: Export as CSV (125 Hz), upload, get prevalence:
            <br>N>90%: Normal &nbsp;|&nbsp; S>5–10%: Supraventricular &nbsp;|&nbsp; V>5%: Ventricular</p>
        </div>

        <div class='warning-box'>
            <h4>⚠️ Important</h4>
            <ul>
                <li>Research use only – consult a cardiologist for clinical decisions</li>
                <li>Model trained on single-lead (Lead II) MIT-BIH data at 125 Hz</li>
                <li>Image digitisation accuracy depends on scan quality and ROI selection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    st.markdown("""
    <hr style='margin-top:3em;border:0;border-top:1px solid #eee;'>
    <div style='text-align:center;color:#888;font-size:.95rem;margin-bottom:1em;'>
    ©Tilendra Choudhary, Ph.D. – Hybrid Morphological Deep Learning for ECG Arrhythmia Detection
    </div>""", unsafe_allow_html=True)
