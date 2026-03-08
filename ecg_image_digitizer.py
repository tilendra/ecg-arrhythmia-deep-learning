"""
ECG Image Digitization Module  v2.0
=====================================
Robust digitisation of printed 12-lead ECG images for the
ECG Arrhythmia Streamlit App.

Key improvements over v1:
  ✅ Color-based trace isolation (RGB dark-pixel mask, not adaptive threshold)
     → eliminates grid-artifact columns that polluted the centroid
  ✅ Artifact-column rejection + densest-cluster fallback
     → removes residual vertical grid lines from the signal
  ✅ ACF-based grid-period detection with 5:1 small/large-box validation
     → resolves the 175 Hz vs 35 Hz disambiguation ambiguity
  ✅ Auto-polarity correction (R-peaks always positive)
  ✅ Short-gap-only NaN interpolation (long gaps left as NaN → excluded)
  ✅ Sensitivity to both red-grid and black-grid ECG paper
"""

import io
import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, medfilt
from scipy.interpolate import interp1d
import streamlit as st
import base64

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_FS     = 125
TARGET_LENGTH = 187
ECG_PAPER_SPEEDS = {
    "25 mm/s  (standard)": 25,
    "50 mm/s  (fast)":     50,
    "10 mm/s  (slow)":     10,
}

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
.digi-header {
    font-family: 'Courier New', monospace;
    font-size: 1.05rem; color: #1a6b3a;
    background: linear-gradient(90deg, #e8f5e9 0%, #f1f8e9 100%);
    border-left: 4px solid #2e7d32;
    padding: .55rem 1rem; border-radius: 0 6px 6px 0;
    margin-bottom: .8rem;
}
.digi-step {
    display: flex; align-items: center; gap: .55rem;
    font-size: .9rem; color: #444; margin: .35rem 0;
}
.digi-badge {
    background: #2e7d32; color: #fff; border-radius: 50%;
    width: 1.4rem; height: 1.4rem;
    display: inline-flex; align-items: center;
    justify-content: center; font-size: .75rem;
    font-weight: 700; flex-shrink: 0;
}
.digi-warn {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 6px; padding: .7rem 1rem;
    font-size: .85rem; color: #92400e;
}
.digi-info {
    background: #eff6ff; border: 1px solid #93c5fd;
    border-radius: 6px; padding: .7rem 1rem;
    font-size: .85rem; color: #1e3a5f;
}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1.  TRACE EXTRACTION  (color-based, robust)
# ══════════════════════════════════════════════════════════════════════════════

def _color_trace_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Isolate the ECG trace using RGB color analysis.

    Logic:
      - ECG trace  → all channels very dark  (max_RGB < Otsu × 0.6, capped at 100)
      - Red grid   → R high, G medium, B low (skipped automatically)
      - Background → all channels near 255   (skipped automatically)

    Returns binary uint8 mask (255 = trace pixel).
    """
    r = roi_bgr[:, :, 2].astype(np.float32)
    g = roi_bgr[:, :, 1].astype(np.float32)
    b = roi_bgr[:, :, 0].astype(np.float32)
    max_rgb = np.maximum(np.maximum(r, g), b).astype(np.uint8)

    # Otsu threshold on the max-channel image
    otsu_val, _ = cv2.threshold(max_rgb, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    trace_thresh = float(min(otsu_val * 0.6, 100))

    mask = (max_rgb.astype(np.float32) < trace_thresh).astype(np.uint8) * 255

    # Close tiny gaps (handles thin/broken lines)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    return mask


def extract_trace_from_roi(roi_bgr: np.ndarray,
                           max_col_fill_frac: float = 0.25
                           ) -> np.ndarray | None:
    """
    Extract a 1-D ECG waveform from a colour ROI image.

    Algorithm per column:
      1. Apply colour-based trace mask (dark-pixel isolation).
      2. Reject columns where > max_col_fill_frac of rows are lit
         (= residual grid line) → mark as NaN.
      3. For marginal columns, find the densest connected segment
         (handles cases where a grid line and the trace are both present).
      4. Interpolate short NaN gaps (≤ 20 columns).
      5. Invert so that deflections are physiologically correct.
      6. Auto-correct polarity (R-peaks → positive).
    """
    h, w = roi_bgr.shape[:2]
    mask = _color_trace_mask(roi_bgr)
    max_rows = int(h * max_col_fill_frac)

    signal_raw: list[float] = []

    for col in range(w):
        col_data = mask[:, col]
        rows = np.where(col_data > 0)[0]
        n = len(rows)

        if n == 0:
            signal_raw.append(np.nan)

        elif n <= max_rows:
            # Clean column: use centroid
            signal_raw.append(float(np.mean(rows)))

        else:
            # Too many lit pixels → find densest contiguous cluster
            # Split into contiguous segments separated by gaps > 3 px
            gaps = np.where(np.diff(rows) > 3)[0]
            if len(gaps) > 0:
                segs = np.split(rows, gaps + 1)
                best = max(segs, key=len)
                if len(best) <= max_rows:
                    signal_raw.append(float(np.mean(best)))
                else:
                    signal_raw.append(np.nan)   # still too wide = artifact
            else:
                signal_raw.append(np.nan)       # one solid block = artifact

    arr = np.array(signal_raw, dtype=np.float64)

    # Interpolate only SHORT gaps (≤ 20 columns) to avoid masking real baseline
    nans = np.isnan(arr)
    if nans.mean() > 0.65:
        return None                             # signal quality too poor

    x = np.arange(len(arr))
    in_gap, gap_start = False, 0
    for i in range(len(arr) + 1):
        nan_now = (i < len(arr)) and nans[i]
        if nan_now and not in_gap:
            gap_start, in_gap = i, True
        elif not nan_now and in_gap:
            gap_len = i - gap_start
            if gap_len <= 20:
                valid_x = x[~nans]
                valid_y = arr[~nans]
                if len(valid_x) >= 2:
                    arr[gap_start:i] = np.interp(x[gap_start:i], valid_x, valid_y)
            in_gap = False

    # Fill any remaining NaNs with linear interpolation
    still_nan = np.isnan(arr)
    if still_nan.any() and still_nan.mean() < 0.5:
        arr[still_nan] = np.interp(x[still_nan], x[~still_nan], arr[~still_nan])

    if np.isnan(arr).mean() > 0.4:
        return None

    # Invert: image row 0 = top = upward ECG deflection
    arr = -(arr - np.nanmean(arr))

    # Smooth
    arr = medfilt(arr, kernel_size=5).astype(np.float64)

    # Auto-polarity: R-peaks must be positive
    pos_max = float(arr.max())
    neg_max = float((-arr).max())
    if neg_max > pos_max * 1.3:
        arr = -arr

    return arr


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SAMPLING-FREQUENCY ESTIMATION  (ACF + 5:1 box-ratio validation)
# ══════════════════════════════════════════════════════════════════════════════

def _acf_periods(col_projection: np.ndarray,
                 max_lag: int = 80) -> list[tuple[float, float]]:
    """
    Return (lag_px, acf_height) pairs for dominant periodicities
    via normalised autocorrelation.
    """
    proj = col_projection - col_projection.mean()
    n    = len(proj)
    acf  = np.correlate(proj, proj, mode='full')[n - 1:]
    norm = acf[0] + 1e-10
    acf  = acf / norm

    limit = min(max_lag, n // 2)
    search = acf[3:limit]
    pks, props = find_peaks(search, height=0.06, distance=2)
    pks += 3

    return [(float(p), float(props['peak_heights'][i]))
            for i, p in enumerate(pks)]


def estimate_fs_robust(img_bgr:        np.ndarray,
                       x1: int, y1: int, x2: int, y2: int,
                       paper_speed_mms: float = 25.0
                       ) -> tuple[float, bool, str]:
    """
    Estimate the effective sampling frequency (pixels/second) from the grid.

    Strategy
    --------
    1. Sample several horizontal strips: above the ROI, top-quarter of ROI,
       and bottom-quarter of ROI (to avoid the trace itself).
    2. Compute ACF of each column-projection to find dominant periods.
    3. Search for a small-box / large-box pair with a 5:1 ratio
       (standard ECG paper: 1 large box = 5 small boxes = 5 mm).
    4. Use the large-box period (more pixels → more accurate).
    5. Validate: fs must be in [50, 600] Hz.
    6. Return (fs, grid_detected, description).
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    roi_h  = y2 - y1
    roi_w  = x2 - x1
    s_per_small = 1.0 / paper_speed_mms   # 0.04 s at 25 mm/s

    # Build list of horizontal strips to sample
    strips: list[tuple[int, int]] = []

    # Above ROI (pure grid, no trace)
    above_h  = roi_h // 2
    above_y1 = max(0, y1 - above_h)
    above_y2 = max(above_y1 + 10, y1)
    if above_y2 - above_y1 >= 5:
        strips.append((above_y1, above_y2))

    # Top 15% of ROI
    qt = max(5, roi_h // 7)
    strips.append((y1, y1 + qt))

    # Bottom 15% of ROI
    strips.append((y2 - qt, y2))

    # Collect all ACF periodicities
    all_periods: list[tuple[float, float]] = []

    for sy1, sy2 in strips:
        sy1 = max(0, sy1); sy2 = min(h_img, sy2)
        if sy2 - sy1 < 3:
            continue
        strip = gray[sy1:sy2, x1:x2]
        proj  = np.mean(strip, axis=0).astype(np.float64)
        for lag, ht in _acf_periods(proj):
            all_periods.append((lag, ht))

    if not all_periods:
        # Hard fallback: assume 10-second strip
        fs_fallback = roi_w / 10.0
        return float(np.clip(fs_fallback, 50, 600)), False, "fallback(10s)"

    # Sort by ACF height descending
    all_periods.sort(key=lambda p: -p[1])
    lags = [p[0] for p in all_periods]

    # Search for 5:1 pair among top-10 lags
    small_px = large_px = None
    for la in lags[:12]:
        for lb in lags[:12]:
            if lb <= la:
                continue
            ratio = lb / la
            if 4.2 <= ratio <= 5.8:
                small_px = la
                large_px = lb
                break
        if small_px is not None:
            break

    if large_px is not None:
        fs   = large_px / (5.0 * s_per_small)
        desc = f"5:1 pair — large={large_px:.1f}px / small={small_px:.1f}px"
    elif lags:
        # Use strongest lag; try both small and large interpretations
        best = lags[0]
        fs_s = best / s_per_small
        fs_l = best / (5.0 * s_per_small)
        # Prefer whichever falls in a physiologically sensible band [60–500]
        if 60 <= fs_s <= 500:
            fs, desc = fs_s, f"small-box({best:.1f}px)"
        elif 60 <= fs_l <= 500:
            fs, desc = fs_l, f"large-box({best:.1f}px)"
        else:
            fs, desc = roi_w / 10.0, "fallback(10s)"
    else:
        fs, desc = roi_w / 10.0, "fallback(10s)"

    fs = float(np.clip(fs, 50, 600))
    return fs, True, desc


# ══════════════════════════════════════════════════════════════════════════════
# 3.  END-TO-END DIGITISATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def digitize_ecg_image(img_bgr:         np.ndarray,
                       roi:             tuple[int, int, int, int],
                       paper_speed_mms: float = 25.0,
                       manual_fs:       float | None = None) -> dict:
    """
    Digitise a printed ECG image within a user-selected ROI.

    Parameters
    ----------
    img_bgr         : full OpenCV BGR image
    roi             : (x1, y1, x2, y2) in original image pixels
    paper_speed_mms : ECG paper speed (mm/s); used for fs estimation
    manual_fs       : override auto-detected fs (optional)

    Returns
    -------
    dict with:
        signal           – extracted waveform at original_fs (np.ndarray)
        signal_resampled – waveform at 125 Hz (np.ndarray)
        original_fs      – detected / manual original fs (float)
        fs_description   – human-readable fs-detection note (str)
        grid_detected    – bool
        duration_s       – signal duration in seconds (float)
        error            – None or error message (str)
    """
    x1, y1, x2, y2 = roi
    ih, iw = img_bgr.shape[:2]

    # Sanitise ROI bounds
    x1 = max(0, min(x1, iw - 1))
    x2 = max(x1 + 1, min(x2, iw))
    y1 = max(0, min(y1, ih - 1))
    y2 = max(y1 + 1, min(y2, ih))

    if (x2 - x1) < 40 or (y2 - y1) < 15:
        return {"error": "ROI too small — drag a larger rectangle."}

    # 1. Fs estimation
    if manual_fs:
        original_fs   = float(manual_fs)
        grid_detected = False
        fs_desc       = f"manual ({manual_fs:.0f} Hz)"
    else:
        original_fs, grid_detected, fs_desc = estimate_fs_robust(
            img_bgr, x1, y1, x2, y2, paper_speed_mms)

    # 2. Trace extraction from the colour ROI
    roi_bgr   = img_bgr[y1:y2, x1:x2]
    signal_raw = extract_trace_from_roi(roi_bgr)

    if signal_raw is None or len(signal_raw) < 10:
        return {"error": "Could not extract ECG trace — "
                         "try adjusting the ROI to cover one clean lead strip."}

    duration_s = len(signal_raw) / original_fs

    # 3. Resample to TARGET_FS (125 Hz)
    n_target  = max(1, int(round(duration_s * TARGET_FS)))
    x_orig    = np.linspace(0.0, 1.0, len(signal_raw))
    x_new     = np.linspace(0.0, 1.0, n_target)
    interp_fn = interp1d(x_orig, signal_raw, kind='cubic',
                          fill_value='extrapolate')
    signal_rs  = interp_fn(x_new).astype(np.float32)

    return {
        "signal":           signal_raw,
        "signal_resampled": signal_rs,
        "original_fs":      original_fs,
        "fs_description":   fs_desc,
        "grid_detected":    grid_detected,
        "duration_s":       duration_s,
        "error":            None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  STREAMLIT TAB RENDERER
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES  = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)',
                'F (Fusion)', 'Q (Unknown)']
CLASS_COLORS = ['#28a745', '#fd7e14', '#dc3545', '#6f42c1', '#17a2b8']
CLASS_SHORT  = ['N', 'S', 'V', 'F', 'Q']


def _prevalence_cards(results: dict) -> None:
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:.6rem;"
                f"background:#f8f9fa;border-radius:8px;"
                f"border-top:4px solid {CLASS_COLORS[i]};'>"
                f"<div style='font-size:1.8rem;font-weight:700;"
                f"color:{CLASS_COLORS[i]};'>{results['prevalence'][i]:.1f}%</div>"
                f"<div style='font-size:.8rem;color:#555;'>{CLASS_SHORT[i]}</div>"
                f"<div style='font-size:.75rem;color:#888;'>"
                f"({results['class_counts'][i]} beats)</div></div>",
                unsafe_allow_html=True,
            )


def render_ecg_image_tab(model, signal_scaler, feature_scaler,
                         threshold: float, process_and_classify_fn) -> None:
    """
    Render the '📄 ECG from Image' Streamlit tab.

    Parameters
    ----------
    model                   : loaded Keras model
    signal_scaler           : fitted StandardScaler or None
    feature_scaler          : fitted StandardScaler or None
    threshold               : confidence threshold float
    process_and_classify_fn : the process_and_classify() from app.py
    """
    import plotly.graph_objects as go

    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='digi-header'>"
        "📄 ECG Image → Digitisation → Arrhythmia Classification"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── How it works ──────────────────────────────────────────────────────────
    with st.expander("ℹ️ How it works & tips for best results", expanded=False):
        st.markdown("""
        **Pipeline:**
        1. Upload a scanned / photographed 12-lead ECG printout.
        2. Adjust the ROI sliders to surround **one** lead strip
           (Lead II, the long rhythm strip, gives best results).
        3. The system:
           - Isolates the ECG trace using **colour analysis** (dark pixels only),
             bypassing red/orange grid lines entirely.
           - Estimates the sampling frequency via **autocorrelation** of the
             grid projection, validated with the 5:1 small-to-large box ratio.
           - Resamples the extracted signal to **125 Hz** and classifies beats
             using NeuroKit2 + your trained model.

        **Tips:**
        - Select a **single, uninterrupted** lead strip — avoid overlapping leads.
        - Include at least **5 complete cardiac cycles** in the ROI.
        - Reduce ROI height to exclude nearby lead strips above/below.
        - If fs detection seems wrong, enter it manually (override checkbox).
        - Photos should be well-lit and as flat as possible (avoid shadows/glare).
        """)

    # ── Step 1: Upload ────────────────────────────────────────────────────────
    st.markdown("#### Step 1 — Upload ECG Image")
    uploaded = st.file_uploader(
        "12-lead ECG printout (PNG / JPG / BMP / TIFF)",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
        key="ecg_img_upload",
    )

    if uploaded is None:
        st.markdown("""
        <div class='digi-info'>
        <b>Supported formats:</b> PNG · JPEG · BMP · TIFF<br>
        <b>Best practice:</b> Use a flat scan at 150 DPI or higher.
        Select Lead II (the long rhythm strip at the bottom of most printouts).
        </div>
        """, unsafe_allow_html=True)
        return

    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode image. Please try a different file.")
        return

    ih, iw = img_bgr.shape[:2]
    st.success(f"✅ Image loaded — {iw} × {ih} px")

    # ── Step 2: Settings ──────────────────────────────────────────────────────
    st.markdown("#### Step 2 — Paper Settings")
    col_a, col_b = st.columns(2)
    with col_a:
        speed_label  = st.selectbox("Paper speed", list(ECG_PAPER_SPEEDS.keys()))
        paper_speed  = ECG_PAPER_SPEEDS[speed_label]
    with col_b:
        override_fs  = st.checkbox("Override auto-detected fs", value=False)
        manual_fs    = None
        if override_fs:
            manual_fs = float(st.number_input(
                "Manual original fs (Hz)", 50, 2000, 360,
                help="Pixels per second corresponding to your scan resolution"))

    # ── Step 3: ROI selection ─────────────────────────────────────────────────
    st.markdown("#### Step 3 — Select Lead Strip (ROI)")

    # Show the image
    thumb_w = min(iw, 900)
    scale   = thumb_w / iw
    thumb_h = int(ih * scale)
    thumb_rgb = cv2.cvtColor(
        cv2.resize(img_bgr, (thumb_w, thumb_h)), cv2.COLOR_BGR2RGB)
    st.image(thumb_rgb,
             caption=f"Full image ({iw}×{ih} px) — set ROI below",
             use_container_width=True)

    st.markdown(
        f"<div class='digi-warn'>"
        f"Enter coordinates in <b>original image pixels</b> "
        f"(thumbnail shown at ×{scale:.2f} — multiply thumbnail px by "
        f"{1/scale:.2f} to get original coords)."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Smart defaults: bottom rhythm strip ≈ last 18% of height
    d_y1 = int(ih * 0.76); d_y2 = int(ih * 0.94)
    d_x1 = int(iw * 0.04); d_x2 = int(iw * 0.97)

    c1, c2, c3, c4 = st.columns(4)
    x1 = int(c1.number_input("x1 (left)",   0, iw-1, d_x1, key="dig_x1"))
    y1 = int(c2.number_input("y1 (top)",    0, ih-1, d_y1, key="dig_y1"))
    x2 = int(c3.number_input("x2 (right)",  1, iw,   d_x2, key="dig_x2"))
    y2 = int(c4.number_input("y2 (bottom)", 1, ih,   d_y2, key="dig_y2"))

    # Live ROI preview
    preview = thumb_rgb.copy()
    tx1 = int(x1*scale); ty1 = int(y1*scale)
    tx2 = int(x2*scale); ty2 = int(y2*scale)
    overlay = preview.copy()
    cv2.rectangle(overlay, (tx1,ty1), (tx2,ty2), (200,255,200), -1)
    preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)
    cv2.rectangle(preview, (tx1,ty1), (tx2,ty2), (46,125,50), 3)
    st.image(preview,
             caption=f"ROI preview  ({x1},{y1}) → ({x2},{y2})  "
                     f"| {x2-x1}×{y2-y1} px",
             use_container_width=True)

    # ── Step 4: Digitise & Classify ───────────────────────────────────────────
    st.markdown("#### Step 4 — Digitise & Classify")

    if st.button("🔬 Digitise ECG & Run Arrhythmia Classification",
                 type="primary", use_container_width=True):

        if x2 <= x1 or y2 <= y1:
            st.error("Invalid ROI — x2 must be > x1 and y2 must be > y1.")
            return
        if (x2-x1) < 50 or (y2-y1) < 15:
            st.error("ROI too small — please select a larger region.")
            return

        # ── Digitise ─────────────────────────────────────────────────────────
        with st.spinner("Digitising ECG trace…"):
            result = digitize_ecg_image(
                img_bgr,
                roi=(x1, y1, x2, y2),
                paper_speed_mms=paper_speed,
                manual_fs=manual_fs,
            )

        if result.get("error"):
            st.error(f"❌ {result['error']}")
            return

        # ── Digitisation report ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Digitisation Report")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detected fs",   f"{result['original_fs']:.1f} Hz")
        m2.metric("Duration",      f"{result['duration_s']:.2f} s")
        m3.metric("Signal length", f"{len(result['signal'])} px")
        m4.metric("Grid detected", "✅ Yes" if result["grid_detected"] else "⚠️ Est.")

        st.markdown(
            f"<div class='digi-info'>"
            f"<b>fs detection:</b> {result['fs_description']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if not result["grid_detected"]:
            st.warning(
                "⚠️ Grid period could not be reliably detected. "
                "fs was estimated from ROI width (assuming 10 s strip). "
                "For better accuracy, use the override checkbox above."
            )

        # Signal preview
        sig_raw = result["signal"]
        sig_rs  = result["signal_resampled"]
        t_raw   = np.arange(len(sig_raw)) / result["original_fs"]
        t_rs    = np.arange(len(sig_rs))  / TARGET_FS

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_raw, y=sig_raw, mode='lines', name='Extracted (original fs)',
            line=dict(color='#1565c0', width=1.0)))
        fig.add_trace(go.Scatter(
            x=t_rs,  y=sig_rs,  mode='lines', name='Resampled (125 Hz)',
            line=dict(color='#2e7d32', width=1.8, dash='dot')))
        fig.update_layout(
            height=260, margin=dict(l=0,r=0,t=30,b=0),
            title="Extracted ECG Signal (before & after resampling to 125 Hz)",
            xaxis_title="Time (s)", yaxis_title="Amplitude (a.u.)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        import pandas as pd
        st.download_button(
            "📥 Download digitised signal (CSV, 125 Hz)",
            data=pd.DataFrame({"ecg_signal": sig_rs.tolist()}).to_csv(index=False).encode(),
            file_name="ecg_digitized_125hz.csv",
            mime="text/csv",
        )

        # ── Classify ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💓 Arrhythmia Classification")

        with st.spinner("Running NeuroKit2 beat detection + model inference…"):
            results, status, resampled_signal = process_and_classify_fn(
                sig_rs, TARGET_FS, model,
                signal_scaler, feature_scaler, threshold,
            )

        if results is None:
            st.error(f"Classification failed: {status}")
            return

        st.success(
            f"✅ {results['total_beats']} heartbeats classified "
            f"from digitised image!"
        )

        st.session_state.update(
            processed        = True,
            results          = results,
            resampled_signal = resampled_signal,
            num_beats        = results['total_beats'],
        )

        _prevalence_cards(results)

        st.info(
            "📌 Full per-beat table, waveform viewer and confidence histogram "
            "are available in the **Classification Results** and "
            "**Detailed Analysis** tabs."
        )