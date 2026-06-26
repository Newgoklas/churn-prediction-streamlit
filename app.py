# =============================================================================
# app.py - Churn Prediction ULTIMATE (7 Fitur + Tampilan Keren)
# =============================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import random

# ─────────────────────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# SUPER MODERN CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }
    
    /* ── MAIN CONTAINER ── */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* ── HEADER GLASS ── */
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.2), transparent 70%);
        border-radius: 50%;
        animation: float 8s ease-in-out infinite;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(236,72,153,0.15), transparent 70%);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite reverse;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(30px, -20px) scale(1.1); }
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #f093fb, #f5576c, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.6);
        margin: 0.5rem 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        padding: 0.4rem 1.5rem;
        border-radius: 30px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.8rem;
        position: relative;
        z-index: 1;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        font-weight: 500;
    }
    
    .header-badge span {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* ── CARDS GLASS ── */
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255,255,255,0.15);
        box-shadow: 0 12px 48px rgba(0,0,0,0.3);
    }
    
    .card-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: rgba(255,255,255,0.5);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .card-title .badge-count {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        font-size: 0.55rem;
        padding: 0.15rem 0.7rem;
        border-radius: 20px;
        font-weight: 600;
        margin-left: auto;
    }
    
    /* ── INPUTS ── */
    .stNumberInput > div > div > input {
        border-radius: 12px !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        background: rgba(255,255,255,0.05) !important;
        color: white !important;
        transition: all 0.3s ease !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1rem !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4facfe !important;
        box-shadow: 0 0 0 4px rgba(79,172,254,0.15) !important;
        background: rgba(255,255,255,0.08) !important;
    }
    
    .stNumberInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.3);
    }
    
    .stNumberInput label {
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        color: rgba(255,255,255,0.7) !important;
    }
    
    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #f093fb, #f5576c, #4facfe) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 1rem 2.5rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 32px rgba(245,87,108,0.35) !important;
        letter-spacing: 0.5px;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 48px rgba(245,87,108,0.5) !important;
    }
    
    .stButton > button:active {
        transform: scale(0.98) !important;
    }
    
    /* ── SAMPLE BUTTONS ── */
    .sample-btn-container {
        display: flex;
        gap: 10px;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .sample-btn {
        flex: 1;
        min-width: 120px;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
    }
    
    .sample-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    /* ── RESULT CARDS ── */
    .result-churn {
        background: linear-gradient(135deg, rgba(255,107,107,0.2), rgba(238,90,36,0.2));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,107,107,0.3);
        color: #ff6b6b;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 12px 48px rgba(255,107,107,0.2);
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 12px 48px rgba(255,107,107,0.2); }
        50% { box-shadow: 0 12px 64px rgba(255,107,107,0.4); }
    }
    
    .result-churn .sub {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.7;
        display: block;
        margin-top: 0.5rem;
        color: rgba(255,255,255,0.6);
    }
    
    .result-ok {
        background: linear-gradient(135deg, rgba(46,204,113,0.2), rgba(0,184,148,0.2));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(46,204,113,0.3);
        color: #2ecc71;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 12px 48px rgba(46,204,113,0.2);
        animation: pulse-glow-green 2s ease-in-out infinite;
    }
    
    @keyframes pulse-glow-green {
        0%, 100% { box-shadow: 0 12px 48px rgba(46,204,113,0.2); }
        50% { box-shadow: 0 12px 64px rgba(46,204,113,0.4); }
    }
    
    .result-ok .sub {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.7;
        display: block;
        margin-top: 0.5rem;
        color: rgba(255,255,255,0.6);
    }
    
    /* ── METRIC BOX ── */
    .metric-box {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        text-align: center;
    }
    
    .metric-box .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: rgba(255,255,255,0.4);
        font-weight: 600;
    }
    
    .metric-box .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        margin-top: 0.2rem;
    }
    
    /* ── PROGRESS BAR ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #f093fb, #f5576c, #4facfe) !important;
        border-radius: 20px !important;
        height: 12px !important;
    }
    
    .stProgress > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 20px !important;
    }
    
    /* ── INFO BOX ── */
    .info-box {
        background: rgba(79,172,254,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(79,172,254,0.15);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
    }
    
    .info-box strong {
        color: #4facfe;
    }
    
    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: rgba(255,255,255,0.7) !important;
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 0 0 12px 12px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-top: none !important;
    }
    
    /* ── FOOTER ── */
    .footer {
        text-align: center;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.2);
        padding: 2rem 0 0.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.03);
        margin-top: 2rem;
    }
    
    /* ── SIDEBAR ── */
    .css-1d391kg {
        background: rgba(0,0,0,0.3) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: rgba(255,255,255,0.7);
    }
    
    /* ── DIVIDER ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    artifacts = {}
    files_needed = {
        'model': MODEL_DIR / 'best_model.pkl',
        'scaler': MODEL_DIR / 'scaler.pkl',
        'scaler_top': MODEL_DIR / 'scaler_top.pkl',
        'label_enc': MODEL_DIR / 'label_encoders.pkl',
        'top_feat': MODEL_DIR / 'top_features.pkl',
        'all_feat': MODEL_DIR / 'all_features.pkl',
        'metadata': MODEL_DIR / 'model_metadata.pkl',
    }
    missing = []
    
    for key, path in files_needed.items():
        if path.exists():
            try:
                artifacts[key] = joblib.load(path)
            except:
                artifacts[key] = None
        else:
            missing.append(str(path))

    if missing:
        artifacts['_missing'] = missing
    return artifacts

arts = load_artifacts()

if '_missing' in arts:
    st.error("⚠️ Model tidak ditemukan. Jalankan `python main.py` dulu!")
    for f in arts['_missing']:
        st.write(f"  ❌ `{f}`")
    st.stop()

if arts.get('model') is None:
    st.error("❌ Model corrupt!")
    st.stop()

model = arts['model']
scaler = arts.get('scaler_top') or arts.get('scaler')
le_map = arts.get('label_enc', {})
top_feat = arts.get('top_feat', [])
all_feat = arts.get('all_feat', [])
meta = arts.get('metadata', {})

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🚀 Churn Predictor Pro</h1>
    <p>Prediksi churn customer dengan 7 fitur utama</p>
    <div class="header-badge">
        ⚡ Akurasi <span>{meta.get('test_accuracy', 0.8923):.1%}</span> · 
        F1 <span>{meta.get('test_f1', 0.8323):.3f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model Info")
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Model</div>
        <div class="value">{meta.get('best_model_name', 'Voting Ensemble')}</div>
        <div style="margin-top:0.8rem;"></div>
        <div class="label">Akurasi</div>
        <div class="value" style="color:#4facfe;">{meta.get('test_accuracy', 0.8923):.2%}</div>
        <div style="margin-top:0.5rem;"></div>
        <div class="label">F1-Score</div>
        <div class="value" style="color:#f093fb;">{meta.get('test_f1', 0.8323):.2%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📌 7 Fitur Utama")
    for i, feat in enumerate(top_feat[:7], 1):
        emojis = ['⭐', '🎫', '⏱️', '🛍️', '💰', '👑', '📦']
        st.markdown(f"{i}. {emojis[i-1]} `{feat}`")
    
    st.divider()
    st.markdown("""
    **📖 Panduan**
    1. Isi 7 data di bawah
    2. Klik **Prediksi Churn**
    3. Lihat hasil
    """)

# ─────────────────────────────────────────────────────────────
# FORM - 7 FITUR SAJA
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 <strong>Isi 7 data berikut</strong> untuk memprediksi churn. Cukup 1 menit!</div>', unsafe_allow_html=True)

# ── TOMBOL CONTOH DATA ──
st.markdown("""
<div class="sample-btn-container">
    <button class="sample-btn" style="background:linear-gradient(135deg,#667eea,#764ba2);" onclick="document.querySelector('[data-testid=stNumberInput]').value=...">
        🎲 Random
    </button>
    <button class="sample-btn" style="background:linear-gradient(135deg,#f093fb,#f5576c);">
        ⚠️ CHURN
    </button>
    <button class="sample-btn" style="background:linear-gradient(135deg,#a8edea,#2ecc71);">
        ✅ TIDAK CHURN
    </button>
</div>
""", unsafe_allow_html=True)

# ── TOMBOL DENGAN STREAMLIT ──
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    if st.button("🎲 Random", use_container_width=True):
        st.session_state['sample_data'] = {
            'satisfaction_score': random.randint(1, 10),
            'support_tickets': random.randint(0, 10),
            'avg_session_time': round(random.uniform(0.5, 15), 1),
            'last_3_month_purchase_freq': random.randint(0, 15),
            'total_spent': random.randint(50, 2000),
            'is_premium_user': random.randint(0, 1),
            'delivery_delay_days': random.randint(0, 15)
        }

with col_s2:
    if st.button("⚠️ CHURN", use_container_width=True):
        st.session_state['sample_data'] = {
            'satisfaction_score': 2,
            'support_tickets': 8,
            'avg_session_time': 1.2,
            'last_3_month_purchase_freq': 0,
            'total_spent': 50,
            'is_premium_user': 0,
            'delivery_delay_days': 12
        }

with col_s3:
    if st.button("✅ TIDAK CHURN", use_container_width=True):
        st.session_state['sample_data'] = {
            'satisfaction_score': 9,
            'support_tickets': 0,
            'avg_session_time': 12.5,
            'last_3_month_purchase_freq': 12,
            'total_spent': 2500,
            'is_premium_user': 1,
            'delivery_delay_days': 1
        }

# ── 7 FITUR ──
FEATURE_CONFIG = {
    'satisfaction_score': {
        'type': 'number', 'min': 1, 'max': 10, 'default': 7,
        'label': '⭐ Skor Kepuasan (1-10)', 'step': 1
    },
    'support_tickets': {
        'type': 'number', 'min': 0, 'max': 20, 'default': 1,
        'label': '🎫 Tiket Support', 'step': 1
    },
    'avg_session_time': {
        'type': 'float', 'min': 0.0, 'max': 60.0, 'default': 5.0,
        'label': '⏱️ Rata-rata Sesi (menit)', 'step': 0.1
    },
    'last_3_month_purchase_freq': {
        'type': 'number', 'min': 0, 'max': 30, 'default': 3,
        'label': '🛍️ Frekuensi Pembelian 3 Bulan', 'step': 1
    },
    'total_spent': {
        'type': 'float', 'min': 0.0, 'max': 10000.0, 'default': 500.0,
        'label': '💰 Total Pengeluaran ($)', 'step': 10.0
    },
    'is_premium_user': {
        'type': 'number', 'min': 0, 'max': 1, 'default': 1,
        'label': '👑 Premium User (0=Tidak, 1=Ya)', 'step': 1
    },
    'delivery_delay_days': {
        'type': 'number', 'min': 0, 'max': 30, 'default': 6,
        'label': '📦 Keterlambatan Pengiriman (hari)', 'step': 1
    }
}

# ── RENDER ──
user_input = {}

# Apply sample data
if 'sample_data' in st.session_state:
    for key, val in st.session_state['sample_data'].items():
        if key in FEATURE_CONFIG:
            FEATURE_CONFIG[key]['default'] = val

col_left, col_right = st.columns(2)

left_keys = ['satisfaction_score', 'support_tickets', 'avg_session_time', 'last_3_month_purchase_freq']
right_keys = ['total_spent', 'is_premium_user', 'delivery_delay_days']

with col_left:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">
            📊 Fitur Utama
            <span class="badge-count">4</span>
        </div>
    """, unsafe_allow_html=True)
    
    for key in left_keys:
        cfg = FEATURE_CONFIG[key]
        if cfg['type'] == 'float':
            user_input[key] = st.number_input(
                cfg['label'],
                min_value=float(cfg['min']),
                max_value=float(cfg['max']),
                value=float(cfg['default']),
                step=float(cfg['step']),
                key=f"l_{key}"
            )
        else:
            user_input[key] = st.number_input(
                cfg['label'],
                min_value=int(cfg['min']),
                max_value=int(cfg['max']),
                value=int(cfg['default']),
                step=int(cfg['step']),
                key=f"l_{key}"
            )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">
            💰 Fitur Tambahan
            <span class="badge-count">3</span>
        </div>
    """, unsafe_allow_html=True)
    
    for key in right_keys:
        cfg = FEATURE_CONFIG[key]
        if cfg['type'] == 'float':
            user_input[key] = st.number_input(
                cfg['label'],
                min_value=float(cfg['min']),
                max_value=float(cfg['max']),
                value=float(cfg['default']),
                step=float(cfg['step']),
                key=f"r_{key}"
            )
        else:
            user_input[key] = st.number_input(
                cfg['label'],
                min_value=int(cfg['min']),
                max_value=int(cfg['max']),
                value=int(cfg['default']),
                step=int(cfg['step']),
                key=f"r_{key}"
            )
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────
def preprocess_input(raw_input: dict) -> np.ndarray:
    row = {}
    for key, val in raw_input.items():
        row[key] = val
    
    ordered = []
    for feat in top_feat:
        if feat in row:
            ordered.append(row[feat])
        else:
            ordered.append(0)
    
    df_input = pd.DataFrame([ordered], columns=top_feat)
    arr_scaled = scaler.transform(df_input)
    return arr_scaled

# ─────────────────────────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────────────────────────
st.divider()

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🚀 Prediksi Churn", use_container_width=True)

if predict_btn:
    with st.spinner("⏳ Memproses..."):
        try:
            X_input = preprocess_input(user_input)
            prediction = model.predict(X_input)[0]

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_input)[0]
                prob_churn = proba[1] * 100
                prob_no_churn = proba[0] * 100
            else:
                prob_churn = 100 if prediction == 1 else 0
                prob_no_churn = 100 - prob_churn

            st.divider()
            
            st.markdown("### 📋 Hasil Prediksi")
            
            col_res, col_prob = st.columns([1, 1])
            
            with col_res:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-churn">
                        ⚠️ CHURN
                        <span class="sub">Pelanggan berpotensi meninggalkan layanan</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-ok">
                        ✅ TIDAK CHURN
                        <span class="sub">Pelanggan kemungkinan tetap bertahan</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_prob:
                st.markdown("#### Probabilitas")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("🔴 Churn", f"{prob_churn:.1f}%")
                with col_m2:
                    st.metric("🟢 Tidak Churn", f"{prob_no_churn:.1f}%")
                
                st.progress(int(prob_churn), text=f"Risiko Churn: {prob_churn:.1f}%")
            
            st.markdown("### 💡 Rekomendasi")
            
            if prediction == 1:
                recs = []
                if user_input.get('satisfaction_score', 10) < 6:
                    recs.append("📞 **Hubungi pelanggan** — skor kepuasan rendah.")
                if user_input.get('support_tickets', 0) > 3:
                    recs.append("🔧 **Selesaikan tiket support** — terlalu banyak keluhan.")
                if user_input.get('last_3_month_purchase_freq', 10) < 2:
                    recs.append("🛍️ **Kirim penawaran khusus** — pembelian rendah.")
                if user_input.get('avg_session_time', 0) < 3:
                    recs.append("⏱️ **Tingkatkan engagement** — sesi terlalu singkat.")
                if user_input.get('delivery_delay_days', 0) > 5:
                    recs.append("🚚 **Perbaiki pengiriman** — keterlambatan tinggi.")
                if not recs:
                    recs.append("🔄 **Jalankan program retensi** — berikan insentif.")
                for r in recs:
                    st.warning(r)
            else:
                st.success("✅ Customer dalam kondisi sehat. Pertahankan kualitas!")
                if user_input.get('satisfaction_score', 0) >= 9:
                    st.balloons()
                    st.info("⭐ Pelanggan sangat puas! Pertahankan.")

            with st.expander("📄 Detail Data Input"):
                df_display = pd.DataFrame([user_input]).T.reset_index()
                df_display.columns = ['Fitur', 'Nilai']
                st.dataframe(df_display, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🚀 Churn Predictor Pro · 7 Fitur Utama · 2024
</div>
""", unsafe_allow_html=True)
