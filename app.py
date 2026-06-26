# =============================================================================
# app.py - Streamlit Deployment: Churn Prediction (MODERN UI)
# =============================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | Sales & Marketing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# MODERN CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #f0f4f8;
    }
    
    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 200px;
        height: 200px;
        background: rgba(255,255,255,0.02);
        border-radius: 50%;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1rem;
        color: rgba(255,255,255,0.7);
        margin: 0.3rem 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        padding: 0.2rem 1rem;
        border-radius: 20px;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.06);
    }
    
    /* ── Card Section ── */
    .card-section {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 1rem;
        transition: box-shadow 0.3s ease;
    }
    
    .card-section:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }
    
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    
    .card-title .icon {
        font-size: 1.2rem;
    }
    
    .card-title .badge-count {
        background: #e8f0fe;
        color: #1a73e8;
        font-size: 0.6rem;
        padding: 0.1rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
        margin-left: auto;
    }
    
    /* ── Form Input ── */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 10px !important;
        border: 1.5px solid #e8edf4 !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 3px rgba(26,115,232,0.1) !important;
    }
    
    .stNumberInput label,
    .stSelectbox label {
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        color: #4a4a6a !important;
    }
    
    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(26,26,46,0.25) !important;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(26,26,46,0.35) !important;
        background: linear-gradient(135deg, #2a2a4e 0%, #1a2a4e 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* ── Result Cards ── */
    .result-churn {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 2rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 8px 30px rgba(238,90,36,0.35);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(4px);
    }
    
    .result-churn .sub {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.85;
        display: block;
        margin-top: 0.3rem;
    }
    
    .result-ok {
        background: linear-gradient(135deg, #00b894, #00a86b);
        color: white;
        padding: 2rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        box-shadow: 0 8px 30px rgba(0,184,148,0.35);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(4px);
    }
    
    .result-ok .sub {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.85;
        display: block;
        margin-top: 0.3rem;
    }
    
    /* ── Metrics ── */
    .metric-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.2rem 1.2rem;
        border-left: 4px solid #1a1a2e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        border: 1px solid #eef2f6;
    }
    
    .metric-box .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8899aa;
        font-weight: 600;
    }
    
    .metric-box .value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    
    /* ── Info Box ── */
    .info-box {
        background: linear-gradient(135deg, #e8f4fd, #d6eaf8);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1a73e8;
        margin-bottom: 1.2rem;
        font-size: 0.9rem;
        color: #1a3a5a;
        border: 1px solid rgba(26,115,232,0.1);
    }
    
    /* ── Sidebar ── */
    .css-1d391kg {
        background: #ffffff !important;
        border-right: 1px solid #eef2f6 !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: #1a1a2e;
    }
    
    /* ── Progress Bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00b894, #00a86b) !important;
        border-radius: 20px !important;
        height: 10px !important;
    }
    
    .stProgress > div {
        background: #eef2f6 !important;
        border-radius: 20px !important;
    }
    
    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        color: #1a1a2e !important;
        background: #f8fafc !important;
        border-radius: 10px !important;
        border: 1px solid #eef2f6 !important;
    }
    
    .streamlit-expanderContent {
        background: #ffffff !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid #eef2f6 !important;
        border-top: none !important;
    }
    
    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e0e6ed, transparent);
        margin: 1.5rem 0;
    }
    
    /* ── Footer ── */
    .footer {
        text-align: center;
        font-size: 0.75rem;
        color: #8899aa;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid #eef2f6;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load Model & Artefak
# ─────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    artifacts = {}
    files_needed = {
        'model': MODEL_DIR / 'best_model.pkl',
        'scaler_top': MODEL_DIR / 'scaler_top.pkl',
        'scaler': MODEL_DIR / 'scaler.pkl',
        'label_enc': MODEL_DIR / 'label_encoders.pkl',
        'top_feat': MODEL_DIR / 'top_features.pkl',
        'all_feat': MODEL_DIR / 'all_features.pkl',
        'metadata': MODEL_DIR / 'model_metadata.pkl',
    }
    missing = []
    corrupt = []
    
    for key, path in files_needed.items():
        if path.exists():
            try:
                artifacts[key] = joblib.load(path)
            except Exception as e:
                corrupt.append(f"{path} ({str(e)[:80]})")
                artifacts[key] = None
        else:
            missing.append(str(path))

    if corrupt:
        st.error("⚠️ File model corrupt! Jalankan ulang `main.py` di local.")
        for c in corrupt:
            st.write(f"  ❌ {c}")
        st.stop()
    
    if missing:
        artifacts['_missing'] = missing
    return artifacts

arts = load_artifacts()

if '_missing' in arts:
    st.error("⚠️ Beberapa file model tidak ditemukan. Jalankan `python main.py` terlebih dahulu!")
    st.code("python main.py", language="bash")
    for f in arts['_missing']:
        st.write(f"  ❌ `{f}`")
    st.stop()

if arts.get('model') is None:
    st.error("❌ Model corrupt! Jalankan ulang `main.py`.")
    st.stop()

model = arts['model']
scaler = arts.get('scaler_top') or arts.get('scaler')
le_map = arts.get('label_enc', {})
top_feat = arts.get('top_feat', [])
all_feat = arts.get('all_feat', [])
meta = arts.get('metadata', {})

# ─────────────────────────────────────────────────────────────
# HEADER MODERN
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>📊 Customer Churn Predictor</h1>
    <p>Prediksi kemungkinan churn customer menggunakan machine learning</p>
    <div class="header-badge">🎯 Akurasi {meta.get('test_accuracy', 0.8923):.1%} · F1 {meta.get('test_f1', 0.8323):.3f}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR MODERN
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model Info")
    
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Model</div>
        <div class="value">{meta.get('best_model_name', 'Voting Ensemble')}</div>
        <div style="margin-top:0.5rem;"></div>
        <div class="label">Test Accuracy</div>
        <div class="value">{meta.get('test_accuracy', 0.8923):.2%}</div>
        <div style="margin-top:0.5rem;"></div>
        <div class="label">Test F1-Score</div>
        <div class="value">{meta.get('test_f1', 0.8323):.2%}</div>
        <div style="margin-top:0.5rem;"></div>
        <div class="label">Fitur Terbaik</div>
        <div class="value">Top {len(top_feat)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📌 Fitur Terpenting")
    for i, feat in enumerate(top_feat[:8], 1):
        st.markdown(f"`{i}. {feat}`")
    
    st.divider()
    st.markdown("""
    **📖 Panduan**
    1. Isi data customer di form
    2. Klik **Prediksi Churn**
    3. Lihat hasil & rekomendasi
    """)

# ─────────────────────────────────────────────────────────────
# FORM INPUT MODERN
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 Isi data customer untuk memprediksi kemungkinan churn.</div>', unsafe_allow_html=True)

# ── FEATURE CONFIG ──
FEATURE_CONFIG = {
    # Demografi
    'age': {'type': 'number', 'min': 18, 'max': 80, 'default': 35, 'label': '🧑 Usia', 'step': 1, 'section': 'demografi'},
    'gender': {'type': 'select', 'options': ['Male', 'Female', 'Other'], 'default': 'Female', 'label': '👤 Gender', 'section': 'demografi'},
    'country': {'type': 'select', 'options': ['USA','UK','Germany','France','India','Australia','Canada','Brazil'], 'default': 'Brazil', 'label': '🌍 Negara', 'section': 'demografi'},
    'city': {'type': 'select', 'options': ['New York','London','Berlin','Paris','Mumbai','Sydney','Toronto','São Paulo'], 'default': 'Mumbai', 'label': '🏙️ Kota', 'section': 'demografi'},
    
    # Aktivitas
    'total_visits': {'type': 'number', 'min': 0, 'max': 500, 'default': 50, 'label': '👀 Total Kunjungan', 'step': 1, 'section': 'aktivitas'},
    'avg_session_time': {'type': 'float', 'min': 0.0, 'max': 60.0, 'default': 5.0, 'label': '⏱️ Rata-rata Sesi (menit)', 'step': 0.1, 'section': 'aktivitas'},
    'pages_per_session': {'type': 'number', 'min': 1, 'max': 50, 'default': 5, 'label': '📄 Halaman per Sesi', 'step': 1, 'section': 'aktivitas'},
    'device_type': {'type': 'select', 'options': ['Mobile','Desktop','Tablet'], 'default': 'Mobile', 'label': '📱 Perangkat', 'section': 'aktivitas'},
    'acquisition_channel': {'type': 'select', 'options': ['Organic','Paid','Referral','Social','Email'], 'default': 'Referral', 'label': '📢 Channel', 'section': 'aktivitas'},
    
    # Email
    'email_open_rate': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.30, 'label': '📧 Email Open Rate', 'step': 0.01, 'section': 'email'},
    'email_click_rate': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.10, 'label': '🖱️ Email Click Rate', 'step': 0.01, 'section': 'email'},
    
    # Pembelian
    'total_spent': {'type': 'float', 'min': 0.0, 'max': 10000.0, 'default': 500.0, 'label': '💰 Total Pengeluaran ($)', 'step': 1.0, 'section': 'pembelian'},
    'avg_order_value': {'type': 'float', 'min': 0.0, 'max': 2000.0, 'default': 100.0, 'label': '🛒 Rata-rata Order ($)', 'step': 1.0, 'section': 'pembelian'},
    'discount_used': {'type': 'number', 'min': 0, 'max': 50, 'default': 3, 'label': '🏷️ Diskon Dipakai', 'step': 1, 'section': 'pembelian'},
    'last_3_month_purchase_freq': {'type': 'number', 'min': 0, 'max': 30, 'default': 3, 'label': '🛍️ Frekuensi Pembelian 3 Bulan', 'step': 1, 'section': 'pembelian'},
    
    # Dukungan
    'support_tickets': {'type': 'number', 'min': 0, 'max': 20, 'default': 1, 'label': '🎫 Tiket Support', 'step': 1, 'section': 'dukungan'},
    'refund_requested': {'type': 'number', 'min': 0, 'max': 10, 'default': 0, 'label': '↩️ Refund Diminta', 'step': 1, 'section': 'dukungan'},
    'delivery_delay_days': {'type': 'number', 'min': 0, 'max': 30, 'default': 6, 'label': '📦 Keterlambatan Pengiriman (hari)', 'step': 1, 'section': 'dukungan'},
    'satisfaction_score': {'type': 'number', 'min': 1, 'max': 10, 'default': 7, 'label': '⭐ Skor Kepuasan (1-10)', 'step': 1, 'section': 'dukungan'},
    'nps_score': {'type': 'number', 'min': -100, 'max': 100, 'default': 30, 'label': '📊 NPS Score', 'step': 1, 'section': 'dukungan'},
    
    # Keuangan
    'marketing_spend_per_user': {'type': 'float', 'min': 0.0, 'max': 500.0, 'default': 30.0, 'label': '📢 Marketing Spend ($)', 'step': 0.5, 'section': 'keuangan'},
    'lifetime_value': {'type': 'float', 'min': 0.0, 'max': 20000.0, 'default': 1500.0, 'label': '💎 Lifetime Value ($)', 'step': 10.0, 'section': 'keuangan'},
    'is_premium_user': {'type': 'number', 'min': 0, 'max': 1, 'default': 1, 'label': '👑 Premium User', 'step': 1, 'section': 'keuangan'},
    'subscription_type': {'type': 'select', 'options': ['Basic','Standard','Premium'], 'default': 'Standard', 'label': '📋 Tipe Langganan', 'section': 'keuangan'},
    'payment_method': {'type': 'select', 'options': ['Credit Card','Debit Card','PayPal','Bank Transfer','Crypto'], 'default': 'Credit Card', 'label': '💳 Metode Pembayaran', 'section': 'keuangan'},
}

# ── RENDER FORM ──
user_input = {}

sections = {}
for key, cfg in FEATURE_CONFIG.items():
    section = cfg.get('section', 'lainnya')
    if section not in sections:
        sections[section] = []
    sections[section].append(key)

section_labels = {
    'demografi': ('👤 Demografi', 6),
    'aktivitas': ('📱 Aktivitas', 5),
    'email': ('📧 Email', 2),
    'pembelian': ('🛒 Pembelian', 4),
    'dukungan': ('🎯 Dukungan & Kepuasan', 5),
    'keuangan': ('💰 Keuangan & Langganan', 5)
}

col_left, col_right = st.columns(2)

# Kiri
with col_left:
    for section in ['demografi', 'aktivitas', 'email']:
        if section in sections:
            label, count = section_labels.get(section, (section, 0))
            st.markdown(f"""
            <div class="card-section">
                <div class="card-title">
                    <span class="icon">{label.split()[0]}</span>
                    {label.split()[1] if len(label.split()) > 1 else label}
                    <span class="badge-count">{len(sections[section])}</span>
                </div>
            """, unsafe_allow_html=True)
            
            for key in sections[section]:
                cfg = FEATURE_CONFIG[key]
                if cfg['type'] == 'select':
                    user_input[key] = st.selectbox(
                        cfg['label'], options=cfg['options'],
                        index=cfg['options'].index(cfg['default']), key=f"cat_{key}"
                    )
                elif cfg['type'] == 'float':
                    user_input[key] = st.number_input(
                        cfg['label'], min_value=float(cfg['min']),
                        max_value=float(cfg['max']), value=float(cfg['default']),
                        step=float(cfg['step']), key=f"num_{key}"
                    )
                else:
                    user_input[key] = st.number_input(
                        cfg['label'], min_value=int(cfg['min']),
                        max_value=int(cfg['max']), value=int(cfg['default']),
                        step=int(cfg['step']), key=f"num_{key}"
                    )
            st.markdown("</div>", unsafe_allow_html=True)

# Kanan
with col_right:
    for section in ['pembelian', 'dukungan', 'keuangan']:
        if section in sections:
            label, count = section_labels.get(section, (section, 0))
            st.markdown(f"""
            <div class="card-section">
                <div class="card-title">
                    <span class="icon">{label.split()[0]}</span>
                    {label.split()[1] if len(label.split()) > 1 else label}
                    <span class="badge-count">{len(sections[section])}</span>
                </div>
            """, unsafe_allow_html=True)
            
            for key in sections[section]:
                cfg = FEATURE_CONFIG[key]
                if cfg['type'] == 'select':
                    user_input[key] = st.selectbox(
                        cfg['label'], options=cfg['options'],
                        index=cfg['options'].index(cfg['default']), key=f"cat_{key}"
                    )
                elif cfg['type'] == 'float':
                    user_input[key] = st.number_input(
                        cfg['label'], min_value=float(cfg['min']),
                        max_value=float(cfg['max']), value=float(cfg['default']),
                        step=float(cfg['step']), key=f"num_{key}"
                    )
                else:
                    user_input[key] = st.number_input(
                        cfg['label'], min_value=int(cfg['min']),
                        max_value=int(cfg['max']), value=int(cfg['default']),
                        step=int(cfg['step']), key=f"num_{key}"
                    )
            st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PREPROCESS & PREDICT
# ─────────────────────────────────────────────────────────────
def preprocess_input(raw_input: dict) -> np.ndarray:
    row = {}
    for key, val in raw_input.items():
        if key in le_map:
            try:
                le = le_map[key]
                val_str = str(val)
                if hasattr(le, 'classes_'):
                    classes = list(le.classes_)
                    if val_str in classes:
                        row[key] = le.transform([val_str])[0]
                    else:
                        row[key] = 0
                else:
                    row[key] = 0
            except Exception:
                row[key] = 0
        else:
            row[key] = val
    
    ordered = []
    for feat in all_feat:
        if feat in row:
            ordered.append(row[feat])
        else:
            ordered.append(0)
    
    df_input = pd.DataFrame([ordered], columns=all_feat)
    available_top = [f for f in top_feat if f in df_input.columns]
    df_top = df_input[available_top]
    arr_scaled = scaler.transform(df_top)
    return arr_scaled

# ─────────────────────────────────────────────────────────────
# BUTTON PREDIKSI
# ─────────────────────────────────────────────────────────────
st.divider()

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔮 Prediksi Churn", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("⏳ Memproses data..."):
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
            
            # ── HASIL ──
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
            
            # ── REKOMENDASI ──
            st.markdown("### 💡 Rekomendasi")
            
            if prediction == 1:
                recs = []
                if user_input.get('satisfaction_score', 10) < 6:
                    recs.append("📞 **Hubungi pelanggan** — skor kepuasan rendah, lakukan survei follow-up.")
                if user_input.get('support_tickets', 0) > 3:
                    recs.append("🔧 **Selesaikan tiket support** — ada banyak tiket yang belum terselesaikan.")
                if user_input.get('last_3_month_purchase_freq', 10) < 2:
                    recs.append("🛍️ **Kirim penawaran khusus** — frekuensi pembelian rendah dalam 3 bulan terakhir.")
                if user_input.get('discount_used', 5) < 1:
                    recs.append("🎁 **Tawarkan diskon personal** — pelanggan belum pernah menggunakan diskon.")
                if user_input.get('delivery_delay_days', 0) > 5:
                    recs.append("🚚 **Perbaiki pengiriman** — keterlambatan pengiriman tinggi.")
                if user_input.get('avg_session_time', 0) < 3:
                    recs.append("⏱️ **Tingkatkan engagement** — sesi terlalu singkat.")
                if not recs:
                    recs.append("🔄 **Jalankan program retensi** — kirim email personal dan tawarkan benefit eksklusif.")
                
                for r in recs:
                    st.warning(r)
            else:
                st.success("✅ Pelanggan dalam kondisi sehat. Pertahankan kualitas layanan dan lanjutkan program loyalitas.")
                if user_input.get('is_premium_user', 0) == 0:
                    st.info("💎 Pertimbangkan untuk menawarkan **upgrade ke Premium** kepada pelanggan ini.")
                if user_input.get('satisfaction_score', 0) >= 9:
                    st.balloons()
                    st.success("⭐ Pelanggan sangat puas! Pertahankan kualitas ini.")

            # ── DETAIL ──
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
    UAS Data Science · Sales & Marketing Churn Prediction · 2024
</div>
""", unsafe_allow_html=True)
