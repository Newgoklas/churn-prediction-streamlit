# =============================================================================
# app.py - Streamlit Deployment: Churn Prediction (ULTIMATE EDITION)
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
# MODERN CSS - SUPER KEREN
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* ── HEADER GLASS ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
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
        background: rgba(255,255,255,0.05);
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
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite reverse;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(30px, -20px) scale(1.1); }
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 20px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.4rem 1.5rem;
        border-radius: 30px;
        font-size: 0.8rem;
        color: #ffffff;
        margin-top: 0.8rem;
        position: relative;
        z-index: 1;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        font-weight: 500;
    }
    
    .header-badge span {
        font-weight: 700;
        color: #ffd700;
    }
    
    /* ── CARDS ── */
    .card-section {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
        border: 1px solid rgba(255,255,255,0.3);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .card-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.1);
        border-color: rgba(102,126,234,0.2);
    }
    
    .card-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #4a4a6a;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .card-title .badge-count {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-size: 0.6rem;
        padding: 0.15rem 0.7rem;
        border-radius: 20px;
        font-weight: 600;
        margin-left: auto;
    }
    
    /* ── INPUT ── */
    .stNumberInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e8edf4 !important;
        transition: all 0.3s ease !important;
        font-size: 0.95rem !important;
        background: white !important;
        padding: 0.6rem 1rem !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102,126,234,0.1) !important;
    }
    
    .stNumberInput label {
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        color: #4a4a6a !important;
    }
    
    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 1rem 2.5rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 30px rgba(102,126,234,0.4) !important;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 48px rgba(102,126,234,0.5) !important;
    }
    
    .stButton > button:active {
        transform: scale(0.98) !important;
    }
    
    /* ── RESULT ── */
    .result-churn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 12px 48px rgba(245,87,108,0.4);
        border: 1px solid rgba(255,255,255,0.1);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .result-churn .sub {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.9;
        display: block;
        margin-top: 0.5rem;
    }
    
    .result-ok {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #1a1a2e;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 12px 48px rgba(168,237,234,0.4);
        border: 1px solid rgba(255,255,255,0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .result-ok .sub {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.7;
        display: block;
        margin-top: 0.5rem;
    }
    
    /* ── METRIC BOX ── */
    .metric-box {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    
    .metric-box .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8899aa;
        font-weight: 600;
    }
    
    .metric-box .value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 0.2rem;
    }
    
    /* ── INFO BOX ── */
    .info-box {
        background: linear-gradient(135deg, #e8f4fd, #d6eaf8);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        border-left: 5px solid #667eea;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        color: #1a3a5a;
        border: 1px solid rgba(102,126,234,0.1);
    }
    
    .info-box strong {
        color: #667eea;
    }
    
    /* ── SAMPLE BUTTON ── */
    .sample-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.5rem 1.2rem !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(245,87,108,0.3) !important;
    }
    
    .sample-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(245,87,108,0.4) !important;
    }
    
    .sample-btn-green {
        background: linear-gradient(135deg, #a8edea 0%, #2ecc71 100%) !important;
        box-shadow: 0 4px 16px rgba(46,204,113,0.3) !important;
    }
    
    .sample-btn-green:hover {
        box-shadow: 0 8px 30px rgba(46,204,113,0.4) !important;
    }
    
    /* ── PROGRESS ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 20px !important;
        height: 12px !important;
    }
    
    .stProgress > div {
        background: #eef2f6 !important;
        border-radius: 20px !important;
    }
    
    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1a1a2e !important;
        background: rgba(255,255,255,0.5) !important;
        border-radius: 12px !important;
        border: 1px solid #eef2f6 !important;
    }
    
    /* ── FOOTER ── */
    .footer {
        text-align: center;
        font-size: 0.75rem;
        color: #8899aa;
        padding: 2rem 0 0.5rem 0;
        border-top: 1px solid rgba(0,0,0,0.05);
        margin-top: 2rem;
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
            except Exception as e:
                artifacts[key] = None
        else:
            missing.append(str(path))

    if missing:
        artifacts['_missing'] = missing
    return artifacts

arts = load_artifacts()

if '_missing' in arts:
    st.error("⚠️ Beberapa file model tidak ditemukan. Jalankan `python main.py` terlebih dahulu!")
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
# HEADER SUPER KEREN
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🚀 Customer Churn Predictor Pro</h1>
    <p>Prediksi churn customer dengan 7 fitur utama — Akurasi <span style="color:#ffd700;">{meta.get('test_accuracy', 0.8923):.1%}</span></p>
    <div class="header-badge">
        ⚡ Model: {meta.get('best_model_name', 'Voting Ensemble')} · 
        F1: <span>{meta.get('test_f1', 0.8323):.2%}</span>
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
        <div class="value" style="color:#667eea;">{meta.get('test_accuracy', 0.8923):.2%}</div>
        <div style="margin-top:0.5rem;"></div>
        <div class="label">F1-Score</div>
        <div class="value" style="color:#764ba2;">{meta.get('test_f1', 0.8323):.2%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📌 7 Fitur Utama")
    for i, feat in enumerate(top_feat[:7], 1):
        emoji = ['⭐', '🎫', '⏱️', '🛍️', '💰', '👑', '📦'][i-1]
        st.markdown(f"{i}. {emoji} `{feat}`")
    
    st.divider()
    st.markdown("""
    **📖 Panduan**
    1. Isi 7 data di bawah
    2. Klik **Prediksi Churn**
    3. Lihat hasil & rekomendasi
    """)

# ─────────────────────────────────────────────────────────────
# FORM INPUT
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 <strong>Isi 7 data berikut</strong> untuk memprediksi churn. Cukup 1 menit!</div>', unsafe_allow_html=True)

# ── CONTOH DATA ──
st.markdown("""
<div style="display:flex; gap:10px; margin-bottom:1rem;">
    <span style="font-size:0.8rem; color:#4a4a6a;">📌 Coba contoh data:</span>
</div>
""", unsafe_allow_html=True)

col_sample1, col_sample2, col_sample3 = st.columns([1, 1, 1])

with col_sample1:
    if st.button("📊 Data Random", key="random_btn", use_container_width=True):
        st.session_state['sample_data'] = {
            'satisfaction_score': random.randint(1, 10),
            'support_tickets': random.randint(0, 10),
            'avg_session_time': round(random.uniform(0.5, 15), 1),
            'last_3_month_purchase_freq': random.randint(0, 15),
            'total_spent': random.randint(50, 2000),
            'is_premium_user': random.randint(0, 1),
            'delivery_delay_days': random.randint(0, 15)
        }

with col_sample2:
    if st.button("⚠️ Data CHURN", key="churn_btn", use_container_width=True):
        st.session_state['sample_data'] = {
            'satisfaction_score': 2,
            'support_tickets': 8,
            'avg_session_time': 1.2,
            'last_3_month_purchase_freq': 0,
            'total_spent': 50,
            'is_premium_user': 0,
            'delivery_delay_days': 12
        }

with col_sample3:
    if st.button("✅ Data TIDAK CHURN", key="nochurn_btn", use_container_width=True):
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
        'label': '⭐ Skor Kepuasan (1-10)', 'step': 1,
        'help': 'Semakin tinggi, semakin puas'
    },
    'support_tickets': {
        'type': 'number', 'min': 0, 'max': 20, 'default': 1, 
        'label': '🎫 Tiket Support', 'step': 1,
        'help': 'Jumlah tiket yang diajukan'
    },
    'avg_session_time': {
        'type': 'float', 'min': 0.0, 'max': 60.0, 'default': 5.0, 
        'label': '⏱️ Rata-rata Sesi (menit)', 'step': 0.1,
        'help': 'Rata-rata waktu per sesi'
    },
    'last_3_month_purchase_freq': {
        'type': 'number', 'min': 0, 'max': 30, 'default': 3, 
        'label': '🛍️ Frekuensi Pembelian 3 Bulan', 'step': 1,
        'help': 'Jumlah pembelian 3 bulan terakhir'
    },
    'total_spent': {
        'type': 'float', 'min': 0.0, 'max': 10000.0, 'default': 500.0, 
        'label': '💰 Total Pengeluaran ($)', 'step': 10.0,
        'help': 'Total uang yang dikeluarkan'
    },
    'is_premium_user': {
        'type': 'number', 'min': 0, 'max': 1, 'default': 1, 
        'label': '👑 Premium User (0=Tidak, 1=Ya)', 'step': 1,
        'help': 'Status premium pelanggan'
    },
    'delivery_delay_days': {
        'type': 'number', 'min': 0, 'max': 30, 'default': 6, 
        'label': '📦 Keterlambatan Pengiriman (hari)', 'step': 1,
        'help': 'Rata-rata keterlambatan'
    }
}

# ── RENDER FORM ──
user_input = {}

# Apply sample data if exists
if 'sample_data' in st.session_state:
    for key, val in st.session_state['sample_data'].items():
        if key in FEATURE_CONFIG:
            FEATURE_CONFIG[key]['default'] = val

col_left, col_right = st.columns(2)

left_keys = ['satisfaction_score', 'support_tickets', 'avg_session_time', 'last_3_month_purchase_freq']
right_keys = ['total_spent', 'is_premium_user', 'delivery_delay_days']

with col_left:
    st.markdown("""
    <div class="card-section">
        <div class="card-title">
            <span>📊</span> Fitur Utama
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
                help=cfg.get('help', ''),
                key=f"left_{key}"
            )
        else:
            user_input[key] = st.number_input(
                cfg['label'],
                min_value=int(cfg['min']),
                max_value=int(cfg['max']),
                value=int(cfg['default']),
                step=int(cfg['step']),
                help=cfg.get('help', ''),
                key=f"left_{key}"
            )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="card-section">
        <div class="card-title">
            <span>💰</span> Fitur Tambahan
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
                help=cfg.get('help', ''),
                key=f"right_{key}"
            )
        else:
            user_input[key] = st.number_input(
                cfg['label'],
                min_value=int(cfg['min']),
                max_value=int(cfg['max']),
                value=int(cfg['default']),
                step=int(cfg['step']),
                help=cfg.get('help', ''),
                key=f"right_{key}"
            )
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PREPROCESS & PREDICT
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
# BUTTON PREDIKSI
# ─────────────────────────────────────────────────────────────
st.divider()

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🚀 Prediksi Churn Sekarang", use_container_width=True)

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
    🚀 UAS Data Science · Churn Prediction Pro · 7 Fitur Utama · 2024
</div>
""", unsafe_allow_html=True)
