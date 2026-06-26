# =============================================================================
# app.py - Churn Prediction PREMIUM (Ringkas & Keren)
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
# SUPER MODERN CSS - DARK GLASS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }
    
    /* ── HEADER GLASS ── */
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 50px -12px rgba(0,0,0,0.5);
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
        background: radial-gradient(circle, rgba(99,102,241,0.15), transparent 70%);
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
        background: radial-gradient(circle, rgba(236,72,153,0.1), transparent 70%);
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
        background: linear-gradient(135deg, #f093fb, #f5576c, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .main-header p {
        font-size: 1rem;
        color: rgba(255,255,255,0.5);
        margin: 0.3rem 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.06);
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.6);
        margin-top: 0.6rem;
        position: relative;
        z-index: 1;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .header-badge span {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* ── GLASS CARD ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.4s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255,255,255,0.12);
        box-shadow: 0 12px 48px rgba(0,0,0,0.3);
    }
    
    .card-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: rgba(255,255,255,0.4);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .card-title .badge {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        font-size: 0.5rem;
        padding: 0.1rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        margin-left: auto;
    }
    
    /* ── INPUTS ── */
    .stNumberInput > div > div > input {
        border-radius: 12px !important;
        border: 1.5px solid rgba(255,255,255,0.06) !important;
        background: rgba(255,255,255,0.03) !important;
        color: white !important;
        transition: all 0.3s ease !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4facfe !important;
        box-shadow: 0 0 0 4px rgba(79,172,254,0.1) !important;
        background: rgba(255,255,255,0.06) !important;
    }
    
    .stNumberInput label {
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        color: rgba(255,255,255,0.5) !important;
    }
    
    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #f093fb, #f5576c, #4facfe) !important;
        color: white !important;
        border-radius: 16px !important;
        padding: 0.9rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 8px 32px rgba(245,87,108,0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 12px 48px rgba(245,87,108,0.4) !important;
    }
    
    /* ── SAMPLE BUTTONS ── */
    .sample-btn-container {
        display: flex;
        gap: 10px;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    
    /* ── RESULT ── */
    .result-churn {
        background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(238,90,36,0.1));
        border: 1px solid rgba(255,107,107,0.2);
        color: #ff6b6b;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        backdrop-filter: blur(20px);
        animation: glow-red 2s ease-in-out infinite;
    }
    
    @keyframes glow-red {
        0%, 100% { box-shadow: 0 8px 32px rgba(255,107,107,0.1); }
        50% { box-shadow: 0 8px 48px rgba(255,107,107,0.25); }
    }
    
    .result-churn .sub {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.6;
        display: block;
        margin-top: 0.3rem;
        color: rgba(255,255,255,0.5);
    }
    
    .result-ok {
        background: linear-gradient(135deg, rgba(46,204,113,0.15), rgba(0,184,148,0.1));
        border: 1px solid rgba(46,204,113,0.2);
        color: #2ecc71;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        backdrop-filter: blur(20px);
        animation: glow-green 2s ease-in-out infinite;
    }
    
    @keyframes glow-green {
        0%, 100% { box-shadow: 0 8px 32px rgba(46,204,113,0.1); }
        50% { box-shadow: 0 8px 48px rgba(46,204,113,0.25); }
    }
    
    .result-ok .sub {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.6;
        display: block;
        margin-top: 0.3rem;
        color: rgba(255,255,255,0.5);
    }
    
    .metric-box {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.04);
    }
    
    .metric-box .label {
        font-size: 0.6rem;
        text-transform: uppercase;
        color: rgba(255,255,255,0.3);
        font-weight: 600;
    }
    
    .metric-box .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        margin-top: 0.1rem;
    }
    
    .info-box {
        background: rgba(79,172,254,0.06);
        border: 1px solid rgba(79,172,254,0.08);
        border-radius: 14px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
    }
    
    .info-box strong {
        color: #4facfe;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #f093fb, #f5576c, #4facfe) !important;
        border-radius: 20px !important;
        height: 10px !important;
    }
    
    .stProgress > div {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 20px !important;
    }
    
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: rgba(255,255,255,0.5) !important;
        background: rgba(255,255,255,0.02) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.01) !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    .footer {
        text-align: center;
        font-size: 0.65rem;
        color: rgba(255,255,255,0.1);
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.02);
        margin-top: 1.5rem;
    }
    
    .css-1d391kg {
        background: rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255,255,255,0.03) !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: rgba(255,255,255,0.6);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    artifacts = {}
    files = {
        'model': MODEL_DIR / 'best_model.pkl',
        'scaler': MODEL_DIR / 'scaler.pkl',
        'scaler_top': MODEL_DIR / 'scaler_top.pkl',
        'label_enc': MODEL_DIR / 'label_encoders.pkl',
        'top_feat': MODEL_DIR / 'top_features.pkl',
        'all_feat': MODEL_DIR / 'all_features.pkl',
        'metadata': MODEL_DIR / 'model_metadata.pkl',
    }
    missing = []
    for key, path in files.items():
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
    <div class="header-badge">⚡ Akurasi <span>{meta.get('test_accuracy', 0.8923):.1%}</span> · F1 <span>{meta.get('test_f1', 0.8323):.3f}</span></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model")
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Model</div>
        <div class="value">{meta.get('best_model_name', 'Voting Ensemble')}</div>
        <div style="margin-top:0.5rem;"></div>
        <div class="label">Akurasi</div>
        <div class="value" style="color:#4facfe;">{meta.get('test_accuracy', 0.8923):.2%}</div>
        <div style="margin-top:0.3rem;"></div>
        <div class="label">F1-Score</div>
        <div class="value" style="color:#f093fb;">{meta.get('test_f1', 0.8323):.2%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📌 Fitur")
    emojis = ['⭐', '🎫', '⏱️', '🛍️', '💰', '👑', '📦']
    for i, feat in enumerate(top_feat[:7], 1):
        st.markdown(f"{emojis[i-1]} `{feat.replace('_', ' ').title()}`")
    
    st.divider()
    st.caption("Isi 7 data → Klik Prediksi")

# ─────────────────────────────────────────────────────────────
# FORM - 7 FITUR RINGKAS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 <strong>Isi 7 data berikut</strong> — cukup 1 menit!</div>', unsafe_allow_html=True)

# ── TOMBOL CONTOH ──
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    if st.button("🎲 Random", use_container_width=True):
        st.session_state['sample'] = {
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
        st.session_state['sample'] = {
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
        st.session_state['sample'] = {
            'satisfaction_score': 9,
            'support_tickets': 0,
            'avg_session_time': 12.5,
            'last_3_month_purchase_freq': 12,
            'total_spent': 2500,
            'is_premium_user': 1,
            'delivery_delay_days': 1
        }

# ── 7 FITUR ──
FEATURES = {
    'satisfaction_score': {'min': 1, 'max': 10, 'default': 7, 'label': '⭐ Skor Kepuasan (1-10)', 'step': 1},
    'support_tickets': {'min': 0, 'max': 20, 'default': 1, 'label': '🎫 Tiket Support', 'step': 1},
    'avg_session_time': {'min': 0.0, 'max': 60.0, 'default': 5.0, 'label': '⏱️ Rata-rata Sesi (menit)', 'step': 0.1},
    'last_3_month_purchase_freq': {'min': 0, 'max': 30, 'default': 3, 'label': '🛍️ Frekuensi Pembelian 3 Bulan', 'step': 1},
    'total_spent': {'min': 0.0, 'max': 10000.0, 'default': 500.0, 'label': '💰 Total Pengeluaran ($)', 'step': 10.0},
    'is_premium_user': {'min': 0, 'max': 1, 'default': 1, 'label': '👑 Premium User (0/1)', 'step': 1},
    'delivery_delay_days': {'min': 0, 'max': 30, 'default': 6, 'label': '📦 Keterlambatan Kirim (hari)', 'step': 1},
}

user_input = {}

# Apply sample
if 'sample' in st.session_state:
    for k, v in st.session_state['sample'].items():
        if k in FEATURES:
            FEATURES[k]['default'] = v

# Render 2 kolom
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📊 Utama <span class="badge">4</span></div>
    """, unsafe_allow_html=True)
    for k in ['satisfaction_score', 'support_tickets', 'avg_session_time', 'last_3_month_purchase_freq']:
        f = FEATURES[k]
        if isinstance(f['default'], float):
            user_input[k] = st.number_input(f['label'], min_value=float(f['min']), max_value=float(f['max']), value=float(f['default']), step=float(f['step']), key=f"a_{k}")
        else:
            user_input[k] = st.number_input(f['label'], min_value=int(f['min']), max_value=int(f['max']), value=int(f['default']), step=int(f['step']), key=f"a_{k}")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">💰 Tambahan <span class="badge">3</span></div>
    """, unsafe_allow_html=True)
    for k in ['total_spent', 'is_premium_user', 'delivery_delay_days']:
        f = FEATURES[k]
        if isinstance(f['default'], float):
            user_input[k] = st.number_input(f['label'], min_value=float(f['min']), max_value=float(f['max']), value=float(f['default']), step=float(f['step']), key=f"b_{k}")
        else:
            user_input[k] = st.number_input(f['label'], min_value=int(f['min']), max_value=int(f['max']), value=int(f['default']), step=int(f['step']), key=f"b_{k}")
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────
def preprocess_input(raw: dict) -> np.ndarray:
    row = {}
    for k, v in raw.items():
        row[k] = v
    ordered = []
    for f in top_feat:
        ordered.append(row.get(f, 0))
    df = pd.DataFrame([ordered], columns=top_feat)
    return scaler.transform(df)

# ─────────────────────────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────────────────────────
st.divider()
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    btn = st.button("🚀 Prediksi Churn", use_container_width=True)

if btn:
    with st.spinner("⏳ Memproses..."):
        try:
            X = preprocess_input(user_input)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            pc = proba[1] * 100
            pn = proba[0] * 100

            st.divider()
            st.markdown("### 📋 Hasil Prediksi")

            c1, c2 = st.columns([1, 1])

            with c1:
                if pred == 1:
                    st.markdown("""
                    <div class="result-churn">
                        ⚠️ CHURN
                        <span class="sub">Pelanggan berpotensi meninggalkan layanan</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-ok">
                        ✅ TIDAK CHURN
                        <span class="sub">Pelanggan kemungkinan tetap bertahan</span>
                    </div>
                    """, unsafe_allow_html=True)

            with c2:
                st.markdown("#### Probabilitas")
                a, b = st.columns(2)
                with a:
                    st.metric("🔴 Churn", f"{pc:.1f}%")
                with b:
                    st.metric("🟢 Tidak Churn", f"{pn:.1f}%")
                st.progress(int(pc), text=f"Risiko Churn: {pc:.1f}%")

            # Rekomendasi
            st.markdown("### 💡 Rekomendasi")
            if pred == 1:
                recs = []
                if user_input.get('satisfaction_score', 10) < 6:
                    recs.append("📞 Hubungi pelanggan — skor kepuasan rendah")
                if user_input.get('support_tickets', 0) > 3:
                    recs.append("🔧 Selesaikan tiket support — terlalu banyak keluhan")
                if user_input.get('last_3_month_purchase_freq', 10) < 2:
                    recs.append("🛍️ Kirim penawaran khusus — pembelian rendah")
                if user_input.get('delivery_delay_days', 0) > 5:
                    recs.append("🚚 Perbaiki pengiriman — keterlambatan tinggi")
                if not recs:
                    recs.append("🔄 Jalankan program retensi")
                for r in recs:
                    st.warning(r)
            else:
                st.success("✅ Customer sehat. Pertahankan kualitas!")
                if user_input.get('satisfaction_score', 0) >= 9:
                    st.balloons()
                    st.info("⭐ Pelanggan sangat puas!")

            with st.expander("📄 Detail Data"):
                df = pd.DataFrame([user_input]).T.reset_index()
                df.columns = ['Fitur', 'Nilai']
                st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🚀 Churn Predictor Pro · 7 Fitur · 2024
</div>
""", unsafe_allow_html=True)
