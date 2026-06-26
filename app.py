# =============================================================================
# app.py - Churn Prediction PREMIUM (Cerah & Lengkap)
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
# MODERN CSS - CERAH & BERSIH
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf5 100%);
    }
    
    /* ── HEADER ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(102,126,234,0.3);
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
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
        animation: float 8s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(30px, -20px); }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1rem;
        color: rgba(255,255,255,0.8);
        margin: 0.3rem 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-size: 0.75rem;
        color: #ffffff;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .header-badge span {
        font-weight: 700;
        color: #ffd700;
    }
    
    /* ── CARDS ── */
    .card-section {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #eef2f6;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .card-section:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
    }
    
    .card-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .card-title .badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-size: 0.55rem;
        padding: 0.1rem 0.7rem;
        border-radius: 20px;
        font-weight: 600;
        margin-left: auto;
    }
    
    .card-title .emoji {
        font-size: 1.2rem;
    }
    
    /* ── INPUTS ── */
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #e8edf4 !important;
        background: #fafbfc !important;
        transition: all 0.3s ease !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        color: #1a1a2e !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102,126,234,0.1) !important;
        background: #ffffff !important;
    }
    
    .stNumberInput label {
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        color: #4a4a6a !important;
    }
    
    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 30px rgba(102,126,234,0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(102,126,234,0.4) !important;
    }
    
    /* ── SAMPLE BUTTONS ── */
    .sample-btn-container {
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    
    /* ── RESULT ── */
    .result-churn {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 8px 30px rgba(238,90,36,0.3);
        border: 1px solid rgba(255,255,255,0.1);
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
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 8px 30px rgba(0,184,148,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .result-ok .sub {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.85;
        display: block;
        margin-top: 0.3rem;
    }
    
    /* ── METRIC ── */
    .metric-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #eef2f6;
        text-align: center;
    }
    
    .metric-box .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        color: #8899aa;
        font-weight: 600;
    }
    
    .metric-box .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 0.1rem;
    }
    
    .metric-box .value.blue { color: #667eea; }
    .metric-box .value.pink { color: #e83e8c; }
    .metric-box .value.green { color: #00b894; }
    
    /* ── INFO BOX ── */
    .info-box {
        background: linear-gradient(135deg, #e8f4fd, #d6eaf8);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1.2rem;
        font-size: 0.9rem;
        color: #1a3a5a;
    }
    
    .info-box strong {
        color: #667eea;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 20px !important;
        height: 10px !important;
    }
    
    .stProgress > div {
        background: #eef2f6 !important;
        border-radius: 20px !important;
    }
    
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1a1a2e !important;
        background: #f8fafc !important;
        border-radius: 10px !important;
        border: 1px solid #eef2f6 !important;
    }
    
    .streamlit-expanderContent {
        background: #ffffff !important;
        border-radius: 0 0 10px 10px !important;
    }
    
    .footer {
        text-align: center;
        font-size: 0.7rem;
        color: #8899aa;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid #eef2f6;
        margin-top: 1.5rem;
    }
    
    .css-1d391kg {
        background: #ffffff !important;
        border-right: 1px solid #eef2f6 !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: #1a1a2e;
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
    <h1>🚀 Customer Churn Predictor</h1>
    <p>Prediksi churn customer dengan machine learning</p>
    <div class="header-badge">⚡ Akurasi <span>{meta.get('test_accuracy', 0.8923):.1%}</span> · F1 <span>{meta.get('test_f1', 0.8323):.3f}</span></div>
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
        <div style="margin-top:0.5rem;"></div>
        <div class="label">Akurasi</div>
        <div class="value blue">{meta.get('test_accuracy', 0.8923):.2%}</div>
        <div style="margin-top:0.3rem;"></div>
        <div class="label">F1-Score</div>
        <div class="value pink">{meta.get('test_f1', 0.8323):.2%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📌 7 Fitur Utama")
    emojis = ['⭐', '🎫', '⏱️', '🛍️', '💰', '👑', '📦']
    for i, feat in enumerate(top_feat[:7], 1):
        st.markdown(f"{emojis[i-1]} `{feat.replace('_', ' ').title()}`")
    
    st.divider()
    st.caption("Isi 7 data → Klik Prediksi")

# ─────────────────────────────────────────────────────────────
# FORM - 7 FITUR
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 <strong>Isi 7 data berikut</strong> untuk memprediksi churn. Cukup 1 menit!</div>', unsafe_allow_html=True)

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
    if st.button("⚠️ Data CHURN", use_container_width=True):
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
    if st.button("✅ Data TIDAK CHURN", use_container_width=True):
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
    'satisfaction_score': {
        'min': 1, 'max': 10, 'default': 7,
        'label': '⭐ Skor Kepuasan (1-10)',
        'step': 1,
        'help': 'Semakin tinggi, semakin puas'
    },
    'support_tickets': {
        'min': 0, 'max': 20, 'default': 1,
        'label': '🎫 Tiket Support',
        'step': 1,
        'help': 'Jumlah tiket yang diajukan'
    },
    'avg_session_time': {
        'min': 0.0, 'max': 60.0, 'default': 5.0,
        'label': '⏱️ Rata-rata Sesi (menit)',
        'step': 0.1,
        'help': 'Rata-rata waktu per sesi'
    },
    'last_3_month_purchase_freq': {
        'min': 0, 'max': 30, 'default': 3,
        'label': '🛍️ Frekuensi Pembelian 3 Bulan',
        'step': 1,
        'help': 'Jumlah pembelian 3 bulan terakhir'
    },
    'total_spent': {
        'min': 0.0, 'max': 10000.0, 'default': 500.0,
        'label': '💰 Total Pengeluaran ($)',
        'step': 10.0,
        'help': 'Total uang yang dikeluarkan'
    },
    'is_premium_user': {
        'min': 0, 'max': 1, 'default': 1,
        'label': '👑 Premium User (0=Tidak, 1=Ya)',
        'step': 1,
        'help': 'Status premium pelanggan'
    },
    'delivery_delay_days': {
        'min': 0, 'max': 30, 'default': 6,
        'label': '📦 Keterlambatan Kirim (hari)',
        'step': 1,
        'help': 'Rata-rata keterlambatan pengiriman'
    },
}

user_input = {}

# Apply sample
if 'sample' in st.session_state:
    for k, v in st.session_state['sample'].items():
        if k in FEATURES:
            FEATURES[k]['default'] = v

# ── RENDER 2 KOLOM ──
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="card-section">
        <div class="card-title">
            <span class="emoji">📊</span> Fitur Utama
            <span class="badge">4</span>
        </div>
    """, unsafe_allow_html=True)
    
    for k in ['satisfaction_score', 'support_tickets', 'avg_session_time', 'last_3_month_purchase_freq']:
        f = FEATURES[k]
        if isinstance(f['default'], float):
            user_input[k] = st.number_input(
                f['label'],
                min_value=float(f['min']),
                max_value=float(f['max']),
                value=float(f['default']),
                step=float(f['step']),
                help=f.get('help', ''),
                key=f"a_{k}"
            )
        else:
            user_input[k] = st.number_input(
                f['label'],
                min_value=int(f['min']),
                max_value=int(f['max']),
                value=int(f['default']),
                step=int(f['step']),
                help=f.get('help', ''),
                key=f"a_{k}"
            )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card-section">
        <div class="card-title">
            <span class="emoji">💰</span> Fitur Tambahan
            <span class="badge">3</span>
        </div>
    """, unsafe_allow_html=True)
    
    for k in ['total_spent', 'is_premium_user', 'delivery_delay_days']:
        f = FEATURES[k]
        if isinstance(f['default'], float):
            user_input[k] = st.number_input(
                f['label'],
                min_value=float(f['min']),
                max_value=float(f['max']),
                value=float(f['default']),
                step=float(f['step']),
                help=f.get('help', ''),
                key=f"b_{k}"
            )
        else:
            user_input[k] = st.number_input(
                f['label'],
                min_value=int(f['min']),
                max_value=int(f['max']),
                value=int(f['default']),
                step=int(f['step']),
                help=f.get('help', ''),
                key=f"b_{k}"
            )
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

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🚀 Prediksi Churn", use_container_width=True)

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
            
            st.markdown("### 📋 Hasil Prediksi")
            
            col_res, col_prob = st.columns([1, 1])
            
            with col_res:
                if prediction == 1:
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
    🚀 Churn Predictor Pro · 7 Fitur Utama · 2024
</div>
""", unsafe_allow_html=True)
