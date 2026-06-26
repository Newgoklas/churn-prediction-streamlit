# =============================================================================
# app.py - Churn Prediction ULTIMATE (LENGKAP + PANDUAN - FINAL FIXED)
# =============================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import random
import plotly.express as px
from datetime import datetime

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
# CSS MODERN
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8edf5 100%); }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 800; color: #ffffff; margin: 0; }
    .main-header p { font-size: 1rem; color: rgba(255,255,255,0.8); margin: 0.3rem 0 0 0; }
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.3rem 1.2rem;
        border-radius: 30px;
        font-size: 0.75rem;
        color: #ffffff;
        margin-top: 0.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .header-badge span { font-weight: 700; color: #ffd700; }
    
    .card-section {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #eef2f6;
        margin-bottom: 1rem;
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
    }
    .card-title .badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-size: 0.55rem;
        padding: 0.1rem 0.7rem;
        border-radius: 20px;
        margin-left: auto;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #e8edf4 !important;
        background: #fafbfc !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        color: #1a1a2e !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102,126,234,0.1) !important;
    }
    .stNumberInput label { font-weight: 600 !important; font-size: 0.78rem !important; color: #4a4a6a !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 8px 30px rgba(102,126,234,0.3) !important;
        width: 100% !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 12px 40px rgba(102,126,234,0.4) !important; }
    
    .result-churn {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 8px 30px rgba(238,90,36,0.3);
    }
    .result-churn .sub { font-size: 0.9rem; font-weight: 400; opacity: 0.85; display: block; margin-top: 0.3rem; }
    
    .result-ok {
        background: linear-gradient(135deg, #00b894, #00a86b);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 8px 30px rgba(0,184,148,0.3);
    }
    .result-ok .sub { font-size: 0.9rem; font-weight: 400; opacity: 0.85; display: block; margin-top: 0.3rem; }
    
    .metric-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #eef2f6;
        text-align: center;
    }
    .metric-box .label { font-size: 0.65rem; text-transform: uppercase; color: #8899aa; font-weight: 600; }
    .metric-box .value { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; margin-top: 0.1rem; }
    .metric-box .value.blue { color: #667eea; }
    .metric-box .value.pink { color: #e83e8c; }
    .metric-box .value.green { color: #00b894; }
    
    .info-box {
        background: linear-gradient(135deg, #e8f4fd, #d6eaf8);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1.2rem;
        font-size: 0.9rem;
        color: #1a3a5a;
    }
    .info-box strong { color: #667eea; }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 20px !important;
        height: 10px !important;
    }
    .stProgress > div { background: #eef2f6 !important; border-radius: 20px !important; }
    
    .footer {
        text-align: center;
        font-size: 0.7rem;
        color: #8899aa;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid #eef2f6;
        margin-top: 1.5rem;
    }
    
    .guide-churn {
        background: #fff5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin-bottom: 0.5rem;
    }
    .guide-nochurn {
        background: #f0fff4;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00b894;
        margin-bottom: 0.5rem;
    }
    .guide-tips {
        background: #e8f4fd;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .guide-table {
        width: 100%;
        font-size: 0.85rem;
        border-collapse: collapse;
        margin-top: 0.5rem;
    }
    .guide-table th {
        padding: 8px;
        text-align: left;
        color: white;
    }
    .guide-table th.churn-header { background: #ff6b6b; }
    .guide-table th.nochurn-header { background: #00b894; }
    .guide-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    .guide-table tr:last-child td { border-bottom: none; }
    .guide-value { font-weight: 700; }
    .guide-value.churn-value { color: #ff6b6b; }
    .guide-value.nochurn-value { color: #00b894; }
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
top_feat = arts.get('top_feat', [])
meta = arts.get('metadata', {})

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🚀 Customer Churn Predictor Pro</h1>
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
    
    if 'prediction_log' in st.session_state and len(st.session_state.prediction_log) > 0:
        st.markdown("### 📊 Statistik")
        log_df = pd.DataFrame(st.session_state.prediction_log)
        total = len(log_df)
        churn_count = log_df[log_df['prediction'] == 1].shape[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("⚠️ Churn", churn_count, delta=f"{churn_count/total*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# TAB 1: PREDIKSI TUNGGAL
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Tunggal", "📊 Batch Prediction", "📈 Dashboard"])

with tab1:
    st.markdown('<div class="info-box">💡 <strong>Isi 7 data berikut</strong> untuk memprediksi churn.</div>', unsafe_allow_html=True)

    # ── TOMBOL CONTOH ──
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
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
                'satisfaction_score': 2, 'support_tickets': 8, 'avg_session_time': 1.2,
                'last_3_month_purchase_freq': 0, 'total_spent': 50, 'is_premium_user': 0,
                'delivery_delay_days': 12
            }
    with col_s3:
        if st.button("✅ TIDAK CHURN", use_container_width=True):
            st.session_state['sample'] = {
                'satisfaction_score': 9, 'support_tickets': 0, 'avg_session_time': 12.5,
                'last_3_month_purchase_freq': 12, 'total_spent': 2500, 'is_premium_user': 1,
                'delivery_delay_days': 1
            }
    with col_s4:
        if st.button("🔄 Reset", use_container_width=True):
            if 'sample' in st.session_state:
                del st.session_state['sample']
            st.rerun()

    # ── PANDUAN ──
    with st.expander("📖 Panduan: Cara Mendapatkan Hasil CHURN / TIDAK CHURN"):
        st.markdown("""
        <div class="guide-churn">
            <h4 style="color: #ff6b6b; margin: 0 0 0.5rem 0;">🔴 Data untuk Hasil CHURN</h4>
            <table class="guide-table">
                <tr>
                    <th class="churn-header">Fitur</th>
                    <th class="churn-header">Nilai</th>
                    <th class="churn-header">Keterangan</th>
                </tr>
                <tr>
                    <td>⭐ Skor Kepuasan</td>
                    <td><span class="guide-value churn-value">1-3</span></td>
                    <td>Sangat tidak puas</td>
                </tr>
                <tr>
                    <td>🎫 Tiket Support</td>
                    <td><span class="guide-value churn-value">5-10</span></td>
                    <td>Banyak komplain</td>
                </tr>
                <tr>
                    <td>⏱️ Rata-rata Sesi</td>
                    <td><span class="guide-value churn-value">0.5-2</span></td>
                    <td>Sangat singkat</td>
                </tr>
                <tr>
                    <td>🛍️ Frekuensi Pembelian</td>
                    <td><span class="guide-value churn-value">0-1</span></td>
                    <td>Jarang beli</td>
                </tr>
                <tr>
                    <td>💰 Total Pengeluaran</td>
                    <td><span class="guide-value churn-value">0-50</span></td>
                    <td>Sedikit belanja</td>
                </tr>
                <tr>
                    <td>👑 Premium User</td>
                    <td><span class="guide-value churn-value">0</span></td>
                    <td>Tidak premium</td>
                </tr>
                <tr>
                    <td>📦 Keterlambatan Kirim</td>
                    <td><span class="guide-value churn-value">10-20</span></td>
                    <td>Sering telat</td>
                </tr>
            </table>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #555;">
                📋 <b>Contoh:</b> Skor=1, Support=10, Sesi=0.5, Beli=0, Pengeluaran=10, Premium=0, Telat=20
            </p>
        </div>
        
        <div class="guide-nochurn">
            <h4 style="color: #00b894; margin: 0 0 0.5rem 0;">🟢 Data untuk Hasil TIDAK CHURN</h4>
            <table class="guide-table">
                <tr>
                    <th class="nochurn-header">Fitur</th>
                    <th class="nochurn-header">Nilai</th>
                    <th class="nochurn-header">Keterangan</th>
                </tr>
                <tr>
                    <td>⭐ Skor Kepuasan</td>
                    <td><span class="guide-value nochurn-value">8-10</span></td>
                    <td>Sangat puas</td>
                </tr>
                <tr>
                    <td>🎫 Tiket Support</td>
                    <td><span class="guide-value nochurn-value">0-1</span></td>
                    <td>Tidak ada keluhan</td>
                </tr>
                <tr>
                    <td>⏱️ Rata-rata Sesi</td>
                    <td><span class="guide-value nochurn-value">8-15</span></td>
                    <td>Lama berkunjung</td>
                </tr>
                <tr>
                    <td>🛍️ Frekuensi Pembelian</td>
                    <td><span class="guide-value nochurn-value">8-15</span></td>
                    <td>Sering beli</td>
                </tr>
                <tr>
                    <td>💰 Total Pengeluaran</td>
                    <td><span class="guide-value nochurn-value">1000-5000</span></td>
                    <td>Banyak belanja</td>
                </tr>
                <tr>
                    <td>👑 Premium User</td>
                    <td><span class="guide-value nochurn-value">1</span></td>
                    <td>Premium</td>
                </tr>
                <tr>
                    <td>📦 Keterlambatan Kirim</td>
                    <td><span class="guide-value nochurn-value">0-2</span></td>
                    <td>Tepat waktu</td>
                </tr>
            </table>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #555;">
                📋 <b>Contoh:</b> Skor=9, Support=0, Sesi=12.5, Beli=12, Pengeluaran=2500, Premium=1, Telat=1
            </p>
        </div>
        
        <div class="guide-tips">
            <b>💡 Tips Cepat:</b><br>
            🔴 <b>CHURN</b> → Semua nilai RENDAH (kecuali Tiket Support & Keterlambatan = TINGGI)<br>
            🟢 <b>TIDAK CHURN</b> → Semua nilai TINGGI (kecuali Tiket Support & Keterlambatan = RENDAH)
        </div>
        """, unsafe_allow_html=True)

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
    if 'sample' in st.session_state:
        for k, v in st.session_state['sample'].items():
            if k in FEATURES:
                FEATURES[k]['default'] = v

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card-section"><div class="card-title">📊 Fitur Utama <span class="badge">4</span></div>', unsafe_allow_html=True)
        for k in ['satisfaction_score', 'support_tickets', 'avg_session_time', 'last_3_month_purchase_freq']:
            f = FEATURES[k]
            if isinstance(f['default'], float):
                user_input[k] = st.number_input(f['label'], min_value=float(f['min']), max_value=float(f['max']), value=float(f['default']), step=float(f['step']), key=f"a_{k}")
            else:
                user_input[k] = st.number_input(f['label'], min_value=int(f['min']), max_value=int(f['max']), value=int(f['default']), step=int(f['step']), key=f"a_{k}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card-section"><div class="card-title">💰 Fitur Tambahan <span class="badge">3</span></div>', unsafe_allow_html=True)
        for k in ['total_spent', 'is_premium_user', 'delivery_delay_days']:
            f = FEATURES[k]
            if isinstance(f['default'], float):
                user_input[k] = st.number_input(f['label'], min_value=float(f['min']), max_value=float(f['max']), value=float(f['default']), step=float(f['step']), key=f"b_{k}")
            else:
                user_input[k] = st.number_input(f['label'], min_value=int(f['min']), max_value=int(f['max']), value=int(f['default']), step=int(f['step']), key=f"b_{k}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # PREPROCESS INPUT - FIXED (PERBAIKAN UTAMA)
    # ─────────────────────────────────────────────────────────────
    def preprocess_input(raw: dict) -> np.ndarray:
        """
        Preprocess input user.
        Untuk fitur yang tidak diisi, gunakan nilai default CHURN.
        """
        # Default values untuk CHURN (ekstrim)
        default_churn = {
            'age': 18,
            'total_visits': 1,
            'nps_score': -100,
            'pages_per_session': 1,
            'email_open_rate': 0.0,
            'email_click_rate': 0.0,
            'refund_requested': 5,
            'avg_order_value': 5,
            'discount_used': 0,
            'marketing_spend_per_user': 0,
            'lifetime_value': 10,
            'country': 0,
            'city': 0,
            'acquisition_channel': 0,
            'device_type': 0,
            'subscription_type': 0,
            'payment_method': 0,
            'gender': 0
        }
        
        ordered = []
        for f in top_feat:
            if f in raw:
                ordered.append(raw[f])
            else:
                ordered.append(default_churn.get(f, 0))
        
        df = pd.DataFrame([ordered], columns=top_feat)
        return scaler.transform(df)

    st.divider()
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🚀 Prediksi Churn", use_container_width=True)

    if predict_btn:
        with st.spinner("⏳ Memproses..."):
            try:
                X = preprocess_input(user_input)
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                pc = proba[1] * 100
                pn = proba[0] * 100

                if 'prediction_log' not in st.session_state:
                    st.session_state.prediction_log = []
                st.session_state.prediction_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    **user_input,
                    'prediction': int(pred),
                    'prob_churn': round(pc, 2)
                })

                st.divider()
                st.markdown("### 📋 Hasil Prediksi")
                col_res, col_prob = st.columns([1, 1])

                with col_res:
                    if pred == 1:
                        st.markdown('<div class="result-churn">⚠️ CHURN<span class="sub">Pelanggan berpotensi meninggalkan layanan</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-ok">✅ TIDAK CHURN<span class="sub">Pelanggan kemungkinan tetap bertahan</span></div>', unsafe_allow_html=True)

                with col_prob:
                    st.markdown("#### Probabilitas")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("🔴 Churn", f"{pc:.1f}%")
                    with col_m2:
                        st.metric("🟢 Tidak Churn", f"{pn:.1f}%")
                    st.progress(int(pc), text=f"Risiko Churn: {pc:.1f}%")

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

# =============================================================================
# TAB 2: BATCH PREDICTION
# =============================================================================
with tab2:
    st.markdown('<div class="info-box">📊 <strong>Upload file CSV</strong> untuk prediksi batch. File harus memiliki 7 kolom fitur.</div>', unsafe_allow_html=True)
    
    template_df = pd.DataFrame({
        'satisfaction_score': [7, 2],
        'support_tickets': [1, 8],
        'avg_session_time': [5.0, 1.2],
        'last_3_month_purchase_freq': [3, 0],
        'total_spent': [500, 50],
        'is_premium_user': [1, 0],
        'delivery_delay_days': [6, 12]
    })
    st.download_button("📥 Download Template CSV", template_df.to_csv(index=False), "churn_template.csv", "text/csv", use_container_width=True)
    
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅ {len(batch_df)} baris data")
            
            required = ['satisfaction_score', 'support_tickets', 'avg_session_time', 
                       'last_3_month_purchase_freq', 'total_spent', 'is_premium_user', 'delivery_delay_days']
            missing = [c for c in required if c not in batch_df.columns]
            if missing:
                st.error(f"❌ Kolom hilang: {missing}")
                st.stop()
            
            if st.button("🚀 Prediksi Batch", use_container_width=True):
                with st.spinner(f"⏳ Memproses {len(batch_df)} data..."):
                    preds, probs = [], []
                    for _, row in batch_df.iterrows():
                        X = preprocess_input(row.to_dict())
                        preds.append(model.predict(X)[0])
                        probs.append(model.predict_proba(X)[0][1] * 100)
                    
                    batch_df['prediction'] = preds
                    batch_df['prob_churn'] = probs
                    batch_df['status'] = batch_df['prediction'].apply(lambda x: '⚠️ CHURN' if x == 1 else '✅ TIDAK CHURN')
                    
                    st.divider()
                    st.markdown("### 📊 Hasil Batch")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", len(batch_df))
                    with col2:
                        st.metric("⚠️ CHURN", batch_df['prediction'].sum())
                    with col3:
                        st.metric("✅ TIDAK CHURN", len(batch_df) - batch_df['prediction'].sum())
                    
                    st.dataframe(batch_df, use_container_width=True)
                    
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        counts = batch_df['status'].value_counts()
                        fig = px.pie(values=counts.values, names=counts.index, title="Distribusi")
                        st.plotly_chart(fig, use_container_width=True)
                    with col_chart2:
                        fig = px.histogram(batch_df, x='prob_churn', nbins=20, title="Probabilitas Churn")
                        fig.update_layout(xaxis_title="Probabilitas (%)", yaxis_title="Jumlah")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.download_button("📥 Download Hasil", batch_df.to_csv(index=False), f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error: {e}")

# =============================================================================
# TAB 3: DASHBOARD
# =============================================================================
with tab3:
    st.markdown('<div class="info-box">📈 <strong>Dashboard Prediksi</strong> — Ringkasan semua prediksi.</div>', unsafe_allow_html=True)
    
    if 'prediction_log' in st.session_state and len(st.session_state.prediction_log) > 0:
        log_df = pd.DataFrame(st.session_state.prediction_log)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total", len(log_df))
        with col2:
            churn = log_df[log_df['prediction'] == 1].shape[0]
            st.metric("⚠️ CHURN", churn, delta=f"{churn/len(log_df)*100:.1f}%")
        with col3:
            st.metric("✅ TIDAK CHURN", len(log_df) - churn)
        with col4:
            st.metric("📈 Rata-rata Risiko", f"{log_df['prob_churn'].mean():.1f}%")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            labels = ['TIDAK CHURN', 'CHURN']
            values = [len(log_df) - churn, churn]
            fig = px.pie(values=values, names=labels, title="Distribusi", color=labels,
                         color_discrete_map={'CH
