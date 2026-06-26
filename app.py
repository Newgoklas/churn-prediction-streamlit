# =============================================================================
# app.py - Streamlit Deployment: Churn Prediction (7 FITUR)
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
# MODERN CSS (disingkat agar lebih ringkas)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #f0f4f8; }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 800; color: #ffffff; margin: 0; }
    .main-header p { font-size: 1rem; color: rgba(255,255,255,0.7); margin: 0.3rem 0 0 0; }
    .header-badge {
        display: inline-block; background: rgba(255,255,255,0.12);
        padding: 0.2rem 1rem; border-radius: 20px; font-size: 0.7rem;
        color: rgba(255,255,255,0.8); margin-top: 0.5rem;
        backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.06);
    }
    
    .card-section {
        background: #ffffff; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04); margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.85rem; font-weight: 600; color: #1a1a2e;
        margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
        text-transform: uppercase; letter-spacing: 0.3px;
    }
    .card-title .badge-count {
        background: #e8f0fe; color: #1a73e8; font-size: 0.6rem;
        padding: 0.1rem 0.6rem; border-radius: 12px; font-weight: 600;
        margin-left: auto;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #e8edf4 !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 3px rgba(26,115,232,0.1) !important;
    }
    .stNumberInput label { font-weight: 500 !important; font-size: 0.8rem !important; color: #4a4a6a !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: white !important; border-radius: 12px !important;
        padding: 0.85rem 2rem !important; font-size: 1.1rem !important;
        font-weight: 600 !important; border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(26,26,46,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(26,26,46,0.35) !important;
    }
    
    .result-churn {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; padding: 2rem; border-radius: 16px;
        text-align: center; font-size: 1.8rem; font-weight: 700;
        box-shadow: 0 8px 30px rgba(238,90,36,0.35);
    }
    .result-churn .sub { font-size: 0.9rem; font-weight: 400; opacity: 0.85; display: block; margin-top: 0.3rem; }
    
    .result-ok {
        background: linear-gradient(135deg, #00b894, #00a86b);
        color: white; padding: 2rem; border-radius: 16px;
        text-align: center; font-size: 1.8rem; font-weight: 700;
        box-shadow: 0 8px 30px rgba(0,184,148,0.35);
    }
    .result-ok .sub { font-size: 0.9rem; font-weight: 400; opacity: 0.85; display: block; margin-top: 0.3rem; }
    
    .metric-box {
        background: #f8fafc; border-radius: 12px; padding: 1.2rem;
        border-left: 4px solid #1a1a2e; border: 1px solid #eef2f6;
    }
    .metric-box .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; color: #8899aa; font-weight: 600; }
    .metric-box .value { font-size: 1.1rem; font-weight: 600; color: #1a1a2e; }
    
    .info-box {
        background: linear-gradient(135deg, #e8f4fd, #d6eaf8);
        border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 4px solid #1a73e8; margin-bottom: 1.2rem;
        font-size: 0.9rem; color: #1a3a5a;
        border: 1px solid rgba(26,115,232,0.1);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #00b894, #00a86b) !important;
        border-radius: 20px !important; height: 10px !important;
    }
    .stProgress > div { background: #eef2f6 !important; border-radius: 20px !important; }
    
    .footer {
        text-align: center; font-size: 0.75rem; color: #8899aa;
        padding: 1.5rem 0 0.5rem 0; border-top: 1px solid #eef2f6;
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
        'scaler': MODEL_DIR / 'scaler.pkl',
        'scaler_top': MODEL_DIR / 'scaler_top.pkl',  # ← PAKAI INI
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
# PRIORITASKAN scaler_top (khusus untuk top_features)
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
    <h1>📊 Customer Churn Predictor</h1>
    <p>Prediksi churn customer dengan 7 fitur utama</p>
    <div class="header-badge">🎯 Akurasi {meta.get('test_accuracy', 0.8923):.1%} · Hanya 7 fitur</div>
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
        <div class="value">{meta.get('test_accuracy', 0.8923):.2%}</div>
        <div style="margin-top:0.5rem;"></div>
        <div class="label">F1-Score</div>
        <div class="value">{meta.get('test_f1', 0.8323):.2%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📌 7 Fitur Utama")
    for i, feat in enumerate(top_feat[:7], 1):
        st.markdown(f"{i}. `{feat}`")
    
    st.divider()
    st.markdown("""
    **📖 Panduan**
    1. Isi 7 data di bawah
    2. Klik **Prediksi Churn**
    3. Lihat hasil
    """)

# ─────────────────────────────────────────────────────────────
# FORM INPUT - HANYA 7 FITUR
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 Isi 7 data berikut untuk memprediksi churn. Hanya 1 menit!</div>', unsafe_allow_html=True)

# ── 7 FITUR TERPENTING ──
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
# PREPROCESS & PREDICT (FIXED)
# ─────────────────────────────────────────────────────────────
def preprocess_input(raw_input: dict) -> np.ndarray:
    """
    Preprocess input user untuk prediksi.
    Hanya menggunakan top_features yang sudah dipilih.
    """
    # Buat dictionary dengan semua nilai
    row = {}
    for key, val in raw_input.items():
        row[key] = val
    
    # Buat dataframe dengan urutan yang benar (sesuai top_feat)
    ordered = []
    for feat in top_feat:
        if feat in row:
            ordered.append(row[feat])
        else:
            # Jika fitur tidak ada, beri nilai 0
            ordered.append(0)
    
    # Buat dataframe dengan nama kolom yang benar
    df_input = pd.DataFrame([ordered], columns=top_feat)
    
    # Scale menggunakan scaler yang sudah dilatih
    # Gunakan scaler (yang seharusnya scaler_top)
    try:
        arr_scaled = scaler.transform(df_input)
    except Exception as e:
        st.error(f"Error saat scaling: {e}")
        st.write("Fitur yang dibutuhkan:", top_feat)
        st.write("Fitur yang diberikan:", list(row.keys()))
        raise e
    
    return arr_scaled

# ─────────────────────────────────────────────────────────────
# BUTTON PREDIKSI
# ─────────────────────────────────────────────────────────────
st.divider()

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔮 Prediksi Churn", use_container_width=True)

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
    UAS Data Science · Churn Prediction · 7 Fitur Utama · 2024
</div>
""", unsafe_allow_html=True)
