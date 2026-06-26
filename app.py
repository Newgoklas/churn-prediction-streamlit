# =============================================================================
# app.py - Streamlit Deployment: Churn Prediction (Numerik Detail)
# =============================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Churn Predictor | Sales & Marketing",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = {}
    model_dir = Path("models")
    
    try:
        artifacts['model'] = joblib.load(model_dir / "best_model.pkl")
        artifacts['scaler'] = joblib.load(model_dir / "scaler.pkl")
        artifacts['scaler_top'] = joblib.load(model_dir / "scaler_top.pkl")
        artifacts['label_encoders'] = joblib.load(model_dir / "label_encoders.pkl")
        artifacts['top_features'] = joblib.load(model_dir / "top_features.pkl")
        artifacts['all_features'] = joblib.load(model_dir / "all_features.pkl")
        artifacts['metadata'] = joblib.load(model_dir / "model_metadata.pkl")
    except:
        artifacts['model'] = joblib.load("best_model.pkl")
        artifacts['scaler'] = joblib.load("scaler.pkl")
        artifacts['scaler_top'] = joblib.load("scaler_top.pkl")
        artifacts['label_encoders'] = joblib.load("label_encoders.pkl")
        artifacts['top_features'] = joblib.load("top_features.pkl")
        artifacts['all_features'] = joblib.load("all_features.pkl")
        artifacts['metadata'] = joblib.load("model_metadata.pkl")
    
    return artifacts

arts = load_artifacts()
model = arts['model']
scaler = arts['scaler']
le_map = arts.get('label_encoders', {})
top_feat = arts.get('top_features', [])
all_feat = arts.get('all_features', [])
meta = arts.get('metadata', {})

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f3a5f; text-align: center; }
    .sub-header { font-size: 1rem; color: #666; text-align: center; margin-bottom: 1.5rem; }
    .result-churn { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; font-size: 1.6rem; font-weight: 700; box-shadow: 0 4px 15px rgba(238,90,36,0.4); }
    .result-ok { background: linear-gradient(135deg, #6ab04c, #2ecc71); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; font-size: 1.6rem; font-weight: 700; box-shadow: 0 4px 15px rgba(46,204,113,0.4); }
    .metric-box { background: #f8f9fa; border-radius: 10px; padding: 1rem; border-left: 4px solid #1f3a5f; }
    .stButton > button { width: 100%; background: #1f3a5f; color: white; border-radius: 8px; padding: 0.7rem; font-size: 1.1rem; font-weight: 600; border: none; transition: 0.3s; }
    .stButton > button:hover { background: #2c5f8a; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
    .info-box { background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem; border-left: 4px solid #3498db; margin-bottom: 1rem; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #1f3a5f; margin-top: 0.5rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sales & Marketing Dataset | UAS Data Science</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    model_name = meta.get('best_model_name', 'Voting Ensemble') if meta else 'Voting Ensemble'
    test_acc = meta.get('test_accuracy', 0.8923) if meta else 0.8923
    test_f1 = meta.get('test_f1', 0.8323) if meta else 0.8323
    
    st.markdown(f"""
    <div class="metric-box">
        <b>Model:</b> {model_name}<br>
        <b>Test Accuracy:</b> {test_acc:.4f}<br>
        <b>Test F1-Score:</b> {test_f1:.4f}<br>
        <b>Fitur:</b> Top {len(top_feat)}
    </div>
    """, unsafe_allow_html=True)

    if top_feat:
        st.subheader("📌 Fitur Terpenting")
        for i, feat in enumerate(top_feat[:10], 1):
            st.write(f"{i}. `{feat}`")

    st.divider()
    st.markdown("**Cara Penggunaan:**")
    st.markdown("""
    1. Isi semua field di form utama
    2. Klik tombol **Prediksi**
    3. Lihat hasil prediksi & probabilitas
    """)

# ─────────────────────────────────────────────────────────────
# Form Input
# ─────────────────────────────────────────────────────────────
st.subheader("🔢 Input Data Customer")
st.markdown('<div class="info-box">Isi data customer di bawah ini untuk memprediksi kemungkinan churn.</div>', unsafe_allow_html=True)

# ── DEFINISI FITUR (DETAIL) ─────────────────────────────────
FEATURE_CONFIG = {
    # ── DATA DEMOGRAFIS ──────────────────────────────────────
    'age': {'type': 'number', 'min': 18, 'max': 80, 'default': 35, 'label': '🧑 Usia (tahun)', 'step': 1, 'section': 'demografi'},
    'gender': {'type': 'select', 'options': ['Male', 'Female', 'Other'], 'default': 'Female', 'label': '👤 Gender', 'section': 'demografi'},
    'country': {'type': 'select', 'options': ['USA', 'UK', 'Germany', 'France', 'India', 'Australia', 'Canada', 'Brazil'], 'default': 'Brazil', 'label': '🌍 Negara', 'section': 'demografi'},
    'city': {'type': 'select', 'options': ['New York', 'London', 'Berlin', 'Paris', 'Mumbai', 'Sydney', 'Toronto', 'São Paulo'], 'default': 'Mumbai', 'label': '🏙️ Kota', 'section': 'demografi'},
    
    # ── AKTIVITAS PENGGUNAAN ────────────────────────────────
    'total_visits': {'type': 'number', 'min': 0, 'max': 500, 'default': 50, 'label': '👀 Total Kunjungan', 'step': 1, 'section': 'aktivitas'},
    'avg_session_time': {'type': 'float', 'min': 0.0, 'max': 60.0, 'default': 5.0, 'label': '⏱️ Rata-rata Sesi (menit)', 'step': 0.1, 'section': 'aktivitas'},
    'pages_per_session': {'type': 'number', 'min': 1, 'max': 50, 'default': 5, 'label': '📄 Halaman per Sesi', 'step': 1, 'section': 'aktivitas'},
    'device_type': {'type': 'select', 'options': ['Mobile', 'Desktop', 'Tablet'], 'default': 'Mobile', 'label': '📱 Tipe Perangkat', 'section': 'aktivitas'},
    'acquisition_channel': {'type': 'select', 'options': ['Organic', 'Paid', 'Referral', 'Social', 'Email'], 'default': 'Referral', 'label': '📢 Channel Akuisisi', 'section': 'aktivitas'},
    
    # ── EMAIL & PEMBELIAN ────────────────────────────────────
    'email_open_rate': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.30, 'label': '📧 Email Open Rate', 'step': 0.01, 'section': 'email'},
    'email_click_rate': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.10, 'label': '🖱️ Email Click Rate', 'step': 0.01, 'section': 'email'},
    'total_spent': {'type': 'float', 'min': 0.0, 'max': 10000.0, 'default': 500.0, 'label': '💰 Total Pengeluaran ($)', 'step': 1.0, 'section': 'pembelian'},
    'avg_order_value': {'type': 'float', 'min': 0.0, 'max': 2000.0, 'default': 100.0, 'label': '🛒 Rata-rata Nilai Order ($)', 'step': 1.0, 'section': 'pembelian'},
    'discount_used': {'type': 'number', 'min': 0, 'max': 50, 'default': 3, 'label': '🏷️ Jumlah Diskon Dipakai', 'step': 1, 'section': 'pembelian'},
    'last_3_month_purchase_freq': {'type': 'number', 'min': 0, 'max': 30, 'default': 3, 'label': '🛍️ Frekuensi Pembelian 3 Bulan', 'step': 1, 'section': 'pembelian'},
    
    # ── DUKUNGAN & KEPUASAN ──────────────────────────────────
    'support_tickets': {'type': 'number', 'min': 0, 'max': 20, 'default': 1, 'label': '🎫 Tiket Support', 'step': 1, 'section': 'dukungan'},
    'refund_requested': {'type': 'number', 'min': 0, 'max': 10, 'default': 0, 'label': '↩️ Refund Diminta', 'step': 1, 'section': 'dukungan'},
    'delivery_delay_days': {'type': 'number', 'min': 0, 'max': 30, 'default': 6, 'label': '📦 Keterlambatan Pengiriman (hari)', 'step': 1, 'section': 'dukungan'},
    'satisfaction_score': {'type': 'number', 'min': 1, 'max': 10, 'default': 7, 'label': '⭐ Skor Kepuasan (1-10)', 'step': 1, 'section': 'dukungan'},
    'nps_score': {'type': 'number', 'min': -100, 'max': 100, 'default': 30, 'label': '📊 NPS Score (-100 - 100)', 'step': 1, 'section': 'dukungan'},
    
    # ── KEUANGAN & LANGANAN ──────────────────────────────────
    'marketing_spend_per_user': {'type': 'float', 'min': 0.0, 'max': 500.0, 'default': 30.0, 'label': '📢 Marketing Spend per User ($)', 'step': 0.5, 'section': 'keuangan'},
    'lifetime_value': {'type': 'float', 'min': 0.0, 'max': 20000.0, 'default': 1500.0, 'label': '💎 Lifetime Value ($)', 'step': 10.0, 'section': 'keuangan'},
    'is_premium_user': {'type': 'number', 'min': 0, 'max': 1, 'default': 1, 'label': '👑 Premium User (0=Tidak, 1=Ya)', 'step': 1, 'section': 'keuangan'},
    'subscription_type': {'type': 'select', 'options': ['Basic', 'Standard', 'Premium'], 'default': 'Standard', 'label': '📋 Tipe Langganan', 'section': 'keuangan'},
    'payment_method': {'type': 'select', 'options': ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer', 'Crypto'], 'default': 'Credit Card', 'label': '💳 Metode Pembayaran', 'section': 'keuangan'},
}

# ── Render Form dengan Section ──────────────────────────────
user_input = {}

# Kategorikan berdasarkan section
sections = {}
for key, cfg in FEATURE_CONFIG.items():
    section = cfg.get('section', 'lainnya')
    if section not in sections:
        sections[section] = []
    sections[section].append(key)

# Tampilkan dalam 2 kolom
col_left, col_right = st.columns(2)

section_colors = {
    'demografi': '🧑‍💼 Data Demografis',
    'aktivitas': '📱 Aktivitas Penggunaan',
    'email': '📧 Email & Komunikasi',
    'pembelian': '🛒 Pembelian & Diskon',
    'dukungan': '🎯 Dukungan & Kepuasan',
    'keuangan': '💰 Keuangan & Langganan'
}

# Tentukan pembagian section ke kiri dan kanan
left_sections = ['demografi', 'aktivitas', 'email']
right_sections = ['pembelian', 'dukungan', 'keuangan']

with col_left:
    for section in left_sections:
        if section in sections:
            st.markdown(f"<div class='section-title'>{section_colors.get(section, section)}</div>", unsafe_allow_html=True)
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
            st.markdown("---")

with col_right:
    for section in right_sections:
        if section in sections:
            st.markdown(f"<div class='section-title'>{section_colors.get(section, section)}</div>", unsafe_allow_html=True)
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
            st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Fungsi Prediksi
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
# Tombol Prediksi
# ─────────────────────────────────────────────────────────────
st.divider()
col_btn, col_empty = st.columns([1, 2])
with col_btn:
    predict_btn = st.button("🔮 Prediksi Churn", use_container_width=True)

if predict_btn:
    with st.spinner("Memproses prediksi..."):
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
            st.subheader("📋 Hasil Prediksi")
            col_res, col_prob = st.columns([1, 1])

            with col_res:
                if prediction == 1:
                    st.markdown(
                        '<div class="result-churn">⚠️ CHURN<br>'
                        '<span style="font-size:0.9rem;font-weight:400;">'
                        'Pelanggan berpotensi meninggalkan layanan</span></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-ok">✅ TIDAK CHURN<br>'
                        '<span style="font-size:0.9rem;font-weight:400;">'
                        'Pelanggan kemungkinan tetap bertahan</span></div>',
                        unsafe_allow_html=True
                    )

            with col_prob:
                st.markdown("**Probabilitas:**")
                st.metric("🔴 Probabilitas Churn", f"{prob_churn:.2f}%")
                st.metric("🟢 Probabilitas Tidak Churn", f"{prob_no_churn:.2f}%")
                st.progress(int(prob_churn), text=f"Risiko Churn: {prob_churn:.1f}%")

            # Rekomendasi
            st.subheader("💡 Rekomendasi")
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
                    st.info("⭐ Pelanggan sangat puas! Pertahankan kualitas ini.")

            with st.expander("📄 Detail Data Input yang Diproses"):
                df_display = pd.DataFrame([user_input]).T.reset_index()
                df_display.columns = ['Fitur', 'Nilai']
                st.dataframe(df_display, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
            st.exception(e)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>UAS Data Science | Sales & Marketing Churn Prediction | 2024</small></center>",
    unsafe_allow_html=True
)