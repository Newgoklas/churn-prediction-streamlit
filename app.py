# =============================================================================
# app.py - Streamlit Deployment: Churn Prediction
# Jalankan: streamlit run app.py
# =============================================================================

import streamlit as st
import joblib  # ← GANTI: dari pickle ke joblib
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
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1f3a5f;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-churn {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: 700;
        box-shadow: 0 4px 15px rgba(238,90,36,0.4);
    }
    .result-ok {
        background: linear-gradient(135deg, #6ab04c, #2ecc71);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: 700;
        box-shadow: 0 4px 15px rgba(46,204,113,0.4);
    }
    .metric-box {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #1f3a5f;
    }
    .stButton > button {
        width: 100%; background: #1f3a5f; color: white;
        border-radius: 8px; padding: 0.7rem; font-size: 1.1rem;
        font-weight: 600; border: none; transition: 0.3s;
    }
    .stButton > button:hover {
        background: #2c5f8a; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .info-box {
        background: #e8f4fd; border-radius: 8px; padding: 0.8rem 1rem;
        border-left: 4px solid #3498db; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load Model & Artefak (Menggunakan joblib)
# ─────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    """Load semua file model menggunakan joblib (lebih kompatibel)"""
    artifacts = {}s
    files_needed = {
        'model'      : MODEL_DIR / 'best_model.pkl',
        'scaler'     : MODEL_DIR / 'scaler.pkl',
        'label_enc'  : MODEL_DIR / 'label_encoders.pkl',
        'top_feat'   : MODEL_DIR / 'top_features.pkl',
        'all_feat'   : MODEL_DIR / 'all_features.pkl',
        'metadata'   : MODEL_DIR / 'model_metadata.pkl',
    }
    missing = []
    corrupt = []
    
    for key, path in files_needed.items():
        if path.exists():
            try:
                # ← PERUBAHAN: pakai joblib.load, bukan pickle.load
                artifacts[key] = joblib.load(path)
            except Exception as e:
                corrupt.append(f"{path} ({str(e)[:80]})")
                artifacts[key] = None
        else:
            missing.append(str(path))

    # Jika ada file corrupt, tampilkan error detail
    if corrupt:
        st.error("⚠️ File model corrupt! Jalankan ulang `main.py` di local.")
        for c in corrupt:
            st.write(f"  ❌ {c}")
        st.stop()
    
    if missing:
        artifacts['_missing'] = missing
    return artifacts

arts = load_artifacts()

# ─────────────────────────────────────────────────────────────
# Cek apakah model sudah ada
# ─────────────────────────────────────────────────────────────
if '_missing' in arts:
    st.error("⚠️ Beberapa file model tidak ditemukan. Jalankan `python main.py` terlebih dahulu!")
    st.code("python main.py", language="bash")
    for f in arts['_missing']:
        st.write(f"  ❌ `{f}`")
    st.stop()

# Cek apakah model None (corrupt)
if arts.get('model') is None:
    st.error("❌ Model corrupt! Jalankan ulang `main.py`.")
    st.stop()

model    = arts['model']
scaler   = arts.get('scaler_top') or arts.get('scaler')
le_map   = arts.get('label_enc', {})
top_feat = arts.get('top_feat', [])
all_feat = arts.get('all_feat', [])
meta     = arts.get('metadata', {})

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sales & Marketing Dataset | UAS Data Science</div>',
            unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────
# Sidebar - Info Model
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    
    # Ambil informasi dari metadata
    model_name = meta.get('best_model_name', 'Voting Ensemble')
    test_acc = meta.get('test_accuracy', 0.8923)
    test_f1 = meta.get('test_f1', 0.8323)
    
    st.markdown(f"""
    <div class="metric-box">
        <b>Model:</b> {model_name}<br>
        <b>Test Accuracy:</b> {test_acc:.4f}<br>
        <b>Test F1-Score:</b> {test_f1:.4f}<br>
        <b>Fitur digunakan:</b> Top {len(top_feat)}<br>
    </div>
    """, unsafe_allow_html=True)

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
st.markdown('<div class="info-box">Isi data customer di bawah ini untuk memprediksi kemungkinan churn.</div>',
            unsafe_allow_html=True)

# Definisi semua fitur & value default yang realistis
FEATURE_CONFIG = {
    # Numerik
    'age'                       : {'type': 'number', 'min': 18, 'max': 80, 'default': 35,   'label': 'Usia', 'step': 1},
    'total_visits'              : {'type': 'number', 'min': 0,  'max': 500, 'default': 50,  'label': 'Total Kunjungan', 'step': 1},
    'avg_session_time'          : {'type': 'float',  'min': 0.0,'max': 60.0,'default': 5.0, 'label': 'Rata-rata Session (menit)', 'step': 0.1},
    'pages_per_session'         : {'type': 'number', 'min': 1,  'max': 50, 'default': 5,   'label': 'Halaman per Session', 'step': 1},
    'email_open_rate'           : {'type': 'float',  'min': 0.0,'max': 1.0,'default': 0.3, 'label': 'Email Open Rate (0–1)', 'step': 0.01},
    'email_click_rate'          : {'type': 'float',  'min': 0.0,'max': 1.0,'default': 0.1, 'label': 'Email Click Rate (0–1)', 'step': 0.01},
    'total_spent'               : {'type': 'float',  'min': 0.0,'max': 10000.0,'default': 500.0,'label': 'Total Pengeluaran ($)', 'step': 1.0},
    'avg_order_value'           : {'type': 'float',  'min': 0.0,'max': 2000.0,'default': 100.0,'label': 'Rata-rata Nilai Order ($)', 'step': 1.0},
    'discount_used'             : {'type': 'number', 'min': 0,  'max': 50, 'default': 3,   'label': 'Jumlah Diskon Dipakai', 'step': 1},
    'support_tickets'           : {'type': 'number', 'min': 0,  'max': 20, 'default': 1,   'label': 'Tiket Support', 'step': 1},
    'refund_requested'          : {'type': 'number', 'min': 0,  'max': 10, 'default': 0,   'label': 'Refund Diminta', 'step': 1},
    'delivery_delay_days'       : {'type': 'number', 'min': 0,  'max': 30, 'default': 2,   'label': 'Keterlambatan Pengiriman (hari)', 'step': 1},
    'satisfaction_score'        : {'type': 'number', 'min': 1,  'max': 10, 'default': 7,   'label': 'Skor Kepuasan (1–10)', 'step': 1},
    'nps_score'                 : {'type': 'number', 'min': -100,'max': 100,'default': 30, 'label': 'NPS Score (-100 – 100)', 'step': 1},
    'marketing_spend_per_user'  : {'type': 'float',  'min': 0.0,'max': 500.0,'default': 30.0,'label': 'Marketing Spend per User ($)', 'step': 0.5},
    'lifetime_value'            : {'type': 'float',  'min': 0.0,'max': 20000.0,'default': 1500.0,'label': 'Lifetime Value ($)', 'step': 10.0},
    'last_3_month_purchase_freq': {'type': 'number', 'min': 0,  'max': 30, 'default': 3,   'label': 'Frekuensi Pembelian 3 Bulan Terakhir', 'step': 1},
    'is_premium_user'           : {'type': 'number', 'min': 0,  'max': 1,  'default': 0,   'label': 'Premium User (0=Tidak, 1=Ya)', 'step': 1},
    # Kategorikal
    'gender'                    : {'type': 'select', 'options': ['Male', 'Female', 'Other'], 'default': 'Male', 'label': 'Gender'},
    'country'                   : {'type': 'select', 'options': ['USA','UK','Germany','France','India','Australia','Canada','Brazil'], 'default': 'USA', 'label': 'Negara'},
    'city'                      : {'type': 'select', 'options': ['New York','London','Berlin','Paris','Mumbai','Sydney','Toronto','São Paulo'], 'default': 'New York', 'label': 'Kota'},
    'acquisition_channel'       : {'type': 'select', 'options': ['Organic','Paid','Referral','Social','Email'], 'default': 'Organic', 'label': 'Channel Akuisisi'},
    'device_type'               : {'type': 'select', 'options': ['Mobile','Desktop','Tablet'], 'default': 'Mobile', 'label': 'Tipe Perangkat'},
    'subscription_type'         : {'type': 'select', 'options': ['Basic','Standard','Premium'], 'default': 'Standard', 'label': 'Tipe Langganan'},
    'payment_method'            : {'type': 'select', 'options': ['Credit Card','Debit Card','PayPal','Bank Transfer','Crypto'], 'default': 'Credit Card', 'label': 'Metode Pembayaran'},
}

# ── Render Form ───────────────────────────────────────────────
user_input = {}

col_left, col_right = st.columns(2)
num_keys = [k for k, v in FEATURE_CONFIG.items() if v['type'] in ('number', 'float')]
cat_keys = [k for k, v in FEATURE_CONFIG.items() if v['type'] == 'select']

with col_left:
    st.markdown("**📈 Data Numerik**")
    for key in num_keys:
        cfg = FEATURE_CONFIG[key]
        if cfg['type'] == 'float':
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

with col_right:
    st.markdown("**🏷️ Data Kategorikal**")
    for key in cat_keys:
        cfg = FEATURE_CONFIG[key]
        user_input[key] = st.selectbox(
            cfg['label'], options=cfg['options'],
            index=cfg['options'].index(cfg['default']), key=f"cat_{key}"
        )

# ─────────────────────────────────────────────────────────────
# Fungsi Prediksi
# ─────────────────────────────────────────────────────────────
def preprocess_input(raw_input: dict) -> np.ndarray:
    """
    Konversi input user → array yang siap masuk model.
    Proses: encode kategorikal → susun kolom sesuai all_features
    → ambil top_features → scale.
    """
    row = {}

    # Encode kategorikal dengan le_map
    for key, val in raw_input.items():
        if key in le_map:
            try:
                le = le_map[key]
                val_str = str(val)
                # Cek apakah value ada di classes
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

    # Susun sesuai all_features (fitur lengkap sebelum seleksi)
    ordered = []
    for feat in all_feat:
        if feat in row:
            ordered.append(row[feat])
        else:
            ordered.append(0)  # default jika tidak ada

    df_input = pd.DataFrame([ordered], columns=all_feat)

    # Ambil top_features
    available_top = [f for f in top_feat if f in df_input.columns]
    df_top = df_input[available_top]

    # Scale
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

            # Probabilitas (jika model support)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_input)[0]
                prob_churn    = proba[1] * 100
                prob_no_churn = proba[0] * 100
            else:
                prob_churn    = 100 if prediction == 1 else 0
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
                st.metric("🔴 Probabilitas Churn",    f"{prob_churn:.2f}%")
                st.metric("🟢 Probabilitas Tidak Churn", f"{prob_no_churn:.2f}%")

                # Progress bar
                st.progress(int(prob_churn),
                            text=f"Risiko Churn: {prob_churn:.1f}%")

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
                if not recs:
                    recs.append("🔄 **Jalankan program retensi** — kirim email personal dan tawarkan benefit eksklusif.")
                for r in recs:
                    st.warning(r)
            else:
                st.success("✅ Pelanggan dalam kondisi sehat. Pertahankan kualitas layanan dan lanjutkan program loyalitas.")
                if user_input.get('is_premium_user', 0) == 0:
                    st.info("💎 Pertimbangkan untuk menawarkan **upgrade ke Premium** kepada pelanggan ini.")

            # Detail input
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