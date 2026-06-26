# =============================================================================
# app.py - Streamlit Deployment: Churn Prediction
# =============================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# Tentukan folder model (Cek beberapa kemungkinan)
# ─────────────────────────────────────────────────────────────
def find_model_folder():
    """Cari folder model di beberapa lokasi"""
    possible_paths = [
        Path("models"),
        Path("model"),
        Path(".")  # root directory
    ]
    for path in possible_paths:
        if path.exists():
            # Cek apakah ada file .pkl di dalamnya
            pkl_files = list(path.glob("*.pkl"))
            if pkl_files:
                return path
    return Path("models")  # default

MODEL_DIR = find_model_folder()
st.sidebar.success(f"📁 Model folder: `{MODEL_DIR}`")

# ─────────────────────────────────────────────────────────────
# Load Model & Artefak
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = {}
    
    # Coba beberapa kemungkinan nama file
    possible_files = {
        'model': ['best_model.pkl', 'best_churn_model.pkl', 'model_churn_terbaik.pkl'],
        'scaler': ['scaler.pkl'],
        'scaler_top': ['scaler_top.pkl'],
        'label_enc': ['label_encoders.pkl'],
        'top_feat': ['top_features.pkl'],
        'all_feat': ['all_features.pkl', 'semua_fitur.pkl'],
        'metadata': ['model_metadata.pkl']
    }
    
    for key, filenames in possible_files.items():
        for fname in filenames:
            path = MODEL_DIR / fname
            if path.exists():
                try:
                    artifacts[key] = joblib.load(path)
                    st.sidebar.write(f"✅ Loaded: `{fname}`")
                    break
                except Exception as e:
                    st.sidebar.write(f"❌ Error loading {fname}: {str(e)[:50]}")
                    continue
        if key not in artifacts:
            artifacts[key] = None
    
    # Cek juga di root
    if artifacts.get('model') is None:
        for fname in ['best_model.pkl', 'best_churn_model.pkl']:
            path = Path(fname)
            if path.exists():
                try:
                    artifacts['model'] = joblib.load(path)
                    st.sidebar.write(f"✅ Loaded from root: `{fname}`")
                    break
                except:
                    pass
    
    return artifacts

arts = load_artifacts()

# ─────────────────────────────────────────────────────────────
# Cek apakah model berhasil di-load
# ─────────────────────────────────────────────────────────────
if arts.get('model') is None:
    st.error("❌ Model tidak ditemukan!")
    st.write(f"📁 Folder yang diperiksa: `{MODEL_DIR}`")
    st.write("File yang dicari:")
    st.write("  - best_model.pkl")
    st.write("  - best_churn_model.pkl")
    st.write("  - model_churn_terbaik.pkl")
    st.write("  - (juga di root directory)")
    st.stop()

model    = arts['model']
scaler   = arts.get('scaler')
scaler_top = arts.get('scaler_top')
le_map   = arts.get('label_enc', {})
top_feat = arts.get('top_feat', [])
all_feat = arts.get('all_feat', [])
meta     = arts.get('metadata', {})

# Jika scaler None, pakai scaler yang ada
if scaler is None and scaler_top is not None:
    scaler = scaler_top

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; text-align: center; }
    .result-churn { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; font-size: 1.6rem; font-weight: 700; }
    .result-ok { background: linear-gradient(135deg, #6ab04c, #2ecc71); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; font-size: 1.6rem; font-weight: 700; }
    .metric-box { background: #f8f9fa; border-radius: 10px; padding: 1rem; border-left: 4px solid #1f3a5f; }
    .stButton > button { width: 100%; background: #1f3a5f; color: white; border-radius: 8px; padding: 0.7rem; font-size: 1.1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown("### Sales & Marketing Dataset | UAS Data Science")
st.divider()

# ─────────────────────────────────────────────────────────────
# Sidebar - Info Model
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    model_name = meta.get('best_model_name', 'Voting Ensemble') if meta else 'Unknown'
    test_acc = meta.get('test_accuracy', 0.8923) if meta else 0.8923
    test_f1 = meta.get('test_f1', 0.8323) if meta else 0.8323
    
    st.markdown(f"""
    <div class="metric-box">
        <b>Model:</b> {model_name}<br>
        <b>Test Accuracy:</b> {test_acc:.4f}<br>
        <b>Test F1-Score:</b> {test_f1:.4f}<br>
        <b>Fitur:</b> Top {len(top_feat) if top_feat else 10}
    </div>
    """, unsafe_allow_html=True)

    if top_feat:
        st.subheader("📌 Fitur Terpenting")
        for i, feat in enumerate(top_feat[:10], 1):
            st.write(f"{i}. `{feat}`")

# ─────────────────────────────────────────────────────────────
# Form Input
# ─────────────────────────────────────────────────────────────
st.subheader("🔢 Input Data Customer")

FEATURE_CONFIG = {
    'age': {'type': 'number', 'min': 18, 'max': 80, 'default': 35, 'label': 'Usia'},
    'total_visits': {'type': 'number', 'min': 0, 'max': 500, 'default': 50, 'label': 'Total Kunjungan'},
    'avg_session_time': {'type': 'float', 'min': 0.0, 'max': 60.0, 'default': 5.0, 'label': 'Rata-rata Session (menit)'},
    'support_tickets': {'type': 'number', 'min': 0, 'max': 20, 'default': 1, 'label': 'Tiket Support'},
    'satisfaction_score': {'type': 'number', 'min': 1, 'max': 10, 'default': 7, 'label': 'Skor Kepuasan (1-10)'},
    'last_3_month_purchase_freq': {'type': 'number', 'min': 0, 'max': 30, 'default': 3, 'label': 'Frekuensi Pembelian 3 Bulan Terakhir'},
    'total_spent': {'type': 'float', 'min': 0, 'max': 10000, 'default': 500, 'label': 'Total Pengeluaran ($)'},
    'nps_score': {'type': 'number', 'min': -100, 'max': 100, 'default': 30, 'label': 'NPS Score'},
    'delivery_delay_days': {'type': 'number', 'min': 0, 'max': 30, 'default': 2, 'label': 'Keterlambatan Pengiriman (hari)'},
    'email_open_rate': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.3, 'label': 'Email Open Rate'},
    'gender': {'type': 'select', 'options': ['Male', 'Female'], 'default': 'Male', 'label': 'Gender'},
    'subscription_type': {'type': 'select', 'options': ['Basic', 'Standard', 'Premium'], 'default': 'Standard', 'label': 'Tipe Langganan'},
}

user_input = {}
col_left, col_right = st.columns(2)

with col_left:
    for key in ['age', 'total_visits', 'avg_session_time', 'support_tickets', 
                'satisfaction_score', 'last_3_month_purchase_freq']:
        cfg = FEATURE_CONFIG[key]
        if cfg['type'] == 'float':
            user_input[key] = st.number_input(cfg['label'], min_value=float(cfg['min']), max_value=float(cfg['max']), value=float(cfg['default']), step=0.1)
        else:
            user_input[key] = st.number_input(cfg['label'], min_value=int(cfg['min']), max_value=int(cfg['max']), value=int(cfg['default']), step=1)

with col_right:
    for key in ['total_spent', 'nps_score', 'delivery_delay_days', 'email_open_rate', 'gender', 'subscription_type']:
        cfg = FEATURE_CONFIG[key]
        if cfg['type'] == 'select':
            user_input[key] = st.selectbox(cfg['label'], options=cfg['options'], index=cfg['options'].index(cfg['default']))
        elif cfg['type'] == 'float':
            user_input[key] = st.number_input(cfg['label'], min_value=float(cfg['min']), max_value=float(cfg['max']), value=float(cfg['default']), step=0.01)
        else:
            user_input[key] = st.number_input(cfg['label'], min_value=int(cfg['min']), max_value=int(cfg['max']), value=int(cfg['default']), step=1)

# ─────────────────────────────────────────────────────────────
# Prediksi
# ─────────────────────────────────────────────────────────────
def preprocess_input(raw_input):
    # Encode categorical
    row = {}
    for key, val in raw_input.items():
        if key in le_map and hasattr(le_map[key], 'classes_'):
            try:
                row[key] = le_map[key].transform([str(val)])[0]
            except:
                row[key] = 0
        else:
            row[key] = val
    
    # Buat DataFrame
    df = pd.DataFrame([row])
    
    # Select top features jika ada
    if top_feat:
        available = [f for f in top_feat if f in df.columns]
        df = df[available]
    
    # Scale
    if scaler is not None:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df.values
    
    return df_scaled

st.divider()
if st.button("🔮 Prediksi Churn", use_container_width=True):
    with st.spinner("Memproses..."):
        try:
            X = preprocess_input(user_input)
            pred = model.predict(X)[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                prob_churn = proba[1] * 100
            else:
                prob_churn = 100 if pred == 1 else 0
            
            st.divider()
            st.subheader("📋 Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            with col1:
                if pred == 1:
                    st.markdown('<div class="result-churn">⚠️ CHURN</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-ok">✅ TIDAK CHURN</div>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Probabilitas Churn", f"{prob_churn:.1f}%")
                st.progress(int(prob_churn))
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

st.divider()
st.markdown("<center><small>UAS Data Science | Sales & Marketing Churn Prediction</small></center>", unsafe_allow_html=True)