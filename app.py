# =============================================================================
# app.py - VERSI RINGKAS (HANYA 5 FITUR UTAMA)
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
# Load Model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_dir = Path("models")
    try:
        model = joblib.load(model_dir / "best_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        return model, scaler
    except:
        # Coba di root
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler

model, scaler = load_model()

# ─────────────────────────────────────────────────────────────
# DAFTAR FITUR (HANYA 5 YANG TERPENTING)
# ─────────────────────────────────────────────────────────────
FITUR = {
    'satisfaction_score': {
        'label': '😊 Skor Kepuasan',
        'help': 'Nilai 1-10, semakin tinggi semakin puas',
        'min': 1, 'max': 10, 'default': 7
    },
    'support_tickets': {
        'label': '🎫 Tiket Support',
        'help': 'Jumlah tiket yang pernah diajukan',
        'min': 0, 'max': 20, 'default': 1
    },
    'avg_session_time': {
        'label': '⏱️ Rata-rata Sesi (menit)',
        'help': 'Rata-rata waktu yang dihabiskan per sesi',
        'min': 0.0, 'max': 60.0, 'default': 5.0
    },
    'last_3_month_purchase_freq': {
        'label': '🛒 Frekuensi Pembelian 3 Bulan',
        'help': 'Berapa kali pembelian dalam 3 bulan terakhir',
        'min': 0, 'max': 30, 'default': 3
    },
    'total_spent': {
        'label': '💰 Total Pengeluaran ($)',
        'help': 'Total uang yang sudah dikeluarkan',
        'min': 0, 'max': 10000, 'default': 500
    }
}

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Predictor")
st.markdown("### Prediksi dengan 5 Fitur Utama")

st.info("""
💡 **Hanya 5 fitur terpenting** yang digunakan berdasarkan analisis data:
1. Skor Kepuasan → semakin tinggi, semakin kecil risiko churn
2. Tiket Support → semakin banyak, semakin tinggi risiko
3. Rata-rata Sesi → semakin lama, semakin kecil risiko
4. Frekuensi Pembelian → semakin sering, semakin kecil risiko
5. Total Pengeluaran → semakin besar, semakin kecil risiko
""")

# ── Form Ringkas ─────────────────────────────────────────────
st.subheader("📝 Masukkan Data Pelanggan")

user_input = {}
cols = st.columns(3)  # 3 kolom agar lebih rapi

for i, (key, cfg) in enumerate(FITUR.items()):
    col = cols[i % 3]
    with col:
        if isinstance(cfg['default'], float):
            user_input[key] = st.number_input(
                cfg['label'],
                min_value=float(cfg['min']),
                max_value=float(cfg['max']),
                value=float(cfg['default']),
                step=0.1,
                help=cfg.get('help', '')
            )
        else:
            user_input[key] = st.number_input(
                cfg['label'],
                min_value=int(cfg['min']),
                max_value=int(cfg['max']),
                value=int(cfg['default']),
                step=1,
                help=cfg.get('help', '')
            )

# ── Tombol Prediksi ──────────────────────────────────────────
st.divider()
if st.button("🔮 Prediksi Churn", use_container_width=True, type="primary"):
    with st.spinner("Memproses..."):
        try:
            # Buat array dengan urutan yang benar
            # Urutan fitur: [satisfaction_score, support_tickets, avg_session_time, 
            #               last_3_month_purchase_freq, total_spent]
            X = np.array([[
                user_input['satisfaction_score'],
                user_input['support_tickets'],
                user_input['avg_session_time'],
                user_input['last_3_month_purchase_freq'],
                user_input['total_spent']
            ]])
            
            # Scale
            X_scaled = scaler.transform(X)
            
            # Prediksi
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            prob_churn = proba[1] * 100
            
            # ── Tampilkan Hasil ──────────────────────────────
            st.divider()
            st.subheader("📋 Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pred == 1:
                    st.error("⚠️ **CHURN**")
                    st.markdown(f"Probabilitas Churn: **{prob_churn:.1f}%**")
                else:
                    st.success("✅ **TIDAK CHURN**")
                    st.markdown(f"Probabilitas Churn: **{prob_churn:.1f}%**")
            
            with col2:
                st.progress(int(prob_churn), text=f"Risiko Churn: {prob_churn:.1f}%")
            
            # ── Rekomendasi ────────────────────────────────────
            st.subheader("💡 Rekomendasi")
            
            recs = []
            if user_input['satisfaction_score'] < 6:
                recs.append("📞 **Hubungi pelanggan** — skor kepuasan rendah")
            if user_input['support_tickets'] > 3:
                recs.append("🔧 **Selesaikan tiket support** — terlalu banyak keluhan")
            if user_input['last_3_month_purchase_freq'] < 2:
                recs.append("🛍️ **Kirim penawaran khusus** — pembelian rendah")
            if user_input['avg_session_time'] < 3:
                recs.append("⏱️ **Tingkatkan engagement** — sesi terlalu singkat")
            if user_input['total_spent'] < 100:
                recs.append("💰 **Tawarkan insentif** — pengeluaran rendah")
            
            if pred == 1:
                if not recs:
                    recs.append("🔄 **Jalankan program retensi**")
                for r in recs:
                    st.warning(r)
            else:
                st.success("✅ Customer dalam kondisi sehat. Pertahankan kualitas layanan!")
                if user_input['satisfaction_score'] >= 8:
                    st.balloons()
                    st.info("⭐ Pelanggan sangat puas! Pertahankan kualitas ini.")
            
            # ── Detail ─────────────────────────────────────────
            with st.expander("📄 Detail Data"):
                st.dataframe(pd.DataFrame([user_input]).T, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>UAS Data Science | Churn Prediction | 5 Fitur Utama</small></center>",
    unsafe_allow_html=True
)