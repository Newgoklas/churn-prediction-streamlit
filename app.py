# ─────────────────────────────────────────────────────────────
# FORM INPUT - HANYA 7 FITUR UTAMA
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 Isi 7 data berikut untuk memprediksi churn. Hanya 1 menit saja!</div>', unsafe_allow_html=True)

# ── 7 FITUR TERPENTING ──
FEATURE_CONFIG = {
    # 1. Skor Kepuasan (paling penting)
    'satisfaction_score': {
        'type': 'number', 'min': 1, 'max': 10, 'default': 7, 
        'label': '⭐ Skor Kepuasan (1-10)', 'step': 1, 
        'help': 'Semakin tinggi, semakin puas pelanggan'
    },
    # 2. Tiket Support
    'support_tickets': {
        'type': 'number', 'min': 0, 'max': 20, 'default': 1, 
        'label': '🎫 Tiket Support', 'step': 1,
        'help': 'Jumlah tiket yang pernah diajukan'
    },
    # 3. Rata-rata Sesi
    'avg_session_time': {
        'type': 'float', 'min': 0.0, 'max': 60.0, 'default': 5.0, 
        'label': '⏱️ Rata-rata Sesi (menit)', 'step': 0.1,
        'help': 'Rata-rata waktu per sesi penggunaan'
    },
    # 4. Frekuensi Pembelian
    'last_3_month_purchase_freq': {
        'type': 'number', 'min': 0, 'max': 30, 'default': 3, 
        'label': '🛍️ Frekuensi Pembelian 3 Bulan', 'step': 1,
        'help': 'Berapa kali pembelian dalam 3 bulan terakhir'
    },
    # 5. Total Pengeluaran
    'total_spent': {
        'type': 'float', 'min': 0.0, 'max': 10000.0, 'default': 500.0, 
        'label': '💰 Total Pengeluaran ($)', 'step': 10.0,
        'help': 'Total uang yang sudah dikeluarkan'
    },
    # 6. Premium User
    'is_premium_user': {
        'type': 'number', 'min': 0, 'max': 1, 'default': 1, 
        'label': '👑 Premium User (0=Tidak, 1=Ya)', 'step': 1,
        'help': 'Apakah pelanggan berlangganan premium?'
    },
    # 7. Keterlambatan Pengiriman
    'delivery_delay_days': {
        'type': 'number', 'min': 0, 'max': 30, 'default': 6, 
        'label': '📦 Keterlambatan Pengiriman (hari)', 'step': 1,
        'help': 'Rata-rata keterlambatan pengiriman'
    }
}

# ── RENDER FORM 2 KOLOM ──
user_input = {}

col_left, col_right = st.columns(2)

# Pisahkan fitur ke kiri dan kanan (4 kiri, 3 kanan)
left_keys = ['satisfaction_score', 'support_tickets', 'avg_session_time', 'last_3_month_purchase_freq']
right_keys = ['total_spent', 'is_premium_user', 'delivery_delay_days']

with col_left:
    st.markdown("""
    <div class="card-section" style="padding:1rem 1.5rem 0.5rem 1.5rem;">
        <div class="card-title" style="margin-bottom:0.5rem;">
            <span class="icon">📊</span> Fitur Utama
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
                key=f"num_{key}"
            )
        else:
            user_input[key] = st.number_input(
                cfg['label'], 
                min_value=int(cfg['min']),
                max_value=int(cfg['max']), 
                value=int(cfg['default']),
                step=int(cfg['step']), 
                help=cfg.get('help', ''),
                key=f"num_{key}"
            )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="card-section" style="padding:1rem 1.5rem 0.5rem 1.5rem;">
        <div class="card-title" style="margin-bottom:0.5rem;">
            <span class="icon">💰</span> Fitur Tambahan
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
                key=f"num_{key}"
            )
        else:
            user_input[key] = st.number_input(
                cfg['label'], 
                min_value=int(cfg['min']),
                max_value=int(cfg['max']), 
                value=int(cfg['default']),
                step=int(cfg['step']), 
                help=cfg.get('help', ''),
                key=f"num_{key}"
            )
    st.markdown("</div>", unsafe_allow_html=True)