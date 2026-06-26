    # ─────────────────────────────────────────────────────────────
    # PANDUAN DATA CHURN & TIDAK CHURN
    # ─────────────────────────────────────────────────────────────
    with st.expander("📖 Panduan: Cara Mendapatkan Hasil CHURN / TIDAK CHURN"):
        st.markdown("""
        <div style="background: #fff5f5; padding: 1rem; border-radius: 10px; border-left: 4px solid #ff6b6b;">
            <h4 style="color: #ff6b6b;">🔴 Data untuk Hasil CHURN</h4>
            <table style="width:100%; font-size:0.85rem; border-collapse: collapse;">
                <tr style="background: #ff6b6b; color: white;">
                    <th style="padding: 8px; text-align: left;">Fitur</th>
                    <th style="padding: 8px; text-align: left;">Nilai</th>
                    <th style="padding: 8px; text-align: left;">Keterangan</th>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>⭐ Skor Kepuasan</td>
                    <td><b>1-3</b></td>
                    <td>Sangat tidak puas</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>🎫 Tiket Support</td>
                    <td><b>5-10</b></td>
                    <td>Banyak komplain</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>⏱️ Rata-rata Sesi</td>
                    <td><b>0.5-2</b></td>
                    <td>Sangat singkat</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>🛍️ Frekuensi Pembelian</td>
                    <td><b>0-1</b></td>
                    <td>Jarang beli</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>💰 Total Pengeluaran</td>
                    <td><b>0-50</b></td>
                    <td>Sedikit belanja</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>👑 Premium User</td>
                    <td><b>0</b></td>
                    <td>Tidak premium</td>
                </tr>
                <tr>
                    <td>📦 Keterlambatan Kirim</td>
                    <td><b>10-20</b></td>
                    <td>Sering telat</td>
                </tr>
            </table>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #555;">
                📋 <b>Contoh:</b> Skor=1, Support=10, Sesi=0.5, Beli=0, Pengeluaran=10, Premium=0, Telat=20
            </p>
        </div>
        <br>
        <div style="background: #f0fff4; padding: 1rem; border-radius: 10px; border-left: 4px solid #00b894;">
            <h4 style="color: #00b894;">🟢 Data untuk Hasil TIDAK CHURN</h4>
            <table style="width:100%; font-size:0.85rem; border-collapse: collapse;">
                <tr style="background: #00b894; color: white;">
                    <th style="padding: 8px; text-align: left;">Fitur</th>
                    <th style="padding: 8px; text-align: left;">Nilai</th>
                    <th style="padding: 8px; text-align: left;">Keterangan</th>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>⭐ Skor Kepuasan</td>
                    <td><b>8-10</b></td>
                    <td>Sangat puas</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>🎫 Tiket Support</td>
                    <td><b>0-1</b></td>
                    <td>Tidak ada keluhan</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>⏱️ Rata-rata Sesi</td>
                    <td><b>8-15</b></td>
                    <td>Lama berkunjung</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>🛍️ Frekuensi Pembelian</td>
                    <td><b>8-15</b></td>
                    <td>Sering beli</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>💰 Total Pengeluaran</td>
                    <td><b>1000-5000</b></td>
                    <td>Banyak belanja</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td>👑 Premium User</td>
                    <td><b>1</b></td>
                    <td>Premium</td>
                </tr>
                <tr>
                    <td>📦 Keterlambatan Kirim</td>
                    <td><b>0-2</b></td>
                    <td>Tepat waktu</td>
                </tr>
            </table>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #555;">
                📋 <b>Contoh:</b> Skor=9, Support=0, Sesi=12.5, Beli=12, Pengeluaran=2500, Premium=1, Telat=1
            </p>
        </div>
        <br>
        <div style="background: #e8f4fd; padding: 0.8rem 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
            <b>💡 Tips Cepat:</b><br>
            🔴 <b>CHURN</b> → Semua nilai RENDAH (kecuali Tiket Support & Keterlambatan = TINGGI)<br>
            🟢 <b>TIDAK CHURN</b> → Semua nilai TINGGI (kecuali Tiket Support & Keterlambatan = RENDAH)
        </div>
        """, unsafe_allow_html=True)
