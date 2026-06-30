# 📊 Churn Prediction — UAS Data Science
### Sales & Marketing Dataset | Machine Learning Pipeline

---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan pipeline lengkap Data Science untuk memprediksi **Customer Churn** pada dataset Sales & Marketing (15.000 baris, 30 kolom). Pipeline mencakup EDA, Preprocessing, Modeling (3 algoritma), Hyperparameter Tuning, Feature Selection, dan Deployment menggunakan Streamlit.

---

## 🗂️ Struktur Direktori

```
uas_datascience/
├── main.py                   # Pipeline utama (EDA → Modeling → Tuning)
├── app.py                    # Streamlit deployment app
├── requirements.txt          # Dependensi Python
├── README.md                 # Dokumentasi ini
├── sales_marketing.csv       # Dataset (otomatis dibuat jika tidak ada)
│
├── output_plots/             # Visualisasi (dibuat otomatis)
│   ├── 1_missing_values.png
│   ├── 2_distribusi_churn.png
│   ├── 3_heatmap_korelasi.png
│   ├── 4_distribusi_fitur_numerik.png
│   ├── 5_perbandingan_direct_vs_prep.png
│   ├── 6_feature_importance.png
│   ├── 7_cm_best_model.png
│   └── 8_ringkasan_tuning.png
│
└── models/                   # Artefak model (dibuat otomatis)
    ├── best_model.pkl
    ├── scaler.pkl
    ├── scaler_top.pkl
    ├── label_encoders.pkl
    ├── top_features.pkl
    ├── all_features.pkl
    └── model_metadata.pkl
```

---

## ⚙️ Cara Menjalankan

### 1. Persiapan Environment

```bash
# Clone / ekstrak folder proyek
cd uas_datascience

# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau
venv\Scripts\activate           # Windows

# Install dependensi
pip install -r requirements.txt
```

### 2. Siapkan Dataset

**Opsi A — Dataset Asli dari Kaggle:**
1. Download dari: https://www.kaggle.com/datasets/bhaskerpaul/sales-and-marketing-dataset
2. Rename file menjadi `sales_marketing.csv`
3. Letakkan di folder yang sama dengan `main.py`

**Opsi B — Dataset Sintetis (Otomatis):**
Jika `sales_marketing.csv` tidak ditemukan, `main.py` akan **otomatis membuat** dataset sintetis 15.000 baris yang mereplikasi struktur aslinya dan menyimpannya sebagai `sales_marketing.csv`.

### 3. Jalankan Pipeline Utama

```bash
python main.py
```

Pipeline ini akan:
- ✅ Melakukan EDA dan menyimpan 4 visualisasi
- ✅ Direct Modeling (3 model, tanpa preprocessing)
- ✅ Preprocessed Modeling (missing value, outlier, encoding, scaling)
- ✅ Feature Importance & Top-10 Feature Selection
- ✅ Hyperparameter Tuning (GridSearchCV)
- ✅ Menyimpan model terbaik ke `models/best_model.pkl`
- ✅ Menyimpan semua artefak preprocessing

⏱️ **Estimasi waktu:** 3–8 menit (tergantung spesifikasi CPU)

### 4. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Buka browser ke: **http://localhost:8501**

---

## 🧪 Model yang Digunakan

| No | Model | Tipe |
|----|-------|------|
| 1 | Logistic Regression | Konvensional |
| 2 | Random Forest | Ensemble Bagging |
| 3 | Voting Classifier (LR + SVM + KNN) | Ensemble Voting |

---

## 🔄 Tahapan Pipeline

```
1. EDA
   ├── 5 baris pertama, info, statistik deskriptif
   ├── Visualisasi missing value
   ├── Distribusi target Churn
   └── Heatmap korelasi

2. Direct Modeling
   └── 3 model tanpa preprocessing → Evaluasi

3. Preprocessing
   ├── Drop fitur tidak relevan (customer_id, signup_date, dll)
   ├── Hapus duplikat
   ├── Imputasi: median (numerik) & modus (kategorikal)
   ├── Outlier handling (IQR clipping)
   ├── LabelEncoder untuk kategorikal
   └── StandardScaler (setelah train-test split)

4. Preprocessed Modeling
   └── 3 model + preprocessing → Evaluasi

5. Feature Selection & Tuning
   ├── Feature Importance (Random Forest)
   ├── Top 10 fitur terpilih
   └── GridSearchCV (5-fold CV, scoring=F1)

6. Deployment (Streamlit)
   ├── Load model terbaik
   ├── Form input semua fitur
   └── Hasil prediksi + probabilitas + rekomendasi
```

---

## 📈 Metrik Evaluasi

- **Accuracy** — proporsi prediksi benar
- **Precision** — dari prediksi Churn, berapa yang benar-benar Churn
- **Recall** — dari pelanggan Churn, berapa yang berhasil dideteksi
- **F1-Score** — harmonic mean dari Precision & Recall
- **Confusion Matrix** — visualisasi TP, TN, FP, FN

---

## 💡 Catatan

- Model terbaik dipilih berdasarkan **Test F1-Score** tertinggi
- Semua plot disimpan otomatis di folder `output_plots/`
- Semua artefak model disimpan otomatis di folder `models/`
- Aplikasi Streamlit memiliki fitur **rekomendasi otomatis** berdasarkan hasil prediksi

---

## 👤 Informasi

| | |
|--|--|
| **Mata Kuliah** | Data Science |
| **Jenis Tugas** | UAS (Ujian Akhir Semester) |
| **Dataset** | Sales & Marketing (Kaggle) |
| **Target** | Prediksi Customer Churn |
