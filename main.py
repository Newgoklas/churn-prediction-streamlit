# =============================================================================
# UAS DATA SCIENCE - SALES & MARKETING CHURN PREDICTION
# Pipeline Lengkap: EDA → Preprocessing → Modeling → Tuning
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk menyimpan PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import os
from pathlib import Path

# Sklearn - Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Sklearn - Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Sklearn - Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────
# Konfigurasi & Setup
# ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
OUTPUT_DIR = Path("output_plots")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size']  = 11
PALETTE = "Set2"

print("=" * 65)
print("   UAS DATA SCIENCE - CHURN PREDICTION PIPELINE")
print("=" * 65)

# =============================================================================
# BAGIAN 0 - LOAD / GENERATE DATASET
# =============================================================================
print("\n[0] Memuat dataset...")

DATASET_PATH = "sales_marketing.csv"

def generate_synthetic_dataset(n=15000, seed=42):
    """
    Buat dataset sintetis yang mereplikasi struktur asli Kaggle
    agar kode bisa dijalankan tanpa file CSV terlebih dahulu.
    """
    rng = np.random.default_rng(seed)

    countries = ['USA', 'UK', 'Germany', 'France', 'India', 'Australia', 'Canada', 'Brazil']
    cities    = ['New York', 'London', 'Berlin', 'Paris', 'Mumbai', 'Sydney', 'Toronto', 'São Paulo']
    channels  = ['Organic', 'Paid', 'Referral', 'Social', 'Email']
    devices   = ['Mobile', 'Desktop', 'Tablet']
    subs      = ['Basic', 'Standard', 'Premium']
    payments  = ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer', 'Crypto']
    coupons   = ['SAVE10', 'DISC20', 'NONE', 'VIP30', 'FIRST50', 'NONE', 'NONE']

    age = rng.integers(18, 70, n)
    total_spent = rng.exponential(500, n).round(2)
    satisfaction = rng.integers(1, 11, n).astype(float)
    nps          = rng.integers(-100, 101, n).astype(float)
    support_tix  = rng.poisson(1.5, n)
    refund_req   = rng.integers(0, 6, n)
    delivery_delay = rng.integers(0, 15, n).astype(float)
    discount_used  = rng.integers(0, 20, n).astype(float)
    lifetime_val   = (total_spent * rng.uniform(1.2, 3, n)).round(2)
    freq_3m        = rng.integers(0, 20, n).astype(float)
    visits         = rng.integers(1, 200, n).astype(float)
    session_time   = rng.exponential(5, n).round(2)
    pages          = rng.integers(1, 30, n).astype(float)
    email_open     = rng.uniform(0, 1, n).round(3)
    email_click    = (email_open * rng.uniform(0.1, 0.6, n)).round(3)
    avg_order      = (total_spent / np.maximum(freq_3m + 1, 1)).round(2)
    mktg_spend     = rng.exponential(30, n).round(2)

    # Logit untuk churn yang realistis
    churn_score = (
        - 0.3 * (satisfaction - 5)
        + 0.4 * support_tix
        + 0.3 * refund_req
        - 0.5 * (freq_3m / 10)
        + rng.normal(0, 1, n)
    )
    churn = (churn_score > 0).astype(int)

    # Injeksi missing values (~3–5%)
    def inject_missing(arr, rate=0.04):
        mask = rng.random(n) < rate
        arr = arr.astype(object)
        arr[mask] = np.nan
        return arr

    signup_base = pd.Timestamp('2018-01-01')
    signup_dates = pd.to_datetime(
        [signup_base + pd.Timedelta(days=int(d)) for d in rng.integers(0, 1800, n)]
    )
    last_purchase = signup_dates + pd.to_timedelta(rng.integers(30, 1500, n), unit='D')

    df = pd.DataFrame({
        'customer_id'             : [f'CUST{i:05d}' for i in range(n)],
        'gender'                  : rng.choice(['Male', 'Female', 'Other'], n),
        'age'                     : inject_missing(age),
        'country'                 : rng.choice(countries, n),
        'city'                    : rng.choice(cities, n),
        'signup_date'             : signup_dates,
        'last_purchase_date'      : last_purchase,
        'acquisition_channel'     : rng.choice(channels, n),
        'device_type'             : rng.choice(devices, n),
        'subscription_type'       : rng.choice(subs, n),
        'is_premium_user'         : rng.integers(0, 2, n),
        'total_visits'            : inject_missing(visits),
        'avg_session_time'        : inject_missing(session_time),
        'pages_per_session'       : inject_missing(pages),
        'email_open_rate'         : inject_missing(email_open),
        'email_click_rate'        : inject_missing(email_click),
        'total_spent'             : inject_missing(total_spent),
        'avg_order_value'         : inject_missing(avg_order),
        'discount_used'           : inject_missing(discount_used),
        'coupon_code'             : rng.choice(coupons, n),
        'support_tickets'         : inject_missing(support_tix),
        'refund_requested'        : inject_missing(refund_req),
        'delivery_delay_days'     : inject_missing(delivery_delay),
        'payment_method'          : rng.choice(payments, n),
        'satisfaction_score'      : inject_missing(satisfaction),
        'nps_score'               : inject_missing(nps),
        'marketing_spend_per_user': inject_missing(mktg_spend),
        'lifetime_value'          : inject_missing(lifetime_val),
        'last_3_month_purchase_freq': inject_missing(freq_3m),
        'churn'                   : churn
    })
    return df


if os.path.exists(DATASET_PATH):
    df_raw = pd.read_csv(DATASET_PATH)
    print(f"   ✔ Dataset dimuat dari '{DATASET_PATH}' → {df_raw.shape}")
else:
    print(f"   ⚠  '{DATASET_PATH}' tidak ditemukan. Menggunakan dataset sintetis (15.000 baris).")
    df_raw = generate_synthetic_dataset(n=15000, seed=RANDOM_STATE)
    df_raw.to_csv(DATASET_PATH, index=False)
    print(f"   ✔ Dataset sintetis disimpan sebagai '{DATASET_PATH}' → {df_raw.shape}")

# =============================================================================
# BAGIAN 1 - EDA (Eksploratory Data Analysis)
# =============================================================================
print("\n" + "=" * 65)
print("   BAGIAN 1 - EKSPLORATORY DATA ANALYSIS (EDA)")
print("=" * 65)

# ── 1.1 Tampilan Awal ────────────────────────────────────────
print("\n[1.1] 5 Baris Pertama:")
print(df_raw.head().to_string())

print("\n[1.2] Info Dataset:")
df_raw.info()

print("\n[1.3] Statistik Deskriptif:")
print(df_raw.describe(include='all').to_string())

# ── 1.2 Missing Value ────────────────────────────────────────
print("\n[1.4] Persentase Missing Value per Kolom:")
missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=False)
missing_df  = missing_pct[missing_pct > 0].reset_index()
missing_df.columns = ['Kolom', 'Persen Missing']
print(missing_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
if len(missing_df) > 0:
    bars = ax.barh(missing_df['Kolom'], missing_df['Persen Missing'],
                   color=sns.color_palette(PALETTE, len(missing_df)))
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)
    ax.set_xlabel('Persentase Missing Value (%)')
    ax.set_title('Missing Value per Kolom', fontweight='bold')
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, 'Tidak ada missing value', ha='center', va='center',
            fontsize=14, transform=ax.transAxes)
    ax.set_title('Missing Value per Kolom')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_missing_values.png', bbox_inches='tight')
plt.close()
print("   → Plot disimpan: 1_missing_values.png")

# ── 1.3 Distribusi Target ────────────────────────────────────
churn_counts  = df_raw['churn'].value_counts()
churn_pct     = df_raw['churn'].value_counts(normalize=True) * 100
churn_labels  = {0: 'Tidak Churn (0)', 1: 'Churn (1)'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = [sns.color_palette(PALETTE)[0], sns.color_palette(PALETTE)[1]]

bars = axes[0].bar([churn_labels.get(i, i) for i in churn_counts.index],
                   churn_counts.values, color=colors)
axes[0].bar_label(bars, labels=[f'{v:,}' for v in churn_counts.values], padding=3)
axes[0].set_title('Distribusi Churn - Jumlah', fontweight='bold')
axes[0].set_ylabel('Jumlah')

wedges, texts, autotexts = axes[1].pie(
    churn_counts.values,
    labels=[churn_labels.get(i, i) for i in churn_counts.index],
    autopct='%1.1f%%', colors=colors, startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
axes[1].set_title('Proporsi Churn', fontweight='bold')

plt.suptitle('Distribusi Target: Churn', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2_distribusi_churn.png', bbox_inches='tight')
plt.close()
print("   → Plot disimpan: 2_distribusi_churn.png")
print(f"   Churn=0: {churn_counts.get(0, 0):,} ({churn_pct.get(0, 0):.1f}%)  "
      f"Churn=1: {churn_counts.get(1, 0):,} ({churn_pct.get(1, 0):.1f}%)")

# ── 1.4 Heatmap Korelasi ─────────────────────────────────────
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df_raw[num_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.4, ax=ax,
            annot_kws={'size': 7}, cbar_kws={'shrink': 0.8})
ax.set_title('Heatmap Korelasi Fitur Numerik', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3_heatmap_korelasi.png', bbox_inches='tight')
plt.close()
print("   → Plot disimpan: 3_heatmap_korelasi.png")

# ── 1.5 Distribusi Fitur Numerik ─────────────────────────────
key_num = [c for c in ['age', 'total_spent', 'satisfaction_score', 'nps_score',
                        'lifetime_value', 'support_tickets', 'total_visits',
                        'last_3_month_purchase_freq'] if c in df_raw.columns]
n_cols_plot = 4
n_rows_plot = (len(key_num) + n_cols_plot - 1) // n_cols_plot
fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                         figsize=(n_cols_plot * 4, n_rows_plot * 3.5))
axes = axes.flatten()
for i, col in enumerate(key_num):
    sns.histplot(df_raw[col].dropna(), kde=True, ax=axes[i],
                 color=sns.color_palette(PALETTE)[i % 8])
    axes[i].set_title(col, fontweight='bold', fontsize=10)
    axes[i].set_xlabel('')
for j in range(len(key_num), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Distribusi Fitur Numerik Utama', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '4_distribusi_fitur_numerik.png', bbox_inches='tight')
plt.close()
print("   → Plot disimpan: 4_distribusi_fitur_numerik.png")

print("\n[EDA] ✔ Selesai.")

# =============================================================================
# BAGIAN 2 - DIRECT MODELING (Tanpa Preprocessing & Tuning)
# =============================================================================
print("\n" + "=" * 65)
print("   BAGIAN 2 - DIRECT MODELING (Tanpa Preprocessing)")
print("=" * 65)

# ── 2.1 Persiapan Data Mentah ────────────────────────────────
df_direct = df_raw.copy()

# Drop kolom non-numerik (datetime, string) agar bisa masuk model
drop_cols_direct = ['customer_id', 'signup_date', 'last_purchase_date',
                    'coupon_code', 'churn']
cat_cols_direct  = df_direct.select_dtypes(include=['object']).columns.tolist()
df_direct        = df_direct.drop(columns=[c for c in drop_cols_direct
                                            if c in df_direct.columns], errors='ignore')

# Encode kategorikal sementara (LabelEncoder cepat)
le_direct = LabelEncoder()
for col in cat_cols_direct:
    if col in df_direct.columns:
        df_direct[col] = df_direct[col].astype(str)
        df_direct[col] = le_direct.fit_transform(df_direct[col])

# Isi missing dengan median (wajib agar model tidak error)
for col in df_direct.columns:
    if df_direct[col].isnull().any():
        df_direct[col].fillna(df_direct[col].median(), inplace=True)

X_direct = df_direct
y_direct = df_raw['churn']

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X_direct, y_direct, test_size=0.2, random_state=RANDOM_STATE, stratify=y_direct
)
print(f"\n   Train: {X_tr_d.shape}  |  Test: {X_te_d.shape}")

# ── 2.2 Definisi 3 Model ─────────────────────────────────────
models_direct = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Voting Classifier'  : VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
            ('svm', SVC(probability=True, random_state=RANDOM_STATE)),
            ('knn', KNeighborsClassifier(n_neighbors=7))
        ],
        voting='soft'
    )
}

def evaluate_model(name, model, X_train, X_test, y_train, y_test,
                   stage='', save_cm=True):
    """Latih, prediksi, hitung metrik, simpan confusion matrix."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n   ┌─ {name} ({stage}) {'─'*(40-len(name)-len(stage))}")
    print(f"   │  Accuracy : {acc:.4f}")
    print(f"   │  Precision: {prec:.4f}")
    print(f"   │  Recall   : {rec:.4f}")
    print(f"   │  F1-Score : {f1:.4f}")
    print(f"   └{'─'*50}")

    if save_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=['Tidak Churn', 'Churn'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{name}\n({stage})', fontweight='bold')
        plt.tight_layout()
        safe_name = name.lower().replace(' ', '_')
        safe_stage = stage.lower().replace(' ', '_')
        plt.savefig(OUTPUT_DIR / f'cm_{safe_stage}_{safe_name}.png', bbox_inches='tight')
        plt.close()

    return {'Model': name, 'Stage': stage,
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

results_all = []

print("\n[2] Melatih model (Direct Modeling)...")
for name, model in models_direct.items():
    result = evaluate_model(name, model, X_tr_d, X_te_d, y_tr_d, y_te_d,
                            stage='Direct')
    results_all.append(result)

# Ringkasan Direct Modeling
df_res_direct = pd.DataFrame([r for r in results_all if r['Stage'] == 'Direct'])
print("\n   [Ringkasan Direct Modeling]")
print(df_res_direct[['Model','Accuracy','Precision','Recall','F1']].to_string(index=False))

# =============================================================================
# BAGIAN 3 - MODELING DENGAN PREPROCESSING
# =============================================================================
print("\n" + "=" * 65)
print("   BAGIAN 3 - MODELING DENGAN PREPROCESSING")
print("=" * 65)

df_prep = df_raw.copy()

# ── 3.1 Hapus Fitur Tidak Relevan ────────────────────────────
drop_irrelevant = ['customer_id', 'signup_date', 'last_purchase_date', 'coupon_code']
df_prep.drop(columns=[c for c in drop_irrelevant if c in df_prep.columns],
             inplace=True, errors='ignore')
print(f"\n[3.1] Fitur dihapus: {drop_irrelevant}")
print(f"      Shape setelah drop: {df_prep.shape}")

# ── 3.2 Hapus Duplikat ───────────────────────────────────────
n_dup = df_prep.duplicated().sum()
df_prep.drop_duplicates(inplace=True)
print(f"\n[3.2] Duplikat dihapus: {n_dup} baris")
print(f"      Shape setelah drop_duplicates: {df_prep.shape}")

# ── 3.3 Penanganan Missing Value ─────────────────────────────
num_cols_prep = df_prep.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_prep = df_prep.select_dtypes(include=['object']).columns.tolist()
# Hapus target dari num_cols
if 'churn' in num_cols_prep:
    num_cols_prep.remove('churn')

imp_num = SimpleImputer(strategy='median')
imp_cat = SimpleImputer(strategy='most_frequent')

df_prep[num_cols_prep] = imp_num.fit_transform(df_prep[num_cols_prep])
if cat_cols_prep:
    df_prep[cat_cols_prep] = imp_cat.fit_transform(df_prep[cat_cols_prep])

missing_after = df_prep.isnull().sum().sum()
print(f"\n[3.3] Missing value setelah imputasi: {missing_after}")

# ── 3.4 Deteksi & Handling Outlier (IQR) ─────────────────────
print("\n[3.4] Handling outlier dengan IQR method (clipping)...")
outlier_report = {}
for col in num_cols_prep:
    Q1  = df_prep[col].quantile(0.25)
    Q3  = df_prep[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df_prep[col] < lower) | (df_prep[col] > upper)).sum()
    if n_out > 0:
        outlier_report[col] = n_out
        df_prep[col] = df_prep[col].clip(lower, upper)

print(f"      Kolom dengan outlier yang di-clip: {len(outlier_report)}")
for col, cnt in list(outlier_report.items())[:5]:
    print(f"      • {col}: {cnt} outlier")
if len(outlier_report) > 5:
    print(f"      ... dan {len(outlier_report)-5} kolom lainnya")

# ── 3.5 Encoding Kategorikal ─────────────────────────────────
print("\n[3.5] Encoding fitur kategorikal...")
le_map = {}
for col in cat_cols_prep:
    le = LabelEncoder()
    df_prep[col] = le.fit_transform(df_prep[col].astype(str))
    le_map[col] = le
print(f"      Kolom di-encode: {cat_cols_prep}")

# Simpan le_map untuk app.py
with open(MODEL_DIR / 'label_encoders.pkl', 'wb') as f:
    pickle.dump(le_map, f)

# ── 3.6 Split Data ───────────────────────────────────────────
X_prep = df_prep.drop(columns=['churn'])
y_prep = df_prep['churn']

feature_names_prep = X_prep.columns.tolist()  # simpan untuk nanti

X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
    X_prep, y_prep, test_size=0.2, random_state=RANDOM_STATE, stratify=y_prep
)

# ── 3.7 Scaling (setelah split) ──────────────────────────────
scaler = StandardScaler()
X_tr_p_sc = scaler.fit_transform(X_tr_p)
X_te_p_sc = scaler.transform(X_te_p)
# Simpan scaler
with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"\n[3.6-3.7] Train: {X_tr_p_sc.shape}  |  Test: {X_te_p_sc.shape}")

# ── 3.8 Latih 3 Model (dengan Preprocessing) ─────────────────
models_prep = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Voting Classifier'  : VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
            ('svm', SVC(probability=True, random_state=RANDOM_STATE)),
            ('knn', KNeighborsClassifier(n_neighbors=7))
        ],
        voting='soft'
    )
}

print("\n[3.8] Melatih model (dengan Preprocessing)...")
trained_models_prep = {}
for name, model in models_prep.items():
    result = evaluate_model(name, model, X_tr_p_sc, X_te_p_sc, y_tr_p, y_te_p,
                            stage='Preprocessed')
    results_all.append(result)
    trained_models_prep[name] = model   # simpan model terlatih

# Ringkasan Preprocessed Modeling
df_res_prep = pd.DataFrame([r for r in results_all if r['Stage'] == 'Preprocessed'])
print("\n   [Ringkasan Preprocessed Modeling]")
print(df_res_prep[['Model','Accuracy','Precision','Recall','F1']].to_string(index=False))

# ── Visualisasi Perbandingan Direct vs Preprocessed ──────────
df_compare = pd.DataFrame(results_all)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
pal = {'Direct': '#5B9BD5', 'Preprocessed': '#ED7D31'}
for i, metric in enumerate(metrics):
    df_plot = df_compare.pivot(index='Model', columns='Stage', values=metric)
    df_plot.plot(kind='bar', ax=axes[i], color=[pal.get(c, '#ccc') for c in df_plot.columns],
                 edgecolor='white', width=0.65)
    axes[i].set_title(metric, fontweight='bold')
    axes[i].set_ylim(0, 1.1)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].legend(title='')
    for bar in axes[i].patches:
        axes[i].annotate(f'{bar.get_height():.3f}',
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontsize=8)
plt.suptitle('Perbandingan: Direct vs Preprocessed Modeling', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '5_perbandingan_direct_vs_prep.png', bbox_inches='tight')
plt.close()
print("\n   → Plot disimpan: 5_perbandingan_direct_vs_prep.png")

# =============================================================================
# BAGIAN 4 - FEATURE IMPORTANCE, SELECTION & HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 65)
print("   BAGIAN 4 - FEATURE SELECTION & HYPERPARAMETER TUNING")
print("=" * 65)

# ── 4.1 Feature Importance dari Random Forest ────────────────
rf_trained = trained_models_prep['Random Forest']
importances = pd.Series(rf_trained.feature_importances_,
                         index=feature_names_prep).sort_values(ascending=False)

print("\n[4.1] Top 15 Fitur Terpenting (Random Forest):")
print(importances.head(15).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
top_n = min(15, len(importances))
top_importances = importances.head(top_n)
bars = ax.barh(top_importances.index[::-1], top_importances.values[::-1],
               color=sns.color_palette('YlOrRd', top_n)[::-1])
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
ax.set_xlabel('Feature Importance Score')
ax.set_title(f'Top {top_n} Fitur Terpenting (Random Forest)', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '6_feature_importance.png', bbox_inches='tight')
plt.close()
print("   → Plot disimpan: 6_feature_importance.png")

# ── 4.2 Pilih Top 10 Fitur ───────────────────────────────────
TOP_N_FEATURES = 10
top10_features = importances.head(TOP_N_FEATURES).index.tolist()
print(f"\n[4.2] Top {TOP_N_FEATURES} fitur terpilih: {top10_features}")

# Simpan daftar fitur untuk app.py
with open(MODEL_DIR / 'top_features.pkl', 'wb') as f:
    pickle.dump(top10_features, f)
with open(MODEL_DIR / 'all_features.pkl', 'wb') as f:
    pickle.dump(feature_names_prep, f)

# Buat dataset dengan top10 fitur
X_top = X_prep[top10_features]
X_tr_top, X_te_top, y_tr_top, y_te_top = train_test_split(
    X_top, y_prep, test_size=0.2, random_state=RANDOM_STATE, stratify=y_prep
)
scaler_top = StandardScaler()
X_tr_top_sc = scaler_top.fit_transform(X_tr_top)
X_te_top_sc  = scaler_top.transform(X_te_top)

# Simpan scaler top features
with open(MODEL_DIR / 'scaler_top.pkl', 'wb') as f:
    pickle.dump(scaler_top, f)

# ── 4.3 Hyperparameter Tuning ────────────────────────────────
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'params': {
            'C'      : [0.01, 0.1, 1, 10],
            'solver' : ['lbfgs', 'liblinear'],
            'penalty': ['l2']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE),
        'params': {
            'n_estimators'     : [50, 100, 200],
            'max_depth'        : [None, 10, 20],
            'min_samples_split': [2, 5],
            'max_features'     : ['sqrt', 'log2']
        }
    },
    'Voting Classifier': {
        'model': VotingClassifier(
            estimators=[
                ('lr',  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                ('svm', SVC(probability=True, random_state=RANDOM_STATE)),
                ('knn', KNeighborsClassifier())
            ],
            voting='soft'
        ),
        'params': {
            'lr__C'          : [0.1, 1, 10],
            'knn__n_neighbors': [5, 7, 11],
            'svm__C'         : [0.1, 1, 5]
        }
    }
}

print(f"\n[4.3] GridSearchCV (cv=5, scoring=f1) — ini membutuhkan waktu beberapa menit...")
best_models = {}
tuning_results = []

for name, cfg in param_grids.items():
    print(f"\n   Tuning: {name} ...")
    gs = GridSearchCV(
        estimator=cfg['model'],
        param_grid=cfg['params'],
        cv=cv_strategy,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X_tr_top_sc, y_tr_top)
    best_models[name] = gs.best_estimator_

    y_pred = gs.best_estimator_.predict(X_te_top_sc)
    acc  = accuracy_score(y_te_top, y_pred)
    prec = precision_score(y_te_top, y_pred, zero_division=0)
    rec  = recall_score(y_te_top, y_pred, zero_division=0)
    f1   = f1_score(y_te_top, y_pred, zero_division=0)

    tuning_results.append({
        'Model'        : name,
        'Best Params'  : gs.best_params_,
        'CV F1 Score'  : gs.best_score_,
        'Test Accuracy': acc,
        'Test Precision': prec,
        'Test Recall'  : rec,
        'Test F1'      : f1
    })

    print(f"   ✔ Best Params : {gs.best_params_}")
    print(f"     CV F1       : {gs.best_score_:.4f}")
    print(f"     Test F1     : {f1:.4f}  |  Acc: {acc:.4f}  |  Rec: {rec:.4f}")

# ── 4.4 Pilih Model Terbaik ──────────────────────────────────
df_tuning = pd.DataFrame(tuning_results)
best_row   = df_tuning.loc[df_tuning['Test F1'].idxmax()]
best_model_name = best_row['Model']
best_model_obj  = best_models[best_model_name]

print(f"\n[4.4] ★ Model Terbaik setelah Tuning: {best_model_name}")
print(f"      Test F1 Score  : {best_row['Test F1']:.4f}")
print(f"      Test Accuracy  : {best_row['Test Accuracy']:.4f}")
print(f"      Best Params    : {best_row['Best Params']}")

# Confusion Matrix model terbaik
y_pred_best = best_model_obj.predict(X_te_top_sc)
cm_best = confusion_matrix(y_te_top, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm_best, display_labels=['Tidak Churn', 'Churn']).plot(
    ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Best Model: {best_model_name}\n(Setelah Tuning + Top-{TOP_N_FEATURES} Fitur)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '7_cm_best_model.png', bbox_inches='tight')
plt.close()
print("   → Plot disimpan: 7_cm_best_model.png")

# Visualisasi ringkasan tuning
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metrics_tuning = ['Test Accuracy', 'Test F1']
for i, metric in enumerate(metrics_tuning):
    bars = axes[i].bar(df_tuning['Model'], df_tuning[metric],
                       color=sns.color_palette(PALETTE, len(df_tuning)),
                       edgecolor='white')
    axes[i].bar_label(bars, fmt='%.4f', padding=3, fontsize=10)
    axes[i].set_ylim(0, 1.12)
    axes[i].set_title(metric, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=15)
plt.suptitle('Ringkasan Model Setelah Hyperparameter Tuning', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '8_ringkasan_tuning.png', bbox_inches='tight')
plt.close()
print("   → Plot disimpan: 8_ringkasan_tuning.png")

# ── 4.5 Simpan Model Terbaik ─────────────────────────────────
model_path = MODEL_DIR / 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model_obj, f)
print(f"\n[4.5] ✔ Model terbaik disimpan: {model_path}")

# Simpan metadata model
metadata = {
    'best_model_name' : best_model_name,
    'top_features'    : top10_features,
    'all_features'    : feature_names_prep,
    'cat_cols'        : cat_cols_prep,
    'test_f1'         : float(best_row['Test F1']),
    'test_accuracy'   : float(best_row['Test Accuracy'])
}
with open(MODEL_DIR / 'model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✔ Metadata disimpan: {MODEL_DIR / 'model_metadata.pkl'}")

# =============================================================================
# RINGKASAN AKHIR
# =============================================================================
print("\n" + "=" * 65)
print("   RINGKASAN LENGKAP SEMUA STAGE")
print("=" * 65)

print("\n[Direct Modeling]")
print(df_res_direct[['Model','Accuracy','Precision','Recall','F1']].to_string(index=False))

print("\n[Preprocessed Modeling]")
print(df_res_prep[['Model','Accuracy','Precision','Recall','F1']].to_string(index=False))

print("\n[Setelah Hyperparameter Tuning + Top-10 Features]")
cols_show = ['Model','Test Accuracy','Test Precision','Test Recall','Test F1','CV F1 Score']
print(df_tuning[cols_show].to_string(index=False))

print(f"\n★ MODEL TERBAIK  : {best_model_name}")
print(f"  Test Accuracy  : {best_row['Test Accuracy']:.4f}")
print(f"  Test Precision : {best_row['Test Precision']:.4f}")
print(f"  Test Recall    : {best_row['Test Recall']:.4f}")
print(f"  Test F1-Score  : {best_row['Test F1']:.4f}")

print(f"\n[Output]")
print(f"  Plots   : {OUTPUT_DIR}/  ({len(list(OUTPUT_DIR.glob('*.png')))} file PNG)")
print(f"  Models  : {MODEL_DIR}/   ({len(list(MODEL_DIR.glob('*.pkl')))} file PKL)")
print("\n✔ Pipeline selesai! Jalankan 'streamlit run app.py' untuk deployment.")
print("=" * 65)
