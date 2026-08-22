import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── PAGE CONFIG ────────────────────────────────────
st.set_page_config(
    page_title="Loan Default — Model Comparison",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CUSTOM CSS ─────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; color: #e2e8f0; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0 !important; }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    .main-title {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.85rem; margin: 0; font-weight: 600; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #f1f5f9; margin: 0.5rem 0; }
    .metric-card .label { font-size: 0.72rem; color: #64748b; margin-top: 0.2rem; }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 0.5rem 0 0.5rem 0.8rem;
        margin: 1.5rem 0 1rem 0;
        background: linear-gradient(90deg, rgba(59,130,246,0.1) 0%, transparent 100%);
        border-radius: 0 8px 8px 0;
    }

    .info-box {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #e2e8f0;
        border: 1px solid #334155;
    }
    .info-box p { color: #cbd5e1 !important; }

    .warn-box {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #e2e8f0;
        border: 1px solid #334155;
    }

    .pred-default {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #fecaca;
        box-shadow: 0 4px 6px rgba(239,68,68,0.2);
    }
    .pred-default h2, .pred-default h3 { color: #fca5a5; margin-bottom: 0.5rem; }

    .pred-safe {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #bbf7d0;
        box-shadow: 0 4px 6px rgba(34,197,94,0.2);
    }
    .pred-safe h2, .pred-safe h3 { color: #86efac; margin-bottom: 0.5rem; }

    p, span, div { color: #e2e8f0; }

    .stButton button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59,130,246,0.3);
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab-list"] { background-color: #0f172a; border-radius: 8px 8px 0 0; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; background-color: transparent; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #e2e8f0;
        background-color: #1e293b;
        border-bottom: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── CONSTANTS ──────────────────────────────────────
MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost", "Weighted Soft Voting Ensemble"]
ENSEMBLE_MODEL_NAME = "Weighted Soft Voting Ensemble"

# Kurs USD → IDR. Model dilatih pada data Lending Club yang berskala USD,
# sehingga seluruh input finansial pada UI ditampilkan dalam Rupiah untuk
# kemudahan pengguna, namun dikonversi kembali ke USD sebelum masuk ke
# preprocessor/model agar tetap sesuai skala data training.
USD_TO_IDR = 17957.70


def idr_to_usd(value_idr):
    """Konversi nilai Rupiah dari input pengguna menjadi USD untuk model."""
    return float(value_idr) / USD_TO_IDR


def usd_to_idr(value_usd):
    """Konversi nilai USD (mis. dari default_vals.pkl) menjadi Rupiah untuk ditampilkan di UI."""
    return float(value_usd) * USD_TO_IDR


def format_idr(value_idr):
    """Format angka Rupiah dengan pemisah ribuan gaya Indonesia, mis. Rp 215.492.400."""
    return "Rp " + f"{value_idr:,.0f}".replace(",", ".")


def parse_rupiah_input(raw_text, min_v, max_v, fallback_v):
    """Parse input Rupiah dari text_input (boleh diketik dengan/tanpa titik/koma
    pemisah ribuan), lalu dibatasi ke rentang [min_v, max_v]. Dipakai sebagai
    pengganti st.number_input untuk field bernilai besar (mis. pendapatan
    tahunan dalam Rupiah), karena input HTML bertipe number dapat kehilangan
    digit pada angka yang sangat besar."""
    if raw_text is None:
        return fallback_v
    cleaned = re.sub(r"[^\d]", "", str(raw_text))
    if cleaned == "":
        return fallback_v
    value = float(cleaned)
    return min(max(value, min_v), max_v)

COLORS = {
    "XGBoost":  "#f59e0b",
    "LightGBM": "#3b82f6",
    "CatBoost": "#10b981",
    ENSEMBLE_MODEL_NAME: "#8b5cf6",
}

# Nilai FALLBACK (mode demo) — hanya dipakai apabila artifact dari notebook
# (model_performance.pkl, baseline_vs_tuned.pkl, feature_importance.pkl) belum
# tersedia di direktori aplikasi. Nilai berikut diambil dari hasil evaluasi
# test set pada Penelitian_Ilmiah_-_Final.ipynb (Cell 179 & Cell 171) per
# tanggal commit terakhir, sebagai jaring pengaman, BUKAN sumber utama.
_MODEL_PERF_FALLBACK = pd.DataFrame([
    {"Model": ENSEMBLE_MODEL_NAME, "Accuracy": 0.6927, "Precision": 0.3492, "Recall": 0.6148, "F1 Score": 0.4454, "ROC AUC": 0.7278, "PR AUC": 0.4029},
    {"Model": "XGBoost",  "Accuracy": 0.6630, "Precision": 0.3316, "Recall": 0.6684, "F1 Score": 0.4433, "ROC AUC": 0.7280, "PR AUC": 0.4027},
    {"Model": "LightGBM", "Accuracy": 0.6864, "Precision": 0.3448, "Recall": 0.6246, "F1 Score": 0.4443, "ROC AUC": 0.7272, "PR AUC": 0.4020},
    {"Model": "CatBoost", "Accuracy": 0.7033, "Precision": 0.3556, "Recall": 0.5884, "F1 Score": 0.4433, "ROC AUC": 0.7270, "PR AUC": 0.4013},
]).set_index("Model")

_BASELINE_PERF_FALLBACK = pd.DataFrame([
    {"Model": "XGBoost",  "Baseline F1": 0.1933, "Optimized F1": 0.4433, "Baseline Recall": 0.1184, "Optimized Recall": 0.6684, "Baseline ROC-AUC": 0.7159, "Optimized ROC-AUC": 0.7280},
    {"Model": "LightGBM", "Baseline F1": 0.1469, "Optimized F1": 0.4443, "Baseline Recall": 0.0844, "Optimized Recall": 0.6246, "Baseline ROC-AUC": 0.7239, "Optimized ROC-AUC": 0.7272},
    {"Model": "CatBoost", "Baseline F1": 0.1778, "Optimized F1": 0.4433, "Baseline Recall": 0.1057, "Optimized Recall": 0.5884, "Baseline ROC-AUC": 0.7262, "Optimized ROC-AUC": 0.7270},
]).set_index("Model")

_TOP_FEATURES_FALLBACK = [
    ("int_rate", 0.111784),
    ("dti", 0.035076),
    ("dti_loan_to_income_interaction", 0.031109),
    ("avg_cur_bal", 0.030156),
    ("installment_to_loan", 0.029923),
    ("loan_to_installment", 0.028032),
    ("acc_open_past_24mths", 0.027990),
    ("tot_hi_cred_lim", 0.027390),
    ("revol_util_open_acc_interaction", 0.026423),
    ("income_per_open_acc", 0.026247),
]

# Fitur numerik & kategorikal utama yang dipakai pada model final (fallback).
# Daftar ini sudah konsisten dengan hasil seleksi 45 fitur final pada subbagian
# 3.4.6 (38 numerik + 7 kategorikal); tetap dipertahankan sebagai deskripsi teks
# karena tidak tersimpan otomatis dalam bentuk deskripsi pada artifact notebook.
NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "credit_history_length", "acc_open_past_24mths", "bc_open_to_buy", "num_tl_op_past_12m",
    "avg_cur_bal", "tot_hi_cred_lim", "total_bc_limit", "num_actv_rev_tl",
    "num_rev_tl_bal_gt_0", "percent_bc_gt_75", "bc_util", "loan_to_income",
    "installment_to_income", "revol_bal_to_income", "loan_to_installment", "open_acc_ratio",
    "revol_bal_per_open_acc", "installment_to_loan", "inq_per_credit_history",
    "delinq_per_credit_history", "income_per_open_acc", "revol_util_open_acc_interaction",
    "dti_loan_to_income_interaction", "log_loan_amnt", "log_annual_inc", "log_revol_bal",
]

CATEGORICAL_FEATURES = [
    "term", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "addr_state",
]

FEATURE_DESCRIPTIONS = {
    "loan_amnt": "Jumlah pinjaman yang diajukan (Rupiah)",
    "term": "Tenor pinjaman (36 atau 60 bulan)",
    "int_rate": "Suku bunga pinjaman (%)",
    "installment": "Cicilan bulanan (Rupiah)",
    "sub_grade": "Sub-grade risiko kredit (A1–G5)",
    "emp_length": "Lama bekerja peminjam",
    "home_ownership": "Status kepemilikan rumah",
    "annual_inc": "Pendapatan tahunan peminjam (Rupiah)",
    "verification_status": "Status verifikasi pendapatan",
    "purpose": "Tujuan pinjaman",
    "addr_state": "Negara bagian alamat peminjam",
    "dti": "Debt-to-Income ratio (%)",
    "delinq_2yrs": "Jumlah keterlambatan pembayaran dalam 2 tahun terakhir",
    "inq_last_6mths": "Jumlah inquiry kredit 6 bulan terakhir",
    "open_acc": "Jumlah akun kredit aktif",
    "revol_bal": "Total saldo revolving kredit (Rupiah)",
    "revol_util": "Tingkat pemanfaatan revolving (%)",
    "total_acc": "Total akun kredit yang pernah dimiliki",
    "mort_acc": "Jumlah akun hipotek",
    "credit_history_length": "Lama riwayat kredit (tahun)",
    "acc_open_past_24mths": "Jumlah akun terbuka 24 bulan terakhir",
    "bc_open_to_buy": "Batas kredit terbuka untuk dibeli",
    "num_tl_op_past_12m": "Jumlah akun kredit yang dibuka 12 bulan terakhir",
    "avg_cur_bal": "Rata-rata saldo rekening saat ini",
    "tot_hi_cred_lim": "Total batas kredit yang tersedia",
    "total_bc_limit": "Total batas kredit bankcard",
    "num_actv_rev_tl": "Jumlah akun revolving aktif",
    "num_rev_tl_bal_gt_0": "Jumlah saldo revolving > 0",
    "percent_bc_gt_75": "Persentase batas kredit bankcard > 75%",
    "bc_util": "Tingkat pemanfaatan bankcard",
    "loan_to_income": "Rasio jumlah pinjaman terhadap pendapatan",
    "installment_to_income": "Rasio cicilan terhadap pendapatan",
    "revol_bal_to_income": "Rasio saldo revolving terhadap pendapatan",
    "loan_to_installment": "Rasio pinjaman terhadap cicilan",
    "open_acc_ratio": "Rasio akun kredit aktif terhadap total akun",
    "revol_bal_per_open_acc": "Saldo revolving per akun terbuka",
    "installment_to_loan": "Rasio cicilan terhadap pinjaman",
    "inq_per_credit_history": "Jumlah inquiry per panjang riwayat kredit",
    "delinq_per_credit_history": "Jumlah delinq per panjang riwayat kredit",
    "income_per_open_acc": "Pendapatan per akun kredit aktif",
    "revol_util_open_acc_interaction": "Interaksi revolving utilization dan open acc ratio",
    "dti_loan_to_income_interaction": "Interaksi DTI dan loan-to-income",
    "log_loan_amnt": "Logaritma pinjaman",
    "log_annual_inc": "Logaritma pendapatan tahunan",
    "log_revol_bal": "Logaritma saldo revolving",
}

# ─────────────────────────── PERFORMANCE ARTIFACT LOADING ───────────────────
# Membaca ringkasan performa & feature importance langsung dari hasil notebook
# (bukan angka yang ditulis manual di kode), agar aplikasi otomatis mengikuti
# hasil pelatihan terbaru setiap kali notebook dijalankan ulang dan artifact
# disalin ke direktori aplikasi.
@st.cache_resource(show_spinner=False)
def load_performance_artifacts():
    import joblib

    model_perf = _MODEL_PERF_FALLBACK
    baseline_perf = _BASELINE_PERF_FALLBACK
    top_features = _TOP_FEATURES_FALLBACK
    metadata = None
    perf_status = []

    for fname in ["model_performance.pkl", "outputs/model_performance.pkl"]:
        if os.path.exists(fname):
            try:
                final_models = joblib.load(fname)
                model_perf = final_models.set_index("Model")[
                    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]
                ]
                perf_status.append(("ok", f"✅ {fname} dimuat (performa realtime dari notebook)"))
                break
            except Exception as exc:
                perf_status.append(("warn", f"⚠️ {fname}: {exc} — memakai nilai fallback"))
    else:
        perf_status.append(("warn", "⚠️ model_performance.pkl tidak ditemukan — memakai nilai fallback demo"))

    for fname in ["baseline_vs_tuned.pkl", "outputs/baseline_vs_tuned.pkl"]:
        if os.path.exists(fname):
            try:
                evaluation_results = joblib.load(fname)
                baseline_perf = evaluation_results.rename(columns={
                    "Tuned F1": "Optimized F1", "Tuned Recall": "Optimized Recall",
                    "Tuned ROC-AUC": "Optimized ROC-AUC",
                }).set_index("Model")[
                    ["Baseline F1", "Optimized F1", "Baseline Recall", "Optimized Recall",
                     "Baseline ROC-AUC", "Optimized ROC-AUC"]
                ]
                perf_status.append(("ok", f"✅ {fname} dimuat (performa realtime dari notebook)"))
                break
            except Exception as exc:
                perf_status.append(("warn", f"⚠️ {fname}: {exc} — memakai nilai fallback"))
    else:
        perf_status.append(("warn", "⚠️ baseline_vs_tuned.pkl tidak ditemukan — memakai nilai fallback demo"))

    for fname in ["feature_importance.pkl", "outputs/feature_importance.pkl"]:
        if os.path.exists(fname):
            try:
                fi_df = joblib.load(fname)
                top_features = list(zip(fi_df["Feature"], fi_df["Importance"]))[:10]
                perf_status.append(("ok", f"✅ {fname} dimuat (feature importance realtime dari notebook)"))
                break
            except Exception as exc:
                perf_status.append(("warn", f"⚠️ {fname}: {exc} — memakai nilai fallback"))
    else:
        perf_status.append(("warn", "⚠️ feature_importance.pkl tidak ditemukan — memakai nilai fallback demo"))

    for fname in ["model_metadata.pkl", "outputs/model_metadata.pkl"]:
        if os.path.exists(fname):
            try:
                metadata = joblib.load(fname)
                perf_status.append(("ok", f"✅ {fname} dimuat"))
                break
            except Exception as exc:
                perf_status.append(("warn", f"⚠️ {fname}: {exc}"))

    return model_perf, baseline_perf, top_features, metadata, perf_status


MODEL_PERF, BASELINE_PERF, TOP_FEATURES, MODEL_METADATA, _PERF_STATUS = load_performance_artifacts()

# Jumlah fitur numerik/kategorikal untuk ditampilkan di halaman "Informasi Model" —
# diambil dari model_metadata.pkl bila tersedia, jika tidak jatuh ke panjang list fallback.
if MODEL_METADATA is not None:
    _N_NUMERIC = len(MODEL_METADATA.get("numeric_features", NUMERIC_FEATURES))
    _N_CATEGORICAL = len(MODEL_METADATA.get("categorical_features", CATEGORICAL_FEATURES))
else:
    _N_NUMERIC = len(NUMERIC_FEATURES)
    _N_CATEGORICAL = len(CATEGORICAL_FEATURES)
_N_FEATURES_TOTAL = _N_NUMERIC + _N_CATEGORICAL

# ─────────────────────────── MODEL LOADING ──────────────────────────────────
def _load_artifact_from_candidates(candidate_paths, loader, missing_message=None):
    """Mencoba memuat artifact dari beberapa path kandidat.

    Mengembalikan tuple (objek, path_terpakai, error) dengan prioritas:
    - jika ada path yang ada dan berhasil dimuat, kembalikan objek itu;
    - jika ada path yang ada tetapi gagal dimuat, kembalikan error terakhir;
    - jika tidak ada path yang ada, kembalikan None dan pesan tidak ditemukan.
    """
    last_error = None
    for fname in candidate_paths:
        if not os.path.exists(fname):
            continue
        try:
            return loader(fname), fname, None
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        return None, None, str(last_error)
    return None, None, missing_message


@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    import joblib
    models = {name: None for name in MODEL_NAMES}
    preprocessor = None
    default_vals = None
    thresholds = {name: 0.5 for name in MODEL_NAMES}
    ensemble_weights = {}
    status = []

    # Preprocessor
    preprocessor, preprocessor_path, preprocessor_error = _load_artifact_from_candidates(
        ["preprocessor.pkl", "outputs/preprocessor.pkl"],
        joblib.load,
    )
    if preprocessor is not None:
        status.append(("ok", f"✅ {preprocessor_path} dimuat"))
    elif preprocessor_error is not None:
        status.append(("info", "ℹ️ Preprocessor tidak tersedia; prediksi memakai fallback heuristik"))
    else:
        status.append(("info", "ℹ️ preprocessor.pkl tidak ditemukan; prediksi memakai fallback heuristik"))

    # Default values
    default_vals, default_vals_path, default_vals_error = _load_artifact_from_candidates(
        ["default_values.pkl", "outputs/default_values.pkl"],
        joblib.load,
    )
    if default_vals is not None:
        status.append(("ok", f"✅ {default_vals_path} dimuat"))
    elif default_vals_error is not None:
        status.append(("warn", f"⚠️ {default_vals_error}"))
    else:
        status.append(("warn", "⚠️ default_values.pkl tidak ditemukan"))

    # Thresholds
    loaded_thresholds, thresholds_path, thresholds_error = _load_artifact_from_candidates(
        ["model_thresholds.pkl", "outputs/model_thresholds.pkl"],
        joblib.load,
    )
    if loaded_thresholds is not None:
        thresholds.update({
            name: float(value)
            for name, value in loaded_thresholds.items()
            if name in thresholds
        })
        status.append(("ok", f"✅ {thresholds_path} dimuat"))
    elif thresholds_error is not None:
        status.append(("warn", f"⚠️ {thresholds_error}"))
    else:
        status.append(("warn", "⚠️ model_thresholds.pkl tidak ditemukan"))

    # Base models
    model_files = {
        "XGBoost": ["xgb_model.pkl", "outputs/xgb_model.pkl"],
        "LightGBM": ["lgbm_model.pkl", "outputs/lgbm_model.pkl"],
        "CatBoost": ["cat_model.pkl", "outputs/cat_model.pkl"],
    }
    ensemble_model_bundle = None
    loaded_bundle, ensemble_path, ensemble_error = _load_artifact_from_candidates(
        ["ensemble_models.pkl", "outputs/ensemble_models.pkl"],
        joblib.load,
    )
    if loaded_bundle is not None:
        if isinstance(loaded_bundle, dict):
            ensemble_model_bundle = loaded_bundle
            status.append(("ok", f"✅ {ensemble_path} dimuat"))
        else:
            status.append(("warn", f"⚠️ {ensemble_path} tidak berisi bundle model"))
    elif ensemble_error is not None:
        status.append(("warn", f"⚠️ {ensemble_error}"))

    for mname, paths in model_files.items():
        loaded_model, model_path, model_error = _load_artifact_from_candidates(paths, joblib.load)
        if loaded_model is not None:
            models[mname] = loaded_model
            status.append(("ok", f"✅ {mname} dimuat dari {model_path}"))
        elif any(os.path.exists(path) for path in paths):
            status.append(("err", f"❌ {mname} gagal dimuat: {model_error}"))
        elif models[mname] is None and isinstance(ensemble_model_bundle, dict):
            bundled_model = ensemble_model_bundle.get(mname)
            if bundled_model is not None:
                models[mname] = bundled_model
                status.append(("ok", f"✅ {mname} dimuat dari ensemble_models.pkl"))
            else:
                status.append(("warn", f"⚠️ {mname}: model tidak tersedia pada ensemble_models.pkl"))
        else:
            status.append(("warn", f"⚠️ {mname}: file .pkl tidak ditemukan"))

    # Ensemble weights
    loaded_weights, weights_path, weights_error = _load_artifact_from_candidates(
        ["ensemble_weights.pkl", "outputs/ensemble_weights.pkl"],
        joblib.load,
    )
    if loaded_weights is not None:
        ensemble_weights.update({
            name: float(value)
            for name, value in loaded_weights.items()
            if name in ["XGBoost", "LightGBM", "CatBoost"]
        })
        status.append(("ok", f"✅ {weights_path} dimuat"))
    elif weights_error is not None:
        status.append(("warn", f"⚠️ {weights_error}"))

    return models, preprocessor, default_vals, thresholds, ensemble_weights, status


def heuristic_prob(row: dict) -> float:
    """Heuristik sederhana berbasis karakteristik peminjam."""
    risk = 0.15

    grade = str(row.get("sub_grade", row.get("grade", "C"))).upper()
    grade_key = grade[0] if grade else "C"
    grade_risk = {"A": -0.10, "B": -0.05, "C": 0.0, "D": 0.08, "E": 0.15, "F": 0.22, "G": 0.28}
    risk += grade_risk.get(grade_key, 0.0)

    dti = float(row.get("dti", 15))
    if dti > 30:  risk += 0.12
    elif dti > 20: risk += 0.06

    int_rate = float(row.get("int_rate", 12))
    if int_rate > 20: risk += 0.10
    elif int_rate > 15: risk += 0.05

    term = str(row.get("term", "36 months"))
    if "60" in term: risk += 0.08

    revol_util = float(row.get("revol_util", 50))
    if revol_util > 80: risk += 0.08
    elif revol_util > 60: risk += 0.04

    pub_rec = float(row.get("pub_rec", 0))
    if pub_rec > 0: risk += 0.05 * min(pub_rec, 3)

    emp_length = str(row.get("emp_length", "")).lower()
    if "< 1" in emp_length or "1 year" in emp_length:
        risk += 0.03

    return float(np.clip(risk + np.random.uniform(-0.03, 0.03), 0.05, 0.95))


def default_values_to_dict(default_vals):
    """Ubah artifact default_values dari notebook menjadi dictionary."""
    if isinstance(default_vals, pd.DataFrame):
        required_cols = {"feature", "default_value"}
        if required_cols.issubset(default_vals.columns):
            return dict(zip(default_vals["feature"], default_vals["default_value"]))

    if isinstance(default_vals, dict):
        return default_vals.copy()

    return {}


def get_widget_default(feature, default_vals, fallback):
    """Ambil nilai default fitur dari artifact training jika tersedia."""
    defaults = default_values_to_dict(default_vals) if default_vals is not None else {}
    if feature in defaults and defaults[feature] is not None:
        value = defaults[feature]
        if isinstance(fallback, float):
            return float(value)
        if isinstance(fallback, int):
            return int(value)
        return value
    return fallback


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_divide(numerator, denominator):
    numerator = safe_float(numerator)
    denominator = safe_float(denominator)

    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan

    return numerator / denominator


def prepare_model_input(row_dict: dict, default_vals=None):
    """Selaraskan input aplikasi dengan fitur yang dipakai notebook."""
    model_row = default_values_to_dict(default_vals)
    # Jangan biarkan nilai kosong/NaN dari CSV menimpa default dari notebook.
    model_row.update({
        key: value
        for key, value in row_dict.items()
        if value is not None and not pd.isna(value)
    })

    loan_amnt = model_row.get("loan_amnt")
    installment = model_row.get("installment")
    annual_inc = model_row.get("annual_inc")
    open_acc = model_row.get("open_acc")
    total_acc = model_row.get("total_acc")
    revol_bal = model_row.get("revol_bal")
    revol_util = model_row.get("revol_util")
    dti = model_row.get("dti")
    inq_last_6mths = model_row.get("inq_last_6mths")
    delinq_2yrs = model_row.get("delinq_2yrs")
    credit_history_length = model_row.get("credit_history_length")

    model_row["loan_to_income"] = safe_divide(loan_amnt, annual_inc)
    model_row["installment_to_income"] = safe_divide(safe_float(installment) * 12, annual_inc)
    model_row["revol_bal_to_income"] = safe_divide(revol_bal, annual_inc)
    model_row["loan_to_installment"] = safe_divide(loan_amnt, installment)
    model_row["open_acc_ratio"] = safe_divide(open_acc, total_acc)
    model_row["revol_bal_per_open_acc"] = safe_divide(revol_bal, open_acc)
    model_row["installment_to_loan"] = safe_divide(installment, loan_amnt)
    model_row["inq_per_credit_history"] = safe_divide(inq_last_6mths, credit_history_length)
    model_row["delinq_per_credit_history"] = safe_divide(delinq_2yrs, credit_history_length)
    model_row["income_per_open_acc"] = safe_divide(annual_inc, open_acc)
    model_row["revol_util_open_acc_interaction"] = (
        safe_float(revol_util) * safe_float(model_row.get("open_acc_ratio"))
    )
    model_row["dti_loan_to_income_interaction"] = (
        safe_float(dti) * safe_float(model_row.get("loan_to_income"))
    )
    model_row["log_loan_amnt"] = np.log1p(max(safe_float(loan_amnt, 0), 0))
    model_row["log_annual_inc"] = np.log1p(max(safe_float(annual_inc, 0), 0))
    model_row["log_revol_bal"] = np.log1p(max(safe_float(revol_bal, 0), 0))

    return model_row


def predict_with_model(model, preprocessor, row_dict: dict, default_vals=None):
    """Prediksi probabilitas default untuk satu model.

    Jika preprocessor tidak bisa dipakai (mis. artifact pickle dari versi sklearn yang berbeda),
    fungsi ini akan kembali ke heuristik sederhana agar aplikasi tetap berjalan.
    """
    if preprocessor is None:
        return heuristic_prob(row_dict)

    try:
        model_input = prepare_model_input(row_dict, default_vals)
        df_input = pd.DataFrame([model_input])
        X_processed = preprocessor.transform(df_input)
        return float(model.predict_proba(X_processed)[0][1])
    except Exception:
        return heuristic_prob(row_dict)


def predict_row(
    model_name: str,
    model,
    preprocessor,
    row_dict: dict,
    default_vals=None,
    threshold=0.5,
    ensemble_models=None,
    ensemble_weights=None,
):
    """Prediksi satu baris data."""
    if preprocessor is None:
        prob = heuristic_prob(row_dict)
        return int(prob >= threshold), prob, True

    if model_name == ENSEMBLE_MODEL_NAME:
        try:
            weights = []
            probs = []
            for base_name in ["XGBoost", "LightGBM", "CatBoost"]:
                base_model = ensemble_models.get(base_name) if ensemble_models else None
                weight = float(ensemble_weights.get(base_name, 0.0)) if ensemble_weights else 0.0
                if base_model is None or weight <= 0:
                    continue
                probs.append(predict_with_model(base_model, preprocessor, row_dict, default_vals) * weight)
                weights.append(weight)
            if not probs or not weights:
                raise ValueError("Ensemble weights tidak tersedia")
            prob = sum(probs) / sum(weights)
            return int(prob >= threshold), float(prob), False
        except Exception:
            prob = heuristic_prob(row_dict)
            return int(prob >= threshold), prob, True

    if model is None:
        prob = heuristic_prob(row_dict)
        return int(prob >= threshold), prob, True

    try:
        prob = predict_with_model(model, preprocessor, row_dict, default_vals)
        return int(prob >= threshold), prob, False
    except Exception:
        prob = heuristic_prob(row_dict)
        return int(prob >= threshold), prob, True


# ─────────────────────────── EXTRA CSS (friendly UI additions) ─────────────
st.markdown("""
<style>
    .pred-warn {
        background: linear-gradient(135deg, #451a03, #78350f);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #fde68a;
        box-shadow: 0 4px 6px rgba(245,158,11,0.2);
    }
    .pred-warn h2, .pred-warn h3 { color: #fbbf24; margin-bottom: 0.5rem; }

    .step-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 0.5rem;
        min-height: 150px;
    }
    .step-card .step-icon { font-size: 1.6rem; }
    .step-card .step-title { font-weight: 700; color: #f1f5f9; margin: 0.4rem 0; font-size: 1.05rem; }
    .step-card .step-desc { font-size: 0.85rem; color: #94a3b8; line-height: 1.45; }
</style>
""", unsafe_allow_html=True)


def friendly_verdict(prob: float):
    """Terjemahkan probabilitas menjadi kartu hasil yang mudah dipahami orang awam."""
    if prob < 0.3:
        return (
            "Risiko Rendah", "pred-safe", "✅",
            "Profil calon peminjam ini mirip dengan pinjaman-pinjaman yang selama ini "
            "berhasil dilunasi dengan lancar."
        )
    elif prob < 0.6:
        return (
            "Risiko Sedang", "pred-warn", "⚠️",
            "Ada beberapa faktor pada profil ini yang perlu diperhatikan lebih lanjut "
            "sebelum keputusan diambil."
        )
    else:
        return (
            "Risiko Tinggi", "pred-default", "🔴",
            "Profil calon peminjam ini memiliki banyak kemiripan dengan pinjaman yang "
            "berakhir gagal bayar pada data historis."
        )


# ─────────────────────────── SIDEBAR ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Cek Risiko Pinjaman")
    st.markdown("---")
    page = st.radio(
        "Menu",
        ["🏠 Beranda",
         "🔍 Cek Risiko Pinjaman",
         "📁 Cek Banyak Data (CSV)",
         "📖 Tentang Aplikasi"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("© 2026 — Muhammad Daffa Alghifari")


# ════════════════════════════════════════════════════════════════════════════
#  HALAMAN: BERANDA
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown('<div class="main-title">🏦 Cek Risiko Pinjaman</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Perkirakan apakah sebuah pinjaman berpotensi lancar atau gagal '
        'dibayar?</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="info-box" style="font-size:1.02rem;line-height:1.7;">
    Sebelum memberikan pinjaman, akan sangat membantu bila kita bisa mengetahui lebih awal
    seberapa besar kemungkinan calon peminjam akan kesulitan membayar cicilannya di kemudian hari.
    <b>Aplikasi ini dibuat untuk membantu memperkirakan hal itu.</b><br><br>
    Anda cukup mengisi beberapa data sederhana tentang calon peminjam seperti jumlah pinjaman,
    pendapatan, dan riwayat kreditnya. Lalu aplikasi akan memberikan perkiraan tingkat risikonya.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Bagaimana cara memakainya?")
    steps = [
        ("1️⃣", "Isi Data Peminjam", "Masukkan jumlah pinjaman, pendapatan, dan beberapa info dasar lain tentang calon peminjam."),
        ("2️⃣", "Klik “Cek Risiko Sekarang”", "Sistem akan membandingkan data tersebut dengan pola dari jutaan data pinjaman sebelumnya."),
        ("3️⃣", "Lihat Hasil & Penjelasannya", "Anda akan mendapat perkiraan risiko (Rendah / Sedang / Tinggi) beserta penjelasan sederhananya."),
    ]
    step_cols = st.columns(3)
    for col, (icon, title, desc) in zip(step_cols, steps):
        col.markdown(f"""
        <div class="step-card">
            <div class="step-icon">{icon}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    _acc_source = MODEL_PERF.loc[ENSEMBLE_MODEL_NAME, "Accuracy"] if ENSEMBLE_MODEL_NAME in MODEL_PERF.index else MODEL_PERF["Accuracy"].mean()
    _correct_guess = round(float(_acc_source) * 100)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="info-box">
        <b>💡 Seberapa bisa dipercaya hasilnya?</b><br><br>
        Sistem ini "belajar" dari data historis lebih dari <b>2 juta pinjaman</b> nyata.
        Dari setiap 100 pinjaman yang diuji, sistem ini menebak dengan benar sekitar
        <b>{_correct_guess} kali</b>.<br><br>
        Karena itu, anggap hasilnya sebagai <b>perkiraan yang membantu</b>, bukan keputusan
        final — tetap gunakan pertimbangan Anda sendiri sebagai pelengkap.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
        <b>🙋 Untuk siapa aplikasi ini?</b><br><br>
        • <b>Petugas / analis kredit</b> yang ingin pemeriksaan cepat atas satu calon peminjam<br>
        • <b>Tim operasional</b> yang perlu memeriksa banyak data pinjaman sekaligus<br>
        • Siapa pun yang ingin tahu, secara sederhana, apa yang membuat sebuah pinjaman
        lebih berisiko dari yang lain
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚀 Mulai Sekarang")
    st.info(
        "👈 Pilih **“🔍 Cek Risiko Pinjaman”** di menu sebelah kiri untuk memeriksa satu calon "
        "peminjam, atau **“📁 Cek Banyak Data (CSV)”** bila Anda ingin memeriksa banyak data "
        "sekaligus dari sebuah file."
    )


# ════════════════════════════════════════════════════════════════════════════
#  HALAMAN: CEK RISIKO PINJAMAN (Prediksi Manual)
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Cek Risiko Pinjaman":
    st.title("🔍 Cek Risiko Pinjaman")
    st.markdown(
        "Isi data calon peminjam di bawah ini, lalu klik tombol **Cek Risiko Sekarang** untuk "
        "melihat perkiraan apakah pinjaman ini berpotensi lancar atau gagal dibayar. "
        "**Catatan:** perkiraan ini berdasarkan kondisi saat pengajuan, bukan setelah pinjaman berjalan."
    )

    models, preprocessor, default_vals, model_thresholds, ensemble_weights, status_msgs = load_models()

    with st.expander("📋 Data Peminjam", expanded=True):
        st.caption("Isi bagian utama di bawah ini. Bagian tambahan bersifat opsional dan sudah terisi otomatis dengan nilai yang umum.")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**💰 Pinjaman & Pendapatan**")
            _loan_amnt_min, _loan_amnt_max = usd_to_idr(500.0), usd_to_idr(40000.0)
            _loan_amnt_default = usd_to_idr(get_widget_default("loan_amnt", default_vals, 12000.0))
            loan_amnt_idr_raw = st.text_input(
                "Jumlah Pinjaman (Rupiah)",
                value=f"{_loan_amnt_default:.0f}",
                help=f"Masukkan angka antara {format_idr(_loan_amnt_min)} – {format_idr(_loan_amnt_max)}",
            )
            loan_amnt_idr = parse_rupiah_input(loan_amnt_idr_raw, _loan_amnt_min, _loan_amnt_max, _loan_amnt_default)
            st.caption(format_idr(loan_amnt_idr))
            loan_amnt = idr_to_usd(loan_amnt_idr)
            int_rate = st.number_input(
                "Suku Bunga (%)",
                min_value=5,
                max_value=31,
                value=int(round(get_widget_default("int_rate", default_vals, 12.5))),
                step=1,
                help="Persentase bunga yang dikenakan atas pinjaman ini per tahun.",
            )
            _installment_min, _installment_max = usd_to_idr(15.0), usd_to_idr(1400.0)
            _installment_default = usd_to_idr(get_widget_default("installment", default_vals, 270.0))
            installment_idr_raw = st.text_input(
                "Cicilan Bulanan (Rupiah)",
                value=f"{_installment_default:.0f}",
                help=f"Masukkan angka antara {format_idr(_installment_min)} – {format_idr(_installment_max)}",
            )
            installment_idr = parse_rupiah_input(installment_idr_raw, _installment_min, _installment_max, _installment_default)
            st.caption(format_idr(installment_idr))
            installment = idr_to_usd(installment_idr)
            term = st.selectbox(
                "Tenor Pinjaman",
                ["36 months", "60 months"],
                index=["36 months", "60 months"].index(get_widget_default("term", default_vals, "36 months")) if get_widget_default("term", default_vals, "36 months") in ["36 months", "60 months"] else 0,
                format_func=lambda x: x.replace("36 months", "36 bulan").replace("60 months", "60 bulan"),
                help="Jangka waktu pengembalian pinjaman: 36 bulan (3 tahun) atau 60 bulan (5 tahun).",
            )
            _annual_inc_min, _annual_inc_max = usd_to_idr(0.0), usd_to_idr(2000000.0)
            _annual_inc_default = usd_to_idr(get_widget_default("annual_inc", default_vals, 65000.0))
            annual_inc_idr_raw = st.text_input(
                "Pendapatan Tahunan (Rupiah)",
                value=f"{_annual_inc_default:.0f}",
                help=f"Masukkan angka antara {format_idr(_annual_inc_min)} – {format_idr(_annual_inc_max)}",
            )
            annual_inc_idr = parse_rupiah_input(annual_inc_idr_raw, _annual_inc_min, _annual_inc_max, _annual_inc_default)
            st.caption(format_idr(annual_inc_idr))
            annual_inc = idr_to_usd(annual_inc_idr)

        with col_b:
            st.markdown("**👤 Profil Peminjam**")
            dti = st.number_input(
                "Rasio Utang terhadap Pendapatan (%)",
                min_value=0,
                max_value=40,
                value=int(round(get_widget_default("dti", default_vals, 15.0))),
                step=1,
                help="Seberapa besar porsi pendapatan yang sudah terpakai untuk membayar utang lain.",
            )
            sub_grade = st.selectbox(
                "Grade Kredit",
                [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)],
                index=[f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)].index(get_widget_default("sub_grade", default_vals, "B3")) if get_widget_default("sub_grade", default_vals, "B3") in [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)] else 0,
                help="A = kualitas kredit terbaik, G = paling berisiko.",
            )
            home_own = st.selectbox(
                "Status Rumah",
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
                index=["RENT", "OWN", "MORTGAGE", "OTHER"].index(get_widget_default("home_ownership", default_vals, "RENT")) if get_widget_default("home_ownership", default_vals, "RENT") in ["RENT", "OWN", "MORTGAGE", "OTHER"] else 0,
                format_func=lambda x: {
                    "RENT": "Sewa/Kontrak",
                    "OWN": "Milik Sendiri",
                    "MORTGAGE": "KPR (Hipotek)",
                    "OTHER": "Lainnya",
                }[x],
                help="Status kepemilikan tempat tinggal calon peminjam saat ini.",
            )
            verif_status = st.selectbox(
                "Verifikasi Pendapatan",
                ["Not Verified", "Verified", "Source Verified"],
                index=["Not Verified", "Verified", "Source Verified"].index(get_widget_default("verification_status", default_vals, "Not Verified")) if get_widget_default("verification_status", default_vals, "Not Verified") in ["Not Verified", "Verified", "Source Verified"] else 0,
                format_func=lambda x: {
                    "Not Verified": "Belum Diverifikasi",
                    "Verified": "Terverifikasi",
                    "Source Verified": "Terverifikasi (Sumber)",
                }[x],
                help="Apakah sumber pendapatan yang dilaporkan peminjam sudah pernah diverifikasi.",
            )
            emp_length = st.selectbox(
                "Lama Bekerja",
                ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
                index=["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"].index(get_widget_default("emp_length", default_vals, "10+ years")) if get_widget_default("emp_length", default_vals, "10+ years") in ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"] else 10,
                format_func=lambda x: x.replace("years", "tahun").replace("year", "tahun"),
                help="Sudah berapa lama calon peminjam bekerja di pekerjaannya saat ini.",
            )

        with col_c:
            st.markdown("**📊 Riwayat Kredit**")
            open_acc = st.number_input(
                "Jumlah Akun Kredit Aktif",
                min_value=0,
                max_value=90,
                value=int(get_widget_default("open_acc", default_vals, 10)),
                help="Jumlah rekening kredit (kartu kredit, pinjaman, dsb.) yang masih aktif digunakan saat ini.",
            )
            _revol_bal_min, _revol_bal_max = usd_to_idr(0.0), usd_to_idr(2500000.0)
            _revol_bal_default = usd_to_idr(get_widget_default("revol_bal", default_vals, 15000.0))
            revol_bal_idr_raw = st.text_input(
                "Total Saldo Kartu Kredit (Rupiah)",
                value=f"{_revol_bal_default:.0f}",
                help=f"Masukkan angka antara {format_idr(_revol_bal_min)} – {format_idr(_revol_bal_max)}",
            )
            revol_bal_idr = parse_rupiah_input(revol_bal_idr_raw, _revol_bal_min, _revol_bal_max, _revol_bal_default)
            st.caption(format_idr(revol_bal_idr))
            revol_bal = idr_to_usd(revol_bal_idr)
            revol_util = st.number_input(
                "Pemakaian Limit Kartu Kredit (%)",
                min_value=0,
                max_value=150,
                value=int(round(get_widget_default("revol_util", default_vals, 45.0))),
                step=1,
                help="Seberapa besar limit kartu kredit yang sudah terpakai.",
            )
            total_acc = st.number_input(
                "Total Akun Kredit (Sepanjang Waktu)",
                min_value=1,
                max_value=150,
                value=int(get_widget_default("total_acc", default_vals, 25)),
                help="Jumlah seluruh rekening kredit yang pernah dimiliki peminjam, termasuk yang sudah ditutup.",
            )
            credit_history_length = st.number_input(
                "Lama Riwayat Kredit (Tahun)",
                min_value=0,
                max_value=50,
                value=int(round(get_widget_default("credit_history_length", default_vals, 15.0))),
                step=1,
                help="Sudah berapa lama peminjam memiliki riwayat penggunaan kredit.",
            )

        with st.expander("🔧 Data Tambahan (opsional)", expanded=False):
            st.caption("Bagian ini tidak wajib diisi; bila dibiarkan, aplikasi memakai nilai umum dari data pelatihan.")
            col_d, col_e, col_f = st.columns(3)
            with col_d:
                delinq_2yrs = st.number_input("Telat Bayar 2 Tahun Terakhir", min_value=0.0, max_value=10.0, value=get_widget_default("delinq_2yrs", default_vals, 0.0), step=1.0, help="Berapa kali peminjam terlambat membayar cicilan/tagihan dalam 2 tahun terakhir.")
                inq_last_6mths = st.number_input("Pengecekan Kredit 6 Bulan Terakhir", min_value=0.0, max_value=20.0, value=get_widget_default("inq_last_6mths", default_vals, 0.0), step=1.0, help="Berapa kali riwayat kredit peminjam dicek oleh pemberi pinjaman lain dalam 6 bulan terakhir.")
                mort_acc = st.number_input("Jumlah Akun KPR", min_value=0, max_value=34, value=int(get_widget_default("mort_acc", default_vals, 2)), help="Jumlah pinjaman KPR/hipotek (cicilan rumah) yang dimiliki peminjam.")
            with col_e:
                acc_open_past_24mths = st.number_input("Akun Baru Dibuka 24 Bulan Terakhir", min_value=0.0, max_value=50.0, value=get_widget_default("acc_open_past_24mths", default_vals, 4.0), step=1.0, help="Jumlah rekening kredit baru yang dibuka peminjam dalam 2 tahun terakhir.")
                _bc_min, _bc_max = usd_to_idr(0.0), usd_to_idr(500000.0)
                _bc_default = usd_to_idr(get_widget_default("bc_open_to_buy", default_vals, 4629.0))
                bc_open_to_buy_idr_raw = st.text_input("Sisa Limit Kartu Kredit (Rupiah)", value=f"{_bc_default:.0f}", help="Sisa limit kartu kredit yang masih bisa dipakai peminjam saat ini.")
                bc_open_to_buy_idr = parse_rupiah_input(bc_open_to_buy_idr_raw, _bc_min, _bc_max, _bc_default)
                st.caption(format_idr(bc_open_to_buy_idr))
                bc_open_to_buy = idr_to_usd(bc_open_to_buy_idr)
                num_tl_op_past_12m = st.number_input("Akun Baru Dibuka 12 Bulan Terakhir", min_value=0.0, max_value=50.0, value=get_widget_default("num_tl_op_past_12m", default_vals, 2.0), step=1.0, help="Jumlah rekening kredit baru yang dibuka peminjam dalam 1 tahun terakhir.")
            with col_f:
                _avg_min, _avg_max = usd_to_idr(0.0), usd_to_idr(1000000.0)
                _avg_default = usd_to_idr(get_widget_default("avg_cur_bal", default_vals, 7433.0))
                avg_cur_bal_idr_raw = st.text_input("Rata-rata Saldo Akun (Rupiah)", value=f"{_avg_default:.0f}", help="Rata-rata saldo yang dimiliki peminjam di seluruh rekening kreditnya.")
                avg_cur_bal_idr = parse_rupiah_input(avg_cur_bal_idr_raw, _avg_min, _avg_max, _avg_default)
                st.caption(format_idr(avg_cur_bal_idr))
                avg_cur_bal = idr_to_usd(avg_cur_bal_idr)
                _tot_min, _tot_max = usd_to_idr(0.0), usd_to_idr(5000000.0)
                _tot_default = usd_to_idr(get_widget_default("tot_hi_cred_lim", default_vals, 112783.5))
                tot_hi_cred_lim_idr_raw = st.text_input("Total Batas Kredit (Rupiah)", value=f"{_tot_default:.0f}", help="Total seluruh batas/limit kredit yang diberikan ke peminjam dari semua rekening.")
                tot_hi_cred_lim_idr = parse_rupiah_input(tot_hi_cred_lim_idr_raw, _tot_min, _tot_max, _tot_default)
                st.caption(format_idr(tot_hi_cred_lim_idr))
                tot_hi_cred_lim = idr_to_usd(tot_hi_cred_lim_idr)
                _totbc_min, _totbc_max = usd_to_idr(0.0), usd_to_idr(1000000.0)
                _totbc_default = usd_to_idr(get_widget_default("total_bc_limit", default_vals, 15000.0))
                total_bc_limit_idr_raw = st.text_input("Total Limit Kartu Kredit (Rupiah)", value=f"{_totbc_default:.0f}", help="Total limit dari seluruh kartu kredit yang dimiliki peminjam.")
                total_bc_limit_idr = parse_rupiah_input(total_bc_limit_idr_raw, _totbc_min, _totbc_max, _totbc_default)
                st.caption(format_idr(total_bc_limit_idr))
                total_bc_limit = idr_to_usd(total_bc_limit_idr)

            col_g, col_h, col_i = st.columns(3)
            with col_g:
                num_actv_rev_tl = st.number_input("Akun Kartu Kredit Aktif", min_value=0.0, max_value=50.0, value=get_widget_default("num_actv_rev_tl", default_vals, 5.0), step=1.0, help="Jumlah kartu kredit (atau kredit serupa) yang masih aktif digunakan.")
                num_rev_tl_bal_gt_0 = st.number_input("Akun dengan Saldo Berjalan", min_value=0.0, max_value=50.0, value=get_widget_default("num_rev_tl_bal_gt_0", default_vals, 5.0), step=1.0, help="Jumlah kartu kredit yang saat ini masih memiliki tagihan berjalan (belum lunas).")
            with col_h:
                percent_bc_gt_75 = st.number_input("% Kartu Kredit Terpakai > 75%", min_value=0.0, max_value=100.0, value=get_widget_default("percent_bc_gt_75", default_vals, 44.4), step=0.1, help="Persentase kartu kredit peminjam yang pemakaiannya sudah di atas 75% dari limitnya.")
                bc_util = st.number_input("Rata-rata Pemakaian Kartu Kredit (%)", min_value=0.0, max_value=100.0, value=get_widget_default("bc_util", default_vals, 63.4), step=0.1, help="Rata-rata persentase limit kartu kredit yang sudah terpakai oleh peminjam.")
            with col_i:
                purpose = st.selectbox(
                    "Tujuan Pinjaman",
                    [
                        "debt_consolidation", "credit_card", "home_improvement", "other",
                        "major_purchase", "small_business", "car", "medical", "vacation",
                        "moving", "house", "wedding", "renewable_energy", "educational",
                    ],
                    index=[
                        "debt_consolidation", "credit_card", "home_improvement", "other",
                        "major_purchase", "small_business", "car", "medical", "vacation",
                        "moving", "house", "wedding", "renewable_energy", "educational",
                    ].index(get_widget_default("purpose", default_vals, "debt_consolidation")) if get_widget_default("purpose", default_vals, "debt_consolidation") in [
                        "debt_consolidation", "credit_card", "home_improvement", "other",
                        "major_purchase", "small_business", "car", "medical", "vacation",
                        "moving", "house", "wedding", "renewable_energy", "educational",
                    ] else 0,
                    help="Tujuan atau rencana penggunaan dana pinjaman ini oleh peminjam.",
                )
                addr_state = st.selectbox(
                    "Negara Bagian (data asal Amerika Serikat)",
                    [
                        "CA", "NY", "TX", "FL", "IL", "NJ", "GA", "PA", "OH", "NC",
                        "MI", "VA", "WA", "CO", "AZ", "MA", "MD", "MN", "IN", "TN",
                        "MO", "WI", "OR", "CT", "SC", "NV", "LA", "AL", "KY", "OK",
                        "UT", "AR", "MS", "KS", "NM", "IA", "NE", "HI", "ID", "MT",
                        "WV", "AK", "RI", "SD", "ND", "DE", "DC", "WY", "VT", "NH", "ME",
                    ],
                    index=[
                        "CA", "NY", "TX", "FL", "IL", "NJ", "GA", "PA", "OH", "NC",
                        "MI", "VA", "WA", "CO", "AZ", "MA", "MD", "MN", "IN", "TN",
                        "MO", "WI", "OR", "CT", "SC", "NV", "LA", "AL", "KY", "OK",
                        "UT", "AR", "MS", "KS", "NM", "IA", "NE", "HI", "ID", "MT",
                        "WV", "AK", "RI", "SD", "ND", "DE", "DC", "WY", "VT", "NH", "ME",
                    ].index(get_widget_default("addr_state", default_vals, "CA")) if get_widget_default("addr_state", default_vals, "CA") in [
                        "CA", "NY", "TX", "FL", "IL", "NJ", "GA", "PA", "OH", "NC",
                        "MI", "VA", "WA", "CO", "AZ", "MA", "MD", "MN", "IN", "TN",
                        "MO", "WI", "OR", "CT", "SC", "NV", "LA", "AL", "KY", "OK",
                        "UT", "AR", "MS", "KS", "NM", "IA", "NE", "HI", "ID", "MT",
                        "WV", "AK", "RI", "SD", "ND", "DE", "DC", "WY", "VT", "NH", "ME",
                    ] else 0,
                    help="Kode negara bagian tempat tinggal peminjam (data pelatihan berasal dari Amerika Serikat).",
                )

    with st.expander("⚙️ Opsi Lanjutan (untuk pengguna teknis)", expanded=False):
        st.caption("Secara default aplikasi memakai model gabungan (ensemble) yang performanya paling seimbang. Ubah di sini bila Anda ingin membandingkan model lain.")
        model_choice = st.radio(
            "Model yang digunakan",
            MODEL_NAMES,
            index=MODEL_NAMES.index(ENSEMBLE_MODEL_NAME),
            horizontal=True,
        )
        st.markdown("**Status Sistem**")
        for level, msg in status_msgs:
            if level == "ok":
                st.success(msg)
            elif level == "warn":
                st.warning(msg)
            elif level == "info":
                st.caption(msg)
            else:
                st.error(msg)

    row = {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "installment": installment,
        "annual_inc": annual_inc,
        "dti": dti,
        "open_acc": open_acc,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "mort_acc": mort_acc,
        "term": term,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_own,
        "verification_status": verif_status,
        "purpose": purpose,
        "addr_state": addr_state,
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "credit_history_length": credit_history_length,
        "acc_open_past_24mths": acc_open_past_24mths,
        "bc_open_to_buy": bc_open_to_buy,
        "num_tl_op_past_12m": num_tl_op_past_12m,
        "avg_cur_bal": avg_cur_bal,
        "tot_hi_cred_lim": tot_hi_cred_lim,
        "total_bc_limit": total_bc_limit,
        "num_actv_rev_tl": num_actv_rev_tl,
        "num_rev_tl_bal_gt_0": num_rev_tl_bal_gt_0,
        "percent_bc_gt_75": percent_bc_gt_75,
        "bc_util": bc_util,
    }

    st.markdown("---")
    if st.button("🔍 Cek Risiko Sekarang", type="primary", width="stretch"):
        label, prob, is_demo = predict_row(
            model_choice,
            models[model_choice],
            preprocessor,
            row,
            default_vals,
            model_thresholds.get(model_choice, 0.5),
            models,
            ensemble_weights,
        )

        level_txt, level_css, level_icon, advice = friendly_verdict(prob)

        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown(f"""
            <div class="{level_css}">
                <h2>{level_icon} {level_txt}</h2>
                <p style="font-size:1.05rem;">{advice}</p>
                <p style="font-size:0.85rem;opacity:0.85;">Perkiraan kemungkinan gagal bayar: <b>{prob:.0%}</b></p>
            </div>""", unsafe_allow_html=True)
            if is_demo:
                st.caption("ℹ️ Model utama belum tersedia di sistem ini, sehingga hasil di atas adalah estimasi awal (mode simulasi).")
            st.caption("Catatan: hasil ini adalah perkiraan berbasis pola data historis, bukan keputusan final. Gunakan sebagai pelengkap, bukan pengganti, pertimbangan Anda.")
        with c2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color="#3b82f6"),
                    steps=[
                        dict(range=[0, 30], color="#dcfce7"),
                        dict(range=[30, 60], color="#fef9c3"),
                        dict(range=[60, 100], color="#fee2e2"),
                    ],
                ),
                number=dict(suffix="%"),
                title=dict(text="Tingkat Risiko"),
            ))
            fig_g.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
#  HALAMAN: CEK BANYAK DATA (CSV)
# ════════════════════════════════════════════════════════════════════════════
elif page == "📁 Cek Banyak Data (CSV)":
    st.title("📁 Cek Banyak Data Sekaligus (CSV)")
    st.markdown(
        "Jika Anda memiliki data banyak calon peminjam sekaligus, unggah file CSV di sini untuk "
        "memeriksa risikonya secara massal — tidak perlu mengisi form satu per satu."
    )

    with st.expander("📥 Langkah 1 — Download Template", expanded=True):
        st.caption(
            "Gunakan file ini sebagai contoh format. Kolom yang berkaitan dengan uang "
            "(jumlah pinjaman, pendapatan, dsb.) diisi dalam Rupiah."
        )
        tpl = pd.DataFrame({
            "loan_amnt": [215492400, 448942500, 89788500, 323238600, 628519500],
            "funded_amnt": [215492400, 448942500, 89788500, 323238600, 628519500],
            "int_rate": [10.5, 18.2, 7.9, 15.0, 22.5],
            "installment": [4848579, 11133774, 2065136, 7811600, 17149604],
            "annual_inc": [1167250500, 1616193000, 808096500, 1292954400, 1975347000],
            "dti": [12.5, 28.3, 8.1, 21.0, 35.2],
            "delinq_2yrs": [0.0, 1.0, 0.0, 0.0, 2.0],
            "inq_last_6mths": [0.0, 1.0, 0.0, 0.0, 2.0],
            "open_acc": [10, 15, 7, 12, 20],
            "pub_rec": [0, 1, 0, 0, 2],
            "revol_bal": [269365500, 808096500, 89788500, 395069400, 1436616000],
            "revol_util": [45.0, 72.0, 20.0, 58.0, 88.0],
            "total_acc": [25, 40, 15, 30, 55],
            "mort_acc": [2, 4, 0, 1, 6],
            "credit_history_length": [15.0, 17.0, 8.0, 12.0, 10.0],
            "acc_open_past_24mths": [4.0, 6.0, 2.0, 3.0, 5.0],
            "bc_open_to_buy": [83126193, 121286306, 22447125, 69675876, 89788500],
            "num_tl_op_past_12m": [2.0, 3.0, 1.0, 2.0, 4.0],
            "avg_cur_bal": [133479584, 194302314, 73626570, 152640450, 215492400],
            "tot_hi_cred_lim": [2025332258, 2783443500, 1167250500, 1759854600, 3052809000],
            "total_bc_limit": [269365500, 413027100, 161619300, 323238600, 448942500],
            "num_actv_rev_tl": [5.0, 7.0, 3.0, 4.0, 6.0],
            "num_rev_tl_bal_gt_0": [5.0, 6.0, 2.0, 3.0, 5.0],
            "percent_bc_gt_75": [44.4, 67.8, 20.0, 55.5, 88.9],
            "bc_util": [63.4, 78.2, 25.5, 56.1, 85.0],
            "term": ["36 months", "60 months", "36 months", "60 months", "60 months"],
            "grade": ["B", "D", "A", "C", "E"],
            "sub_grade": ["B3", "D2", "A1", "C4", "E5"],
            "emp_length": ["10+ years", "2 years", "5 years", "3 years", "1 year"],
            "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT", "MORTGAGE"],
            "verification_status": ["Not Verified", "Verified", "Source Verified", "Not Verified", "Verified"],
            "purpose": ["debt_consolidation", "credit_card", "home_improvement", "debt_consolidation", "other"],
            "addr_state": ["CA", "NY", "TX", "FL", "IL"],
        })
        derived_cols = {
            "loan_to_income": [np.nan] * len(tpl),
            "installment_to_income": [np.nan] * len(tpl),
            "revol_bal_to_income": [np.nan] * len(tpl),
            "loan_to_installment": [np.nan] * len(tpl),
            "open_acc_ratio": [np.nan] * len(tpl),
            "revol_bal_per_open_acc": [np.nan] * len(tpl),
            "installment_to_loan": [np.nan] * len(tpl),
            "inq_per_credit_history": [np.nan] * len(tpl),
            "delinq_per_credit_history": [np.nan] * len(tpl),
            "income_per_open_acc": [np.nan] * len(tpl),
            "revol_util_open_acc_interaction": [np.nan] * len(tpl),
            "dti_loan_to_income_interaction": [np.nan] * len(tpl),
            "log_loan_amnt": [np.nan] * len(tpl),
            "log_annual_inc": [np.nan] * len(tpl),
            "log_revol_bal": [np.nan] * len(tpl),
        }
        for k, v in derived_cols.items():
            tpl[k] = v
        st.download_button(
            "⬇️ Download Template CSV", tpl.to_csv(index=False),
            "loan_default_template.csv", "text/csv", type="primary",
        )
        with st.expander("🔧 Lihat arti setiap kolom (detail teknis)", expanded=False):
            st.markdown("""
            <div class="warn-box">
            <b>⚠️ Syarat file CSV</b>
            <ul>
            <li>Format file harus <code>.csv</code>, baris pertama adalah nama kolom</li>
            <li>Nama kolom harus <b>persis sama</b> dengan template (huruf besar/kecil berpengaruh)</li>
            <li>Gunakan <b>titik</b> untuk angka desimal, bukan koma</li>
            <li>Kolom bernilai uang diisi dalam <b>Rupiah</b></li>
            <li>Jangan sertakan kolom hasil (misalnya <code>loan_status</code>)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"Kolom": k, "Deskripsi": v}
                for k, v in FEATURE_DESCRIPTIONS.items()
            ]).set_index("Kolom"), width="stretch")
        st.dataframe(tpl, width="stretch")

    st.markdown("### 📤 Langkah 2 — Upload File CSV Anda")
    with st.expander("⚙️ Opsi Lanjutan (untuk pengguna teknis)", expanded=False):
        model_batch = st.selectbox(
            "Model yang digunakan",
            MODEL_NAMES,
            index=MODEL_NAMES.index(ENSEMBLE_MODEL_NAME),
        )
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file:
        try:
            df_up = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil dibaca: **{len(df_up)} baris data**, {len(df_up.columns)} kolom")
            with st.expander("👀 Lihat 5 baris pertama", expanded=False):
                st.dataframe(df_up.head(5), width="stretch")

            required_cols = list(FEATURE_DESCRIPTIONS.keys())
            missing_cols = [f for f in required_cols if f not in df_up.columns]
            if missing_cols:
                st.warning(f"⚠️ Beberapa kolom tidak ditemukan dan akan diisi otomatis dengan nilai umum: {missing_cols}")
                for col in missing_cols:
                    df_up[col] = np.nan

            if st.button("🚀 Cek Risiko Semua Data", type="primary"):
                models, preprocessor, default_vals, model_thresholds, ensemble_weights, _ = load_models()
                model_obj = models[model_batch]

                _MONETARY_COLS = [
                    "loan_amnt", "funded_amnt", "installment", "annual_inc",
                    "revol_bal", "bc_open_to_buy", "avg_cur_bal",
                    "tot_hi_cred_lim", "total_bc_limit",
                ]
                df_pred_input = df_up.copy()
                for col in _MONETARY_COLS:
                    if col in df_pred_input.columns:
                        df_pred_input[col] = df_pred_input[col].apply(
                            lambda v: idr_to_usd(v) if pd.notna(v) else v
                        )

                probs, labels = [], []
                bar = st.progress(0, text="Memeriksa data...")
                total = len(df_pred_input)
                for i, (_, r) in enumerate(df_pred_input.iterrows()):
                    lbl, prb, _ = predict_row(
                        model_batch,
                        model_obj,
                        preprocessor,
                        r.to_dict(),
                        default_vals,
                        model_thresholds.get(model_batch, 0.5),
                        models,
                        ensemble_weights,
                    )
                    probs.append(prb)
                    labels.append(lbl)
                    bar.progress((i + 1) / total, text=f"Memeriksa baris {i + 1}/{total}...")
                bar.empty()

                df_res = df_up.copy()
                df_res["default_probability"] = [round(p, 4) for p in probs]
                df_res["prediction"] = labels
                df_res["verdict"] = ["Berpotensi Gagal Bayar" if l else "Berpotensi Lancar" for l in labels]

                default_n = sum(labels)
                st.markdown("### 📊 Hasil")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Data", len(df_res))
                c2.metric("Berpotensi Bermasalah", default_n)
                c3.metric("Berpotensi Aman", len(df_res) - default_n)
                c4.metric("Persentase Bermasalah", f"{default_n / len(df_res):.0%}")

                st.dataframe(
                    df_res[["loan_amnt", "int_rate", "sub_grade", "dti", "default_probability", "verdict"]].rename(columns={
                        "loan_amnt": "Jumlah Pinjaman",
                        "int_rate": "Suku Bunga (%)",
                        "sub_grade": "Grade Kredit",
                        "dti": "Rasio Utang (%)",
                        "default_probability": "Perkiraan Risiko",
                        "verdict": "Kesimpulan",
                    }),
                    width="stretch",
                )
                st.download_button(
                    "⬇️ Download Hasil Lengkap",
                    df_res.to_csv(index=False),
                    "hasil_cek_risiko_pinjaman.csv", "text/csv",
                )
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  HALAMAN: TENTANG APLIKASI
# ════════════════════════════════════════════════════════════════════════════
elif page == "📖 Tentang Aplikasi":
    st.markdown('<div class="main-title">📖 Tentang Aplikasi Ini</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Ringkasan sederhana tentang tujuan dan keandalan sistem ini</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🎯 Apa tujuan aplikasi ini?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-box">
        <b>🎯 Tujuan</b><br><br>
        Aplikasi ini membantu memperkirakan kemungkinan sebuah pinjaman akan gagal dibayar
        (<i>loan default</i>), berdasarkan pola dari <b>Lending Club Loan Data</b> — kumpulan data
        historis lebih dari 2,26 juta pinjaman nyata.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>🏦 Manfaatnya</b><br><br>
        • <b>Lembaga keuangan</b> — mengurangi kerugian akibat kredit macet<br>
        • <b>Peminjam</b> — proses evaluasi kredit lebih transparan<br>
        • <b>Masyarakat umum</b> — akses kredit yang lebih adil dan efisien
        </div>
        """, unsafe_allow_html=True)
    with c2:
        _acc_e = MODEL_PERF.loc[ENSEMBLE_MODEL_NAME, "Accuracy"] if ENSEMBLE_MODEL_NAME in MODEL_PERF.index else MODEL_PERF["Accuracy"].mean()
        _rec_e = MODEL_PERF.loc[ENSEMBLE_MODEL_NAME, "Recall"] if ENSEMBLE_MODEL_NAME in MODEL_PERF.index else MODEL_PERF["Recall"].mean()
        st.markdown(f"""
        <div class="info-box">
        <b>📊 Seberapa akurat sistem ini?</b><br><br>
        Dari 100 pinjaman yang diuji, sistem ini menebak dengan benar sekitar
        <b>{_acc_e:.0%}</b> di antaranya, dan berhasil mengenali sekitar <b>{_rec_e:.0%}</b>
        dari pinjaman yang memang akhirnya gagal dibayar.<br><br>
        Sistem ini sengaja dibuat lebih "waspada" — lebih memilih menandai calon risiko
        daripada melewatkannya — sesuai kebutuhan kehati-hatian di industri keuangan.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>📐 Bagaimana cara membuatnya?</b><br><br>
        Penelitian ini melalui tahapan standar analisis data (metodologi CRISP-DM):
        memahami kebutuhan bisnis → memahami data → menyiapkan data → melatih model →
        mengevaluasi hasil → menerapkan ke aplikasi ini.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🔧 Detail Teknis (untuk developer / peneliti)", expanded=False):
        st.caption("Bagian ini berisi rincian teknis pelatihan & evaluasi model — tidak diperlukan untuk penggunaan sehari-hari.")

        with st.expander("📦 Sumber Data Performa & Feature Importance", expanded=False):
            for level, msg in _PERF_STATUS:
                if level == "ok":
                    st.success(msg)
                elif level == "warn":
                    st.warning(msg)
                else:
                    st.error(msg)

        tech_tabs = st.tabs([
            "🏆 Performa Model", "📈 Distribusi Data", "📉 Kurva Evaluasi",
            "⚙️ Parameter Model", "🖥️ Pipeline & Requirements",
        ])

        # ---- Tab: Performa Model (gabungan Dashboard + Visualisasi t2) ----
        with tech_tabs[0]:
            st.markdown("#### Model Performance Overview")
            metric_tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
            for tab, metric in zip(metric_tabs, ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]):
                with tab:
                    perf_rows = MODEL_PERF.sort_values(metric, ascending=False).reset_index()
                    cols = st.columns(len(perf_rows))
                    for j, prow in perf_rows.iterrows():
                        medal = ["🥇", "🥈", "🥉", "🏅"][j] if j < 4 else f"{j + 1}."
                        cols[j].markdown(f"""
                        <div class="metric-card">
                            <h3>{medal} {metric}</h3>
                            <div class="value">{prow[metric]:.4f}</div>
                            <div class="label">{prow['Model']}</div>
                        </div>""", unsafe_allow_html=True)

            col_r, col_b = st.columns(2)
            with col_r:
                st.markdown("##### Radar Chart")
                cats = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
                fig = go.Figure()
                for mname, mrow in MODEL_PERF.iterrows():
                    vals = [mrow[c] for c in cats] + [mrow[cats[0]]]
                    fig.add_trace(go.Scatterpolar(
                        r=vals, theta=cats + [cats[0]], fill="toself", name=mname,
                        line=dict(color=COLORS[mname], width=2),
                        fillcolor=COLORS[mname], opacity=0.2,
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(
                        visible=True, range=[0.0, 1.0],
                        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        tickfont=dict(size=10, color="#7B7B7B"),
                    )),
                    height=380, paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.18),
                )
                st.plotly_chart(fig, width="stretch")
            with col_b:
                st.markdown("##### Perbandingan Bar")
                fig2 = go.Figure()
                for mname, mrow in MODEL_PERF.iterrows():
                    vals = [mrow[m] for m in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]]
                    fig2.add_trace(go.Bar(
                        name=mname,
                        x=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                        y=vals,
                        marker_color=COLORS[mname],
                        text=[f"{v:.4f}" for v in vals],
                        textposition="outside",
                    ))
                fig2.update_layout(
                    barmode="group", height=380,
                    yaxis=dict(range=[0.0, 1.15]),
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.28),
                )
                st.plotly_chart(fig2, width="stretch")

            st.markdown("##### Tabel Performa Model (Setelah Optimasi)")
            st.caption(
                "Metrik dihitung menggunakan threshold hasil validation set. "
                "Weighted Soft Voting Ensemble dipilih sebagai model terbaik berdasarkan F1 Score tertinggi."
            )
            st.dataframe(
                MODEL_PERF.style.background_gradient(cmap="YlGn", axis=0).format("{:.4f}"),
                width="stretch",
            )

            st.markdown("##### Perbandingan Baseline vs Optimized")
            st.caption(
                "Peningkatan Recall setelah penerapan class weight/scale_pos_weight sangat signifikan "
                "pada ketiga model dasar. Ensemble tidak memiliki baris baseline karena baru dibentuk "
                "setelah proses tuning ketiga model dasar selesai."
            )
            st.dataframe(
                BASELINE_PERF.style.background_gradient(cmap="YlGn", axis=0).format("{:.4f}"),
                width="stretch",
            )

            st.markdown("##### Peningkatan F1 Score & Recall")
            fig_cmp = go.Figure()
            for mname in MODEL_NAMES:
                if mname not in BASELINE_PERF.index:
                    continue
                fig_cmp.add_trace(go.Bar(
                    name=f"{mname} — Baseline",
                    x=["F1 Score", "Recall"],
                    y=[BASELINE_PERF.loc[mname, "Baseline F1"], BASELINE_PERF.loc[mname, "Baseline Recall"]],
                    marker_color=COLORS[mname], opacity=0.4,
                ))
                fig_cmp.add_trace(go.Bar(
                    name=f"{mname} — Optimized",
                    x=["F1 Score", "Recall"],
                    y=[BASELINE_PERF.loc[mname, "Optimized F1"], BASELINE_PERF.loc[mname, "Optimized Recall"]],
                    marker_color=COLORS[mname], opacity=1.0,
                ))
            fig_cmp.update_layout(
                barmode="group", height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.35),
            )
            st.plotly_chart(fig_cmp, width="stretch")

        # ---- Tab: Distribusi Data ----
        with tech_tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Distribusi Loan Status")
                fig = px.bar(
                    x=["Fully Paid", "Charged Off", "Current", "Late", "Default", "Others"],
                    y=[1041952, 261655, 878317, 43726, 11855, 23163],
                    color=["Fully Paid", "Charged Off", "Current", "Late", "Default", "Others"],
                    color_discrete_sequence=["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#64748b"],
                    labels={"x": "Status", "y": "Jumlah"},
                )
                fig.update_layout(showlegend=False, height=320, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.markdown("##### Distribusi Target Biner")
                st.caption("Hanya Fully Paid (0) dan Charged Off (1) yang digunakan sebagai target")
                fig2 = px.pie(
                    values=[1041952, 261655],
                    names=["Fully Paid — Lancar (0)", "Charged Off — Gagal Bayar (1)"],
                    color_discrete_sequence=["#3b82f6", "#ef4444"],
                    hole=0.4,
                )
                fig2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, width="stretch")

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("##### Default Rate per Grade Kredit")
                grades = ["A", "B", "C", "D", "E", "F", "G"]
                default_rates = [0.060, 0.118, 0.165, 0.214, 0.266, 0.318, 0.367]
                fig3 = px.bar(
                    x=grades, y=default_rates,
                    color=default_rates,
                    color_continuous_scale=["#3b82f6", "#f59e0b", "#ef4444"],
                    labels={"x": "Grade", "y": "Default Rate"},
                )
                fig3.update_layout(showlegend=False, height=300, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, width="stretch")
            with c4:
                st.markdown("##### Top Feature Importance (XGBoost)")
                feat_names = [f[0] for f in TOP_FEATURES]
                feat_vals = [f[1] for f in TOP_FEATURES]
                fig4 = go.Figure(go.Bar(
                    x=feat_vals[::-1],
                    y=feat_names[::-1],
                    orientation="h",
                    marker_color="#3b82f6",
                ))
                fig4.update_layout(height=370, paper_bgcolor="rgba(0,0,0,0)",
                                   xaxis_title="Importance Score")
                st.plotly_chart(fig4, width="stretch")

        # ---- Tab: Kurva Evaluasi ----
        with tech_tabs[2]:
            st.markdown("##### Kurva ROC (Simulasi berdasarkan nilai AUC)")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"), name="Random (AUC=0.50)",
            ))
            for mname, auc in [("XGBoost", 0.7365), ("LightGBM", 0.7361), ("CatBoost", 0.7317)]:
                t = np.linspace(0, 1, 300)
                fpr = np.sort(t ** (1 / max(auc * 2.5, 0.01)))
                tpr = np.sort(np.clip(t ** ((1 - auc) * 1.8), 0, 1))
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{mname} (AUC={auc:.4f})",
                    line=dict(color=COLORS[mname], width=2.5),
                ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                height=420, paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_roc, width="stretch")

            st.markdown("##### Simulasi Confusion Matrix (Optimized Models)")
            st.caption("Recall tinggi menunjukkan model lebih baik mendeteksi peminjam berisiko (Charged Off).")
            cm_data = {
                "XGBoost": [[138096, 70295], [16810, 35521]],
                "LightGBM": [[137153, 71238], [16646, 35685]],
                "CatBoost": [[135490, 72901], [16544, 35787]],
            }
            cm_cols = st.columns(3)
            for i, (mname, cm) in enumerate(cm_data.items()):
                with cm_cols[i]:
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Prediksi", y="Aktual"),
                        x=["Fully Paid", "Charged Off"],
                        y=["Fully Paid", "Charged Off"],
                        text_auto=True,
                        color_continuous_scale=["#f8fafc", COLORS[mname]],
                        title=mname,
                    )
                    fig_cm.update_layout(height=280, margin=dict(t=55, b=5, l=5, r=5))
                    st.plotly_chart(fig_cm, width="stretch")

        # ---- Tab: Parameter Model (dari Informasi Model) ----
        with tech_tabs[3]:
            model_tabs = st.tabs(["🟡 XGBoost", "🔵 LightGBM", "🟢 CatBoost"])

            with model_tabs[0]:
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("#### XGBoost Optimized")
                    st.markdown("**Tipe:** Extreme Gradient Boosting")
                    st.markdown("Model gradient boosting berbasis pohon keputusan dengan regularisasi L1/L2 bawaan.")
                    st.markdown("**⚙️ Parameter Utama:**")
                    st.code("""n_estimators     = 700
max_depth        = 5
learning_rate    = 0.037516
subsample        = 0.718293
colsample_bytree = 0.741574
min_child_weight = 7
gamma            = 3.272793
reg_alpha        = 2.291493
reg_lambda       = 11.986861
scale_pos_weight = 3.016943
random_state     = 42""", language="python")
                    st.markdown("**✨ Keunggulan:**")
                    st.markdown("""
                    - **Recall tertinggi antar model tunggal** → 66,84%, paling sensitif terhadap risiko gagal bayar
                    - **Regularisasi bawaan** → Mencegah overfitting pada data tabular
                    - **Inference cepat** → Cocok untuk deployment production
                    - **Feature importance** → Mudah diinterpretasi untuk audit kredit
                    - Salah satu model penyusun **Weighted Soft Voting Ensemble** (model pilihan akhir)
                    """)
                with c2:
                    st.markdown("**📊 Performa Aktual**")
                    for m, v in MODEL_PERF.loc["XGBoost"].items():
                        st.metric(m, f"{v:.4f}")

            with model_tabs[1]:
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("#### LightGBM Optimized")
                    st.markdown("**Tipe:** Light Gradient Boosting Machine")
                    st.markdown("Gradient boosting berbasis histogram yang dioptimalkan untuk kecepatan dan efisiensi memori.")
                    st.markdown("**⚙️ Parameter Utama:**")
                    st.code("""n_estimators      = 700
learning_rate     = 0.032917
max_depth         = 5
num_leaves        = 18
subsample         = 0.705968
colsample_bytree  = 0.795486
min_child_samples = 219
reg_alpha         = 2.773426
reg_lambda        = 11.589733
scale_pos_weight  = 2.926106
random_state      = 42""", language="python")
                    st.markdown("**✨ Keunggulan:**")
                    st.markdown("""
                    - **F1 Score tertinggi antar model tunggal** → 0,4443 (validation: 0,4446)
                    - **Recall kompetitif** → 62,46%, hampir setara XGBoost
                    - **Pelatihan lebih cepat** → Histogram-based splitting
                    - **Efisiensi memori** → Cocok untuk dataset besar
                    - Salah satu model penyusun **Weighted Soft Voting Ensemble** (bobot terbesar, 60%)
                    """)
                with c2:
                    st.markdown("**📊 Performa Aktual**")
                    for m, v in MODEL_PERF.loc["LightGBM"].items():
                        st.metric(m, f"{v:.4f}")

            with model_tabs[2]:
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("#### CatBoost Optimized")
                    st.markdown("**Tipe:** Categorical Boosting")
                    st.markdown("Gradient boosting dengan penanganan fitur kategorikal bawaan menggunakan Ordered Boosting.")
                    st.markdown("**⚙️ Parameter Utama:**")
                    st.code("""iterations           = 700
depth                = 6
learning_rate        = 0.042082
l2_leaf_reg          = 6.825588
bagging_temperature  = 0.223182
random_strength      = 1.554043
loss_function        = 'Logloss'
eval_metric          = 'F1'
class_weights        = [1.0, 3.0678]
random_state         = 42
verbose              = 0""", language="python")
                    st.markdown("**✨ Keunggulan:**")
                    st.markdown("""
                    - **Accuracy tertinggi antar model tunggal** → 70,33%
                    - **Penanganan kategorik bawaan** → Tidak perlu encoding manual
                    - **Ordered Boosting** → Mengurangi target leakage selama training
                    - **Stabil** → Performa konsisten lintas fold
                    - Salah satu model penyusun **Weighted Soft Voting Ensemble** (bobot 40%)
                    """)
                with c2:
                    st.markdown("**📊 Performa Aktual**")
                    for m, v in MODEL_PERF.loc["CatBoost"].items():
                        st.metric(m, f"{v:.4f}")

            st.markdown("---")
            st.markdown("##### 📌 Daftar Fitur Model")
            st.caption(
                f"{_N_FEATURES_TOTAL} fitur final sebelum encoding "
                f"({_N_NUMERIC} numerik + {_N_CATEGORICAL} kategorikal) setelah feature engineering, "
                "seleksi fitur, dan penghapusan data leakage. Setelah One-Hot Encoding, "
                "jumlah kolom aktual yang masuk ke model bertambah menjadi 179 kolom."
            )
            st.dataframe(pd.DataFrame([
                {
                    "No": i + 1,
                    "Fitur": k,
                    "Tipe": "float64" if k in NUMERIC_FEATURES else "object/encoded",
                    "Deskripsi": v,
                }
                for i, (k, v) in enumerate(FEATURE_DESCRIPTIONS.items())
            ]).set_index("No"), width="stretch")

        # ---- Tab: Pipeline & Requirements ----
        with tech_tabs[4]:
            st.markdown("##### 🔧 Preprocessing Pipeline")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="info-box"><b>🔢 Fitur Numerik</b><br><br>
                Pipeline: <code>SimpleImputer(strategy='median', add_indicator=True)</code><br><br>
                • Missing value diisi dengan nilai median<br>
                • Tidak dilakukan normalisasi (boosting tidak memerlukannya)<br>
                • {_N_NUMERIC} fitur numerik setelah seleksi
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="info-box"><b>🏷️ Fitur Kategorikal</b><br><br>
                Pipeline: <code>SimpleImputer(fill='Unknown')</code> + <code>OneHotEncoder</code><br><br>
                • Missing value diisi dengan <i>"Unknown"</i><br>
                • Dikodekan dengan <code>OneHotEncoder(handle_unknown='ignore')</code><br>
                • {_N_CATEGORICAL} fitur kategorikal, kategori tak dikenal saat inferensi diabaikan (bukan error)
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### 🖥️ System Requirements")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="info-box"><b>🐍 Python & Core</b><br><br>
                • Python 3.9+<br>
                • streamlit >= 1.28<br>
                • pandas >= 1.5<br>
                • numpy >= 1.23<br>
                • plotly >= 5.0<br>
                • scikit-learn >= 1.2<br>
                • joblib >= 1.2
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="info-box"><b>🤖 Model Libraries</b><br><br>
                • xgboost >= 1.7<br>
                • lightgbm >= 3.3<br>
                • catboost >= 1.2<br>
                • optuna >= 3.0<br>
                • imbalanced-learn >= 0.11<br>
                • shap >= 0.41
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### 📁 File yang Diperlukan")
            st.markdown("""
            <div class="warn-box">
            Letakkan file berikut di direktori yang sama dengan <code>app.py</code>:
            <ul>
            <li><code>preprocessor.pkl</code> — Pipeline preprocessing (ColumnTransformer)</li>
            <li><code>xgb_model.pkl</code> — Model XGBoost terlatih</li>
            <li><code>lgbm_model.pkl</code> — Model LightGBM terlatih</li>
            <li><code>cat_model.pkl</code> — Model CatBoost terlatih</li>
            <li><code>default_values.pkl</code> — Nilai default fitur (opsional)</li>
            </ul>
            Jika file tidak tersedia, aplikasi berjalan dalam <b>mode demo (heuristik)</b>.
            </div>
            """, unsafe_allow_html=True)