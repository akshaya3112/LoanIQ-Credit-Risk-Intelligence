import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="LoanIQ · Smart Loan Decisions",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #060A14;
  --surface: #0D1525;
  --surface2: #111C30;
  --border: rgba(99,179,237,0.12);
  --border2: rgba(99,179,237,0.22);
  --accent: #3B82F6;
  --accent2: #06B6D4;
  --green: #10B981;
  --red: #F43F5E;
  --yellow: #F59E0B;
  --text: #E2E8F0;
  --muted: #64748B;
  --card-shadow: 0 0 0 1px rgba(99,179,237,0.08), 0 8px 32px rgba(0,0,0,0.4);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, [data-testid="collapsedControl"],
[data-testid="stSidebar"] { display: none !important; }

[data-testid="stAppViewContainer"] > .main {
    padding: 0 0 0 2.5rem !important;
    background: var(--bg) !important;
}
.block-container {
    padding: 0 0 0 2rem !important;
    max-width: 100% !important;
}

/* ── ANIMATED BACKGROUND ── */
.bg-grid {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}
.bg-orb1 {
    position: fixed; top: -200px; right: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
    animation: orbFloat 8s ease-in-out infinite;
}
.bg-orb2 {
    position: fixed; bottom: -200px; left: -200px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(6,182,212,0.06) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
    animation: orbFloat 10s ease-in-out infinite reverse;
}
@keyframes orbFloat {
    0%,100% { transform: translate(0,0) scale(1); }
    50% { transform: translate(30px,-30px) scale(1.05); }
}

/* ── NAVBAR ── */
.navbar {
    position: relative; z-index: 100;
    background: rgba(6,10,20,0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    padding: 0 2.5rem;
    height: 68px;
    display: flex; align-items: center; justify-content: space-between;
}
.navbar-brand { display: flex; align-items: center; gap: 12px; }
.navbar-logo {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 16px; font-weight: 800; color: white;
    box-shadow: 0 0 20px rgba(59,130,246,0.4);
    position: relative; overflow: hidden;
}
.navbar-logo::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.15));
}
.navbar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 800; color: white; letter-spacing: -0.5px;
}
.navbar-title span { color: #06B6D4; }
.navbar-subtitle {
    font-size: 0.65rem; color: var(--muted); letter-spacing: 2px;
    text-transform: uppercase; margin-top: -2px;
}
.navbar-right { display: flex; align-items: center; gap: 0.75rem; }
.nav-pill {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    color: #93C5FD; padding: 5px 12px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.5px;
    display: flex; align-items: center; gap: 5px;
}
.nav-pill .dot {
    width: 6px; height: 6px; background: #10B981; border-radius: 50%;
    box-shadow: 0 0 6px #10B981;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}
.nav-avatar {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 0.82rem;
    box-shadow: 0 0 12px rgba(59,130,246,0.3);
}

/* ── PAGE LAYOUT ── */
.page-wrapper {
    position: relative; z-index: 1;
    padding: 2rem 2.5rem;
    display: grid;
    grid-template-columns: 1.7fr 1fr;
    gap: 1.5rem;
    min-height: calc(100vh - 68px);
}

/* ── SECTION LABELS ── */
.sec-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: var(--accent2); margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 8px;
}
.sec-label::before {
    content: ''; display: inline-block;
    width: 16px; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 999px;
}
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 800; color: white;
    letter-spacing: -0.8px; margin-bottom: 1.25rem;
    line-height: 1.2;
}

/* ── CARDS ── */
.card {
    background: var(--surface);
    border-radius: 18px;
    border: 1px solid var(--border);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--card-shadow);
    position: relative; overflow: hidden;
    transition: border-color 0.3s;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
}
.card:hover { border-color: rgba(99,179,237,0.2); }

.card-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1.25rem;
}
.card-icon {
    width: 32px; height: 32px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
}
.card-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--muted);
}

/* ── INFO BANNER ── */
.info-banner {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.06));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 1.2rem;
    display: flex; gap: 10px; align-items: flex-start;
}
.info-banner-icon { font-size: 1rem; margin-top: 1px; flex-shrink: 0; }
.info-banner-text { font-size: 0.78rem; color: #93C5FD; line-height: 1.7; }
.info-banner-text strong { color: #E0F2FE; font-weight: 600; }

/* ── STAT CARDS ── */
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem; }
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 1rem 1.1rem;
    box-shadow: var(--card-shadow);
    position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.stat-card:hover { transform: translateY(-2px); border-color: var(--border2); }
.stat-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
}
.stat-card.blue::after  { background: linear-gradient(90deg, #3B82F6, #06B6D4); }
.stat-card.green::after { background: linear-gradient(90deg, #10B981, #34D399); }
.stat-card.yellow::after{ background: linear-gradient(90deg, #F59E0B, #FBBF24); }
.stat-card.purple::after{ background: linear-gradient(90deg, #8B5CF6, #A78BFA); }
.stat-val { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: white; letter-spacing: -1px; }
.stat-lbl { font-size: 0.68rem; color: var(--muted); font-weight: 500; margin-top: 3px; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-trend { font-size: 0.68rem; font-weight: 600; margin-top: 5px; }
.trend-up { color: #34D399; } .trend-down { color: #FB7185; } .trend-neutral { color: #94A3B8; }

/* ── RESULT PANEL ── */
.result-idle {
    background: var(--surface);
    border: 1px dashed rgba(59,130,246,0.2);
    border-radius: 18px; padding: 3rem 2rem;
    text-align: center;
    box-shadow: var(--card-shadow);
}
.result-idle-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.6; }
.result-idle-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #94A3B8; margin-bottom: 0.5rem; }
.result-idle-sub { font-size: 0.78rem; color: var(--muted); line-height: 1.7; max-width: 220px; margin: 0 auto; }

/* Result cards */
.result-card {
    border-radius: 18px; padding: 2rem 1.5rem;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    animation: resultReveal 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}
.result-card.approved {
    background: linear-gradient(160deg, #022C22 0%, #064E3B 50%, #065F46 100%);
    border: 1px solid rgba(16,185,129,0.3);
}
.result-card.rejected {
    background: linear-gradient(160deg, #1C0B14 0%, #4C0519 50%, #881337 100%);
    border: 1px solid rgba(244,63,94,0.3);
}
@keyframes resultReveal {
    from { opacity: 0; transform: scale(0.9) translateY(20px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.result-card::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 0%, rgba(255,255,255,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.result-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 999px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 1rem;
}
.badge-ok  { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.badge-no  { background: rgba(244,63,94,0.15);  color: #FB7185; border: 1px solid rgba(244,63,94,0.3);  }
.result-headline {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem; font-weight: 800; color: white;
    letter-spacing: -0.8px; margin-bottom: 0.3rem;
}
.result-sub { font-size: 0.8rem; color: rgba(255,255,255,0.5); margin-bottom: 1.5rem; }

/* Gauge */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; margin: 1rem 0; }
.gauge-ring {
    width: 110px; height: 110px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    position: relative; margin-bottom: 0.5rem;
}
.gauge-inner {
    width: 82px; height: 82px; border-radius: 50%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    position: absolute;
}
.gauge-pct {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem; font-weight: 700; color: white; line-height: 1;
}
.gauge-txt { font-size: 0.6rem; color: rgba(255,255,255,0.45); margin-top: 2px; letter-spacing: 0.5px; }
.gauge-label { font-size: 0.7rem; color: rgba(255,255,255,0.4); font-weight: 500; letter-spacing: 0.5px; }

/* Prob bars */
.prob-row { display: flex; align-items: center; gap: 10px; margin: 0.45rem 0; }
.prob-name { font-size: 0.72rem; color: rgba(255,255,255,0.55); width: 65px; text-align: left; font-weight: 500; }
.prob-bg { flex: 1; height: 5px; background: rgba(255,255,255,0.07); border-radius: 999px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 999px; }
.fill-g { background: linear-gradient(90deg, #10B981, #34D399); }
.fill-r { background: linear-gradient(90deg, #F43F5E, #FB7185); }
.prob-pct { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: rgba(255,255,255,0.7); width: 38px; text-align: right; }

/* Factor chips */
.factor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem; margin-top: 1rem; }
.factor-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.6rem 0.8rem;
    text-align: left;
}
.chip-lbl { font-size: 0.6rem; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 1px; }
.chip-val { font-family: 'Space Mono', monospace; font-size: 0.9rem; font-weight: 700; color: white; margin-top: 3px; }

/* ── KEY FACTORS CARD ── */
.factors-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem;
    margin-top: 1rem;
    box-shadow: var(--card-shadow);
}
.factor-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.factor-row:last-child { border-bottom: none; }
.factor-name { font-size: 0.78rem; color: #94A3B8; flex: 1; font-weight: 500; }
.factor-bar-bg { width: 80px; height: 4px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow: hidden; }
.factor-bar-fill { height: 100%; border-radius: 999px; }
.impact-high   { background: linear-gradient(90deg, #3B82F6, #06B6D4); width: 100%; }
.impact-medium { background: linear-gradient(90deg, #8B5CF6, #A78BFA); width: 66%;  }
.impact-low    { background: linear-gradient(90deg, #F59E0B, #FBBF24); width: 33%;  }
.factor-badge {
    font-size: 0.62rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.fb-high   { background: rgba(59,130,246,0.12);  color: #93C5FD; }
.fb-medium { background: rgba(139,92,246,0.12);  color: #C4B5FD; }
.fb-low    { background: rgba(245,158,11,0.12);  color: #FCD34D; }

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stWidgetLabel"] label,
.stSelectbox label, .stNumberInput label, .stSlider label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important; font-weight: 600 !important;
    color: #64748B !important; text-transform: uppercase !important;
    letter-spacing: 1px !important; margin-bottom: 4px !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.88rem !important; font-weight: 400 !important;
    color: #E2E8F0 !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.85rem !important;
    transition: all 0.2s !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: rgba(59,130,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
    background: rgba(59,130,246,0.05) !important;
    outline: none !important;
}

[data-testid="stSelectbox"] > div > div {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(59,130,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #3B82F6, #06B6D4) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: white !important;
    border: 2px solid #3B82F6 !important;
    box-shadow: 0 0 12px rgba(59,130,246,0.5) !important;
    width: 18px !important; height: 18px !important;
}

[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important;
    background: linear-gradient(135deg, #1D4ED8, #0891B2) !important;
    color: white !important; font-weight: 700 !important;
    font-size: 0.95rem !important; letter-spacing: 0.5px !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.85rem 1.5rem !important; width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.35), 0 0 0 1px rgba(59,130,246,0.2) !important;
    position: relative !important; overflow: hidden !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.5), 0 0 0 1px rgba(59,130,246,0.3) !important;
    background: linear-gradient(135deg, #2563EB, #0E7490) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

hr { border-color: rgba(99,179,237,0.08) !important; margin: 0.75rem 0 !important; }
[data-testid="column"] { padding: 0 0.3rem !important; }
[data-testid="stVerticalBlock"] > div { gap: 0.4rem !important; }

[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #93C5FD !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
}

[data-testid="stSpinner"] { color: #3B82F6 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.2); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.35); }

/* Step indicator */
.steps-row { display: flex; align-items: center; gap: 0; margin-bottom: 1.5rem; }
.step {
    display: flex; flex-direction: column; align-items: center; gap: 4px; flex: 1;
}
.step-dot {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    border: 2px solid rgba(99,179,237,0.2);
    color: var(--muted); background: var(--surface2);
    position: relative; z-index: 1;
}
.step-dot.active {
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    border-color: transparent; color: white;
    box-shadow: 0 0 16px rgba(59,130,246,0.5);
}
.step-line { flex: 1; height: 1px; background: rgba(99,179,237,0.1); margin-top: -14px; }
.step-lbl { font-size: 0.6rem; color: var(--muted); font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; text-align: center; }
.step-lbl.active { color: #93C5FD; }
</style>

<div class="bg-grid"></div>
<div class="bg-orb1"></div>
<div class="bg-orb2"></div>
""", unsafe_allow_html=True)


# ── Train model ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Initializing LoanIQ engine…")
def train_model():
    df = pd.read_csv("loan_approval_data.csv")
    df = df.drop(columns=["Applicant_ID"])

    income_mean  = int(df["Applicant_Income"].mean())
    savings_mean = int(df["Savings"].mean())
    loan_mean    = int(df["Loan_Amount"].mean())

    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(exclude=["object"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[num_cols] = num_imp.fit_transform(df[num_cols])
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    df["DTI_Ratio_sq"]         = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"]      = df["Credit_Score"] ** 2
    df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

    le_edu = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = (df["Loan_Approved"] == "Yes").astype(int)

    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
                "Property_Area","Gender","Employer_Category"]
    one = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded    = one.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=one.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    return model, scaler, one, le_edu, X.columns.tolist(), income_mean, savings_mean, loan_mean

model, scaler, one_enc, le_edu, feature_cols, income_mean, savings_mean, loan_mean = train_model()

def build_input(data):
    row = {
        "Applicant_Income":   data["applicant_income"],
        "Coapplicant_Income": data["coapplicant_income"],
        "Age":                data["age"],
        "Dependents":         data["dependents"],
        "Credit_Score":       data["credit_score"],
        "Existing_Loans":     data["existing_loans"],
        "DTI_Ratio":          data["dti_ratio"],
        "Savings":            data["savings"],
        "Collateral_Value":   data["collateral_value"],
        "Loan_Amount":        data["loan_amount"],
        "Loan_Term":          data["loan_term"],
        "Education_Level":    int(le_edu.transform([data["education"]])[0]),
    }
    row["DTI_Ratio_sq"]         = row["DTI_Ratio"] ** 2
    row["Credit_Score_sq"]      = row["Credit_Score"] ** 2
    row["Applicant_Income_log"] = np.log1p(row["Applicant_Income"])
    base_df = pd.DataFrame([row])
    ohe_input = pd.DataFrame([{
        "Employment_Status": data["employment_status"],
        "Marital_Status":    data["marital_status"],
        "Loan_Purpose":      data["loan_purpose"],
        "Property_Area":     data["property_area"],
        "Gender":            data["gender"],
        "Employer_Category": data["employer_category"],
    }])
    ohe_df = pd.DataFrame(
        one_enc.transform(ohe_input),
        columns=one_enc.get_feature_names_out(ohe_input.columns)
    )
    final = pd.concat([base_df, ohe_df], axis=1).reindex(columns=feature_cols, fill_value=0)
    return final


# ── NAVBAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-brand">
    <div class="navbar-logo">L</div>
    <div>
      <div class="navbar-title">Loan<span>IQ</span></div>
      <div class="navbar-subtitle">Credit Risk Intelligence</div>
    </div>
  </div>
  <div class="navbar-right">
    <div class="nav-pill"><span class="dot"></span> Model Active</div>
    <div class="nav-pill">86% Accuracy</div>
    <div class="nav-pill">v2.1</div>
    <div class="nav-avatar">A</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ───────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.75, 1], gap="large")

# ════════════════════════════════════
# LEFT — FORM
# ════════════════════════════════════
with left_col:
    st.markdown('<div class="sec-label">Application Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">New Loan Assessment</div>', unsafe_allow_html=True)

    # Step indicator
    st.markdown("""
    <div class="steps-row">
      <div class="step">
        <div class="step-dot active">1</div>
        <div class="step-lbl active">Personal</div>
      </div>
      <div class="step-line"></div>
      <div class="step">
        <div class="step-dot active">2</div>
        <div class="step-lbl active">Financial</div>
      </div>
      <div class="step-line"></div>
      <div class="step">
        <div class="step-dot active">3</div>
        <div class="step-lbl active">Loan</div>
      </div>
      <div class="step-line"></div>
      <div class="step">
        <div class="step-dot">4</div>
        <div class="step-lbl">Result</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Info banner
    st.markdown("""
    <div class="info-banner">
      <div class="info-banner-icon">⚡</div>
      <div class="info-banner-text">
        Fill all fields for best accuracy.
        <strong>Credit score 700+</strong> is strong &nbsp;·&nbsp;
        <strong>DTI below 0.40</strong> is healthy &nbsp;·&nbsp;
        <strong>Income 2,500–20,000</strong> typical range
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CARD 1: Personal ──
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon" style="background:rgba(59,130,246,0.12)">👤</div>
        <div class="card-label">Personal Information</div>
      </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    age            = p1.number_input("Age", 18, 80, 35)
    gender         = p2.selectbox("Gender", ["Male", "Female"])
    marital_status = p3.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents     = p4.number_input("Dependents", 0, 10, 0)

    p5, p6, p7, p8 = st.columns(4)
    education         = p5.selectbox("Education", ["Graduate", "Not Graduate"])
    employment_status = p6.selectbox("Employment", ["Salaried", "Self-employed"])
    employer_category = p7.selectbox("Employer Type", ["Private", "Government", "MNC"])
    property_area     = p8.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CARD 2: Financial ──
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon" style="background:rgba(16,185,129,0.12)">💰</div>
        <div class="card-label">Financial Profile</div>
      </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    applicant_income   = f1.number_input("Monthly Income", 1000, 25000, income_mean)
    coapplicant_income = f2.number_input("Co-applicant Income", 0, 25000, 2000)
    savings            = f3.number_input("Savings", 0, 25000, savings_mean)

    f4, f5, f6 = st.columns(3)
    collateral_value = f4.number_input("Collateral Value", 0, 100000, 20000)
    credit_score     = f5.number_input("Credit Score", 300, 900, 724)
    existing_loans   = f6.number_input("Existing Loans", 0, 10, 1)

    # Credit score health bar
    cs_pct = int(((credit_score - 300) / 600) * 100)
    cs_color = "#10B981" if credit_score >= 700 else "#F59E0B" if credit_score >= 580 else "#F43F5E"
    cs_label = "Excellent" if credit_score >= 750 else "Good" if credit_score >= 700 else "Fair" if credit_score >= 580 else "Poor"
    st.markdown(f"""
    <div style="margin: 0.5rem 0 0.25rem; padding: 0 0.2rem;">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
        <span style="font-size:0.68rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Credit Health</span>
        <span style="font-size:0.68rem;font-weight:700;color:{cs_color};">{cs_label}</span>
      </div>
      <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:999px;overflow:hidden;">
        <div style="width:{cs_pct}%;height:100%;background:linear-gradient(90deg,{cs_color},{cs_color}CC);border-radius:999px;transition:width 0.4s ease;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    dti_ratio = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 1.0, 0.25, step=0.01,
                          help="Below 0.40 is healthy. Above 0.50 is risky.")
    dti_color = "#10B981" if dti_ratio < 0.35 else "#F59E0B" if dti_ratio < 0.5 else "#F43F5E"
    dti_label = "Healthy" if dti_ratio < 0.35 else "Moderate" if dti_ratio < 0.5 else "High Risk"
    st.markdown(f"""
    <div style="margin:-0.25rem 0 0.25rem;padding:0 0.2rem;display:flex;justify-content:flex-end;">
      <span style="font-size:0.7rem;font-weight:700;color:{dti_color};">DTI: {dti_label}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── CARD 3: Loan ──
    st.markdown("""
    <div class="card">
      <div class="card-header">
        <div class="card-icon" style="background:rgba(245,158,11,0.12)">📋</div>
        <div class="card-label">Loan Details</div>
      </div>
    """, unsafe_allow_html=True)

    l1, l2, l3 = st.columns(3)
    loan_amount  = l1.number_input("Loan Amount", 1000, 50000, loan_mean)
    loan_term    = l2.selectbox("Loan Term (months)", [12,24,36,48,60,72,84,96,108,120,180,240,360])
    loan_purpose = l3.selectbox("Loan Purpose", ["Personal","Business","Car","Home","Education"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("⚡  Run AI Loan Assessment", use_container_width=True)


# ════════════════════════════════════
# RIGHT — STATS + RESULT
# ════════════════════════════════════
with right_col:
    st.markdown('<div class="sec-label">Dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-grid">
      <div class="stat-card blue">
        <div class="stat-val">86%</div>
        <div class="stat-lbl">Model Accuracy</div>
        <div class="stat-trend trend-up">↑ Logistic Reg</div>
      </div>
      <div class="stat-card green">
        <div class="stat-val">1,000</div>
        <div class="stat-lbl">Training Records</div>
        <div class="stat-trend trend-up">↑ Balanced</div>
      </div>
      <div class="stat-card yellow">
        <div class="stat-val">31%</div>
        <div class="stat-lbl">Approval Rate</div>
        <div class="stat-trend trend-neutral">Dataset avg</div>
      </div>
      <div class="stat-card purple">
        <div class="stat-val">19</div>
        <div class="stat-lbl">Features Used</div>
        <div class="stat-trend trend-up">↑ Engineered</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-top:0.5rem">AI Assessment</div>', unsafe_allow_html=True)

    if predict_clicked:
        input_data = {
            "age": age, "gender": gender, "marital_status": marital_status,
            "dependents": dependents, "education": education,
            "employment_status": employment_status, "employer_category": employer_category,
            "property_area": property_area, "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income, "savings": savings,
            "collateral_value": collateral_value, "credit_score": credit_score,
            "existing_loans": existing_loans, "dti_ratio": dti_ratio,
            "loan_amount": loan_amount, "loan_term": loan_term, "loan_purpose": loan_purpose,
        }
        X_input       = build_input(input_data)
        X_scaled      = scaler.transform(X_input)
        prediction    = model.predict(X_scaled)[0]
        proba         = model.predict_proba(X_scaled)[0]
        approved_prob = proba[1] * 100
        rejected_prob = proba[0] * 100

        ap = f"{approved_prob:.0f}%"
        rp = f"{rejected_prob:.0f}%"

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card approved">
              <div class="result-badge badge-ok">✦ APPROVED</div>
              <div class="result-headline">Loan Eligible</div>
              <div class="result-sub">Profile meets lending criteria</div>
              <div class="gauge-wrap">
                <div class="gauge-ring" style="background: conic-gradient(#10B981 {ap}, rgba(255,255,255,0.06) {ap});">
                  <div class="gauge-inner" style="background:linear-gradient(135deg,#022C22,#064E3B);">
                    <div class="gauge-pct">{approved_prob:.0f}%</div>
                    <div class="gauge-txt">CONFIDENCE</div>
                  </div>
                </div>
                <div class="gauge-label">Approval Confidence</div>
              </div>
              <div class="prob-row">
                <span class="prob-name">Approved</span>
                <div class="prob-bg"><div class="prob-fill fill-g" style="width:{ap}"></div></div>
                <span class="prob-pct">{approved_prob:.1f}%</span>
              </div>
              <div class="prob-row">
                <span class="prob-name">Rejected</span>
                <div class="prob-bg"><div class="prob-fill fill-r" style="width:{rp}"></div></div>
                <span class="prob-pct">{rejected_prob:.1f}%</span>
              </div>
              <div class="factor-grid">
                <div class="factor-chip"><div class="chip-lbl">Credit Score</div><div class="chip-val">{credit_score}</div></div>
                <div class="factor-chip"><div class="chip-lbl">DTI Ratio</div><div class="chip-val">{dti_ratio:.2f}</div></div>
                <div class="factor-chip"><div class="chip-lbl">Income</div><div class="chip-val">₹{applicant_income:,}</div></div>
                <div class="factor-chip"><div class="chip-lbl">Loan Amt</div><div class="chip-val">₹{loan_amount:,}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card rejected">
              <div class="result-badge badge-no">✕ REJECTED</div>
              <div class="result-headline">Not Eligible</div>
              <div class="result-sub">Profile doesn't meet criteria</div>
              <div class="gauge-wrap">
                <div class="gauge-ring" style="background: conic-gradient(#F43F5E {rp}, rgba(255,255,255,0.06) {rp});">
                  <div class="gauge-inner" style="background:linear-gradient(135deg,#1C0B14,#4C0519);">
                    <div class="gauge-pct">{rejected_prob:.0f}%</div>
                    <div class="gauge-txt">CONFIDENCE</div>
                  </div>
                </div>
                <div class="gauge-label">Rejection Confidence</div>
              </div>
              <div class="prob-row">
                <span class="prob-name">Approved</span>
                <div class="prob-bg"><div class="prob-fill fill-g" style="width:{ap}"></div></div>
                <span class="prob-pct">{approved_prob:.1f}%</span>
              </div>
              <div class="prob-row">
                <span class="prob-name">Rejected</span>
                <div class="prob-bg"><div class="prob-fill fill-r" style="width:{rp}"></div></div>
                <span class="prob-pct">{rejected_prob:.1f}%</span>
              </div>
              <div class="factor-grid">
                <div class="factor-chip"><div class="chip-lbl">Credit Score</div><div class="chip-val">{credit_score}</div></div>
                <div class="factor-chip"><div class="chip-lbl">DTI Ratio</div><div class="chip-val">{dti_ratio:.2f}</div></div>
                <div class="factor-chip"><div class="chip-lbl">Income</div><div class="chip-val">₹{applicant_income:,}</div></div>
                <div class="factor-chip"><div class="chip-lbl">Loan Amt</div><div class="chip-val">₹{loan_amount:,}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if prediction == 0:
            with st.expander("💡 How to improve your profile"):
                st.markdown("""
                - **Raise credit score** above 700 (target 750+)
                - **Lower DTI ratio** below 0.35 by reducing debts
                - **Reduce existing loans** to 0 or 1
                - **Increase savings** relative to loan amount
                - Add a **co-applicant** with stable income
                """)
    else:
        st.markdown("""
        <div class="result-idle">
          <div class="result-idle-icon">🔮</div>
          <div class="result-idle-title">Awaiting Assessment</div>
          <div class="result-idle-sub">Complete the form and run the AI assessment to receive an instant decision.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="factors-card">
          <div class="card-label" style="margin-bottom:0.85rem">Key Decision Factors</div>
          <div class="factor-row">
            <span class="factor-name">Credit Score</span>
            <div class="factor-bar-bg"><div class="factor-bar-fill impact-high"></div></div>
            <span class="factor-badge fb-high">High</span>
          </div>
          <div class="factor-row">
            <span class="factor-name">DTI Ratio</span>
            <div class="factor-bar-bg"><div class="factor-bar-fill impact-high"></div></div>
            <span class="factor-badge fb-high">High</span>
          </div>
          <div class="factor-row">
            <span class="factor-name">Monthly Income</span>
            <div class="factor-bar-bg"><div class="factor-bar-fill impact-medium"></div></div>
            <span class="factor-badge fb-medium">Medium</span>
          </div>
          <div class="factor-row">
            <span class="factor-name">Existing Loans</span>
            <div class="factor-bar-bg"><div class="factor-bar-fill impact-medium"></div></div>
            <span class="factor-badge fb-medium">Medium</span>
          </div>
          <div class="factor-row">
            <span class="factor-name">Collateral Value</span>
            <div class="factor-bar-bg"><div class="factor-bar-fill impact-low"></div></div>
            <span class="factor-badge fb-low">Low</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
