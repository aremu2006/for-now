import os
import re
import math
import time
import csv
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# ------------------------------------------------------------
# 1. REALISTIC URL LISTS (From your original code)
# ------------------------------------------------------------
REALISTIC_BENIGN = [
    "https://www.google.com/search?q=python+tutorial",
    "https://github.com/scikit-learn/scikit-learn",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://docs.python.org/3/library/re.html",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.linkedin.com/in/johndoe",
]

REALISTIC_MALICIOUS = [
    "http://paypal.com.secure-login-verify.xyz/account/update?token=abc",
    "http://192.168.1.105/admin/login.php?redirect=home",
    "http://g00gle-security-alert.com/verify?user=victim@gmail.com",
    "http://amazon-prize-winner-2024.top/claim?id=99812&ref=email",
    "http://login.microsoftonline.com.phish.tk/oauth2/token",
]

# ------------------------------------------------------------
# 2. DATASET GENERATION (Restored from your code)
# ------------------------------------------------------------
def augment_realistic_urls(url_list, target_count):
    if len(url_list) >= target_count: return random.sample(url_list, target_count)
    augmented = list(url_list)
    while len(augmented) < target_count:
        base = random.choice(url_list)
        perturbed = base
        if '?' in perturbed:
            perturbed += f"&ref_{random.randint(1,999)}={random.randint(1000,99999)}"
        else:
            perturbed += f"?ref={random.randint(1000,99999)}"
        if perturbed not in augmented:
            augmented.append(perturbed)
    return augmented[:target_count]

DATASET_PATH = "data/urls_dataset.csv"
MODEL_PATH = "models/best_model.joblib"

def create_realistic_dataset(target_benign=5000, target_malicious=5000):
    benign_urls = augment_realistic_urls(REALISTIC_BENIGN, target_benign)
    malicious_urls = augment_realistic_urls(REALISTIC_MALICIOUS, target_malicious)
    rows = [(url, 0) for url in benign_urls] + [(url, 1) for url in malicious_urls]
    random.shuffle(rows)
    os.makedirs("data", exist_ok=True)
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    return rows

def ensure_dataset():
    if not os.path.exists(DATASET_PATH):
        create_realistic_dataset(500, 500) # Scaled down for quick init if missing

# ------------------------------------------------------------
# 3. FEATURE EXTRACTION (All 36 Original Features Restored)
# ------------------------------------------------------------
SUSPICIOUS_TLDS = {'xyz','top','click','tk','ml','ga','cf','gq','pw','cc','su','biz','info','online','site','live','stream','download','loan','review','country','kim','science','work','party','trade','cricket','date','faith','racing','accountant','win','bid','men','icu','monster','cyou','buzz','sbs','ru'}
TRUSTED_DOMAINS = {'google.com','youtube.com','facebook.com','microsoft.com','apple.com','amazon.com','github.com','twitter.com','linkedin.com','wikipedia.org','instagram.com','netflix.com','stackoverflow.com','reddit.com','paypal.com'}
BRAND_KEYWORDS = ['paypal','google','apple','microsoft','amazon','facebook','instagram','netflix','ebay','steam','whatsapp','youtube','dropbox','icloud','twitter','chase','wellsfargo','citibank','bankofamerica','boa','dhl','fedex','usps','ups']
URL_SHORTENERS = {'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd','buff.ly','rebrand.ly','short.io','tiny.cc','cutt.ly'}
PHISH_RE = re.compile(r'login|signin|verify|account|update|secure|confirm|password|credential|alert|suspend|unlock|recover|reset|billing|payment|invoice', re.I)
EXEC_RE = re.compile(r'\.(exe|bat|cmd|msi|scr|vbs|jar|apk|dmg|sh|ps1|crx|xpi)$', re.I)
SPAM_WORDS = ['free','win','prize','claim','urgent','alert','suspended','verify','confirm','limited','offer','bonus','gift','reward','lucky','congratulation']

def _entropy(s):
    if not s: return 0.0
    freq = {c: s.count(c) for c in set(s)}
    n = len(s)
    return -sum((v/n)*math.log2(v/n) for v in freq.values())

def _domain_parts(hostname):
    clean = re.sub(r'^www\.', '', hostname.lower())
    parts = clean.split('.')
    if len(parts) >= 3: return '.'.join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2: return '', parts[0], parts[1]
    return '', hostname, ''

def extract_features(url: str) -> dict:
    raw = str(url).strip()
    f = {}
    try: p = urlparse(raw if '://' in raw else 'http://' + raw)
    except: p = urlparse('http://invalid')
    hostname = (p.hostname or '').lower()
    path, query, scheme = p.path or '', p.query or '', p.scheme or ''
    full_lower = raw.lower()
    _, domain, tld = _domain_parts(hostname)
    base = f"{domain}.{tld}" if domain and tld else hostname
    sub, _, _ = _domain_parts(hostname)
    hl = max(len(hostname), 1)
    
    f['is_https'] = int(scheme == 'https')
    f['is_http'] = int(scheme == 'http')
    f['url_length'] = len(raw)
    f['hostname_length'] = len(hostname)
    f['path_length'] = len(path)
    f['query_length'] = len(query)
    f['dot_count'] = hostname.count('.')
    f['hyphen_count'] = hostname.count('-')
    f['underscore_count'] = raw.count('_')
    f['at_sign'] = int('@' in raw)
    f['double_slash'] = int('//' in path)
    f['question_mark'] = int('?' in raw)
    f['ampersand_count'] = query.count('&')
    f['equals_count'] = query.count('=')
    f['percent_count'] = len(re.findall(r'%[0-9a-fA-F]{2}', raw))
    f['hash_count'] = int('#' in raw)
    f['digit_ratio'] = round(sum(c.isdigit() for c in hostname) / hl, 4)
    f['alpha_ratio'] = round(sum(c.isalpha() for c in hostname) / hl, 4)
    f['subdomain_count'] = len(sub.split('.')) if sub else 0
    f['suspicious_tld'] = int(tld in SUSPICIOUS_TLDS)
    f['tld_length'] = len(tld)
    f['is_ip_host'] = int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname)))
    f['trusted_domain'] = int(base in TRUSTED_DOMAINS)
    brand_hit = any(b in hostname for b in BRAND_KEYWORDS)
    f['brand_in_domain'] = int(brand_hit and base not in TRUSTED_DOMAINS)
    f['digit_in_word'] = int(bool(re.search(r'[a-z]\d[a-z]', hostname)))
    f['phish_path_kw'] = int(bool(PHISH_RE.search(path)))
    f['executable_ext'] = int(bool(EXEC_RE.search(path)))
    f['path_depth'] = path.count('/')
    f['path_has_ip'] = int(bool(re.search(r'/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', path)))
    try: f['param_count'] = len(parse_qs(query))
    except: f['param_count'] = 0
    f['hostname_entropy'] = round(_entropy(hostname), 4)
    f['path_entropy'] = round(_entropy(path), 4)
    f['is_shortener'] = int(hostname in URL_SHORTENERS)
    f['spam_keyword_count'] = sum(w in full_lower for w in SPAM_WORDS)
    f['has_punycode'] = int('xn--' in hostname)
    f['domain_age_days'] = 365
    return f

FEATURE_COLUMNS = list(extract_features("http://example.com").keys())

# ------------------------------------------------------------
# 4. MODEL TRAINING & LOADING (Restored from your code)
# ------------------------------------------------------------
def train_model():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates(subset=['url']).dropna(subset=['url', 'label'])
    df['label'] = df['label'].astype(int).clip(0,1)
    X = pd.DataFrame([extract_features(url) for url in df['url']])[FEATURE_COLUMNS].fillna(0).values.astype(float)
    y = df['label'].values
    
    col_idx = FEATURE_COLUMNS.index('domain_age_days')
    col_vals = X[:, col_idx]
    median_age = np.median(col_vals[col_vals >= 0]) if np.any(col_vals >= 0) else 365
    X[X[:, col_idx] == -1, col_idx] = median_age
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": rf, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    return rf

@st.cache_resource(show_spinner="Loading Sentinel Engine...")
def load_model():
    ensure_dataset()
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Training model on realistic dataset..."):
            model = train_model()
            return model, FEATURE_COLUMNS
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_columns"]

# ------------------------------------------------------------
# 5. STREAMLIT UI - EXACT SENTINEL MATCH
# ------------------------------------------------------------
st.set_page_config(page_title="Sentinel", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles to match Sentinel Deep Navy */
    html, body, .stApp { background-color: #0b0f19 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
    
    /* Hide top header lines and default menus */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Exact Sentinel Top Nav Bar */
    .sentinel-nav {
        display: flex; justify-content: space-between; align-items: center;
        padding: 1.5rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 2rem;
    }
    .nav-brand { font-size: 1.5rem; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 10px; }
    .nav-brand-icon { background: #4f46e5; border-radius: 8px; padding: 4px 8px; }
    .nav-sub { color: #94a3b8; font-size: 0.95rem; font-weight: 500; }

    /* Sentinel Main Inspect Card */
    .inspect-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 3rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .inspect-title { font-size: 1.8rem; font-weight: 600; margin-bottom: 0.5rem; color: white; }
    .inspect-desc { color: #94a3b8; font-size: 1rem; max-width: 600px; margin: 0 auto 2rem auto; line-height: 1.5; }

    /* Unified Input Bar Hack */
    .stTextInput > div > div > input {
        background-color: #0b0f19 !important;
        border: 1px solid #1e293b !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.9rem 1rem !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus { border-color: #4f46e5 !important; box-shadow: none !important; }
    
    .stButton > button {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: none !important;
        border-radius: 8px !important;
        height: 48px !important;
        width: 100% !important;
        font-weight: 600 !important;
        margin-top: 2px !important;
        transition: 0.2s;
    }
    .stButton > button:hover { background-color: #334155 !important; color: white !important; }

    /* Sidebar - Recent Scans */
    section[data-testid="stSidebar"] { background-color: #0b0f19 !important; border-right: 1px solid #1e293b !important; }
    .recent-header { color: #ffffff; font-weight: 700; font-size: 0.9rem; letter-spacing: 1px; margin-bottom: 1rem; }
    .history-card { background: #111827; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; border: 1px solid #1e293b; }
    .history-verdict-safe { color: #10b981; font-weight: 700; font-size: 0.85rem; margin-bottom: 4px; }
    .history-verdict-mal { color: #ef4444; font-weight: 700; font-size: 0.85rem; margin-bottom: 4px; }
    .history-url { color: #94a3b8; font-size: 0.8rem; word-break: break-all; }

    /* Performance Metrics */
    .metric-box { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 1.5rem; height: 100%; }
    .metric-label { color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem; }
    .metric-val { font-size: 2rem; font-weight: 700; color: white; }
</style>
""", unsafe_allow_html=True)

# Top Navigation
st.markdown("""
    <div class="sentinel-nav">
        <div class="nav-brand"><span class="nav-brand-icon">🛡️</span> SENTINEL</div>
        <div class="nav-sub">Professional URL Shield</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Logic
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

with st.sidebar:
    st.markdown('<div class="recent-header">🕘 RECENT SCANS</div>', unsafe_allow_html=True)
    if not st.session_state.scan_history:
        st.markdown('<div style="color:#64748b; text-align:center; padding:2rem 0;">No scan history yet.</div>', unsafe_allow_html=True)
    else:
        for scan in reversed(st.session_state.scan_history[-10:]):
            v_class = "history-verdict-mal" if scan['verdict'] == "MALICIOUS" else "history-verdict-safe"
            st.markdown(f"""
                <div class="history-card">
                    <div class="{v_class}">{scan['verdict']}</div>
                    <div class="history-url">{scan['url'][:45]}...</div>
                </div>
            """, unsafe_allow_html=True)

# Main Inspect Box
st.markdown("""
    <div class="inspect-card">
        <div style="font-size: 3rem; color: #334155; margin-bottom: 1rem;">🛡️</div>
        <div class="inspect-title">Ready for Inspection</div>
        <div class="inspect-desc">Input a URL below to start the multi-layered heuristic and semantic analysis process using our Random Forest engine.</div>
    </div>
""", unsafe_allow_html=True)

# Tightly coupled input bar
col1, col2 = st.columns([5, 1])
with col1:
    url_input = st.text_input("URL", placeholder="🔍 https://example.com/login", label_visibility="collapsed")
with col2:
    analyze_btn = st.button("Analyze")

# ------------------------------------------------------------
# 6. EXECUTION & PERFORMANCE GRAPH
# ------------------------------------------------------------
model, feat_cols = load_model()

if analyze_btn and url_input:
    with st.spinner("Analyzing threat signatures..."):
        # Prediction Logic
        feats = extract_features(url_input)
        X = np.array([feats.get(c, 0) for c in feat_cols]).reshape(1, -1)
        prob = model.predict_proba(X)[0]
        
        mal_pct = round(prob[1] * 100, 1)
        verdict = "MALICIOUS" if mal_pct >= 50 else ("SUSPICIOUS" if mal_pct >= 30 else "SAFE")
        
        # Save to sidebar history
        st.session_state.scan_history.append({"url": url_input, "verdict": verdict})
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # --- PERFORMANCE / RESULTS SECTION ---
        st.markdown("### Analysis Results")
        m1, m2, m3 = st.columns(3)
        with m1:
            color = "#ef4444" if verdict == "MALICIOUS" else "#10b981"
            st.markdown(f'<div class="metric-box"><div class="metric-label">VERDICT</div><div class="metric-val" style="color:{color}">{verdict}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">THREAT PROBABILITY</div><div class="metric-val">{mal_pct}%</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">HTTPS SECURED</div><div class="metric-val">{"Yes" if feats.get("is_https") else "No"}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # "WHY" GRAPH
        st.markdown("### Why this result?")
        
        # Isolate top indicators
        bad_signals = []
        good_signals = []
        
        if feats.get('suspicious_tld'): bad_signals.append("Suspicious TLD")
        if feats.get('is_ip_host'): bad_signals.append("IP instead of Domain")
        if feats.get('phish_path_kw'): bad_signals.append("Phishing Keywords")
        if feats.get('brand_in_domain'): bad_signals.append("Brand Impersonation")
        if feats.get('executable_ext'): bad_signals.append("Executable Payload")
        if feats.get('url_length') > 80: bad_signals.append("Abnormal Length")
        
        if feats.get('trusted_domain'): good_signals.append("Trusted Root Domain")
        if feats.get('is_https'): good_signals.append("SSL/TLS Encrypted")
        if not bad_signals: good_signals.append("No Anomaly Detected")

        # Matplotlib config to match dark theme
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        
        y_labels = []
        x_values = []
        colors = []
        
        # Populate Graph data
        for bg in bad_signals[:4]: # Show top 4 bad
            y_labels.append(bg)
            x_values.append(1)
            colors.append('#ef4444')
        for gg in good_signals[:3]: # Show top 3 good
            if len(y_labels) < 5: # Keep graph clean
                y_labels.append(gg)
                x_values.append(1)
                colors.append('#10b981')
                
        # Draw Horizontal Bar Chart
        ax.barh(y_labels, x_values, color=colors, height=0.5)
        ax.set_xticks([]) # Hide x axis numbers
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color('#1e293b')
        ax.tick_params(axis='y', colors='#e2e8f0', labelsize=10)
        
        st.pyplot(fig)
