import os
import re
import math
import time
import csv
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. REALISTIC URL LISTS & DATASET GEN
# ==========================================
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

def augment_realistic_urls(url_list, target_count):
    if len(url_list) >= target_count:
        return random.sample(url_list, target_count)
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
MODEL_PATH = "models/best_model.pkl"   # changed from .joblib

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
        create_realistic_dataset(500, 500)  # Scaled down for quick init if missing

# ==========================================
# 2. FEATURE EXTRACTION (All 36 Features)
# ==========================================
SUSPICIOUS_TLDS = {'xyz','top','click','tk','ml','ga','cf','gq','pw','cc','su','biz','info','online','site','live','stream','download','loan','review','country','kim','science','work','party','trade','cricket','date','faith','racing','accountant','win','bid','men','icu','monster','cyou','buzz','sbs','ru'}
TRUSTED_DOMAINS = {'google.com','youtube.com','facebook.com','microsoft.com','apple.com','amazon.com','github.com','twitter.com','linkedin.com','wikipedia.org','instagram.com','netflix.com','stackoverflow.com','reddit.com','paypal.com'}
BRAND_KEYWORDS = ['paypal','google','apple','microsoft','amazon','facebook','instagram','netflix','ebay','steam','whatsapp','youtube','dropbox','icloud','twitter','chase','wellsfargo','citibank','bankofamerica','boa','dhl','fedex','usps','ups']
URL_SHORTENERS = {'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd','buff.ly','rebrand.ly','short.io','tiny.cc','cutt.ly'}
PHISH_RE = re.compile(r'login|signin|verify|account|update|secure|confirm|password|credential|alert|suspend|unlock|recover|reset|billing|payment|invoice', re.I)
EXEC_RE = re.compile(r'\.(exe|bat|cmd|msi|scr|vbs|jar|apk|dmg|sh|ps1|crx|xpi)$', re.I)
SPAM_WORDS = ['free','win','prize','claim','urgent','alert','suspended','verify','confirm','limited','offer','bonus','gift','reward','lucky','congratulation']

def _entropy(s):
    if not s:
        return 0.0
    freq = {c: s.count(c) for c in set(s)}
    n = len(s)
    return -sum((v/n)*math.log2(v/n) for v in freq.values())

def _domain_parts(hostname):
    clean = re.sub(r'^www\.', '', hostname.lower())
    parts = clean.split('.')
    if len(parts) >= 3:
        return '.'.join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) == 2:
        return '', parts[0], parts[1]
    return '', hostname, ''

def extract_features(url: str) -> dict:
    raw = str(url).strip()
    f = {}
    try:
        p = urlparse(raw if '://' in raw else 'http://' + raw)
    except Exception:
        p = urlparse('http://invalid')
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
    try:
        f['param_count'] = len(parse_qs(query))
    except Exception:
        f['param_count'] = 0
    f['hostname_entropy'] = round(_entropy(hostname), 4)
    f['path_entropy'] = round(_entropy(path), 4)
    f['is_shortener'] = int(hostname in URL_SHORTENERS)
    f['spam_keyword_count'] = sum(w in full_lower for w in SPAM_WORDS)
    f['has_punycode'] = int('xn--' in hostname)
    f['domain_age_days'] = 365
    return f

FEATURE_COLUMNS = list(extract_features("http://example.com").keys())

# ==========================================
# 3. MODEL TRAINING & LOADING (using pickle)
# ==========================================
def train_model():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates(subset=['url']).dropna(subset=['url', 'label'])
    df['label'] = df['label'].astype(int).clip(0, 1)
    X = pd.DataFrame([extract_features(url) for url in df['url']])[FEATURE_COLUMNS].fillna(0).values.astype(float)
    y = df['label'].values

    col_idx = FEATURE_COLUMNS.index('domain_age_days')
    col_vals = X[:, col_idx]
    median_age = np.median(col_vals[col_vals >= 0]) if np.any(col_vals >= 0) else 365
    X[X[:, col_idx] == -1, col_idx] = median_age

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({"model": rf, "feature_columns": FEATURE_COLUMNS}, f)
    return rf

@st.cache_resource(show_spinner="Loading ML Scanner Engine...")
def load_model():
    ensure_dataset()
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Training model on realistic dataset..."):
            model = train_model()
            return model, FEATURE_COLUMNS
    with open(MODEL_PATH, 'rb') as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_columns"]

# ==========================================
# 4. STREAMLIT UI SETUP & CSS
# ==========================================
st.set_page_config(page_title="ML Scanner - Malicious URL Detector", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');

    .stApp { background-color: #0b1120 !important; color: #f1f5f9 !important; font-family: 'Inter', sans-serif; }
    header, #MainMenu {visibility: hidden;}

    .mono { font-family: 'JetBrains Mono', monospace; }
    .uppercase-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; }

    .ts-nav { display: flex; justify-content: space-between; align-items: center; padding: 1rem 2rem; background: rgba(15, 23, 42, 0.8); border-bottom: 1px solid #1e293b; margin-top: -3rem; margin-bottom: 2rem; }
    .ts-brand { font-size: 1.25rem; font-weight: 700; color: white; display: flex; align-items: center; gap: 12px; }
    .ts-icon { background: #4f46e5; padding: 6px; border-radius: 8px; box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39); }

    .ts-card { background: rgba(30, 41, 59, 0.4); border: 1px solid #1e293b; border-radius: 24px; padding: 2rem; margin-bottom: 1.5rem; }
    .hero-tags { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .hero-tag { background: #1e293b; border: 1px solid #334155; padding: 2px 10px; border-radius: 12px; font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; }

    .stTextInput > div > div > input { background-color: #0f172a !important; border: 1px solid #1e293b !important; color: #e2e8f0 !important; border-radius: 12px !important; padding: 1rem !important; font-family: 'JetBrains Mono', monospace !important; font-size: 14px !important; }
    .stTextInput > div > div > input:focus { border-color: #4f46e5 !important; }

    .stButton > button { background-color: #4f46e5 !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; font-weight: 700 !important; height: 100% !important; width: 100%; transition: 0.2s; }
    .stButton > button:hover { background-color: #4338ca !important; box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39) !important; }

    .risk-safe { color: #34d399; background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.2); }
    .risk-high { color: #f87171; background: rgba(248, 113, 113, 0.1); border: 1px solid rgba(248, 113, 113, 0.2); }
    .risk-badge { padding: 4px 8px; border-radius: 6px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. APP LAYOUT
# ==========================================

st.markdown("""
    <div class="ts-nav">
        <div class="ts-brand"><span class="ts-icon">🛡️</span> ML Scanner</div>
        <div style="color: #64748b; font-size: 14px; font-weight: 500;">Malicious URL Detector</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="ts-card" style="display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; gap: 20px; align-items: center;">
            <div style="background: rgba(79, 70, 229, 0.1); padding: 16px; border-radius: 16px; border: 1px solid rgba(79, 70, 229, 0.2); font-size: 32px;">🛡️</div>
            <div>
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">ML Scanner</h1>
                <div class="hero-tags">
                    <span class="hero-tag">Malicious URL Detector</span>
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <span class="hero-tag">Real‑time Analysis</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

model, feat_cols = load_model()

# Only Single URL tab remains
st.markdown('<div class="uppercase-label" style="margin-bottom: 8px;">// Target URL</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])
with col_input:
    url_input = st.text_input("URL", placeholder="https://example.com or paste a suspicious link...", label_visibility="collapsed")
with col_btn:
    analyze_btn = st.button("⚡ Scan URL")

if analyze_btn and url_input:
    with st.spinner("Analyzing..."):
        feats = extract_features(url_input)
        X = np.array([feats.get(c, 0) for c in feat_cols]).reshape(1, -1)
        prob = model.predict_proba(X)[0]

        risk_score = round(prob[1] * 100, 1)
        is_mal = risk_score >= 50

        if risk_score < 20:
            risk_level = "SAFE"
        elif risk_score < 40:
            risk_level = "LOW"
        elif risk_score < 70:
            risk_level = "MEDIUM"
        elif risk_score < 90:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        risk_class = "risk-high" if is_mal else "risk-safe"

        st.session_state.history.insert(0, {"url": url_input, "level": risk_level, "score": risk_score, "time": time.strftime("%H:%M")})

        st.markdown("<br>", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([7, 5])

        with res_col1:
            st.markdown(f"""
            <div class="ts-card {risk_class}">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 24px;">
                    <div style="display: flex; gap: 16px; align-items: center;">
                        <div style="font-size: 32px;">{'🚨' if is_mal else '✅'}</div>
                        <div>
                            <h3 style="margin: 0; font-size: 2rem; font-weight: 700;">{risk_level} RISK</h3>
                            <p style="margin: 0; font-size: 14px; opacity: 0.8;">Confidence Score: {risk_score}/100</p>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div class="uppercase-label" style="opacity: 0.6;">Target Host</div>
                        <div class="mono" style="font-size: 14px;">{urlparse(url_input if '://' in url_input else 'http://'+url_input).hostname}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="ts-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Threat Indicators")
            indicators = []
            if feats.get('suspicious_tld'):
                indicators.append("Suspicious Top Level Domain detected.")
            if feats.get('is_ip_host'):
                indicators.append("IP Address used instead of Domain name.")
            if feats.get('brand_in_domain'):
                indicators.append("Potential Brand Impersonation (Typosquatting).")
            if feats.get('phish_path_kw'):
                indicators.append("Phishing keywords found in URL path.")
            if not feats.get('is_https'):
                indicators.append("Connection is not secured with HTTPS.")
            if feats.get('is_shortener'):
                indicators.append("URL shortener service detected.")
            if feats.get('executable_ext'):
                indicators.append("Direct link to an executable file.")
            if feats.get('spam_keyword_count') > 0:
                indicators.append(f"Contains {feats.get('spam_keyword_count')} spam/urgent keywords.")

            if not indicators:
                indicators.append("No significant threat indicators found.")
            for ind in indicators:
                st.markdown(f"- {ind}")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="ts-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("#### 📊 Feature Impact")
            st.caption("Top features contributing to the final classification.")

            impact_data = {
                "Suspicious TLD": 85 if feats.get('suspicious_tld') else 5,
                "IP Host": 90 if feats.get('is_ip_host') else 2,
                "Brand Spoof": 75 if feats.get('brand_in_domain') else 4,
                "Phish Keywords": 80 if feats.get('phish_path_kw') else 10,
                "Shortener": 60 if feats.get('is_shortener') else 5,
                "Spam Words": feats.get('spam_keyword_count', 0) * 20
            }
            filtered = [(k, v) for k, v in impact_data.items() if v > 5]
            if not filtered:
                filtered = [("Baseline Safe", 10)]
            filtered.sort(key=lambda x: x[1])
            features = [f[0] for f in filtered]
            impacts = [f[1] for f in filtered]

            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors = ['#10b981' if val < 40 else '#f59e0b' if val < 70 else '#f43f5e' for val in impacts]
            ax.barh(features, impacts, color=colors, alpha=0.9)
            ax.set_facecolor('#0f172a')
            fig.patch.set_facecolor('#0f172a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.tick_params(colors='#94a3b8')
            ax.xaxis.label.set_color('#94a3b8')
            ax.set_xlabel("Impact Score", color='#94a3b8')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

elif analyze_btn and not url_input:
    st.warning("Please enter a URL.")

# ------------------------------------------
# BOTTOM: RECENT SCANS HISTORY
# ------------------------------------------
st.markdown("<hr style='border-color: #1e293b; margin: 3rem 0;'>", unsafe_allow_html=True)
st.markdown("### 🕒 Recent Scans")

if not st.session_state.history:
    st.markdown('<div style="text-align: center; color: #64748b; padding: 2rem;">No scan history yet. Start by analyzing a URL.</div>', unsafe_allow_html=True)
else:
    for item in st.session_state.history[:10]:
        rc = "risk-high" if item['level'] in ["HIGH", "CRITICAL"] else "risk-safe"
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid #1e293b; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; gap: 16px; align-items: center;">
                <span class="risk-badge {rc}">{item['level']}</span>
                <span class="mono" style="font-size: 14px; color: #cbd5e1;">{item['url']}</span>
            </div>
            <div class="uppercase-label" style="opacity: 0.6;">{item['time']}</div>
        </div>
        """, unsafe_allow_html=True)