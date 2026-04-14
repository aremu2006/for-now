import os
import re
import math
import time
import csv
import random
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ==========================================
# 3. MODEL TRAINING & LOADING
# ==========================================
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

@st.cache_resource(show_spinner="Loading ThreatScan Engine...")
def load_model():
    ensure_dataset()
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Training model on realistic dataset..."):
            model = train_model()
            return model, FEATURE_COLUMNS
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_columns"]

# ==========================================
# 4. STREAMLIT UI SETUP & CSS (ThreatScan Theme)
# ==========================================
st.set_page_config(page_title="ThreatScan - Professional URL Shield", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');
    
    .stApp { background-color: #0b1120 !important; color: #f1f5f9 !important; font-family: 'Inter', sans-serif; }
    header, #MainMenu {visibility: hidden;}
    
    .mono { font-family: 'JetBrains Mono', monospace; }
    .uppercase-label { font-size
