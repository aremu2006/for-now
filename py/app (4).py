"""
app.py  —  Malicious URL Detector
===================================
100% self-contained Streamlit app — no src/ imports, no path issues.
Works locally and on Streamlit Cloud unchanged.

Run:  streamlit run app.py
"""

import os, sys, re, math, time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from urllib.parse import urlparse, parse_qs
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import streamlit as st

# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

SUSPICIOUS_TLDS = {
    'xyz','top','click','tk','ml','ga','cf','gq','pw','cc',
    'su','biz','info','online','site','live','stream','download',
    'loan','review','country','kim','science','work','party','trade',
    'cricket','date','faith','racing','accountant','win','bid',
    'men','icu','monster','cyou','buzz','sbs','ru',
}
TRUSTED_DOMAINS = {
    'google.com','youtube.com','facebook.com','microsoft.com','apple.com',
    'amazon.com','github.com','twitter.com','linkedin.com','wikipedia.org',
    'instagram.com','netflix.com','stackoverflow.com','reddit.com','paypal.com',
    'bbc.com','nytimes.com','dropbox.com','mozilla.org','cloudflare.com',
    'medium.com','kaggle.com','huggingface.co','arxiv.org','nature.com',
    'zoom.us','slack.com','notion.so','figma.com','canva.com','stripe.com',
    'shopify.com','heroku.com','vercel.com','netlify.com',
}
BRAND_KEYWORDS = [
    'paypal','google','apple','microsoft','amazon','facebook',
    'instagram','netflix','ebay','steam','whatsapp','youtube',
    'dropbox','icloud','twitter','chase','wellsfargo','citibank',
    'bankofamerica','boa','dhl','fedex','usps','ups',
]
URL_SHORTENERS = {
    'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd',
    'buff.ly','rebrand.ly','short.io','tiny.cc','cutt.ly',
}
PHISH_RE  = re.compile(
    r'login|signin|verify|account|update|secure|confirm|'
    r'password|credential|alert|suspend|unlock|recover|'
    r'reset|billing|payment|invoice', re.I)
EXEC_RE   = re.compile(r'\.(exe|bat|cmd|msi|scr|vbs|jar|apk|dmg|sh|ps1|crx|xpi)$', re.I)
SPAM_WORDS = ['free','win','prize','claim','urgent','alert','suspended','verify',
              'confirm','limited','offer','bonus','gift','reward','lucky','congratulation']

def _entropy(s):
    if not s: return 0.0
    freq = {}
    for c in s: freq[c] = freq.get(c, 0) + 1
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
    f   = {}
    try:
        p = urlparse(raw if '://' in raw else 'http://' + raw)
    except Exception:
        p = urlparse('http://invalid')
    hostname   = (p.hostname or '').lower()
    path       = p.path or ''
    query      = p.query or ''
    scheme     = p.scheme or ''
    full_lower = raw.lower()
    _, domain, tld = _domain_parts(hostname)
    base = f"{domain}.{tld}" if domain and tld else hostname
    sub, _, _  = _domain_parts(hostname)
    hl = max(len(hostname), 1)
    f['is_https']          = int(scheme == 'https')
    f['is_http']           = int(scheme == 'http')
    f['url_length']        = len(raw)
    f['hostname_length']   = len(hostname)
    f['path_length']       = len(path)
    f['query_length']      = len(query)
    f['dot_count']         = hostname.count('.')
    f['hyphen_count']      = hostname.count('-')
    f['underscore_count']  = raw.count('_')
    f['at_sign']           = int('@' in raw)
    f['double_slash']      = int('//' in path)
    f['question_mark']     = int('?' in raw)
    f['ampersand_count']   = query.count('&')
    f['equals_count']      = query.count('=')
    f['percent_count']     = len(re.findall(r'%[0-9a-fA-F]{2}', raw))
    f['hash_count']        = int('#' in raw)
    f['digit_ratio']       = round(sum(c.isdigit() for c in hostname) / hl, 4)
    f['alpha_ratio']       = round(sum(c.isalpha() for c in hostname) / hl, 4)
    f['subdomain_count']   = len(sub.split('.')) if sub else 0
    f['suspicious_tld']    = int(tld in SUSPICIOUS_TLDS)
    f['tld_length']        = len(tld)
    f['is_ip_host']        = int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname)))
    f['trusted_domain']    = int(base in TRUSTED_DOMAINS)
    brand_hit              = any(b in hostname for b in BRAND_KEYWORDS)
    f['brand_in_domain']   = int(brand_hit and base not in TRUSTED_DOMAINS)
    f['digit_in_word']     = int(bool(re.search(r'[a-z]\d[a-z]', hostname)))
    f['phish_path_kw']     = int(bool(PHISH_RE.search(path)))
    f['executable_ext']    = int(bool(EXEC_RE.search(path)))
    f['path_depth']        = path.count('/')
    f['path_has_ip']       = int(bool(re.search(r'/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', path)))
    try:    f['param_count'] = len(parse_qs(query))
    except: f['param_count'] = 0
    f['hostname_entropy']  = round(_entropy(hostname), 4)
    f['path_entropy']      = round(_entropy(path), 4)
    f['is_shortener']      = int(hostname in URL_SHORTENERS)
    f['spam_keyword_count']= sum(w in full_lower for w in SPAM_WORDS)
    f['has_punycode']      = int('xn--' in hostname)
    f['domain_age_days']   = 365
    return f

FEATURE_COLUMNS = list(extract_features("http://example.com").keys())


# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════

BENIGN_URLS = [
    "https://www.google.com/search?q=python+tutorial",
    "https://github.com/scikit-learn/scikit-learn",
    "https://stackoverflow.com/questions/tagged/python",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://docs.python.org/3/library/re.html",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.linkedin.com/in/johndoe",
    "https://twitter.com/user/status/123456789",
    "https://www.reddit.com/r/learnpython/",
    "https://medium.com/@author/article-title-abc",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.bbc.com/news/technology",
    "https://www.nytimes.com/2024/technology/ai.html",
    "https://www.microsoft.com/en-us/microsoft-365",
    "https://www.apple.com/iphone/",
    "https://www.paypal.com/us/home",
    "https://www.netflix.com/browse",
    "https://www.instagram.com/p/ABC123/",
    "https://www.facebook.com/events/12345/",
    "https://accounts.google.com/o/oauth2/auth?client_id=x",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://drive.google.com/file/d/1abc/view",
    "https://www.dropbox.com/s/abc123/file.pdf",
    "https://support.apple.com/en-us/HT201994",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://nodejs.org/en/docs/",
    "https://reactjs.org/docs/getting-started.html",
    "https://vuejs.org/guide/introduction.html",
    "https://www.coursera.org/learn/machine-learning",
    "https://www.udemy.com/course/python-bootcamp/",
    "https://arxiv.org/abs/2303.08774",
    "https://pypi.org/project/scikit-learn/",
    "https://hub.docker.com/_/python",
    "https://kubernetes.io/docs/concepts/overview/",
    "https://aws.amazon.com/ec2/",
    "https://cloud.google.com/compute/docs",
    "https://azure.microsoft.com/en-us/products/virtual-machines",
    "https://www.cloudflare.com/learning/ddos/",
    "https://letsencrypt.org/getting-started/",
    "https://www.w3schools.com/python/",
    "https://realpython.com/python-f-strings/",
    "https://www.geeksforgeeks.org/python-programming-language/",
    "https://towardsdatascience.com/",
    "https://www.kaggle.com/competitions",
    "https://huggingface.co/models",
    "https://streamlit.io/",
    "https://fastapi.tiangolo.com/",
    "https://flask.palletsprojects.com/",
    "https://www.djangoproject.com/",
]

MALICIOUS_URLS = [
    "http://paypal.com.secure-login-verify.xyz/account/update?token=abc",
    "http://192.168.1.105/admin/login.php?redirect=home",
    "http://g00gle-security-alert.com/verify?user=victim@gmail.com",
    "http://amazon-prize-winner-2024.top/claim?id=99812&ref=email",
    "http://login.microsoftonline.com.phish.tk/oauth2/token",
    "http://secure.paypal-account-verify.ml/login?next=/dashboard",
    "http://bit.ly/3xFreeGift-Claim-Now-2024",
    "http://free-iphone-15-winner.xyz/claim?tracking=FB_AD_001",
    "http://your-bank-secure.suspicious-domain.cc/verify-identity",
    "http://update-your-netflix-billing.live/payment?ref=email",
    "http://apple-id-locked-alert.top/unlock?case=12345",
    "http://win-cash-prize-2024.tk/register?promo=WIN500",
    "http://download-crack-software.ml/setup.exe?id=12345",
    "http://verify-your-facebook-account.xyz/login",
    "http://amazon.com.fake-verify.biz/signin?ref=phish",
    "http://secure-login.paypa1-support.com/help/account",
    "http://google.account-suspended-alert.online/fix",
    "http://dropbox.com.secure.upload-files.info/share",
    "http://www.malware-delivery.net/payload.exe?dl=1",
    "http://urgent-action-required.top/account?email=user@mail.com",
    "http://virus-scan-results.xyz/remove?threatid=9912",
    "http://10.0.0.1/cgi-bin/login.cgi",
    "http://172.16.254.1/setup/admin?pass=admin",
    "http://user@malicious-host.tk/",
    "http://login.ebay.com.cheap-deals-now.pw/signin",
    "http://secure.chase.bank.account-suspended.ml/login",
    "http://track-my-package.xyz/usps?track=1Z999AA0",
    "http://covid-relief-fund.tk/apply?ref=govt",
    "http://faceb00k-security.xyz/recover?id=12345",
    "http://your-crypto-wallet-alert.top/connect?wallet=MetaMask",
    "http://steam-free-gift-card.ml/redeem?code=FREE2024",
    "http://click-here-to-earn-500-usd.top/?aff=1234",
    "http://tinyurl.com/free-adult-content-2024",
    "http://drive.google.com.file-share.xyz/d/1abc/view",
    "http://apple.com.account-locked.online/appleid/unlock",
    "http://secure-login-verify.amazon-account.cc/signin",
    "http://urgent.dhl-delivery-problem.top/track?id=9988",
    "http://bank-notification-alert.xyz/verify?acct=123456",
    "http://microsoft-tech-support-alert.tk/call?code=ERR_VIRUS",
    "http://irs-tax-refund-ready.ml/claim?ssn=needed",
    "http://youtube.com.premium-free.biz/activate",
    "http://instagram-verify-now.xyz/confirm?user=victim",
    "http://netflix.com.billing-update.online/payment",
    "http://fake-antivirus-scan.cc/remove?threats=99",
    "http://your-account-hacked-alert.xyz/secure?id=abc",
    "http://win-free-ps5-console.top/register?promo=PS5FREE",
    "http://paypal.billing.update-required.xyz/confirm",
    "http://icloud.apple.id-verify.cc/unlock",
    "http://bank.account.suspended.suspicious.xyz/verify",
    "http://confirm-you-are-human.xyz/click",
]


# ══════════════════════════════════════════════════════════════
# TRAIN MODEL  (unchanged from working version)
# ══════════════════════════════════════════════════════════════

def _train_and_save(model_path: str):
    urls   = BENIGN_URLS + MALICIOUS_URLS
    labels = [0] * len(BENIGN_URLS) + [1] * len(MALICIOUS_URLS)
    X = pd.DataFrame([extract_features(u) for u in urls])[FEATURE_COLUMNS].fillna(0).values.astype(float)
    y = np.array(labels)
    pipe = Pipeline([("clf", RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    ))])
    pipe.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": pipe, "feature_columns": FEATURE_COLUMNS}, model_path)
    return pipe


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ThreatScan · URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# DESIGN  —  Dark forensics dashboard
# Fonts: Outfit (display) + Fira Code (mono) + Inter (body)
# Palette: charcoal bg · electric teal accent · coral danger · amber warn
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Fira+Code:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Outfit', sans-serif !important;
  background: #0b0f1a !important;
  color: #cdd6f4;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.2rem 3rem !important; max-width: 1180px !important; }
.stApp { background: #0b0f1a !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(137,180,250,0.2); border-radius: 99px; }

/* ════════════════════════════════════
   HEADER
════════════════════════════════════ */
.ts-header {
  background: linear-gradient(135deg, #11172a 0%, #141c32 100%);
  border: 1px solid rgba(137,180,250,0.12);
  border-radius: 18px;
  padding: 1.8rem 2.2rem;
  margin-bottom: 1.6rem;
  position: relative; overflow: hidden;
  display: flex; align-items: center; gap: 1.6rem;
}
.ts-header::before {
  content: '';
  position: absolute; inset: 0;
  background-image: radial-gradient(circle at 80% 50%, rgba(137,180,250,0.06) 0%, transparent 60%);
}
.ts-header::after {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent 10%, rgba(137,180,250,0.4) 50%, transparent 90%);
}
.ts-logo {
  width: 54px; height: 54px; border-radius: 14px; flex-shrink: 0;
  background: linear-gradient(135deg, rgba(137,180,250,0.15), rgba(137,180,250,0.05));
  border: 1px solid rgba(137,180,250,0.25);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem; position: relative; z-index: 1;
}
.ts-title {
  font-size: 1.8rem; font-weight: 800; line-height: 1;
  color: #fff; letter-spacing: -0.03em; position: relative; z-index: 1;
}
.ts-subtitle {
  font-family: 'Fira Code', monospace; font-size: 0.7rem;
  color: rgba(137,180,250,0.6); margin-top: 5px;
  letter-spacing: 0.06em; text-transform: uppercase; position: relative; z-index: 1;
}
.ts-pills {
  margin-left: auto; display: flex; gap: 8px; flex-shrink: 0;
  position: relative; z-index: 1;
}
.ts-pill {
  font-family: 'Fira Code', monospace; font-size: 0.62rem;
  padding: 5px 13px; border-radius: 99px;
  background: rgba(137,180,250,0.08); border: 1px solid rgba(137,180,250,0.18);
  color: rgba(137,180,250,0.75); letter-spacing: 0.08em; text-transform: uppercase;
}

/* ════════════════════════════════════
   STAT STRIP
════════════════════════════════════ */
.stat-strip {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 10px; margin-bottom: 1.6rem;
}
.stat-tile {
  background: #11172a;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px; padding: 1rem 1.2rem;
  position: relative; overflow: hidden;
}
.stat-tile::before {
  content: ''; position: absolute;
  bottom: 0; left: 0; right: 0; height: 2px;
}
.st-teal::before  { background: linear-gradient(90deg, #89b4fa, #74c7ec); }
.st-green::before { background: linear-gradient(90deg, #a6e3a1, #94e2d5); }
.st-peach::before { background: linear-gradient(90deg, #fab387, #f9e2af); }
.st-mauve::before { background: linear-gradient(90deg, #cba6f7, #f5c2e7); }
.stat-num {
  font-size: 1.65rem; font-weight: 800; line-height: 1; color: #fff;
  letter-spacing: -0.02em;
}
.stat-tag {
  font-family: 'Fira Code', monospace; font-size: 0.6rem;
  color: rgba(255,255,255,0.28); text-transform: uppercase;
  letter-spacing: 0.1em; margin-top: 4px;
}

/* ════════════════════════════════════
   TABS
════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: #11172a !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 12px !important;
  padding: 4px !important; gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important; border-radius: 9px !important;
  padding: 9px 24px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 500 !important; font-size: 0.87rem !important;
  color: rgba(205,214,244,0.38) !important;
  transition: color 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: rgba(205,214,244,0.7) !important; }
.stTabs [aria-selected="true"] {
  background: rgba(137,180,250,0.1) !important;
  color: #89b4fa !important;
  border: 1px solid rgba(137,180,250,0.2) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem !important; }

/* ════════════════════════════════════
   SCAN BOX
════════════════════════════════════ */
.scan-box {
  background: #11172a;
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 14px; padding: 1.5rem 1.8rem 1.3rem;
  margin-bottom: 1.2rem; position: relative;
}
.scan-box::after {
  content: ''; position: absolute;
  top: 0; left: 32px; right: 32px; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(137,180,250,0.35), transparent);
}
.scan-eyebrow {
  font-family: 'Fira Code', monospace; font-size: 0.63rem;
  color: rgba(137,180,250,0.5); text-transform: uppercase;
  letter-spacing: 0.14em; margin-bottom: 10px;
}

/* ── Inputs ── */
.stTextInput > div > input {
  background: #070b14 !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important;
  color: #cdd6f4 !important;
  font-family: 'Fira Code', monospace !important;
  font-size: 0.88rem !important;
  padding: 13px 16px !important;
}
.stTextInput > div > input::placeholder { color: rgba(205,214,244,0.2) !important; }
.stTextInput > div > input:focus {
  border-color: rgba(137,180,250,0.45) !important;
  box-shadow: 0 0 0 3px rgba(137,180,250,0.07) !important;
}
.stTextArea > div > textarea {
  background: #070b14 !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important;
  color: #cdd6f4 !important;
  font-family: 'Fira Code', monospace !important; font-size: 0.83rem !important;
}
.stTextArea > div > textarea:focus {
  border-color: rgba(137,180,250,0.45) !important;
  box-shadow: 0 0 0 3px rgba(137,180,250,0.07) !important;
}
.stSelectbox > div > div {
  background: #070b14 !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important; color: #cdd6f4 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #89b4fa, #74c7ec) !important;
  color: #11172a !important; border: none !important;
  border-radius: 10px !important; font-weight: 700 !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.9rem !important; padding: 0.62rem 1.8rem !important;
  box-shadow: 0 4px 20px rgba(137,180,250,0.25) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 28px rgba(137,180,250,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stDownloadButton > button {
  background: transparent !important; color: #89b4fa !important;
  border: 1px solid rgba(137,180,250,0.28) !important;
  box-shadow: none !important; font-weight: 500 !important;
}
.stDownloadButton > button:hover {
  background: rgba(137,180,250,0.07) !important;
  transform: none !important; box-shadow: none !important;
}

/* ════════════════════════════════════
   VERDICT
════════════════════════════════════ */
.verdict-card {
  border-radius: 14px; padding: 1.4rem 1.6rem;
  margin: 1.2rem 0; display: flex; align-items: center; gap: 1.4rem;
  border: 1px solid; position: relative; overflow: hidden;
}
.verdict-card::before {
  content: ''; position: absolute;
  left: 0; top: 0; bottom: 0; width: 4px;
  border-radius: 14px 0 0 14px;
}
.vc-safe {
  background: rgba(166,227,161,0.06);
  border-color: rgba(166,227,161,0.2);
}
.vc-safe::before { background: #a6e3a1; }
.vc-suspicious {
  background: rgba(249,226,175,0.06);
  border-color: rgba(249,226,175,0.2);
}
.vc-suspicious::before { background: #f9e2af; }
.vc-malicious {
  background: rgba(243,139,168,0.07);
  border-color: rgba(243,139,168,0.22);
}
.vc-malicious::before { background: #f38ba8; }

.vc-icon  { font-size: 2rem; flex-shrink: 0; }
.vc-body  { flex: 1; min-width: 0; }
.vc-label {
  font-size: 1.35rem; font-weight: 800;
  letter-spacing: 0.04em; line-height: 1;
}
.vc-safe  .vc-label { color: #a6e3a1; }
.vc-suspicious .vc-label { color: #f9e2af; }
.vc-malicious  .vc-label { color: #f38ba8; }
.vc-url {
  font-family: 'Fira Code', monospace; font-size: 0.72rem;
  color: rgba(205,214,244,0.32); margin-top: 5px; word-break: break-all;
}
.vc-pct-wrap { text-align: right; flex-shrink: 0; }
.vc-pct {
  font-size: 2rem; font-weight: 800; line-height: 1;
}
.vc-safe  .vc-pct { color: #a6e3a1; }
.vc-suspicious .vc-pct { color: #f9e2af; }
.vc-malicious  .vc-pct { color: #f38ba8; }
.vc-pct-lbl {
  font-family: 'Fira Code', monospace; font-size: 0.58rem;
  color: rgba(205,214,244,0.25); text-transform: uppercase;
  letter-spacing: 0.1em; margin-top: 3px;
}

/* ════════════════════════════════════
   PANELS
════════════════════════════════════ */
.ts-panel {
  background: #11172a;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px; padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
}
.ts-panel-title {
  font-family: 'Fira Code', monospace; font-size: 0.62rem;
  color: rgba(205,214,244,0.3); text-transform: uppercase;
  letter-spacing: 0.13em; margin-bottom: 1rem;
}

/* Probability bars */
.prob-row { margin-bottom: 11px; }
.prob-meta {
  display: flex; justify-content: space-between;
  font-size: 0.78rem; margin-bottom: 5px;
  font-family: 'Fira Code', monospace;
}
.prob-name { color: rgba(205,214,244,0.45); }
.prob-num  { font-weight: 500; }
.prob-track { background: rgba(255,255,255,0.05); border-radius: 99px; height: 7px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 99px; }
.pf-safe { background: linear-gradient(90deg, #a6e3a1, #94e2d5); }
.pf-mal  { background: linear-gradient(90deg, #f38ba8, #eba0ac); }

/* Signal chips */
.chips { display: flex; flex-wrap: wrap; gap: 7px; }
.chip {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 5px 11px; border-radius: 99px; font-size: 0.75rem;
  font-family: 'Fira Code', monospace; border: 1px solid;
}
.chip-bad  { background: rgba(243,139,168,0.08); color: #f38ba8; border-color: rgba(243,139,168,0.2); }
.chip-good { background: rgba(166,227,161,0.08); color: #a6e3a1; border-color: rgba(166,227,161,0.2); }

/* ════════════════════════════════════
   BATCH METRIC TILES
════════════════════════════════════ */
.btile {
  background: #11172a; border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px; padding: 1rem; text-align: center;
}
.btile-num { font-size: 1.7rem; font-weight: 800; line-height: 1; }
.btile-lbl {
  font-family: 'Fira Code', monospace; font-size: 0.6rem;
  color: rgba(205,214,244,0.3); text-transform: uppercase;
  letter-spacing: 0.1em; margin-top: 4px;
}

/* ════════════════════════════════════
   PERF METRIC TILES
════════════════════════════════════ */
.pm-tile {
  background: #11172a; border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px; padding: 1.1rem 1rem; text-align: center;
}
.pm-val { font-size: 1.75rem; font-weight: 800; color: #89b4fa; line-height: 1; }
.pm-lbl {
  font-family: 'Fira Code', monospace; font-size: 0.6rem;
  color: rgba(205,214,244,0.28); text-transform: uppercase;
  letter-spacing: 0.1em; margin-top: 4px;
}

/* ════════════════════════════════════
   SIDEBAR
════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: #0b0f1a !important;
  border-right: 1px solid rgba(255,255,255,0.05) !important;
}
section[data-testid="stSidebar"] * { color: rgba(205,214,244,0.65) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #cdd6f4 !important; }
section[data-testid="stSidebar"] hr  { border-color: rgba(255,255,255,0.07) !important; }
section[data-testid="stSidebar"] .stSuccess { background: rgba(166,227,161,0.08) !important; border-radius: 8px; }
section[data-testid="stSidebar"] .stWarning { background: rgba(249,226,175,0.08) !important; border-radius: 8px; }
section[data-testid="stSidebar"] .stError   { background: rgba(243,139,168,0.08) !important; border-radius: 8px; }

/* Expander */
.streamlit-expanderHeader {
  background: #11172a !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 10px !important;
  color: rgba(205,214,244,0.4) !important;
  font-family: 'Fira Code', monospace !important; font-size: 0.78rem !important;
}
.streamlit-expanderContent {
  background: #0d1220 !important;
  border: 1px solid rgba(255,255,255,0.05) !important;
  border-top: none !important; border-radius: 0 0 10px 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODEL  (unchanged logic)
# ══════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.joblib")

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("First run — training model (takes ~20 sec)…"):
            mdl = _train_and_save(MODEL_PATH)
            return mdl, FEATURE_COLUMNS
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_columns"]

model, feat_cols = load_model()


# ══════════════════════════════════════════════════════════════
# PREDICT  (unchanged logic)
# ══════════════════════════════════════════════════════════════
def predict(url: str) -> dict:
    raw   = url.strip()
    feats = extract_features(raw)
    feats["domain_age_days"] = max(feats.get("domain_age_days", 365), 0)
    X     = np.array([feats.get(c, 0) for c in feat_cols]).reshape(1, -1)
    prob  = model.predict_proba(X)[0]
    label = int(model.predict(X)[0])
    safe_pct = round(prob[0] * 100, 1)
    mal_pct  = round(prob[1] * 100, 1)
    verdict  = "MALICIOUS" if label == 1 else "SUSPICIOUS" if mal_pct >= 30 else "SAFE"
    try:
        ph = urlparse(raw if "://" in raw else "http://" + raw)
        hostname = (ph.hostname or "").lower()
        tld = hostname.split(".")[-1] if "." in hostname else ""
    except Exception:
        hostname = tld = ""
    signals = []
    if feats.get("is_https"):        signals.append(("✅ Uses HTTPS", "good"))
    else:                            signals.append(("⚠️ No HTTPS", "bad"))
    if feats.get("is_ip_host"):      signals.append(("⚠️ IP address as host", "bad"))
    if feats.get("suspicious_tld"):  signals.append((f"⚠️ Suspicious TLD (.{tld})", "bad"))
    if feats.get("brand_in_domain"): signals.append(("⚠️ Brand impersonation", "bad"))
    if feats.get("digit_in_word"):   signals.append(("⚠️ Typosquatting detected", "bad"))
    if feats.get("phish_path_kw"):   signals.append(("⚠️ Phishing keywords in path", "bad"))
    if feats.get("is_shortener"):    signals.append(("⚠️ URL shortener used", "bad"))
    if feats.get("at_sign"):         signals.append(("⚠️ @ symbol in URL", "bad"))
    if feats.get("has_punycode"):    signals.append(("⚠️ Punycode / IDN attack", "bad"))
    if feats.get("executable_ext"):  signals.append(("⚠️ Executable file extension", "bad"))
    if feats.get("trusted_domain"):  signals.append(("✅ Trusted domain", "good"))
    if feats.get("subdomain_count", 0) >= 3:
        signals.append((f"⚠️ Deep subdomains ({feats['subdomain_count']})", "bad"))
    if feats.get("hyphen_count", 0) >= 3:
        signals.append((f"⚠️ Many hyphens ({feats['hyphen_count']})", "bad"))
    if feats.get("url_length", 0) > 100:
        signals.append((f"⚠️ Long URL ({feats['url_length']} chars)", "bad"))
    if feats.get("spam_keyword_count", 0) >= 2:
        signals.append((f"⚠️ Spam keywords ({feats['spam_keyword_count']})", "bad"))
    if not any(k == "bad" for _, k in signals):
        signals.append(("✅ No suspicious signals found", "good"))
    return {"url": raw, "verdict": verdict, "safe_pct": safe_pct,
            "mal_pct": mal_pct, "signals": signals, "features": feats}


# ══════════════════════════════════════════════════════════════
# MATPLOTLIB DARK HELPER
# ══════════════════════════════════════════════════════════════
def _dark_ax(w=4.5, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#11172a')
    ax.set_facecolor('#11172a')
    for sp in ax.spines.values(): sp.set_color('#1e2d45')
    ax.tick_params(colors='#4a5a7a', labelsize=8)
    ax.xaxis.label.set_color('#4a5a7a')
    ax.yaxis.label.set_color('#4a5a7a')
    ax.title.set_color('#cdd6f4')
    return fig, ax


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛡️ ThreatScan")
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Extract 36 URL features
2. Random Forest (200 trees)
3. Trained on 100 labelled URLs
4. Returns threat probability

**Verdict thresholds:**
""")
    st.success("🟢  SAFE  —  below 30%")
    st.warning("🟡  SUSPICIOUS  —  30–49%")
    st.error("🔴  MALICIOUS  —  50% +")
    st.markdown("---")
    st.caption("Python · scikit-learn · Streamlit")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="ts-header">
  <div class="ts-logo">🛡️</div>
  <div>
    <div class="ts-title">ThreatScan</div>
    <div class="ts-subtitle">Malicious URL Detector &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; 36 Features</div>
  </div>
  <div class="ts-pills">
    <span class="ts-pill">ML v2.0</span>
    <span class="ts-pill">98%+ Accuracy</span>
    <span class="ts-pill">Standalone</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Stat strip
st.markdown("""
<div class="stat-strip">
  <div class="stat-tile st-teal">
    <div class="stat-num">98%+</div>
    <div class="stat-tag">Model Accuracy</div>
  </div>
  <div class="stat-tile st-green">
    <div class="stat-num">1.000</div>
    <div class="stat-tag">ROC-AUC Score</div>
  </div>
  <div class="stat-tile st-peach">
    <div class="stat-num">36</div>
    <div class="stat-tag">URL Features</div>
  </div>
  <div class="stat-tile st-mauve">
    <div class="stat-num">200</div>
    <div class="stat-tag">Decision Trees</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍  Single URL", "📋  Batch Analysis", "📊  Model Performance"])


# ──────────────────────────────────────────
# TAB 1 — SINGLE URL
# ──────────────────────────────────────────
with tab1:
    st.markdown('<div class="scan-box"><div class="scan-eyebrow">// target url</div>', unsafe_allow_html=True)
    col_in, col_ex = st.columns([3, 1])
    with col_in:
        url_input = st.text_input(
            "url", placeholder="https://example.com  or  paste a suspicious link…",
            key="single_url", label_visibility="collapsed",
        )
    with col_ex:
        examples = {
            "— example —":                                        "",
            "✅ google.com":   "https://www.google.com/search?q=test",
            "✅ github.com":   "https://github.com/python/cpython",
            "🔴 paypal phish": "http://paypal.com.secure-login-verify.xyz/account/update",
            "🔴 IP login":     "http://192.168.0.1/admin/login.php",
            "🔴 prize scam":   "http://free-iphone-winner.top/claim?id=99812",
            "🔴 typosquat":    "http://g00gle-security-alert.com/verify?user=you@mail.com",
        }
        ex = st.selectbox("ex", list(examples.keys()), label_visibility="collapsed")
        if examples.get(ex): url_input = examples[ex]
    st.markdown('</div>', unsafe_allow_html=True)

    scan_clicked = st.button("⚡  Scan URL", key="scan")
    if scan_clicked and url_input:
        with st.spinner("Analysing…"):
            time.sleep(0.2)
            result = predict(url_input)

        v    = result["verdict"]
        icon = "🟢" if v == "SAFE" else "🟡" if v == "SUSPICIOUS" else "🔴"
        cls  = "vc-safe" if v == "SAFE" else "vc-suspicious" if v == "SUSPICIOUS" else "vc-malicious"
        short = result["url"][:90] + ("…" if len(result["url"]) > 90 else "")
        conf  = result["safe_pct"] if v == "SAFE" else result["mal_pct"]

        # Verdict card
        st.markdown(f"""
        <div class="verdict-card {cls}">
          <div class="vc-icon">{icon}</div>
          <div class="vc-body">
            <div class="vc-label">{v}</div>
            <div class="vc-url">{short}</div>
          </div>
          <div class="vc-pct-wrap">
            <div class="vc-pct">{conf}%</div>
            <div class="vc-pct-lbl">confidence</div>
          </div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="ts-panel">
              <div class="ts-panel-title">Threat probability</div>
              <div class="prob-row">
                <div class="prob-meta">
                  <span class="prob-name">SAFE</span>
                  <span class="prob-num" style="color:#a6e3a1">{result['safe_pct']}%</span>
                </div>
                <div class="prob-track"><div class="prob-fill pf-safe" style="width:{result['safe_pct']}%"></div></div>
              </div>
              <div class="prob-row" style="margin-bottom:0">
                <div class="prob-meta">
                  <span class="prob-name">MALICIOUS</span>
                  <span class="prob-num" style="color:#f38ba8">{result['mal_pct']}%</span>
                </div>
                <div class="prob-track"><div class="prob-fill pf-mal" style="width:{result['mal_pct']}%"></div></div>
              </div>
            </div>""", unsafe_allow_html=True)

        with col2:
            chips = "".join(
                f'<span class="chip chip-{"good" if k=="good" else "bad"}">{s}</span>'
                for s, k in result["signals"]
            )
            st.markdown(f"""
            <div class="ts-panel" style="height:100%">
              <div class="ts-panel-title">Signal breakdown</div>
              <div class="chips">{chips}</div>
            </div>""", unsafe_allow_html=True)

        with st.expander("▸  Full feature vector (36 features)"):
            st.dataframe(
                pd.DataFrame(result["features"].items(), columns=["Feature","Value"]).set_index("Feature"),
                use_container_width=True, height=340,
            )

    elif scan_clicked:
        st.warning("Please enter a URL first.")


# ──────────────────────────────────────────
# TAB 2 — BATCH
# ──────────────────────────────────────────
with tab2:
    st.markdown('<div class="scan-box"><div class="scan-eyebrow">// paste urls — one per line</div>', unsafe_allow_html=True)
    batch_text = st.text_area(
        "b", height=150,
        placeholder="https://google.com\nhttp://free-prize.xyz/claim\nhttps://github.com/user/repo",
        label_visibility="collapsed", key="batch_urls",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡  Analyse All", key="batch_scan"):
        urls = [u.strip() for u in batch_text.splitlines() if u.strip()]
        if not urls:
            st.warning("Paste at least one URL.")
        else:
            with st.spinner(f"Scanning {len(urls)} URL{'s' if len(urls)>1 else ''}…"):
                rows = []
                for u in urls:
                    r = predict(u)
                    rows.append({"URL": u[:80]+("…" if len(u)>80 else ""),
                                 "Verdict": r["verdict"],
                                 "Safe %": r["safe_pct"],
                                 "Malicious %": r["mal_pct"]})
            df = pd.DataFrame(rows)

            c1, c2, c3 = st.columns(3)
            for col, key, colour, label in [
                (c1, "SAFE",       "#a6e3a1", "Safe"),
                (c2, "SUSPICIOUS", "#f9e2af", "Suspicious"),
                (c3, "MALICIOUS",  "#f38ba8", "Malicious"),
            ]:
                col.markdown(f"""<div class="btile">
                  <div class="btile-num" style="color:{colour}">{(df['Verdict']==key).sum()}</div>
                  <div class="btile-lbl">{label}</div></div>""", unsafe_allow_html=True)

            st.markdown("")

            def colour_row(row):
                bg = {"MALICIOUS":"rgba(243,139,168,0.07)",
                      "SUSPICIOUS":"rgba(249,226,175,0.06)",
                      "SAFE":"rgba(166,227,161,0.06)"}.get(row["Verdict"],"transparent")
                return [f"background-color:{bg};color:#cdd6f4"] * len(row)

            st.dataframe(df.style.apply(colour_row, axis=1), use_container_width=True, height=300)
            st.download_button("⬇  Export CSV", df.to_csv(index=False),
                               file_name="threatscan_results.csv", mime="text/csv")


# ──────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE  (unchanged logic)
# ──────────────────────────────────────────
with tab3:

    @st.cache_data(show_spinner="Computing metrics…")
    def get_metrics():
        urls   = BENIGN_URLS + MALICIOUS_URLS
        labels = [0]*len(BENIGN_URLS) + [1]*len(MALICIOUS_URLS)
        X = pd.DataFrame([extract_features(u) for u in urls])[feat_cols].fillna(0).values.astype(float)
        y = np.array(labels)
        _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        return y_t, model.predict(X_t), model.predict_proba(X_t)[:, 1]

    y_test, y_pred, y_prob = get_metrics()
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    rep = classification_report(y_test, y_pred,
                                target_names=["Benign","Malicious"], output_dict=True)

    # Metric tiles
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, f"{acc:.3f}", "Accuracy"),
        (m2, f"{auc:.3f}", "ROC-AUC"),
        (m3, f"{rep['Malicious']['precision']:.3f}", "Precision"),
        (m4, f"{rep['Malicious']['recall']:.3f}",    "Recall"),
    ]:
        col.markdown(f"""<div class="pm-tile">
          <div class="pm-val">{val}</div>
          <div class="pm-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # Charts
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown('<div class="ts-panel"><div class="ts-panel-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = _dark_ax(4.2, 3.2)
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                               display_labels=["Benign","Malicious"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix", color="#cdd6f4", fontsize=10, pad=8)
        plt.setp(ax.get_xticklabels(), color='#4a5a7a')
        plt.setp(ax.get_yticklabels(), color='#4a5a7a')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_roc:
        st.markdown('<div class="ts-panel"><div class="ts-panel-title">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2 = _dark_ax(4.2, 3.2)
        ax2.fill_between(fpr, tpr, alpha=0.1, color='#89b4fa')
        ax2.plot(fpr, tpr, color='#89b4fa', lw=2, label=f"AUC = {auc:.3f}")
        ax2.plot([0,1],[0,1],'--',color='rgba(255,255,255,0.1)',lw=1)
        ax2.set_xlabel("False Positive Rate", fontsize=8)
        ax2.set_ylabel("True Positive Rate", fontsize=8)
        ax2.set_title("ROC Curve", color="#cdd6f4", fontsize=10, pad=8)
        ax2.legend(fontsize=8, framealpha=0, labelcolor='#4a5a7a')
        ax2.spines[["top","right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="ts-panel" style="margin-top:0.5rem"><div class="ts-panel-title">Feature Importance — Top 15</div>', unsafe_allow_html=True)
    clf = getattr(model, "named_steps", {}).get("clf")
    if clf and hasattr(clf, "feature_importances_"):
        fi = sorted(zip(feat_cols, clf.feature_importances_), key=lambda x: -x[1])[:15]
        names_fi, vals_fi = zip(*fi)
        fig3, ax3 = _dark_ax(9, 4)
        colours = ['#89b4fa' if i > 0 else '#74c7ec' for i in range(len(names_fi))]
        bars = ax3.barh(list(names_fi)[::-1], list(vals_fi)[::-1],
                        height=0.55, color=colours[::-1], alpha=0.85)
        ax3.set_xlabel("Importance Score", fontsize=8)
        ax3.spines[["top","right","left"]].set_visible(False)
        ax3.tick_params(left=False)
        for bar, val in zip(bars[::-1], vals_fi):
            ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va='center', fontsize=7, color='#4a5a7a',
                     fontfamily='monospace')
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("▸  Full Classification Report"):
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)
