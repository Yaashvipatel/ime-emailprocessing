"""
IME ML Classifier — Pure Python (no scikit-learn)
===================================================
TF-IDF + Logistic Regression implemented from scratch.
Zero binary dependencies. Works on any Python 3.8+.
"""

import os
import json
import math
import pickle
import sqlite3
from collections import defaultdict
from datetime import datetime

import tempfile
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DB_PATH = os.path.join(_DATA_DIR, 'ime.db')
# Store model files in temp dir so Flask reloader doesn't detect writes and restart
_MODEL_DIR = os.path.join(tempfile.gettempdir(), 'ime_ml')
MODEL_PATH = os.path.join(_MODEL_DIR, 'ml_model.pkl')
METRICS_PATH = os.path.join(_MODEL_DIR, 'ml_metrics.json')

BUILTIN_TRAINING_DATA = [
    # Tonnage
    ("MV PACIFIC GLORY DWT 56000 OPEN SINGAPORE O/A 5 JUNE 2026 BULK CARRIER LOA 189M FLAG PANAMA CLASS NK", "tonnage"),
    ("VESSEL OPEN: MV ATLANTIC HOPE 82000 DWT BUILT 2015 OPEN ROTTERDAM NETHERLANDS 10-15 JUNE SCRUBBER FITTED", "tonnage"),
    ("M/V GOLDEN STAR DWT 32000 MT FLAG LIBERIA OPEN GUANGZHOU CHINA 20 MAY GRAIN CAP 42000 CBM 4 CRANES", "tonnage"),
    ("MV OCEAN BREEZE 63000 DWT OPEN BUSAN KOREA O/A 3RD JUNE 2026 CLASS LR LOA 199M BEAM 32M", "tonnage"),
    ("MV CORAL SEA DWT 37500 OPEN CHITTAGONG BANGLADESH 8-12 JUNE 2026 SDBC GEAR 4X30T", "tonnage"),
    ("OPEN VESSEL: MV SHINING STAR 56500 DWT OPEN PORT KELANG MALAYSIA 1ST JUNE 2026 CCS CLASS", "tonnage"),
    ("MV NORDIC SPIRIT DWT 76000 BUILT 2017 FLAG HONG KONG OPEN NEWCASTLE AUSTRALIA 25 MAY", "tonnage"),
    ("VESSEL PARTICULARS: MV COSMOS BUILT 2009 DWT 53000 LOA 190M OPEN DAMPIER AUSTRALIA LAYCAN JUNE", "tonnage"),
    # Cargo VC
    ("CARGO OFFER 50000 MTS COAL LOAD PORT NEWCASTLE AUSTRALIA DISCHARGE PORT MUNDRA INDIA LAYCAN 10-20 JUNE 2026 FIOS 3.75 PCT TTL", "cargo_vc"),
    ("25000 MT UREA IN BULK LP BANDAR IMAM KHOMEINI DP KANDLA LAYCAN 15-25 JUNE 2026 PWWD SSHEX 2PCT TTL COMMISSION", "cargo_vc"),
    ("FIRM CARGO: 35000 MTS IRON ORE POL PORT HEDLAND POD QINGDAO CHINA LAYCAN EARLY JULY 2026 MOLCO 5PCT", "cargo_vc"),
    ("30000 MTS GRAIN CARGO LOAD PORT SANTOS BRAZIL DISCHARGE VERACRUZ MEXICO LAYCAN 20-30 JUNE CQD 3.75 TTL", "cargo_vc"),
    ("CARGO: 15000-20000 MTS CLINKER LP AQABA DP KARACHI LAYCAN MID JULY 2026 1000 MTS PWWD FHINC COMM 2.5PCT", "cargo_vc"),
    ("PLEASE OFFER FIRM CARGO 40000 MTS COAL LOADING PORT MUARA INDONESIA DISCHARGE PORT KRISHNAPATNAM INDIA LAYCAN JUNE 2026", "cargo_vc"),
    ("IRON ORE CARGO 65000 MTS POL DAMPIER AUSTRALIA POD BEILUN CHINA LC 5-15 JULY FIOS 3PCT TOTAL COMMISSION", "cargo_vc"),
    ("15000 MT HRC STEEL JEDDAH TO BILBAO LAYCAN 25 JUNE 5 JULY FIOS CQD 3.75 PCT HERE", "cargo_vc"),
    # Cargo TC
    ("TCT CARGO DELIVERY SINGAPORE REDELIVERY JAPAN DURATION 30-35 DAYS WOG LC 10-15 JUNE 2026 GRAIN SMX-UMX 3.75 ADDCOM", "cargo_tc"),
    ("1 TIME CHARTER TRIP DELY ROTTERDAM REDEL SINGAPORE 45-50 DAYS WOG LAWFULS GENS SUPRA CLASS LAYCAN JULY 2026 5PCT ADC", "cargo_tc"),
    ("TC: DELIVERY TIANJIN REDELIVERY WEST AFRICA DURATION ABT 60 DAYS WITH CLINKER PANAMAX 3.75PCT ADDCOM LAYCAN JUNE 2026", "cargo_tc"),
    ("ACCOUNT ABC SHIPPING DELY MUMBAI INDIA REDEL EAST COAST AUSTRALIA DURATION 40 DAYS WOG WITH GRAIN ULTRA MAX 3.75PCT", "cargo_tc"),
    ("1 TCT WITH STEELS DELIVERY ECI REDELIVERY MED VIA GOA 35-40 DAYS WOG LC 15-18 JULY HANDYMAX 3.75 ADC", "cargo_tc"),
    ("TIME CHARTER TRIP SUPRA/ULTRA DELY WORLDWIDE DURATION 1-3 YEARS TRY SHORT PERIOD FLAT OR INDEX 3.75 ADDCOM", "cargo_tc"),
    ("ACC DAI AN TCT WITH GRAINS DELIVERY VANCOUVER LC 10-17 JUNE SMX-UMX REDELIVERY CHITTAGONG 3.75 ADDCOM", "cargo_tc"),
    ("A/C SEA SCHIFFE 1 TCT STEELS GENS LAWFULS 33K DWT DELIVERY ECI LAYCAN 21-23 JULY REDEL ARAG 50-55 DAYS WOG 3.75 ADC", "cargo_tc"),
]

CLASSES = ['tonnage', 'cargo_vc', 'cargo_tc']


# ─────────────────────────────────────────────
# Pure-Python TF-IDF
# ─────────────────────────────────────────────

def tokenize(text):
    """Simple whitespace + punctuation tokenizer, produces unigrams + bigrams."""
    import re
    words = re.findall(r'[a-z0-9]+', text.lower())
    unigrams = [w for w in words if len(w) > 1]
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams)-1)]
    return unigrams + bigrams


def build_tfidf(corpus, max_features=2000):
    """Build TF-IDF vocabulary and document-term matrix."""
    # Count document frequencies
    df = defaultdict(int)
    tokenized = [tokenize(doc) for doc in corpus]
    for tokens in tokenized:
        for t in set(tokens):
            df[t] += 1

    n_docs = len(corpus)
    # Keep top max_features by df
    sorted_vocab = sorted(df.items(), key=lambda x: -x[1])[:max_features]
    vocab = {term: idx for idx, (term, _) in enumerate(sorted_vocab)}

    # Compute IDF
    idf = {}
    for term, idx in vocab.items():
        idf[term] = math.log((1 + n_docs) / (1 + df[term])) + 1.0  # smooth

    # Build TF-IDF vectors (as dicts for sparsity)
    vectors = []
    for tokens in tokenized:
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        vec = {}
        for term, idx in vocab.items():
            if term in tf:
                tfidf_val = (1 + math.log(tf[term])) * idf[term]  # sublinear TF
                vec[idx] = tfidf_val
        # L2 normalize
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        vectors.append({k: v/norm for k, v in vec.items()})

    return vocab, idf, vectors


def transform_tfidf(text, vocab, idf):
    """Transform a single document using existing vocab/idf."""
    import re
    tokens = tokenize(text)
    tf = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    vec = {}
    for term, idx in vocab.items():
        if term in tf:
            tfidf_val = (1 + math.log(tf[term])) * idf[term]
            vec[idx] = tfidf_val
    norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
    return {k: v/norm for k, v in vec.items()}


# ─────────────────────────────────────────────
# Pure-Python Logistic Regression (one-vs-rest)
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def dot_sparse(vec, weights):
    """Dot product of a sparse dict vector with a dense list of weights."""
    return sum(weights[idx] * val for idx, val in vec.items() if idx < len(weights))


def train_logistic(X, y_binary, n_features, lr=0.1, epochs=200, reg=0.01):
    """Train one logistic regression binary classifier."""
    weights = [0.0] * n_features
    bias = 0.0
    for _ in range(epochs):
        for vec, label in zip(X, y_binary):
            pred = sigmoid(dot_sparse(vec, weights) + bias)
            error = pred - label
            for idx, val in vec.items():
                if idx < n_features:
                    weights[idx] -= lr * (error * val + reg * weights[idx])
            bias -= lr * error
    return weights, bias


def softmax(scores):
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


# ─────────────────────────────────────────────
# Cross-validation (stratified k-fold)
# ─────────────────────────────────────────────

def stratified_kfold(X, y, n_splits=5):
    """Yield (train_indices, test_indices) for stratified k-fold."""
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)

    folds = [[] for _ in range(n_splits)]
    for label, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % n_splits].append(idx)

    for k in range(n_splits):
        test = folds[k]
        train = [idx for j, fold in enumerate(folds) if j != k for idx in fold]
        yield train, test


# ─────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_training_data_from_db():
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT raw_text, category FROM emails WHERE raw_text IS NOT NULL AND raw_text != ''")
        rows = c.fetchall()
        conn.close()
        return [(row['raw_text'], row['category']) for row in rows]
    except:
        return []


# ─────────────────────────────────────────────
# Train & persist
# ─────────────────────────────────────────────

def train_classifier():
    """Train pure-Python TF-IDF + Logistic Regression classifier."""
    db_data = get_training_data_from_db()
    all_data = BUILTIN_TRAINING_DATA + db_data

    texts = [d[0] for d in all_data]
    labels = [d[1] for d in all_data]
    n = len(texts)

    # Build vocabulary and TF-IDF vectors
    vocab, idf, X = build_tfidf(texts, max_features=2000)
    n_features = len(vocab)

    # Cross-validation accuracy
    n_splits = min(5, min(labels.count(c) for c in CLASSES))
    n_splits = max(2, n_splits)
    cv_scores = []
    for train_idx, test_idx in stratified_kfold(X, labels, n_splits):
        X_train = [X[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [labels[i] for i in test_idx]

        # Train one-vs-rest
        clfs = {}
        for cls in CLASSES:
            yb = [1 if lbl == cls else 0 for lbl in y_train]
            clfs[cls] = train_logistic(X_train, yb, n_features, epochs=100)

        # Predict
        correct = 0
        for vec, true_label in zip(X_test, y_test):
            scores = [dot_sparse(vec, clfs[cls][0]) + clfs[cls][1] for cls in CLASSES]
            pred = CLASSES[scores.index(max(scores))]
            if pred == true_label:
                correct += 1
        cv_scores.append(correct / len(y_test) if y_test else 0)

    # Train final model on all data
    final_clfs = {}
    for cls in CLASSES:
        yb = [1 if lbl == cls else 0 for lbl in labels]
        final_clfs[cls] = train_logistic(X, yb, n_features, epochs=200)

    # In-sample predictions for confusion matrix
    y_pred = []
    for vec in X:
        scores = [dot_sparse(vec, final_clfs[cls][0]) + final_clfs[cls][1] for cls in CLASSES]
        y_pred.append(CLASSES[scores.index(max(scores))])

    # Confusion matrix
    cm = [[0]*3 for _ in range(3)]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    for true, pred in zip(labels, y_pred):
        cm[cls_idx[true]][cls_idx[pred]] += 1

    # Per-class precision/recall/f1
    per_class = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(3)) - tp
        fn = sum(cm[i][j] for j in range(3)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            'precision': round(precision, 3),
            'recall':    round(recall, 3),
            'f1':        round(f1, 3),
            'support':   labels.count(cls),
        }

    # Top features per class (highest logistic weights)
    top_features = {}
    inv_vocab = {idx: term for term, idx in vocab.items()}
    for cls in CLASSES:
        weights, _ = final_clfs[cls]
        indexed = [(i, w) for i, w in enumerate(weights) if i in inv_vocab]
        indexed.sort(key=lambda x: -x[1])
        top_features[cls] = [
            {'word': inv_vocab[i].replace('_', ' '), 'weight': round(w, 3)}
            for i, w in indexed[:15] if w > 0
        ]

    acc = sum(cv_scores) / len(cv_scores)
    std = math.sqrt(sum((s - acc)**2 for s in cv_scores) / len(cv_scores))

    metrics = {
        'accuracy':         round(acc, 4),
        'cv_scores':        [round(s, 4) for s in cv_scores],
        'cv_std':           round(std, 4),
        'training_samples': n,
        'confusion_matrix': cm,
        'classes':          CLASSES,
        'per_class':        per_class,
        'top_features':     top_features,
        'trained_at':       datetime.now().isoformat(),
        'model_ready':      True,
        'engine':           'Pure-Python TF-IDF + Logistic Regression',
    }

    os.makedirs(_MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'vocab': vocab, 'idf': idf, 'classifiers': final_clfs}, f)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def predict_with_model(text):
    """Predict category using trained ML model."""
    if not os.path.exists(MODEL_PATH):
        train_classifier()

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    vocab  = model['vocab']
    idf    = model['idf']
    clfs   = model['classifiers']

    vec = transform_tfidf(text, vocab, idf)
    raw_scores = [dot_sparse(vec, clfs[cls][0]) + clfs[cls][1] for cls in CLASSES]
    probs = softmax(raw_scores)

    pred_idx = probs.index(max(probs))
    pred = CLASSES[pred_idx]
    confidence = round(max(probs) * 100, 1)

    label_map = {
        'tonnage':  'Tonnage (Open Vessels)',
        'cargo_vc': 'Cargo VC (Voyage Charter)',
        'cargo_tc': 'Cargo TC (Time Charter)',
    }

    return {
        'ml_prediction':    pred,
        'ml_label':         label_map.get(pred, pred),
        'ml_confidence':    confidence,
        'ml_probabilities': {CLASSES[i]: round(probs[i] * 100, 1) for i in range(len(CLASSES))},
        'model_used':       'Pure-Python TF-IDF + Logistic Regression',
    }


def get_model_metrics():
    """Get stored model metrics, training if necessary."""
    if not os.path.exists(METRICS_PATH):
        try:
            return train_classifier()
        except Exception as e:
            return {'model_ready': False, 'error': str(e)}
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)
