import time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

def parse_ts(path):
    values = []
    labels = []
    active = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not active:
                active = line.lower() == "@data"
                continue
            sig, lab = line.rsplit(":", 1)
            values.append([float(x) for x in sig.split(",")])
            labels.append(int(float(lab)))
    return np.array(values, dtype=np.float32), np.array(labels)

def extract_groups(X):
    X = np.asarray(X, dtype=np.float64)
    eps = 1e-12
    absx = np.abs(X)
    peak = absx.max(axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    meanabs = absx.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        temporal = np.column_stack([
            skew(X, axis=1, bias=False),
            kurtosis(X, axis=1, fisher=True, bias=False),
            peak / np.maximum(rms, eps),
            peak / np.maximum(meanabs, eps),
            rms / np.maximum(meanabs, eps),
            peak / np.maximum(np.mean(np.sqrt(absx), axis=1)**2, eps),
            np.mean(np.diff(np.signbit(X), axis=1), axis=1),
            np.sqrt(np.mean(np.diff(X, axis=1)**2, axis=1))
        ])
    f, p = welch(X, fs=1., window="hann", nperseg=256, noverlap=128, nfft=256, detrend="constant", axis=1, scaling="density")
    total = p.sum(axis=1, keepdims=True)
    q = p / np.maximum(total, eps)
    freq = f / .5
    centroid = (q * freq).sum(axis=1)
    center = freq[None, :] - centroid[:, None]
    bw = np.sqrt((q * center**2).sum(axis=1))
    spec_skew = (q * center**3).sum(axis=1) / np.maximum(bw**3, eps)
    spec_kurt = (q * center**4).sum(axis=1) / np.maximum(bw**4, eps) - 3
    flat = np.exp(np.mean(np.log(np.maximum(p, eps)), axis=1)) / np.maximum(p.mean(axis=1), eps)
    entropy = -(q * np.log(np.maximum(q, eps))).sum(axis=1) / np.log(len(f))
    roll = freq[np.argmax(np.cumsum(q, axis=1) >= .85, axis=1)]
    peakfreq = freq[np.argmax(p, axis=1)]
    spectral = np.column_stack([centroid, bw, spec_skew, spec_kurt, flat, entropy, roll, peakfreq])
    edges = np.linspace(0, .5, 33)
    bands = np.column_stack([q[:, (f >= a) & ((f < b) if i < 31 else (f <= b))].sum(axis=1) for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))])
    
    temporal = np.nan_to_num(temporal, nan=0., posinf=0., neginf=0.)
    spectral = np.nan_to_num(spectral, nan=0., posinf=0., neginf=0.)
    bands = np.nan_to_num(bands, nan=0., posinf=0., neginf=0.)
    
    return temporal.astype(np.float32), spectral.astype(np.float32), bands.astype(np.float32)

print("Loading train & test sets...")
X_train, y_train = parse_ts("data/FordA_TRAIN.ts")
X_test, y_test = parse_ts("data/FordA_TEST.ts")

print("Extracting feature groups...")
t_tr, s_tr, b_tr = extract_groups(X_train)
t_te, s_te, b_te = extract_groups(X_test)

feature_sets = {
    "Full 48D (Bentuk + PSD + Bandpower)": (np.column_stack([t_tr, s_tr, b_tr]), np.column_stack([t_te, s_te, b_te])),
    "Hanya 8D Statistik Bentuk": (t_tr, t_te),
    "Hanya 8D Statistik Spektrum PSD": (s_tr, s_te),
    "Hanya 32D Bandpower Spektral": (b_tr, b_te),
    "16D (Bentuk + Spektrum PSD)": (np.column_stack([t_tr, s_tr]), np.column_stack([t_te, s_te])),
    "40D (Spektrum PSD + Bandpower)": (np.column_stack([s_tr, b_tr]), np.column_stack([s_te, b_te]))
}

print("\n=== HASIL ABLASI FITUR (Split Resmi: 3601 train, 1320 test) ===")
print(f"{'Konfigurasi Fitur':<40} | {'Dim':<4} | {'Akurasi':<8} | {'Macro-F1':<8} | {'Waktu Fit':<10}")
print("-" * 80)

for name, (F_tr, F_te) in feature_sets.items():
    clf = ExtraTreesClassifier(n_estimators=200, max_features="sqrt", criterion="entropy", bootstrap=False, random_state=0, n_jobs=-1)
    t0 = time.perf_counter()
    clf.fit(F_tr, y_train)
    fit_time = time.perf_counter() - t0
    y_pred = clf.predict(F_te)
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average="macro") * 100
    dim = F_tr.shape[1]
    print(f"{name:<40} | {dim:<4} | {acc:.2f}%  | {f1:.2f}%  | {fit_time:.3f} s")
