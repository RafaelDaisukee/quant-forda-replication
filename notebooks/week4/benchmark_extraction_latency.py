import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import skew, kurtosis

sys.path.append(str(Path("minggu 1/vendor/code").resolve()))
from quant import Quant

def parse_ts(path):
    values = []
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
    return np.array(values, dtype=np.float32)

def compact_features(X):
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
    F = np.nan_to_num(np.column_stack([temporal, spectral, bands]), nan=0., posinf=0., neginf=0.)
    F[total[:, 0] <= eps] = 0
    return F.astype(np.float32)

print("Loading test data...")
X_test = parse_ts("data/FordA_TEST.ts")
N = len(X_test)
print(f"Loaded {N} test samples of length {X_test.shape[1]}")

# 1. Benchmark Welch 48D
_ = compact_features(X_test[:20]) # warmup
welch_times = []
for _ in range(3):
    t0 = time.perf_counter()
    F_welch = compact_features(X_test)
    welch_times.append(time.perf_counter() - t0)

welch_mean = np.mean(welch_times)
welch_per_sample_ms = (welch_mean / N) * 1000

print(f"\n--- Welch 48D Extraction ---")
print(f"Total time for {N} samples: {welch_mean:.4f} s (± {np.std(welch_times):.4f})")
print(f"Latency per sample: {welch_per_sample_ms:.3f} ms")
print(f"Throughput: {N / welch_mean:.1f} samples/sec")
print(f"Output shape: {F_welch.shape}")

# 2. Benchmark QUANT
X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
quant_model = Quant(depth=6, div=4)
_ = quant_model.fit_transform(X_tensor[:20], None) # warmup

quant_times = []
for _ in range(3):
    qm = Quant(depth=6, div=4)
    t0 = time.perf_counter()
    F_quant = qm.fit_transform(X_tensor, None)
    quant_times.append(time.perf_counter() - t0)

quant_mean = np.mean(quant_times)
quant_per_sample_ms = (quant_mean / N) * 1000

print(f"\n--- QUANT 4469D Extraction ---")
print(f"Total time for {N} samples: {quant_mean:.4f} s (± {np.std(quant_times):.4f})")
print(f"Latency per sample: {quant_per_sample_ms:.3f} ms")
print(f"Throughput: {N / quant_mean:.1f} samples/sec")
print(f"Output shape: {F_quant.shape}")

ratio_speedup = quant_mean / welch_mean
print(f"\nFeature Extraction Speedup: Welch is {ratio_speedup:.1f}x FASTER than QUANT")
