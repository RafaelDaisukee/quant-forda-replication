"""QUANT/FordA: unchanged author transform, full data, fixed benchmark protocol."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_URL = 'https://www.timeseriesclassification.com/aeon-toolkit/FordA.zip'
ARCHIVE_SHA = '4d78a72e5165e8da40c51115a8219c9d01978aafac6334ed8dfabd29da359ffb'
CONFIG = {'dataset': 'FordA', 'train': 3601, 'test': 1320, 'length': 500,
          'quant_depth': 6, 'quant_div': 4, 'classifier': 'ExtraTreesClassifier',
          'n_estimators': 200, 'max_features': .1, 'criterion': 'entropy',
          'bootstrap': False, 'resample_ids': list(range(30)), 'classifier_seed_policy': 'resample_id',
          'threads': 2, 'preprocessing': 'archive_values_to_float32_without_new_scaling',
          'resampling': 'resample0_original_other_ids_aeon_classwise_RandomState_shuffle',
          'scope': 'one_complete_dataset_experiment_not_all_142_datasets'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parse_ts(path):
    import numpy as np
    values, labels, active = [], [], False
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        if not active:
            active = line.lower() == '@data'
            continue
        signal, label = line.rsplit(':', 1)
        values.append(np.fromstring(signal, sep=',', dtype=np.float32))
        labels.append(int(label))
    X, y = np.stack(values), np.asarray(labels, dtype=np.int64)
    if X.shape[1] != 500 or not np.isfinite(X).all() or set(y) != {-1, 1}:
        raise ValueError('Isi FordA tidak sesuai format benchmark.')
    return X, y


def data(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    archive = root / 'FordA.zip'
    if not archive.exists() or sha(archive) != ARCHIVE_SHA:
        print('Mengunduh arsip FordA lengkap (sekitar 36 MB)...', flush=True)
        tmp = archive.with_suffix('.partial')
        with urllib.request.urlopen(DATA_URL, timeout=120) as response, tmp.open('wb') as f:
            while chunk := response.read(1048576): f.write(chunk)
        if sha(tmp) != ARCHIVE_SHA:
            raise ValueError('Checksum arsip berbeda; versi dataset berubah atau transfer rusak.')
        os.replace(tmp, archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip(): raise ValueError('CRC arsip gagal.')
        for name in ['FordA_TRAIN.ts', 'FordA_TEST.ts']:
            raw = z.read(name); p = root / name
            if not p.exists() or p.read_bytes() != raw: p.write_bytes(raw)
    Xtr, ytr = parse_ts(root / 'FordA_TRAIN.ts'); Xte, yte = parse_ts(root / 'FordA_TEST.ts')
    assert Xtr.shape == (3601, 500) and Xte.shape == (1320, 500)
    return Xtr, ytr, Xte, yte


def splits(ytr, yte):
    """Classwise shuffle matches aeon stratified_resample_data_indices v1.0.0."""
    import numpy as np
    y = np.r_[ytr, yte]; n = len(ytr)
    result = [(np.arange(n), np.arange(n, len(y)))]
    labels, counts = np.unique(ytr, return_counts=True)
    for seed in range(1, 30):
        rng = np.random.RandomState(seed); train, test = [], []
        for label, count in zip(labels, counts):
            indices = np.flatnonzero(y == label); rng.shuffle(indices)
            train.extend(indices[:count]); test.extend(indices[count:])
        result.append((np.array(train), np.array(test)))
    for a, b in result:
        assert len(a) == 3601 and len(b) == 1320
        assert len(np.unique(np.r_[a, b])) == 4921 and not np.intersect1d(a, b).size
        assert np.array_equal(np.unique(y[a], return_counts=True)[1], counts)
    return result


def environment():
    names = ['numpy', 'scipy', 'scikit-learn', 'torch', 'joblib', 'threadpoolctl', 'matplotlib', 'psutil']
    return {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
            'packages': {k: importlib.metadata.version(k) for k in names},
            'runner_sha256': sha(__file__), 'source_lock': read(BASE / 'source_lock.json')}


def backend():
    lock = read(BASE / 'source_lock.json'); path = BASE / 'vendor/code/quant.py'
    if sha(path) != lock['source_sha256']: raise ValueError('Kode QUANT asli berubah.')
    sys.path.insert(0, str(path.parent))
    from quant import Quant
    return Quant


def metrics(y, pred, prob):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    return {'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro')), 'roc_auc': float(roc_auc_score(y, prob)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[-1, 1]).tolist()}


def run(root, out, stop_after):
    import numpy as np
    import torch
    import joblib
    import psutil
    from sklearn.ensemble import ExtraTreesClassifier
    from threadpoolctl import threadpool_limits
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    Xtr, ytr, Xte, yte = data(root)
    X = np.concatenate([Xtr, Xte]); y = np.r_[ytr, yte]
    partitions = splits(ytr, yte)
    env = environment(); identity = digest({'config': CONFIG, 'archive': ARCHIVE_SHA, 'environment': env})
    if (out / 'identity.json').exists() and read(out / 'identity.json')['identity'] != identity:
        raise ValueError('Kode/config/lingkungan berbeda. Gunakan folder run baru agar cache tidak tercampur.')
    save(out / 'identity.json', {'identity': identity}); save(out / 'config.json', CONFIG); save(out / 'environment.json', env)
    (out / 'requirements_actual.txt').write_text('\n'.join(f'{k}=={v}' for k, v in env['packages'].items()) + '\n')
    save(out / 'data_audit.json', {'archive_sha256': ARCHIVE_SHA, 'train_shape': list(Xtr.shape), 'test_shape': list(Xte.shape),
         'original_train_labels': {str(k): int(v) for k,v in zip(*np.unique(ytr, return_counts=True))},
         'original_test_labels': {str(k): int(v) for k,v in zip(*np.unique(yte, return_counts=True))},
         'all_cases_used': True, 'new_scaling_applied': False})
    Quant = backend()
    feature_file, meta_file = out / 'features.npy', out / 'features_meta.json'
    if feature_file.exists() and meta_file.exists():
        meta = read(meta_file)
        if meta['identity'] != identity or meta['sha256'] != sha(feature_file): raise ValueError('Cache fitur rusak/usang.')
        print('Cache fitur valid; transformasi dilewati.', flush=True)
    else:
        start = time.perf_counter()
        with torch.no_grad(), threadpool_limits(limits=2):
            transform = Quant(depth=6, div=4)
            tensor = torch.from_numpy(X[:, None, :])
            F = transform.fit_transform(tensor, torch.from_numpy(y)).numpy()
            # QUANT.fit uses only lengths: verify fitting on train gives identical rows.
            separate = Quant(depth=6, div=4)
            sample = separate.fit_transform(torch.from_numpy(Xtr[:3, None, :]), torch.from_numpy(ytr[:3])).numpy()
            equivalence_error = float(np.max(np.abs(F[:3] - sample)))
            if not np.allclose(F[:3], sample, rtol=1e-5, atol=1e-5): raise AssertionError('Transform global berbeda secara numerik dari transform train terpisah.')
        assert len(F) == 4921 and np.isfinite(F).all()
        tmp = feature_file.with_name(feature_file.name + '.partial')
        with tmp.open('wb') as f: np.save(f, F)
        os.replace(tmp, feature_file)
        save(meta_file, {'identity': identity, 'sha256': sha(feature_file), 'shape': list(F.shape),
             'seconds': time.perf_counter()-start, 'train_only_transform_equivalence_verified': True,
             'equivalence_check_max_abs_float32_error': equivalence_error,
             'no_learned_statistics_or_labels_used_in_transform': True})
        del F, tensor
    F = np.load(feature_file, mmap_mode='r')
    with (BASE / 'vendor/results/accuracy_resamples_ucr142.csv').open() as f:
        reference_row = next(r for r in csv.DictReader(f) if r['dataset'] == 'FordA')
    ref = np.array([float(reference_row[str(i)]) for i in range(30)])
    results = []
    folds = out / 'resamples'; folds.mkdir(exist_ok=True)
    for seed, (tr, te) in enumerate(partitions[:stop_after]):
        p, preds = folds / f'{seed:02d}.json', folds / f'{seed:02d}_predictions.npz'
        if p.exists() and preds.exists():
            r = read(p)
            if r['identity'] != identity or r['predictions_sha256'] != sha(preds): raise ValueError('Cache resample rusak/usang.')
            print(f'Resample {seed:02d}: cache valid, accuracy={r["accuracy"]:.6f}', flush=True)
        else:
            clf = ExtraTreesClassifier(n_estimators=200, max_features=.1, criterion='entropy',
                                      bootstrap=False, random_state=seed, n_jobs=2)
            start = time.perf_counter()
            with threadpool_limits(limits=2): clf.fit(F[tr], y[tr])
            fit_s = time.perf_counter() - start; start = time.perf_counter()
            prob = clf.predict_proba(F[te])[:, 1]; pred = clf.classes_[np.argmax(clf.predict_proba(F[te]), axis=1)]
            predict_s = time.perf_counter() - start
            tmp = preds.with_name(preds.name + '.partial')
            with tmp.open('wb') as f: np.savez_compressed(f, train_indices=tr, test_indices=te, y_true=y[te], pred=pred, probability=prob)
            os.replace(tmp, preds)
            if seed == 0: joblib.dump(clf, out / 'model_original_split.joblib')
            r = {'resample': seed, 'identity': identity, **metrics(y[te], pred, prob),
                 'reference_accuracy': float(ref[seed]), 'fit_seconds': fit_s, 'predict_seconds': predict_s,
                 'rss_mib_at_end': psutil.Process().memory_info().rss / 2**20,
                 'predictions_sha256': sha(preds), 'train_count': len(tr), 'test_count': len(te)}
            r['delta_pp'] = (r['accuracy'] - r['reference_accuracy']) * 100
            save(p, r)
            print(f'Resample {seed:02d}: accuracy={r["accuracy"]:.6f}, reference={ref[seed]:.6f}, delta={r["delta_pp"]:+.4f} pp, fit={fit_s:.2f}s', flush=True)
        results.append(r)
        save(out / 'run_status.json', {'completed_resamples': len(results), 'total_resamples': 30, 'complete': len(results)==30})
    with (out / 'paper_vs_run.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, ['resample','accuracy','reference_accuracy','delta_pp','balanced_accuracy','macro_f1','roc_auc','fit_seconds','predict_seconds'], extrasaction='ignore'); w.writeheader(); w.writerows(results)
    vals = np.array([r['accuracy'] for r in results]); f1 = [r['macro_f1'] for r in results]
    summary = {'completed_resamples': len(results), 'complete_30_resamples': len(results)==30,
               'original_split_accuracy': results[0]['accuracy'], 'original_split_reference_accuracy': float(ref[0]),
               'mean_accuracy': float(vals.mean()), 'std_accuracy_ddof1': float(vals.std(ddof=1)) if len(vals)>1 else None,
               'reference_mean_30': float(ref.mean()), 'reference_std_30_ddof1': float(ref.std(ddof=1)),
               'mean_macro_f1': float(np.mean(f1)), 'mean_roc_auc': float(np.mean([r['roc_auc'] for r in results])),
               'classifier_seed_matches_authors_random_draws': False,
               'note': 'Classifier seeds fixed before scoring; article section4.1/Fig17 acknowledges seed-induced variation. Original author package versions unspecified; no tuning based on test.'}
    summary['original_split_delta_pp'] = (vals[0]-ref[0])*100
    summary['mean_delta_pp'] = (vals.mean()-ref.mean())*100 if len(results)==30 else None
    save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    plot(out)


def plot(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    out = Path(out)
    with np.load(out / 'resamples/00_predictions.npz') as z: y, p, pr = z['y_true'], z['pred'], z['probability']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(y, p, labels=[-1,1], display_labels=['Label -1','Label +1'], cmap='Greys', colorbar=False, ax=axes[0])
    RocCurveDisplay.from_predictions(y, pr, pos_label=1, color='black', ax=axes[1])
    axes[0].set_title('FordA: original split'); axes[1].set_title('QUANT + ExtraTrees200')
    fig.tight_layout(); fig.savefig(out / 'confusion_roc_original.png', dpi=150); plt.close(fig)
    rows = list(csv.DictReader((out / 'paper_vs_run.csv').open()))
    if len(rows)>1:
        fig, ax = plt.subplots(figsize=(6,4)); ax.boxplot([[100*float(r['reference_accuracy']) for r in rows],[100*float(r['accuracy']) for r in rows]], tick_labels=['Author reference','Replication'], patch_artist=False)
        ax.set(ylabel='Accuracy (%)', title=f'FordA | {len(rows)} resamples'); fig.tight_layout(); fig.savefig(out / 'boxplot_resamples.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=BASE/'data'); parser.add_argument('--out', type=Path, default=BASE/'results_local_cpu'); parser.add_argument('--stop-after', type=int, default=30)
    a = parser.parse_args()
    if not 1 <= a.stop_after <= 30: parser.error('stop-after must be1..30')
    run(a.root, a.out, a.stop_after)
