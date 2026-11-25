"""Standard-library audit of completed evidence, without importing ML modules."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def rows(path):
    with (ROOT/path).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-10):
    if abs(float(a)-float(b))>tol:raise ValueError((a,b))
m1=read('results/m1/summary.json');assert m1['complete_30_resamples'] and m1['completed_resamples']==30
summary=rows('results/m3/summary.csv');folds=rows('results/m3/fold_metrics.csv')
assert len(summary)==3 and len(folds)==15
for s in summary:
    fs=[f for f in folds if f['method']==s['method']];assert len(fs)==5
    vals=[float(f['macro_f1']) for f in fs]
    close(statistics.mean(vals),s['macro_f1_mean']);close(statistics.stdev(vals),s['macro_f1_std'])
pairs=rows('results/m3/paired_tests.csv');assert len(pairs)==2
close(pairs[0]['p_holm'],.748049833345736);close(pairs[1]['p_holm'],.07148064796798619)
assert all(p['significant_holm_005']=='False' for p in pairs)
status=read('results/m3/run_status.json');assert status['complete'] and status['completed_fold_jobs']==15 and status['OOF_cases_per_method']==4921
for p in (ROOT/'src').glob('*/source_lock.json'):
    lock=json.loads(p.read_text());q=p.parent/'vendor/code/quant.py'
    assert hashlib.sha256(q.read_bytes()).hexdigest()=='a46094499033443342f048a65770d10b70e9cc64f27145e3f035ab08b54ef817'
manifest=ROOT/'MANIFEST.json'
if manifest.exists():
    for name,expected in read('MANIFEST.json')['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
print('PASS: 30 resamples; 15 fold metrics; 4921 OOF cases/method; paired tests; locked QUANT; manifest. No training executed.')
