"""Run a stage from a Colab terminal after approval; --help does not fit."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description='Reproduce FordA stages on Colab CPU')
p.add_argument('stage',choices=['m1','m2','m3'])
p.add_argument('--root',type=Path,required=True,help='Folder dataset dan arsip FordA')
p.add_argument('--out',type=Path,required=True,help='Folder hasil persisten masing-masing tahap')
p.add_argument('--m1-features',type=Path,help='File fitur NPY M1 beserta sidecar aslinya')
a=p.parse_args();base=Path(__file__).resolve().parent
script={'m1':'run_replication.py','m2':'run_comparison.py','m3':'run_analysis.py'}[a.stage]
cmd=[sys.executable,str(base/'src'/f'{a.stage}_quant_forda'/script),'--root',str(a.root),'--out',str(a.out)]
if a.stage=='m1':cmd+=['--stop-after','30']
elif a.m1_features:cmd+=['--m1-features',str(a.m1_features)]
subprocess.run(cmd,check=True)
