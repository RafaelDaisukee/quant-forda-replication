"""M2 controlled comparisons. Full FordA original split; training-only learning curves."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from scipy.stats import skew,kurtosis
import baseline_runner as baseline

BASE=Path(__file__).resolve().parent
METHODS=['QUANT_ExtraTrees','Welch48_ExtraTrees','QUANT_Ridge']
CONFIG={'scope':'M2_original_split_not_M3_5fold','train':3601,'test':1320,'length':500,
        'seed':0,'feature_comparator':{'time_features':8,'spectral_features':8,'relative_bands':32,'total':48,
          'welch':{'fs':1.0,'units':'cycles_per_sample_not_physical_Hz','window':'hann','nperseg':256,'noverlap':128,'nfft':256,'detrend':'constant'}},
        'ExtraTrees':{'n_estimators':200,'max_features':.1,'criterion':'entropy','bootstrap':False,'random_state':0,'n_jobs':2},
        'Ridge':{'scaler':'StandardScaler_fit_train','alphas':np.logspace(-3,3,10).tolist(),'cv':None,'selection':'efficient_LOOCV_negative_mean_squared_error_training_only'},
        'learning_curve':{'main_train_only':True,'validation_fraction':.2,'random_state':42,'fractions':[.25,.5,.75,1.]}}


def compact_features(X):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=2 or X.shape[1]!=500 or not np.isfinite(X).all():raise ValueError('Input must be finite N x500.')
    eps=1e-12;absx=np.abs(X);peak=absx.max(axis=1);rms=np.sqrt(np.mean(X**2,axis=1));meanabs=absx.mean(axis=1)
    with np.errstate(invalid='ignore',divide='ignore'):
        temporal=np.column_stack([skew(X,axis=1,bias=False),kurtosis(X,axis=1,fisher=True,bias=False),
           peak/np.maximum(rms,eps),peak/np.maximum(meanabs,eps),rms/np.maximum(meanabs,eps),
           peak/np.maximum(np.mean(np.sqrt(absx),axis=1)**2,eps),np.mean(np.diff(np.signbit(X),axis=1),axis=1),
           np.sqrt(np.mean(np.diff(X,axis=1)**2,axis=1))])
    f,p=welch(X,fs=1.,window='hann',nperseg=256,noverlap=128,nfft=256,detrend='constant',axis=1,scaling='density')
    total=p.sum(axis=1,keepdims=True);q=p/np.maximum(total,eps);freq=f/.5
    centroid=(q*freq).sum(axis=1);center=freq[None,:]-centroid[:,None];bw=np.sqrt((q*center**2).sum(axis=1))
    spec_skew=(q*center**3).sum(axis=1)/np.maximum(bw**3,eps)
    spec_kurt=(q*center**4).sum(axis=1)/np.maximum(bw**4,eps)-3
    flat=np.exp(np.mean(np.log(np.maximum(p,eps)),axis=1))/np.maximum(p.mean(axis=1),eps)
    entropy=-(q*np.log(np.maximum(q,eps))).sum(axis=1)/np.log(len(f))
    roll=freq[np.argmax(np.cumsum(q,axis=1)>=.85,axis=1)];peakfreq=freq[np.argmax(p,axis=1)]
    spectral=np.column_stack([centroid,bw,spec_skew,spec_kurt,flat,entropy,roll,peakfreq])
    edges=np.linspace(0,.5,33)
    bands=np.column_stack([q[:,(f>=a)&((f<b) if i<31 else (f<=b))].sum(axis=1) for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]))])
    F=np.nan_to_num(np.column_stack([temporal,spectral,bands]),nan=0.,posinf=0.,neginf=0.)
    F[total[:,0]<=eps]=0
    assert F.shape==(len(X),48) and np.isfinite(F).all()
    return F.astype(np.float32)


def model(name):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name.endswith('ExtraTrees'):return ExtraTreesClassifier(**CONFIG['ExtraTrees'])
    return make_pipeline(StandardScaler(),RidgeClassifierCV(alphas=np.array(CONFIG['Ridge']['alphas']),cv=None))


def scores(est,X):
    return est.predict_proba(X)[:,1] if hasattr(est,'predict_proba') else est.decision_function(X)


def run(root,out,m1_features,learning=True):
    import torch,joblib
    from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
    from threadpoolctl import threadpool_limits
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    Xtr,ytr,Xte,yte=baseline.data(root);X=np.r_[Xtr,Xte];y=np.r_[ytr,yte]
    environment=baseline.environment();environment['comparison_sha256']=baseline.sha(__file__)
    identity=baseline.digest({'config':CONFIG,'env':environment,'archive':baseline.ARCHIVE_SHA})
    if (out/'identity.json').exists() and baseline.read(out/'identity.json')['identity']!=identity:raise ValueError('Cache/config/environment changed; use new RUN.')
    baseline.save(out/'identity.json',{'identity':identity});baseline.save(out/'config.json',CONFIG);baseline.save(out/'environment.json',environment)
    (out/'requirements_actual.txt').write_text('\n'.join(f'{k}=={v}' for k,v in environment['packages'].items())+'\n')
    t=time.perf_counter();qpath=Path(m1_features) if m1_features else None
    reused=False
    if qpath and qpath.is_file() and (qpath.parent/'features_meta.json').is_file():
        meta=baseline.read(qpath.parent/'features_meta.json');config=baseline.read(qpath.parent/'config.json');env=baseline.read(qpath.parent/'environment.json')
        if config!=baseline.CONFIG or env['source_lock']!=environment['source_lock'] or env['packages']!=environment['packages'] or meta['sha256']!=baseline.sha(qpath):raise ValueError('M1 cache incompatible; set M1_FEATURES=None to recompute.')
        Q=np.load(qpath,mmap_mode='r');reused=True
    else:
        qpath=out/'quant_features.npy';cache=out/'quant_features_meta.json'
        if qpath.exists() and cache.exists():
            meta=baseline.read(cache)
            if meta['identity']!=identity or meta['sha256']!=baseline.sha(qpath):raise ValueError('QUANT cache mismatch.')
            Q=np.load(qpath,mmap_mode='r');reused=True
        else:
            with torch.no_grad():Q=baseline.backend()(depth=6,div=4).fit_transform(torch.from_numpy(X[:,None,:]),torch.from_numpy(y)).numpy()
            np.save(qpath,Q);baseline.save(cache,{'identity':identity,'sha256':baseline.sha(qpath)})
    qt=time.perf_counter()-t
    t=time.perf_counter();H=compact_features(X);ht=time.perf_counter()-t
    assert Q.shape==(4921,4469) and np.isfinite(Q).all()
    baseline.save(out/'feature_audit.json',{'quant_shape':list(Q.shape),'compact_shape':list(H.shape),'quant_seconds_this_run':qt,
       'quant_cache_reused':reused,'compact_seconds':ht,'compact_dimensions_reduction_pct':100*(1-48/4469),
       'timing_note':'A reused cache time is not a fresh extraction benchmark.'})
    train=np.arange(3601);test=np.arange(3601,4921)
    baseline.save(out/'split.json',{'train_indices':train.tolist(),'test_indices':test.tolist(),'split':'original_FordA','shared_by_all_methods':True})
    folder=out/'methods';folder.mkdir(exist_ok=True);rows=[]
    with threadpool_limits(limits=2):
        for name in METHODS:
            F=H if name.startswith('Welch') else Q
            metric=folder/(name+'.json');predpath=folder/(name+'_predictions.npz')
            if metric.exists() and predpath.exists():
                row=baseline.read(metric)
                if row['identity']!=identity or row['predictions_sha256']!=baseline.sha(predpath):raise ValueError('Invalid result cache.')
                print(name+': cache valid',flush=True)
            else:
                est=model(name);t=time.perf_counter();est.fit(F[train],y[train]);fit=time.perf_counter()-t
                t=time.perf_counter();pred=est.predict(F[test]);continuous=scores(est,F[test]);predict=time.perf_counter()-t
                np.savez_compressed(predpath,y_true=y[test],y_pred=pred,score=continuous,test_indices=test)
                joblib.dump(est,folder/(name+'.joblib'))
                row={'method':name,'identity':identity,'features':F.shape[1],**baseline.metrics(y[test],pred,continuous),
                     'fit_seconds':fit,'predict_seconds':predict,'score_type':'decision_function' if name.endswith('Ridge') else 'probability',
                     'alpha':float(est[-1].alpha_) if name.endswith('Ridge') else None,'predictions_sha256':baseline.sha(predpath)}
                baseline.save(metric,row)
            rows.append(row);print(f'{name}: accuracy={row["accuracy"]:.6f}, Macro-F1={row["macro_f1"]:.6f}, AUC={row["roc_auc"]:.6f}',flush=True)
        if learning:
            inner,val=train_test_split(train,test_size=.2,stratify=y[train],random_state=42)
            curvefile=out/'learning_curves.json'
            if curvefile.exists():
                curves=baseline.read(curvefile)
                if curves['identity']!=identity:raise ValueError('Learning cache mismatch.')
            else:
                records=[]
                for frac in [.25,.5,.75,1.]:
                    if frac==1.:sub=inner
                    else:
                        a,_=next(StratifiedShuffleSplit(n_splits=1,train_size=frac,random_state=42).split(inner,y[inner]));sub=inner[a]
                    assert not np.intersect1d(np.r_[sub,val],test).size
                    for name in METHODS:
                        F=H if name.startswith('Welch') else Q;est=model(name);est.fit(F[sub],y[sub])
                        from sklearn.metrics import f1_score
                        records.append({'method':name,'n_train':len(sub),'train_macro_f1':float(f1_score(y[sub],est.predict(F[sub]),average='macro')),
                              'validation_macro_f1':float(f1_score(y[val],est.predict(F[val]),average='macro'))})
                    print(f'Learning curve {len(sub)} train / {len(val)} validation; test untouched.',flush=True)
                curves={'identity':identity,'records':records,'validation_indices':val.tolist(),'main_test_used':False};baseline.save(curvefile,curves)
    baseline.save(out/'results.json',{'methods':rows,'scope':'M2_original_split','data_complete':True,'test_used_for_tuning':False})
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,['method','features','accuracy','balanced_accuracy','macro_f1','roc_auc','fit_seconds','predict_seconds','alpha'],extrasaction='ignore');w.writeheader();w.writerows(rows)
    plot(out)


def plot(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
    out=Path(out);fig,axes=plt.subplots(1,3,figsize=(13,4));figroc,axr=plt.subplots(figsize=(6,5))
    for i,name in enumerate(METHODS):
        with np.load(out/'methods'/(name+'_predictions.npz')) as z:y,p,s=z['y_true'],z['y_pred'],z['score']
        ConfusionMatrixDisplay.from_predictions(y,p,labels=[-1,1],cmap='Greys',colorbar=False,ax=axes[i]);axes[i].set_title(name,fontsize=11)
        RocCurveDisplay.from_predictions(y,s,pos_label=1,name=name,ax=axr,color='black',linestyle=['-','--',':'][i])
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=150);figroc.tight_layout();figroc.savefig(out/'roc.png',dpi=150);plt.close('all')
    if (out/'learning_curves.json').exists():
        records=baseline.read(out/'learning_curves.json')['records'];fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,name in zip(axes,METHODS):
            pts=[x for x in records if x['method']==name]
            for key,style,c,label in [('train_macro_f1','o-','black','Train'),('validation_macro_f1','s--','gray','Validation')]:ax.plot([x['n_train'] for x in pts],[x[key] for x in pts],style,color=c,label=label)
            ax.set(xlabel='Training cases',ylabel='Macro-F1',title=name,ylim=(0,1.02));ax.legend()
        fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--m1-features',type=Path);p.add_argument('--no-learning',action='store_true');a=p.parse_args()
    run(a.root,a.out,a.m1_features,not a.no_learning)
