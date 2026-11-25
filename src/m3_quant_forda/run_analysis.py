"""M3: complete FordA, paired five-fold comparison, resumable CPU experiments."""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,os,time
from pathlib import Path
import numpy as np
import baseline_runner as base
import comparison_methods as methods

BASE=Path(__file__).resolve().parent
METHODS=methods.METHODS
LABELS=['QUANT + ExtraTrees','Welch48 + ExtraTrees','QUANT + Ridge']
CONFIG={'scope':'M3_pooled_4921_cases_not_official_holdout','n_splits':5,'shuffle':True,'split_seed':42,
 'methods':methods.CONFIG,'primary_metric':'macro_f1','classifier_seed':0,'alpha':.05,
 'paired_test':'two-sided ttest_rel(comparator,baseline), exploratory correlated CV folds',
 'multiplicity':'Holm two prespecified comparisons','std_ddof':1,
 'learning_curve':{'diagnostic_outer_fold':1,'inner_validation_fraction':.2,'seed':1042,'fractions':[.25,.5,.75,1.]},
 'error_examples_method':'Welch48_ExtraTrees','error_examples':'first 2 FP and first 2 FN by global index, if available'}

def make_folds(y):
    from sklearn.model_selection import StratifiedKFold
    out=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    seen=np.zeros(len(y),int)
    for tr,te in out:
        if np.intersect1d(tr,te).size or len(tr)+len(te)!=len(y):raise ValueError('Fold overlap.')
        seen[te]+=1
    if not np.all(seen==1):raise ValueError('Each case must be held out exactly once.')
    return out

def holm_adjust(p):
    p=np.asarray(p,float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid p-values.')
    order=np.argsort(p);adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);out=np.empty_like(p)
    out[order]=np.minimum(adjusted,1.);return out.tolist()

def paired_statistics(baseline,comparator):
    from scipy.stats import ttest_rel
    b,c=np.asarray(baseline,float),np.asarray(comparator,float)
    if b.shape!=(5,) or c.shape!=(5,) or not np.isfinite(np.r_[b,c]).all():raise ValueError('Exactly five complete paired folds required.')
    delta=c-b;mean=float(delta.mean());sd=float(delta.std(ddof=1))
    result={'n_pairs':5,'df':4,'mean_delta':mean,'delta_std':sd,'direction':'comparator_minus_baseline'}
    if sd<1e-14:
        zero=abs(mean)<1e-14
        return dict(result,t_statistic=0. if zero else None,p_raw=1. if zero else None,
                    ci95_low=0. if zero else None,ci95_high=0. if zero else None,
                    test_status='identical_scores' if zero else 'undefined_zero_variance_difference')
    t=ttest_rel(c,b,alternative='two-sided',nan_policy='raise');ci=t.confidence_interval(.95)
    return dict(result,t_statistic=float(t.statistic),p_raw=float(t.pvalue),ci95_low=float(ci.low),ci95_high=float(ci.high),test_status='computed')

def assemble_oof(parts,y):
    n=len(y);seen=np.zeros(n,int);pred=np.empty(n,np.int64);score=np.empty(n,float);fold=np.empty(n,np.int64)
    for z in parts:
        idx=np.asarray(z['test_indices'],int)
        if len(np.unique(idx))!=len(idx) or (idx<0).any() or (idx>=n).any():raise ValueError('Invalid held-out indices.')
        if not np.array_equal(np.asarray(z['y_true']),y[idx]):raise ValueError('Label/index mismatch.')
        if not (len(idx)==len(z['y_pred'])==len(z['score'])) or not np.isfinite(z['score']).all():raise ValueError('Invalid predictions.')
        seen[idx]+=1;pred[idx]=z['y_pred'];score[idx]=z['score'];fold[idx]=z['fold']
    if not np.all(seen==1):raise ValueError('Missing or duplicate OOF predictions.')
    return {'y_true':y,'y_pred':pred,'score':score,'fold':fold,'global_indices':np.arange(n)}

def verify_cached(row,identity,model,predictions):
    if row['identity']!=identity:raise ValueError('Cache identity changed; use a new output folder.')
    for path,key in [(model,'model_sha256'),(predictions,'predictions_sha256')]:
        if not Path(path).is_file() or base.sha(path)!=row[key]:raise ValueError('Cache incomplete or corrupt: '+str(path))
    return True

def save_npz(path,**arrays):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

def write_csv(path,rows,fields):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    os.replace(tmp,path)

def signal_audit(X,y,folds):
    groups={}
    for i,x in enumerate(X):groups.setdefault(hashlib.sha256(x.tobytes()).hexdigest(),[]).append(i)
    duplicate=[v for v in groups.values() if len(v)>1]
    fold_id=np.empty(len(y),int)
    for i,(_,te) in enumerate(folds):fold_id[te]=i+1
    return {'n_cases':len(y),'signal_length':X.shape[1],'all_cases_used':True,'original_train':3601,'original_test':1320,
       'archive_sha256':base.ARCHIVE_SHA,'exact_duplicate_groups':duplicate,
       'cross_fold_duplicate_groups':[g for g in duplicate if len(np.unique(fold_id[g]))>1],
       'conflicting_label_duplicate_groups':[g for g in duplicate if len(np.unique(y[g]))>1],
       'limitations':'Exact duplicate audit only; no engine/session grouping metadata. Original official test is pooled for M3.'}

def prepare_features(X,y,out,m1_features,env,identity):
    import torch
    quant_path=Path(m1_features) if m1_features else None
    reused=False;t=time.perf_counter()
    if quant_path and quant_path.is_file():
        folder=quant_path.parent;meta=base.read(folder/'features_meta.json');config=base.read(folder/'config.json');oldenv=base.read(folder/'environment.json')
        if config!=base.CONFIG or oldenv['packages']!=env['packages'] or oldenv['source_lock']!=env['source_lock'] or base.sha(quant_path)!=meta['sha256']:
            raise ValueError('M1 feature cache incompatible. Set M1_FEATURES=None to recompute without changing the method.')
        reused=True
    else:
        quant_path=out/'quant_features.npy';meta_path=out/'quant_features_meta.json'
        if quant_path.exists() and meta_path.exists():
            meta=base.read(meta_path)
            if meta['identity']!=identity or base.sha(quant_path)!=meta['sha256']:raise ValueError('M3 feature cache invalid.')
            reused=True
        else:
            with torch.no_grad():
                Q=base.backend()(depth=6,div=4).fit_transform(torch.from_numpy(X[:,None,:]),torch.from_numpy(y)).numpy()
            tmp=quant_path.with_name(quant_path.name+'.partial')
            with tmp.open('wb') as f:np.save(f,Q)
            os.replace(tmp,quant_path);base.save(meta_path,{'identity':identity,'sha256':base.sha(quant_path),'shape':list(Q.shape)})
            del Q
    Q=np.load(quant_path,mmap_mode='r');quant_seconds=time.perf_counter()-t
    if Q.shape!=(4921,4469) or not np.isfinite(Q).all():raise ValueError('QUANT dimensions/content invalid.')
    h_path=out/'welch48.npy';h_meta=out/'welch48_meta.json';t=time.perf_counter()
    if h_path.exists() and h_meta.exists():
        hm=base.read(h_meta)
        if hm['identity']!=identity or hm['sha256']!=base.sha(h_path):raise ValueError('Welch cache invalid.')
    else:
        H=methods.compact_features(X);tmp=h_path.with_name(h_path.name+'.partial')
        with tmp.open('wb') as f:np.save(f,H)
        os.replace(tmp,h_path);base.save(h_meta,{'identity':identity,'sha256':base.sha(h_path)})
    H=np.load(h_path,mmap_mode='r')
    if H.shape!=(4921,48) or not np.isfinite(H).all():raise ValueError('Welch features invalid.')
    audit={'quant_shape':list(Q.shape),'welch_shape':list(H.shape),'quant_cache_reused':reused,
       'quant_seconds_this_run':quant_seconds,'welch_seconds_this_run':time.perf_counter()-t,
       'quant_sha256':base.sha(quant_path),'welch_sha256':base.sha(h_path),
       'sample_independent_transform':True,'timing_note':'Cache reads are not fresh extraction timings.'}
    base.save(out/'feature_audit.json',audit)
    return Q,H

def fit_job(name,F,y,tr,te,path,identity,fold):
    import joblib
    from sklearn.metrics import f1_score,recall_score
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    mf,pf,rf=path/'model.joblib',path/'predictions.npz',path/'metrics.json'
    split_hash=base.digest({'train':tr.tolist(),'test':te.tolist()})
    if rf.exists():
        row=base.read(rf);verify_cached(row,identity,mf,pf)
        if row['split_sha256']!=split_hash or row['method']!=name:raise ValueError('Cached split/method mismatch.')
        with np.load(pf) as z:
            if not np.array_equal(z['test_indices'],te) or not np.array_equal(z['y_true'],y[te]):raise ValueError('Cached predictions have wrong cases.')
            measured=base.metrics(z['y_true'],z['y_pred'],z['score'])
            for key,value in measured.items():
                if not np.allclose(row[key],value,rtol=0,atol=1e-12):raise ValueError('Cached metrics disagree with predictions: '+key)
        print(f'Fold {fold} / {name}: cache valid',flush=True);return row,True
    start=time.perf_counter();est=methods.model(name);est.fit(F[tr],y[tr]);fit=time.perf_counter()-start
    start=time.perf_counter();pred=est.predict(F[te]);score=methods.scores(est,F[te]);predict=time.perf_counter()-start
    save_npz(pf,y_true=y[te],y_pred=pred,score=score,test_indices=te)
    temp=mf.with_name(mf.name+'.partial');joblib.dump(est,temp);os.replace(temp,mf)
    f1=f1_score(y[te],pred,labels=[-1,1],average=None,zero_division=0);rec=recall_score(y[te],pred,labels=[-1,1],average=None,zero_division=0)
    row={'identity':identity,'method':name,'fold':fold,'n_train':len(tr),'n_test':len(te),'features':F.shape[1],
       'split_sha256':split_hash,**base.metrics(y[te],pred,score),'f1_minus1':float(f1[0]),'f1_plus1':float(f1[1]),
       'recall_minus1':float(rec[0]),'recall_plus1':float(rec[1]),'train_macro_f1':float(f1_score(y[tr],est.predict(F[tr]),average='macro')),
       'fit_seconds':fit,'predict_seconds':predict,'score_type':'decision_function' if name.endswith('Ridge') else 'probability',
       'alpha':float(est[-1].alpha_) if name.endswith('Ridge') else None,
       'model_sha256':base.sha(mf),'predictions_sha256':base.sha(pf)}
    base.save(rf,row);del est;gc.collect()
    print(f'Fold {fold} / {name}: F1={row["macro_f1"]:.6f}, AUC={row["roc_auc"]:.6f}, fit={fit:.2f}s',flush=True)
    return row,False

def summary_and_tests(rows):
    summary=[];paired=[]
    for name in METHODS:
        own=sorted([r for r in rows if r['method']==name],key=lambda r:r['fold'])
        if [r['fold'] for r in own]!=list(range(1,6)):raise ValueError('Exactly one record per method/fold required.')
        row={'method':name,'folds':5}
        for metric in ['accuracy','balanced_accuracy','macro_f1','roc_auc','fit_seconds','predict_seconds','f1_minus1','f1_plus1']:
            values=np.array([r[metric] for r in own]);row[metric+'_mean']=float(values.mean());row[metric+'_std']=float(values.std(ddof=1))
        summary.append(row)
    b=sorted([r for r in rows if r['method']==METHODS[0]],key=lambda r:r['fold'])
    for name in METHODS[1:]:
        p=sorted([r for r in rows if r['method']==name],key=lambda r:r['fold'])
        if [r['split_sha256'] for r in p]!=[r['split_sha256'] for r in b]:raise ValueError('Paired folds do not share indices.')
        paired.append({'baseline':METHODS[0],'comparator':name,'metric':'macro_f1',**paired_statistics([r['macro_f1'] for r in b],[r['macro_f1'] for r in p])})
    # A non-estimable test remains in the family with p=1 for adjustment, while its reported p stays null.
    adjusted=holm_adjust([r['p_raw'] if r['p_raw'] is not None else 1. for r in paired])
    for row,p in zip(paired,adjusted):
        row['p_holm']=p if row['p_raw'] is not None else None
        row['significant_holm_005']=row['p_holm'] is not None and p<.05
        row['interpretation']=('different_mean_in_exploratory_test' if row['significant_holm_005'] else 'no_significant_difference_established')
    return summary,paired

def error_analysis(out,X,y,oofs):
    all_errors=[];summary=[]
    for name,z in oofs.items():
        pred=z['y_pred'];fp=np.flatnonzero((y==-1)&(pred==1));fn=np.flatnonzero((y==1)&(pred==-1))
        summary.append({'method':name,'false_positive':len(fp),'false_negative':len(fn),'total_errors':len(fp)+len(fn),'positive_label':1})
        for i in np.flatnonzero(pred!=y):
            all_errors.append({'method':name,'global_index':int(i),'source_file':'FordA_TRAIN.ts' if i<3601 else 'FordA_TEST.ts',
              'source_case_1based':int(i+1 if i<3601 else i-3601+1),'fold':int(z['fold'][i]),'y_true':int(y[i]),'y_pred':int(pred[i]),
              'score':float(z['score'][i]),'error_type':'FP' if y[i]==-1 else 'FN'})
    write_csv(out/'errors.csv',all_errors,['method','global_index','source_file','source_case_1based','fold','y_true','y_pred','score','error_type'])
    write_csv(out/'error_summary.csv',summary,['method','false_positive','false_negative','total_errors','positive_label'])
    subset=[]
    for kind in ['FP','FN']:subset.extend([e for e in all_errors if e['method']==CONFIG['error_examples_method'] and e['error_type']==kind][:2])
    base.save(out/'error_examples.json',{'selection':CONFIG['error_examples'],'examples':subset,'caution':'Sample descriptions do not identify physical defect causes.'})
    return summary

def diagnostic_learning(Q,H,y,train,test,out,identity):
    from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
    inner,val=train_test_split(train,test_size=.2,stratify=y[train],random_state=1042)
    records=[];new=cached=0
    for frac in [.25,.5,.75,1.]:
        if frac==1.:sub=inner
        else:
            a,_=next(StratifiedShuffleSplit(n_splits=1,train_size=frac,random_state=1042).split(inner,y[inner]));sub=inner[a]
        if np.intersect1d(np.r_[sub,val],test).size:raise ValueError('Outer test leaked into learning curve.')
        for name in METHODS:
            F=H if name.startswith('Welch') else Q
            row,used=fit_job(name,F,y,sub,val,out/'learning_jobs'/f'n{len(sub)}'/name,identity,1)
            cached+=int(used);new+=int(not used)
            records.append({'method':name,'n_train':len(sub),'n_validation':len(val),'train_macro_f1':row['train_macro_f1'],'validation_macro_f1':row['macro_f1']})
    base.save(out/'learning_curves.json',{'identity':identity,'diagnostic_outer_fold':1,'validation_indices':val.tolist(),
       'excluded_outer_test_indices':test.tolist(),'records':records,'used_for_parameter_selection':False})
    return new,cached

def run(root,out,m1_features=None):
    import torch
    from threadpoolctl import threadpool_limits
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    env=base.environment();env['m3_source_sha256']=base.sha(__file__);env['m2_methods_sha256']=base.sha(BASE/'comparison_methods.py')
    identity=base.digest({'config':CONFIG,'environment':env,'archive':base.ARCHIVE_SHA})
    if (out/'identity.json').exists() and base.read(out/'identity.json')['identity']!=identity:raise ValueError('Run identity changed. Use another RUN folder; do not mix caches.')
    base.save(out/'identity.json',{'identity':identity});base.save(out/'config.json',CONFIG);base.save(out/'environment.json',env)
    (out/'requirements_actual.txt').write_text('\n'.join(f'{k}=={v}' for k,v in env['packages'].items())+'\n')
    base.save(out/'run_status.json',{'status':'running','completed_fold_jobs':0,'total_fold_jobs':15,'complete':False})
    Xtr,ytr,Xte,yte=base.data(root);X=np.r_[Xtr,Xte];y=np.r_[ytr,yte];folds=make_folds(y)
    audit=signal_audit(X,y,folds);base.save(out/'data_audit.json',audit)
    if audit['cross_fold_duplicate_groups']:print('Warning: exact duplicate signals occur across folds; inspect data_audit.json.',flush=True)
    base.save(out/'folds.json',{'seed':42,'pool':'official train + official test','shared_by_all_methods':True,
       'folds':[{'fold':i+1,'train_indices':tr.tolist(),'test_indices':te.tolist()} for i,(tr,te) in enumerate(folds)]})
    new=cached=0
    with threadpool_limits(limits=2):
        Q,H=prepare_features(X,y,out,m1_features,env,identity);rows=[];parts={n:[] for n in METHODS}
        for fold,(tr,te) in enumerate(folds,1):
            for name in METHODS:
                path=out/'folds'/f'{fold:02d}'/name;F=H if name.startswith('Welch') else Q
                row,used=fit_job(name,F,y,tr,te,path,identity,fold);rows.append(row);cached+=int(used);new+=int(not used)
                with np.load(path/'predictions.npz') as z:parts[name].append({k:z[k].copy() for k in z.files}|{'fold':fold})
                base.save(out/'run_status.json',{'status':'running','completed_fold_jobs':len(rows),'total_fold_jobs':15,'complete':False})
        oofs={name:assemble_oof(parts[name],y) for name in METHODS}
        for name,z in oofs.items():save_npz(out/(name+'_oof.npz'),**z)
        summary,paired=summary_and_tests(rows)
        errors=error_analysis(out,X,y,oofs)
        n,c=diagnostic_learning(Q,H,y,*folds[0],out,identity);new+=n;cached+=c
    write_csv(out/'fold_metrics.csv',rows,['method','fold','n_train','n_test','accuracy','balanced_accuracy','macro_f1','roc_auc','f1_minus1','f1_plus1','fit_seconds','predict_seconds','alpha'])
    fields=['method','folds']+[m+s for m in ['accuracy','balanced_accuracy','macro_f1','roc_auc','fit_seconds','predict_seconds','f1_minus1','f1_plus1'] for s in ['_mean','_std']]
    write_csv(out/'summary.csv',summary,fields)
    write_csv(out/'paired_tests.csv',paired,['baseline','comparator','metric','n_pairs','df','mean_delta','delta_std','t_statistic','p_raw','p_holm','ci95_low','ci95_high','significant_holm_005','test_status','interpretation'])
    base.save(out/'results.json',{'summary':summary,'paired_tests':paired,'error_summary':errors,'complete':True,'folds':5,'scope':CONFIG['scope'],
       'statistics_limitations':'Exploratory paired t-tests with five correlated CV folds. Holm corrects two comparisons, not fold dependence. No external confirmation.'})
    plot(out,X)
    base.save(out/'run_status.json',{'status':'complete','complete':True,'completed_fold_jobs':15,'total_fold_jobs':15,
       'learning_jobs':12,'new_fits_this_invocation':new,'cached_fits_this_invocation':cached,'OOF_cases_per_method':len(y)})
    print(json.dumps({'complete':True,'summary':summary,'paired_tests':paired,'new_fits':new,'cached_fits':cached},indent=2),flush=True)

def plot(out,X):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay,roc_curve,auc
    from scipy.signal import welch
    out=Path(out);results=base.read(out/'results.json');rows=list(csv.DictReader((out/'fold_metrics.csv').open()))
    plt.rcParams.update({'font.size':10,'axes.labelcolor':'black','text.color':'black','axes.edgecolor':'black'})
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['balanced_accuracy','macro_f1','roc_auc'],['Macro-Accuracy','Macro-F1','ROC-AUC']):
        vals=[[float(r[key]) for r in rows if r['method']==n] for n in METHODS]
        ax.boxplot(vals,tick_labels=['QUANT+ET','Welch48+ET','QUANT+Ridge'],medianprops={'color':'black'})
        for i,v in enumerate(vals,1):ax.plot(np.full(len(v),i),v,'o',color='gray',markersize=4)
        ax.set_title(title);ax.tick_params(axis='x',labelrotation=12);ax.set_ylabel('Score (0–1)')
    fig.tight_layout();fig.savefig(out/'boxplot_metrics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,label in zip(axes,['f1_minus1','f1_plus1'],[-1,1]):
        ax.boxplot([[float(r[key]) for r in rows if r['method']==n] for n in METHODS],tick_labels=['QUANT+ET','Welch48+ET','QUANT+Ridge'],medianprops={'color':'black'})
        ax.set(title=f'F1 kelas {label}',ylabel='F1 (0–1)');ax.tick_params(axis='x',labelrotation=12)
    fig.tight_layout();fig.savefig(out/'boxplot_per_class.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4));figroc,ar=plt.subplots(1,3,figsize=(14,4))
    for i,(name,label) in enumerate(zip(METHODS,LABELS)):
        with np.load(out/(name+'_oof.npz')) as z:y,p,s,f=z['y_true'],z['y_pred'],z['score'],z['fold']
        ConfusionMatrixDisplay.from_predictions(y,p,labels=[-1,1],cmap='Greys',colorbar=False,ax=axes[i]);axes[i].set_title(label,fontsize=11)
        grid=np.linspace(0,1,201);interpolated=[];aucs=[]
        for fold in range(1,6):
            a,b,_=roc_curve(y[f==fold],s[f==fold],pos_label=1);ar[i].plot(a,b,color='gray',alpha=.5,linewidth=.8)
            line=np.interp(grid,a,b);line[0]=0;interpolated.append(line);aucs.append(auc(a,b))
        mean=np.mean(interpolated,axis=0);mean[-1]=1
        ar[i].plot(grid,mean,color='black',label=f'Mean fold AUC {np.mean(aucs):.4f} ± {np.std(aucs,ddof=1):.4f}')
        ar[i].set(title=label,xlabel='False Positive Rate',ylabel='True Positive Rate',xlim=(0,1),ylim=(0,1.01));ar[i].legend(fontsize=8,loc='lower right')
    fig.tight_layout();fig.savefig(out/'confusion_oof.png',dpi=170);figroc.tight_layout();figroc.savefig(out/'roc_multifold.png',dpi=170);plt.close('all')
    curves=base.read(out/'learning_curves.json')['records'];fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,name,label in zip(axes,METHODS,LABELS):
        pts=[r for r in curves if r['method']==name]
        ax.plot([r['n_train'] for r in pts],[r['train_macro_f1'] for r in pts],'o-',color='black',label='Train')
        ax.plot([r['n_train'] for r in pts],[r['validation_macro_f1'] for r in pts],'s--',color='gray',label='Inner validation')
        ax.set(title=label,xlabel='Training cases (outer fold 1 only)',ylabel='Macro-F1',ylim=(.8,1.01));ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(out/'learning_curve_internal.png',dpi=170);plt.close(fig)
    examples=base.read(out/'error_examples.json')['examples']
    if examples:
        fig,axes=plt.subplots(len(examples),2,figsize=(12,2.4*len(examples)),squeeze=False)
        for pair,e in zip(axes,examples):
            x=X[e['global_index']];f,p=welch(x,fs=1.,window='hann',nperseg=256,noverlap=128,nfft=256,detrend='constant')
            pair[0].plot(x,color='black',linewidth=.7);pair[0].set(title=f"{e['error_type']} / index {e['global_index']} / fold {e['fold']} / true {e['y_true']} predicted {e['y_pred']}",xlabel='Sample index',ylabel='Amplitude')
            pair[1].plot(f,p/max(p.sum(),1e-12),color='black');pair[1].set(xlabel='Normalized frequency (cycles/sample)',ylabel='Relative PSD bin mass')
        fig.tight_layout();fig.savefig(out/'error_examples.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--m1-features',type=Path)
    args=parser.parse_args();run(args.root,args.out,args.m1_features)
