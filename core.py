"""Reproducible synthetic procurement ML. No organizational data used."""
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
import joblib, sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
FEATURES=['cycle_ratio','variance_pct','change_orders','late_days']

def simulate():
    rng=np.random.default_rng(8920)
    rows=[]
    for i in range(1200):
     month=i//134+1
     month=min(9,month)
     cat=['Professional services','IT & digital','Facilities','General supplies'][int(rng.integers(4))]
     planned=int(rng.integers(25,61)); cycle=max(7,int(rng.normal(planned-4,8)))
     variance=round(float(rng.normal(0,3)),1); changes=int(rng.poisson(.5)); late=max(0,int(rng.normal(1,3)))
     anomaly=bool(rng.random()<.08)
     if anomaly:
      kind=int(rng.integers(3))
      if kind==0: cycle+=int(rng.integers(28,65));late+=int(rng.integers(12,30))
      if kind==1: variance+=float(rng.integers(16,35));changes+=int(rng.integers(3,7))
      if kind==2: cycle+=int(rng.integers(12,25)); variance+=float(rng.integers(10,20));changes+=3
     approved=int(rng.integers(80,1800))*1000;actual=round(approved*(1+variance/100))
     missing=bool(rng.random()<.035)
     rows.append(dict(id=f'PR-26-{i+1:04}',month=f'2026-{month:02}',supplier=f'Supplier {int(rng.integers(1,31)):02}',category=cat,system=['TMT','TMS','Oracle'][i%3],planned_days=planned,cycle_days=cycle,budget_aed=approved,actual_aed=actual,variance_pct=round(variance,1),change_orders=changes,late_days=None if missing else late,quality_issue=missing,injected_anomaly=anomaly))
    return pd.DataFrame(rows)

def features(df):
    required=['id','month','planned_days','cycle_days','budget_aed','actual_aed','variance_pct','change_orders','late_days']
    missing=set(required)-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.empty: raise ValueError('Dataset is empty.')
    if len(df)>50000: raise ValueError('Use at most 50,000 rows for this desktop prototype.')
    if df.id.isna().any() or df.id.duplicated().any(): raise ValueError('IDs must be unique and present.')
    if not df.month.astype(str).str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all(): raise ValueError('month must be YYYY-MM.')
    for k in required[2:]:
        v=pd.to_numeric(df[k],errors='coerce')
        if (df[k].notna() & v.isna()).any() or np.isinf(v.dropna()).any(): raise ValueError(f'Invalid number in {k}.')
        if k!='late_days' and v.isna().any(): raise ValueError(f'Missing {k}; resolve before training.')
        if k in ['planned_days','cycle_days','budget_aed'] and (v<=0).any(): raise ValueError(f'{k} must be positive.')
        if k in ['actual_aed','change_orders','late_days'] and (v.dropna()<0).any(): raise ValueError(f'{k} cannot be negative.')
    if not np.allclose((df.actual_aed/df.budget_aed-1)*100,df.variance_pct,atol=.11): raise ValueError('Spend variance does not reconcile.')
    x=df[['variance_pct','change_orders','late_days']].astype(float).copy()
    x['cycle_ratio']=df.cycle_days/df.planned_days
    return x[FEATURES]

def explain(r):
    signals=[]
    if r.cycle_days/r.planned_days>1.35: signals.append('Cycle time above plan')
    if r.variance_pct>10: signals.append('Spend variance above 10%')
    if r.change_orders>=3: signals.append('Frequent change orders')
    if r.imputed_late_days>10: signals.append('Delivery delay above 10 days')
    return '; '.join(signals) or 'No individual context-rule limit exceeded'

MODEL_NAMES={'isolation_forest':'Isolation Forest','one_class_svm':'One-Class SVM','rules':'Context rules'}
DEFAULT_PERCENTILE=90

def data_report(df):
    """Raw-data diagnostics. No rows are silently deleted or silently repaired."""
    features(df)
    return dict(rows=len(df),columns=len(df.columns),missing_late_days=int(df.late_days.isna().sum()),
        delivery_completeness=float(df.late_days.notna().mean()),duplicate_ids=int(df.id.duplicated().sum()),
        start=str(df.month.min()),end=str(df.month.max()))

def rules(df,imputed):
    return ((df.cycle_days/df.planned_days>1.35)|(df.variance_pct>10)|
            (df.change_orders>=3)|(imputed>10)).to_numpy()

def score(df,bundle,model_name='isolation_forest',percentile=None):
    """Score without retraining; ranks/cutoffs use the separate calibration cohort."""
    if model_name not in MODEL_NAMES: raise ValueError(f'Unknown model: {model_name}')
    p=bundle['percentile'] if percentile is None else float(percentile)
    if not 0<p<100: raise ValueError('Percentile must be between 0 and 100.')
    x=bundle['imputer'].transform(features(df))
    result=df.copy();result['imputed_late_days']=x[:,3]
    if model_name=='rules':
        flag=rules(df,x[:,3]);scores=flag.astype(float)
        ranks=np.full(len(df),np.nan)
    else:
        transformed=bundle['scaler'].transform(x) if model_name=='one_class_svm' else x
        scores=-bundle['models'][model_name].score_samples(transformed)
        reference=bundle['calibration_scores'][model_name]
        flag=scores>np.percentile(reference,p)
        ranks=np.searchsorted(reference,scores)/len(reference)*100
    result['anomaly_score']=scores;result['risk_percentile']=ranks
    result['flagged']=flag;result['model']=MODEL_NAMES[model_name]
    result['cohort']=np.where(df.month<=bundle['train_end'],'Training',np.where(df.month<=bundle['calibration_end'],'Calibration','Held-out test'))
    result['signals']=result.apply(explain,axis=1)
    return result

def evaluate(scored):
    if scored.empty:return dict(cases=0,note='No held-out cases in this selection.')
    if 'injected_anomaly' not in scored:return dict(cases=len(scored),note='Labels absent; precision/recall cannot be measured.')
    truth=scored.injected_anomaly.astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
    if truth.isna().any():raise ValueError('Labels must be true/false or 1/0.')
    truth=truth.astype(bool);pred=scored.flagged.astype(bool)
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[False,True]).ravel()
    return dict(cases=len(scored),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),
        alerts=int(pred.sum()),alert_rate=float(pred.mean()),
        precision=float(precision_score(truth,pred,zero_division=0)) if tp+fp else None,
        recall=float(recall_score(truth,pred,zero_division=0)) if tp+fn else None,
        f1=float(f1_score(truth,pred,zero_division=0)) if 2*tp+fp+fn else None,
        auc=float(roc_auc_score(truth,scored.anomaly_score)) if truth.nunique()==2 and scored.model.iloc[0]!='Context rules' else None)

def train(df,output_dir,percentile=DEFAULT_PERCENTILE,train_end='2026-05',calibration_end='2026-06'):
    """Chronological fit/calibration/test separation; no labels tune model/cutoff."""
    import re
    for d in [train_end,calibration_end]:
        if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])',d):raise ValueError('Split dates must be YYYY-MM.')
    if train_end>=calibration_end:raise ValueError('Calibration must end after training.')
    if not 0<percentile<100:raise ValueError('Percentile must be between 0 and 100.')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    x=features(df)
    fit=df.month<=train_end;cal=(df.month>train_end)&(df.month<=calibration_end);test=df.month>calibration_end
    if fit.sum()<20 or cal.sum()<20 or not test.any():raise ValueError('Need at least 20 training, 20 calibration and 1 held-out case.')
    if x.loc[fit].isna().all().any():raise ValueError('A training feature is entirely missing.')
    imputer=SimpleImputer(strategy='median');xf=imputer.fit_transform(x.loc[fit]);xc=imputer.transform(x.loc[cal])
    scaler=RobustScaler().fit(xf)
    models={'isolation_forest':IsolationForest(n_estimators=250,random_state=8920,contamination='auto').fit(xf),
            'one_class_svm':OneClassSVM(kernel='rbf',gamma='scale',nu=.08).fit(scaler.transform(xf))}
    calibration_scores={name:np.sort(-model.score_samples(scaler.transform(xc) if name=='one_class_svm' else xc)) for name,model in models.items()}
    bundle=dict(models=models,imputer=imputer,scaler=scaler,calibration_scores=calibration_scores,
        percentile=float(percentile),train_end=train_end,calibration_end=calibration_end,
        features=FEATURES,seed=8920,sklearn_version=sklearn.__version__,data_status='SYNTHETIC')
    all_scored=[];metrics={}
    for name in MODEL_NAMES:
        s=score(df,bundle,name);all_scored.append(s)
        metrics[name]=evaluate(s.loc[test])
    report=dict(seed=8920,data_status='SYNTHETIC',features=FEATURES,percentile=float(percentile),
        train_end=train_end,calibration_end=calibration_end,split=dict(training=int(fit.sum()),calibration=int(cal.sum()),test=int(test.sum())),
        quality=data_report(df),models=metrics,
        note='Threshold and hyperparameters fixed without test labels. Model comparison does not select an operational winner.')
    scored=pd.concat(all_scored,ignore_index=True)
    joblib.dump(bundle,output_dir/'model_bundle.joblib')
    scored.to_csv(output_dir/'all_model_scores.csv',index=False)
    score(df,bundle).to_csv(output_dir/'scored_procurement.csv',index=False)
    (output_dir/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scikit_learn=sklearn.__version__,joblib=joblib.__version__,matplotlib=matplotlib.__version__)
    (output_dir/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    save_charts(scored,report,output_dir)
    return bundle,scored,report

def save_charts(scored,report,folder):
    """Publication-ready PNGs, generated from actual computed outputs."""
    first=scored[scored.model=='Isolation Forest']
    fig,ax=plt.subplots(figsize=(9,4));first.groupby('month')[['cycle_days','planned_days']].mean().plot(ax=ax,color=['#087f78','#9badb4'],marker='o')
    ax.set(title='Synthetic procurement cycle time',ylabel='Average days',xlabel='Completion cohort');ax.legend(['Actual','Planned']);ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(folder/'cycle_time.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(name,m) in zip(axes,report['models'].items()):
        if 'tp' not in m:continue
        matrix=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        ax.imshow(matrix,cmap='Blues')
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>matrix.max()/2 else '#183640',fontsize=17)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Alert'],yticklabels=['Normal','Injected'],xlabel='Prediction',ylabel='Synthetic truth',title=MODEL_NAMES[name])
    fig.suptitle('Held-out confusion matrices • Jul–Sep 2026 (synthetic)',fontsize=12)
    fig.tight_layout();fig.savefig(folder/'model_comparison.png',dpi=180);plt.close(fig)
    comparison=pd.DataFrame({MODEL_NAMES[n]:{k:m.get(k) for k in ['precision','recall','f1','alerts','tp','fp','fn','tn']} for n,m in report['models'].items()}).T
    comparison.to_csv(folder/'model_comparison.csv',index_label='model')
