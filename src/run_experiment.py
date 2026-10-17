"""Reproducible Cell Painting MoA course project. Run from any directory."""
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.packages').exists(): sys.path.insert(0, str(ROOT / '.packages'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json, hashlib, time, platform, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

CFG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
CLASSES = CFG['classes']
OUT = ROOT/'results'; OUT.mkdir(exist_ok=True)
(ROOT/'models').mkdir(exist_ok=True)
(ROOT/'data/processed').mkdir(exist_ok=True)

def savejson(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x)), encoding='utf-8')

def compound_table(meta, y):
    t = pd.DataFrame({'group':np.asarray(meta['group']), 'label':np.asarray(y)})
    assert t.groupby('group').label.nunique().max()==1
    return t.drop_duplicates('group').sort_values('group').reset_index(drop=True)

def grouped_folds(meta, y, n=4, seed=42):
    t=compound_table(meta,y)
    sk=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)
    folds=[]
    for a,b in sk.split(t.group,t.label):
        ia=np.flatnonzero(meta.group.isin(t.group.iloc[a]).to_numpy())
        ib=np.flatnonzero(meta.group.isin(t.group.iloc[b]).to_numpy())
        assert not set(meta.group.iloc[ia]) & set(meta.group.iloc[ib])
        folds.append((ia,ib))
    return folds

def make_model(name, meta, y):
    pre=[('impute',SimpleImputer(strategy='median')),('variance',VarianceThreshold(1e-8)),('scale',StandardScaler()),('select',SelectKBest(f_classif,k=CFG['feature_selection_k']))]
    if name.startswith('LR'):
        clf=LogisticRegression(C=float(name.split('_C')[1]),max_iter=3000,class_weight='balanced',random_state=42)
    elif name.startswith('SVM'):
        clf=SVC(C=float(name.split('_C')[1]),kernel='rbf',class_weight='balanced',probability=False,random_state=42)
    elif name.startswith('RF'):
        clf=RandomForestClassifier(n_estimators=300,min_samples_leaf=int(name.split('leaf')[1]),class_weight='balanced',random_state=42,n_jobs=2,max_features='sqrt')
    else:
        clf=DummyClassifier(strategy='prior')
    p=Pipeline(pre+[('classifier',clf)])
    if name.startswith('SVM'):
        p=CalibratedClassifierCV(p,method='sigmoid',cv=grouped_folds(meta,y,3),n_jobs=1)
    return p

def predict_table(model, X, meta, y):
    proba=model.predict_proba(X)
    q=meta[['group','compound_id','compound_name','dose_code']].copy().reset_index(drop=True)
    q['true']=np.asarray(y)
    for i,c in enumerate(model.classes_): q[c]=proba[:,i]
    q['pred']=q[CLASSES].idxmax(axis=1)
    return q

def aggregate(q):
    p=q.groupby('group',sort=True)[CLASSES].mean()
    m=q.groupby('group',sort=True).agg(true=('true','first'),compound_name=('compound_name','first'),compound_id=('compound_id','first'))
    m=m.join(p);m['pred']=p.idxmax(axis=1);m['confidence']=p.max(axis=1)
    return m.reset_index()

def metrics(t):
    return {'accuracy':accuracy_score(t.true,t.pred),'macro_f1':f1_score(t.true,t.pred,labels=CLASSES,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(t.true,t.pred)}

def evaluate_cv(name,X,y,meta,folds):
    records=[]; predictions=[]
    for k,(a,b) in enumerate(folds):
        m=make_model(name,meta.iloc[a],y[a]);m.fit(X.iloc[a],y[a])
        p=predict_table(m,X.iloc[b],meta.iloc[b],y[b]); p['fold']=k+1
        predictions.append(p)
        records.append({'model':name,'fold':k+1,**metrics(aggregate(p))})
    return records,pd.concat(predictions,ignore_index=True)

def prepare():
    raw=ROOT/'data/raw/lincs_consensus_median_dmso.csv.gz'
    d=pd.read_csv(raw,low_memory=False)
    mask=d.Metadata_moa.isin(CLASSES)&d.Metadata_dose_recode.isin(CFG['dose_recodes'])
    s=d.loc[mask].copy()
    ann=pd.read_csv(ROOT/'data/raw/repurposing_info.tsv',sep='\t',low_memory=False)
    ann['key']=ann.broad_id.str[:13]
    names=ann.drop_duplicates('key').set_index('key')
    ids=s.Metadata_broad_sample.str[:13]
    meta=pd.DataFrame({'compound_id':ids,'compound_name':ids.map(names.pert_iname),'structure_key':ids.map(names.InChIKey14),'dose_code':s.Metadata_dose_recode,'moa':s.Metadata_moa,'plate_map':s.Metadata_Plate_Map_Name})
    meta['compound_name']=meta.compound_name.fillna(meta.compound_id)
    meta['group']=meta.structure_key.fillna(meta.compound_id)
    feature_cols=[c for c in d if c.startswith(('Cells_','Cytoplasm_','Nuclei_'))]
    blockfile=ROOT/'references/utils_consensus_blocklist.txt'
    blocked=set(blockfile.read_text().splitlines()) if blockfile.exists() else set()
    features=[c for c in feature_cols if c not in blocked]
    X=s[features].replace([np.inf,-np.inf],np.nan).reset_index(drop=True)
    meta=meta.reset_index(drop=True)
    assert not meta.duplicated(['group','dose_code']).any(), 'Unexpected duplicate structure/dose; audit before running'
    assert meta.groupby('group').moa.nunique().max()==1
    y=meta.moa.to_numpy()
    t=compound_table(meta,y)
    train,test=train_test_split(t.group,test_size=CFG['test_fraction'],stratify=t.label,random_state=CFG['seed'])
    meta['split']=np.where(meta.group.isin(test),'test','train')
    meta.to_csv(ROOT/'data/processed/sample_metadata.csv',index=False,encoding='utf-8-sig')
    X.to_csv(ROOT/'data/processed/morphology_features.csv.gz',index=False,compression={'method':'gzip','mtime':1})
    savejson(ROOT/'data/processed/feature_names.json',features)
    counts=meta.groupby(['moa','split']).group.nunique().unstack(fill_value=0).reindex(CLASSES)
    counts.to_csv(OUT/'class_counts.csv',encoding='utf-8-sig')
    audit={'source_rows':len(d),'source_columns':len(d.columns),'selected_rows':len(meta),'compound_ids':meta.compound_id.nunique(),'independent_groups':meta.group.nunique(),'train_groups':len(train),'test_groups':len(test),'features_before_blocklist':len(feature_cols),'features_after_blocklist':len(features),'blocked_features':sorted(set(feature_cols)&blocked),'missing_name_groups':int(meta[meta.compound_name.eq(meta.compound_id)].group.nunique()),'missing_structure_groups':int(meta[meta.structure_key.isna()].group.nunique()),'source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),'split_overlap':len(set(train)&set(test)),'nonfinite_fraction':float(X.isna().to_numpy().mean()),'class_counts':counts.reset_index().to_dict('records')}
    savejson(OUT/'data_audit.json',audit)
    return X,y,meta,audit

def main():
    start=time.time()
    X,y,meta,audit=prepare()
    tr=np.flatnonzero(meta.split.eq('train')); te=np.flatnonzero(meta.split.eq('test'))
    xt=X.iloc[tr].reset_index(drop=True); mt=meta.iloc[tr].reset_index(drop=True); yt=y[tr]
    xv=X.iloc[te].reset_index(drop=True); mv=meta.iloc[te].reset_index(drop=True); yv=y[te]
    folds=grouped_folds(mt,yt,CFG['cv_folds'])
    foldmeta=[]
    for k,(a,b) in enumerate(folds):
        foldmeta.append({'fold':k+1,'train_groups':sorted(mt.group.iloc[a].unique()),'validation_groups':sorted(mt.group.iloc[b].unique()),'overlap':len(set(mt.group.iloc[a])&set(mt.group.iloc[b]))})
    savejson(OUT/'cv_splits.json',foldmeta)
    allscores=[]; cvpred={}
    for name in CFG['candidates']:
        print('CV',name,flush=True)
        scores,p=evaluate_cv(name,xt,yt,mt,folds);allscores.extend(scores);cvpred[name]=p
    scores=pd.DataFrame(allscores);scores.to_csv(OUT/'cv_fold_metrics.csv',index=False)
    summary=scores.groupby('model').agg(macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),accuracy_mean=('accuracy','mean'),balanced_accuracy_mean=('balanced_accuracy','mean')).reindex(CFG['candidates']).reset_index()
    summary.to_csv(OUT/'cv_summary.csv',index=False)
    eligible=summary[summary.model.ne('Dummy')]
    winner=eligible.loc[eligible.macro_f1_mean.idxmax(),'model']
    # Selection is recorded before any test prediction.
    savejson(OUT/'selected_model.json',{'model':winner,'criterion':'training grouped CV mean compound macro-F1','test_used_for_selection':False})
    print('SELECTED',winner,flush=True)
    cvpred[winner].to_csv(OUT/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    selected=make_model(winner,mt,yt);selected.fit(xt,yt)
    dose_pred=predict_table(selected,xv,mv,yv); cp=aggregate(dose_pred)
    dose_pred.to_csv(OUT/'test_dose_predictions.csv',index=False,encoding='utf-8-sig')
    cp.to_csv(OUT/'test_compound_predictions.csv',index=False,encoding='utf-8-sig')
    testmetrics=metrics(cp);dosemetrics=metrics(dose_pred)
    dummy=make_model('Dummy',mt,yt); dummy.fit(xt,yt)
    baseline=metrics(aggregate(predict_table(dummy,xv,mv,yv)))
    cm=confusion_matrix(cp.true,cp.pred,labels=CLASSES)
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(OUT/'test_confusion_matrix.csv',encoding='utf-8-sig')
    savejson(OUT/'test_classification_report.json',classification_report(cp.true,cp.pred,labels=CLASSES,output_dict=True,zero_division=0))
    rng=np.random.default_rng(42); boots=[]
    strata=[cp.index[cp.true.eq(c)].to_numpy() for c in CLASSES]
    for _ in range(CFG['bootstrap_repeats']):
        sample=np.concatenate([rng.choice(a,len(a),replace=True) for a in strata])
        boots.append(metrics(cp.loc[sample]))
    bdf=pd.DataFrame(boots);bdf.to_csv(OUT/'bootstrap_metrics.csv',index=False)
    intervals={c:np.quantile(bdf[c],[.025,.975]).tolist() for c in bdf}
    joblib.dump({'model':selected,'features':list(X.columns),'classes':CLASSES,'training_groups':sorted(mt.group.unique()),'test_groups':sorted(mv.group.unique()),'config':CFG},ROOT/'models/selected_model.joblib')
    # Same fixed LR used for all compartment ablations. Training data only.
    ablation=[]
    for compartment in ['All','Nuclei','Cytoplasm','Cells']:
        xc=xt if compartment=='All' else xt[[c for c in xt if c.startswith(compartment+'_')]]
        rec,_=evaluate_cv('LR_C1',xc,yt,mt,folds)
        ablation.append({'compartment':compartment,'n_features':len(xc.columns),'cv_macro_f1':float(np.mean([r['macro_f1'] for r in rec])),'cv_std':float(np.std([r['macro_f1'] for r in rec],ddof=1))})
    pd.DataFrame(ablation).to_csv(OUT/'ablation.csv',index=False)
    print('Permutation control',flush=True)
    observed=float(summary.loc[summary.model.eq('LR_C1'),'macro_f1_mean'].iloc[0]); null=[]
    groups=compound_table(mt,yt)
    for k in range(CFG['permutation_repeats']):
        mapping=dict(zip(groups.group,rng.permutation(groups.label)))
        yp=mt.group.map(mapping).to_numpy()
        rec,_=evaluate_cv('LR_C1',xt,yp,mt,grouped_folds(mt,yp,CFG['cv_folds']))
        null.append(float(np.mean([r['macro_f1'] for r in rec])))
        if (k+1)%5==0: print('Permutation',k+1,flush=True)
    pd.DataFrame({'permutation_macro_f1':null}).to_csv(OUT/'permutation.csv',index=False)
    permutation={'model':'LR_C1','observed_cv_macro_f1':observed,'null_mean':float(np.mean(null)),'null_max':float(np.max(null)),'n_permutations':len(null),'p_value':float((1+np.sum(np.array(null)>=observed))/(len(null)+1))}
    # Interpretation model is labelled separately if winner is not LR_C1.
    lr=make_model('LR_C1',mt,yt);lr.fit(xt,yt)
    names=np.array(lr[:-1].get_feature_names_out())
    coef=lr[-1].coef_
    coeff=pd.DataFrame(coef.T,index=names,columns=lr.classes_);coeff.index.name='feature';coeff['mean_abs_coefficient']=np.abs(coef).mean(axis=0)
    coeff.sort_values('mean_abs_coefficient',ascending=False).to_csv(OUT/'lr_feature_coefficients.csv',encoding='utf-8-sig')
    z=lr[:-1].transform(xt)
    centroid=pd.DataFrame(z,columns=names).assign(label=yt).groupby('label').mean()
    centroid.to_csv(OUT/'train_feature_centroids.csv',encoding='utf-8-sig')
    pca=PCA(n_components=2,random_state=42).fit(z)
    coords=pca.transform(lr[:-1].transform(X))
    pcameta=meta.copy();pcameta['PC1']=coords[:,0];pcameta['PC2']=coords[:,1]
    pcameta.to_csv(OUT/'pca_coordinates.csv',index=False,encoding='utf-8-sig')
    fullpred=predict_table(selected,X,meta,y);fullpred['split']=meta.split.to_numpy()
    fullpred.to_csv(OUT/'demo_predictions.csv',index=False,encoding='utf-8-sig')
    # Self-contained sample input for live prediction. Ground truth is in separate metadata.
    example_ids=[]
    for correct in [True,False]:
        q=cp[(cp.true==cp.pred)==correct]
        if len(q): example_ids.append(q.iloc[0].group)
    exidx=np.flatnonzero(meta.group.isin(example_ids))
    examples=X.iloc[exidx].copy();examples.insert(0,'sample_id',[f'sample_{i+1:02d}' for i in range(len(examples))]);examples.to_csv(ROOT/'data/processed/example_input.csv',index=False)
    meta.iloc[exidx].reset_index(drop=True).assign(sample_id=examples.sample_id.to_numpy()).to_csv(ROOT/'data/processed/example_answers.csv',index=False,encoding='utf-8-sig')
    import sklearn, scipy
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__}
    result={'selected_model':winner,'test_compound_metrics':testmetrics,'test_dose_metrics':dosemetrics,'test_baseline_metrics':baseline,'test_ci_95':intervals,'permutation':permutation,'pca_variance_ratio':pca.explained_variance_ratio_.tolist(),'data':audit,'versions':versions,'runtime_seconds':round(time.time()-start,2),'correct_test_compounds':int((cp.true==cp.pred).sum()),'test_compounds':len(cp),'interpretation_model':'LR_C1','selected_feature_count_lr':len(names)}
    savejson(OUT/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
