"""Predict MoA from precomputed, source-compatible Cell Painting feature CSV."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.packages').exists(): sys.path.insert(0,str(ROOT/'.packages'))
import argparse, joblib, pandas as pd, numpy as np

def predict_frame(data):
    bundle=joblib.load(ROOT/'models/selected_model.joblib')
    names=bundle['features'];missing=set(names)-set(data.columns)
    if missing: raise ValueError(f'Missing {len(missing)} required feature columns. Example: {sorted(missing)[:3]}')
    if len(data)==0 or len(data)>5000: raise ValueError('CSV must contain 1 to 5000 rows')
    X=data[names].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    if X.isna().mean().mean()>0.2: raise ValueError('Too many missing values (>20%); check input format')
    p=bundle['model'].predict_proba(X)
    out=pd.DataFrame(p,columns=bundle['model'].classes_)
    out.insert(0,'sample_id',data['sample_id'].astype(str).to_numpy() if 'sample_id' in data else [str(i+1) for i in range(len(data))])
    out['predicted_moa']=bundle['model'].classes_[p.argmax(axis=1)]
    out['max_model_probability']=p.max(axis=1)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'results/new_predictions.csv')
    a=p.parse_args();out=predict_frame(pd.read_csv(a.input));a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False,encoding='utf-8-sig');print(out.to_string(index=False));print('Saved:',a.output)
