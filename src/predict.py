"""Predict MoA from precomputed, source-compatible Cell Painting feature CSV."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.packages').exists(): sys.path.insert(0,str(ROOT/'.packages'))
import argparse, joblib, pandas as pd, numpy as np

def predict_frame(data):
    bundle=joblib.load(ROOT/'models/selected_model.joblib')
    names=bundle['features'];missing=set(names)-set(data.columns)
    if missing: raise ValueError(f'缺少 {len(missing)} 个必需的形态特征列。请使用兼容的特征表或运行内置示例。缺失列示例：{sorted(missing)[:3]}')
    if len(data)==0 or len(data)>5000: raise ValueError('CSV 需要包含 1 至 5000 行样本。')
    try:X=data[names].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    except (ValueError,TypeError) as e:raise ValueError('形态特征列包含非数值内容，请检查 CSV 格式。') from e
    if X.isna().mean().mean()>0.2: raise ValueError('缺失或无效的特征值超过 20%，请检查输入文件。')
    p=bundle['model'].predict_proba(X)
    out=pd.DataFrame(p,columns=bundle['model'].classes_)
    out.insert(0,'sample_id',data['sample_id'].astype(str).to_numpy() if 'sample_id' in data else [str(i+1) for i in range(len(data))])
    out['predicted_moa']=bundle['model'].classes_[p.argmax(axis=1)]
    out['max_model_probability']=p.max(axis=1)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'results/new_predictions.csv')
    a=p.parse_args();out=predict_frame(pd.read_csv(a.input));a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False,encoding='utf-8-sig');print(out.to_string(index=False));print('Saved:',a.output)
