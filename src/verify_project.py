"""Validate saved predictions against the actual model and independent split."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.packages').exists():sys.path.insert(0,str(ROOT/'.packages'))
import numpy as np,pandas as pd,joblib
from sklearn.metrics import accuracy_score,f1_score
from predict import predict_frame

def main():
    meta=pd.read_csv(ROOT/'data/processed/sample_metadata.csv')
    X=pd.read_csv(ROOT/'data/processed/morphology_features.csv.gz')
    bundle=joblib.load(ROOT/'models/selected_model.joblib')
    train=meta[meta.split=='train'];test=meta[meta.split=='test']
    assert len(meta)==192 and meta.group.nunique()==64
    assert set(train.group).isdisjoint(test.group)
    assert set(bundle['training_groups'])==set(train.group)
    assert set(bundle['test_groups'])==set(test.group)
    assert all(c.startswith(('Cells_','Nuclei_','Cytoplasm_')) for c in bundle['features'])
    saved=pd.read_csv(ROOT/'results/test_dose_predictions.csv')
    actual=bundle['model'].predict_proba(X.loc[test.index,bundle['features']])
    assert np.allclose(actual,saved[bundle['model'].classes_].values,atol=1e-12)
    assert test.compound_id.tolist()==saved.compound_id.tolist()
    assert test.dose_code.tolist()==saved.dose_code.tolist()
    assert test.moa.tolist()==saved['true'].tolist()
    scores=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    assert np.isclose(accuracy_score(saved['true'],saved.pred),scores['test_dose_metrics']['accuracy'])
    assert np.isclose(f1_score(saved['true'],saved.pred,average='macro'),scores['test_dose_metrics']['macro_f1'])
    compounds=pd.read_csv(ROOT/'results/test_compound_predictions.csv').set_index('group')
    grouped=saved.groupby('group')[bundle['model'].classes_].mean()
    assert np.allclose(grouped,compounds.loc[grouped.index,bundle['model'].classes_])
    assert (grouped.idxmax(axis=1)==compounds.loc[grouped.index,'pred']).all()
    assert len(compounds)==16 and (compounds['true']==compounds.pred).all()
    example=predict_frame(pd.read_csv(ROOT/'data/processed/example_input.csv'))
    previous=pd.read_csv(ROOT/'results/example_predictions.csv')
    assert example.predicted_moa.tolist()==previous.predicted_moa.tolist()
    assert np.allclose(example[bundle['model'].classes_],previous[bundle['model'].classes_])
    try:predict_frame(pd.DataFrame({'wrong_column':[1]}))
    except ValueError:pass
    else:raise AssertionError('Invalid CSV was accepted')
    result={'status':'PASS','checks':['192 profiles / 64 drugs','disjoint train/test groups','model training groups match','only morphology inputs','48 saved probabilities reproduced','dose metrics recomputed','16 drug probability averages reproduced','12 example predictions reproduced','invalid CSV rejected']}
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
