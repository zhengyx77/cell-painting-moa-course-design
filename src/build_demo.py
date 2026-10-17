"""Build a portable offline results viewer plus optional live CSV inference."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.packages').exists():sys.path.insert(0,str(ROOT/'.packages'))
import pandas as pd

def main():
    out=ROOT/'demo';out.mkdir(exist_ok=True)
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    data={'summary':summary,'cv':pd.read_csv(ROOT/'results/cv_summary.csv').to_dict('records'),'compounds':pd.read_csv(ROOT/'results/test_compound_predictions.csv').fillna('').to_dict('records'),'doses':pd.read_csv(ROOT/'results/test_dose_predictions.csv').fillna('').to_dict('records'),'features':pd.read_csv(ROOT/'results/rf_feature_importance.csv').head(10).to_dict('records')}
    (out/'data.js').write_text('window.PROJECT_DATA='+json.dumps(data,ensure_ascii=False)+';',encoding='utf-8')

if __name__=='__main__':main()
