"""Audit frozen predictions and produce scientific figures, without changing models."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.packages').exists():sys.path.insert(0,str(ROOT/'.packages'))
import numpy as np,pandas as pd,json,joblib,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import norm

OUT=ROOT/'results'; FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'));CLASSES=cfg['classes']
SHORT=['EGFR','HDAC','MEK','拓扑异构酶','微管聚合']
ZH=dict(zip(CLASSES,[c+'抑制剂' for c in SHORT]))
COLORS=['#2563EB','#0D9488','#8B5CF6','#E68A2E','#DB4C77']
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.facecolor':'white','figure.facecolor':'white'})

def save(name):
    plt.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close()

def main():
    r=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    X=pd.read_csv(ROOT/'data/processed/morphology_features.csv.gz');meta=pd.read_csv(ROOT/'data/processed/sample_metadata.csv')
    bundle=joblib.load(ROOT/'models/selected_model.joblib');m=bundle['model']
    tr=meta.split.eq('train');te=~tr
    hashes=pd.util.hash_pandas_object(X,index=False)
    exact_overlap=len(set(hashes[tr])&set(hashes[te]))
    # Structure keys and IDs are independently checked, not only group values.
    audit={'exact_feature_row_overlap':exact_overlap,'broad_id_overlap':len(set(meta.loc[tr,'compound_id'])&set(meta.loc[te,'compound_id'])),'known_structure_overlap':len(set(meta.loc[tr,'structure_key'].dropna())&set(meta.loc[te,'structure_key'].dropna())),'metadata_feature_columns':[c for c in X if not c.startswith(('Cells_','Nuclei_','Cytoplasm_'))]}
    assert exact_overlap==0 and audit['broad_id_overlap']==0 and audit['known_structure_overlap']==0 and not audit['metadata_feature_columns']
    (OUT/'independence_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    # A perfect test causes a degenerate empirical bootstrap. Wilson communicates finite n.
    n=r['test_compounds'];p=r['correct_test_compounds']/n;z=norm.ppf(.975)
    center=(p+z*z/(2*n))/(1+z*z/n); half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    r['test_accuracy_wilson_95']=[float(center-half),float(min(1,center+half))]
    r['bootstrap_warning']='All compound predictions are correct, so the empirical bootstrap interval degenerates to [1,1]. It is not an uncertainty-free conclusion. Report Wilson accuracy interval and CV variability.'
    r['independence_checks']=audit
    names=m[:-1].get_feature_names_out()
    imp=pd.DataFrame({'feature':names,'importance':m[-1].feature_importances_}).sort_values('importance',ascending=False)
    imp['compartment']=imp.feature.str.split('_').str[0];imp['measurement']=imp.feature.str.split('_').str[1]
    imp.to_csv(OUT/'rf_feature_importance.csv',index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')

    counts=pd.read_csv(OUT/'class_counts.csv').set_index('moa').loc[CLASSES]
    fig,ax=plt.subplots(figsize=(9,4.4));x=np.arange(5)
    ax.bar(x,counts.train,label='训练药物',color='#193D65',width=.58)
    ax.bar(x,counts.test,bottom=counts.train,label='测试药物',color='#28B7A6',width=.58)
    for i,a in enumerate(counts.train+counts.test):ax.text(i,a+.3,str(a),ha='center')
    ax.set(xticks=x,xticklabels=SHORT,ylabel='独立药物数',ylim=(0,20));ax.legend(frameon=False,ncol=2);fig.tight_layout();save('class_counts')

    cv=pd.read_csv(OUT/'cv_summary.csv');labels={'LR_C0.1':'逻辑回归 C=0.1','LR_C1':'逻辑回归 C=1','SVM_C1':'SVM C=1','SVM_C10':'SVM C=10','RF_leaf1':'随机森林 叶节点≥1','RF_leaf3':'随机森林 叶节点≥3','Dummy':'多数类基线'}
    fig,ax=plt.subplots(figsize=(9.5,4.9));y=np.arange(len(cv))
    ax.barh(y,cv.macro_f1_mean,xerr=cv.macro_f1_std,color=['#0D9488' if a==r['selected_model'] else '#A8BDD2' for a in cv.model],capsize=3)
    ax.set(yticks=y,yticklabels=[labels[a] for a in cv.model],xlim=(0,1.03),xlabel='训练集四折交叉验证宏平均 F1');ax.invert_yaxis()
    ax.text(.02,-.14,'误差线为折间标准差，评测单位为药物；测试集未参与模型选择。',transform=ax.transAxes,fontsize=10,color='#586579');fig.tight_layout();save('cv_comparison')

    cm=pd.read_csv(OUT/'test_confusion_matrix.csv',index_col=0).to_numpy()
    fig,ax=plt.subplots(figsize=(7,5.7));im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.max(),1))
    for i in range(5):
        for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=20,color='white' if cm[i,j]>=3 else '#25415C')
    ax.set(xticks=np.arange(5),yticks=np.arange(5),xticklabels=SHORT,yticklabels=SHORT,xlabel='预测机制',ylabel='真实机制');plt.setp(ax.get_xticklabels(),rotation=20,ha='right');fig.tight_layout();save('confusion_matrix')

    pc=pd.read_csv(OUT/'pca_coordinates.csv');fig,ax=plt.subplots(figsize=(9,5.6))
    for c,col,short in zip(CLASSES,COLORS,SHORT):
        for split,mk in [('train','o'),('test','^')]:
            a=pc[(pc.moa==c)&(pc.split==split)];ax.scatter(a.PC1,a.PC2,c=col,marker=mk,alpha=.7,s=35,label=short if split=='train' else None)
    v=r['pca_variance_ratio'];ax.set(xlabel=f'PC1（{v[0]:.1%}）',ylabel=f'PC2（{v[1]:.1%}）');ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1,1));ax.text(0,-.16,'圆点：训练剂量样本；三角：测试剂量样本。PCA拟合训练集筛选后的特征，仅用于展示。',transform=ax.transAxes,fontsize=9);fig.tight_layout();save('pca')

    top=imp.head(10).iloc[::-1];fig,ax=plt.subplots(figsize=(11,5.3));ax.barh(np.arange(len(top)),top.importance,color='#0D9488');ax.set(yticks=np.arange(len(top)),yticklabels=top.feature,xlabel='随机森林不纯度重要性（探索性）');ax.tick_params(axis='y',labelsize=9);fig.tight_layout();save('feature_importance')
    grp=imp.groupby('compartment').importance.sum().reindex(['Nuclei','Cells','Cytoplasm']);fig,ax=plt.subplots(figsize=(7,4.1));ax.bar(['细胞核','整细胞','细胞质'],grp,color=['#2563EB','#0D9488','#8B5CF6']);ax.set(ylabel='入选特征的重要性之和',ylim=(0,1));
    for i,v in enumerate(grp):ax.text(i,v+.02,f'{v:.1%}',ha='center')
    fig.tight_layout();save('compartment_importance')

    ab=pd.read_csv(OUT/'ablation.csv');fig,ax=plt.subplots(figsize=(8,4.3));ax.bar(['全部特征','仅细胞核','仅细胞质','仅整细胞'],ab.cv_macro_f1,yerr=ab.cv_std,capsize=4,color=['#193D65','#2563EB','#8B5CF6','#0D9488']);ax.set(ylabel='训练集四折宏平均 F1',ylim=(0,1));
    for i,v in enumerate(ab.cv_macro_f1):ax.text(i,.05,f'{v:.3f}',ha='center',color='white',weight='bold')
    ax.text(0,-.18,'统一使用逻辑回归 C=1；不用于重新选择最终测试模型。',transform=ax.transAxes,fontsize=10);fig.tight_layout();save('ablation')

    null=pd.read_csv(OUT/'permutation.csv').iloc[:,0];fig,ax=plt.subplots(figsize=(8,4.2));ax.hist(null,bins=8,color='#A8BDD2',edgecolor='white');ax.axvline(r['permutation']['observed_cv_macro_f1'],color='#D66A38',linewidth=3,label='真实标签');ax.set(xlim=(0,1),xlabel='逻辑回归训练集四折宏平均 F1',ylabel='置换次数');ax.legend(frameon=False);fig.tight_layout();save('permutation')
    d=pd.read_csv(OUT/'test_dose_predictions.csv');errors=d[d.true!=d.pred];errors.to_csv(OUT/'test_error_cases.csv',index=False,encoding='utf-8-sig')
    # Always export at least one real erroneous dose for the demo, if available.
    example_groups=list(dict.fromkeys([d.iloc[0].group]+errors.group.tolist()))
    idx=meta.index[meta.group.isin(example_groups)]
    example=X.loc[idx].copy();example.insert(0,'sample_id',[f'sample_{i+1:02d}' for i in range(len(example))]);example.to_csv(ROOT/'data/processed/example_input.csv',index=False)
    ans=meta.loc[idx].copy().reset_index(drop=True);ans.insert(0,'sample_id',example.sample_id.to_numpy());ans.to_csv(ROOT/'data/processed/example_answers.csv',index=False,encoding='utf-8-sig')
    print('Wilson',r['test_accuracy_wilson_95']);print('Errors',errors[['compound_name','dose_code','true','pred']].to_string(index=False));print('Top features',imp.head(8).to_string(index=False));print('Audit',audit)

if __name__=='__main__':main()
