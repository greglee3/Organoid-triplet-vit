"""CPU-only follow-up; frozen probabilities, no training or inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,csv,time,traceback,hashlib,importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score,confusion_matrix
ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'runs/reviewer_cpu_followup_ten_seeds_20261102'
SEEDS=[70384,12731,76313,98831,80324,104729,130363,155921,180749,205019]
CONDS=['lambda_0','lambda_0p0025'];NAMES=['CE-only ViT','Triplet-ViT']
def write(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(state):write('STATUS.json',dict(state=state,time=time.time()));print(state,flush=True)
def main():
 OUT.mkdir(exist_ok=False)
 try:
  status('selecting_oof_thresholds')
  helper=Path(__file__).with_name('threshold_transfer_analysis.py')
  spec=importlib.util.spec_from_file_location('historical_threshold',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
  thresholds=[];hashes={};identity=None
  for seed in SEEDS:
   base=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/runs' if seed in SEEDS[:5] else ROOT/'runs/reviewer_ten_seeds_extension_20261030/training'
   for cond in CONDS:
    p=base/f'seed_{seed}/{cond}/oof_predictions.csv';hashes[str(p)]=digest(p)
    with p.open() as f:rows=sorted(csv.DictReader(f),key=lambda r:int(r['train_index']))
    ids=[(r['train_index'],r['sample'],r['true_label']) for r in rows]
    assert len(rows)==1200
    if identity is None:identity=ids
    assert identity==ids
    y=np.array([r['true_label']=='C' for r in rows]);score=np.array([float(r['prob_C']) for r in rows]);assert y.sum()==400
    for target in [.9,.925]:
     t=h.choose(y,score,target)
     fpr,tpr,thr=roc_curve(y,score);eligible=np.flatnonzero(fpr<=1-target+1e-12);best=eligible[np.isclose(tpr[eligible],tpr[eligible].max())];j=best[np.argmax(fpr[best])]
     assert np.isclose(t,thr[j])
     thresholds.append(dict(seed=seed,condition=cond,target=target,threshold=t,oof=h.metrics(y,score>=t)))
  write('FROZEN_OOF_THRESHOLDS.json',dict(seeds=SEEDS,thresholds=thresholds,source_hashes=hashes,policy='Identical historical 90%/92.5% OOF specificity targets; maximize OOF recall; tie maximum eligible FPR. No test adjustment. Exploratory.'))
  status('test_threshold_and_error_analysis')
  reference=None;allprob={};curves=[];rank=[];errors=[]
  saved=json.loads((ROOT/'runs/reviewer_ten_seeds_extension_20261030/TEN_SEED_RESULTS.json').read_text())
  lookup={(r['seed'],r['model']):r['metrics'] for r in saved['rows']}
  for seed in SEEDS:
   base=ROOT/'runs/reviewer1_multiseed_test_20261029' if seed in SEEDS[:5] else ROOT/'runs/reviewer_ten_seeds_extension_20261030/test'
   yl=np.load(base/'test_labels.npy')
   if reference is None:reference=yl
   assert np.array_equal(reference,yl) and np.bincount(yl).tolist()==[100,100,100]
   for cond,name in zip(CONDS,NAMES):
    p=base/f'seed_{seed}/{cond}_probabilities.npy';hashes[str(p)]=digest(p);prob=np.load(p)
    assert prob.shape==(300,3) and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
    allprob[seed,cond]=prob;cm=confusion_matrix(yl,prob.argmax(1),labels=[0,1,2]);assert np.array_equal(cm,lookup[seed,name]['confusion_matrix'])
    errs={f'{a}_to_{b}':int(cm[i,j]) for i,a in enumerate('ABC') for j,b in enumerate('ABC') if i!=j}
    errors.append(dict(seed=seed,model=name,confusion_matrix=cm,**errs,AB_total=int(cm[0,1]+cm[1,0]),BC_total=int(cm[1,2]+cm[2,1]),AC_total=int(cm[0,2]+cm[2,0])))
    y=yl==2;s=prob[:,2];fpr,tpr,rt=roc_curve(y,s);pr,rec,pt=precision_recall_curve(y,s)
    curves.append(dict(seed=seed,model=name,fpr=fpr,tpr=tpr,precision=pr,recall=rec))
    rank.append(dict(seed=seed,model=name,roc_auc=roc_auc_score(y,s),average_precision=average_precision_score(y,s)))
  result=[dict(**r,test=h.metrics(reference==2,allprob[r['seed'],r['condition']][:,2]>=r['threshold'])) for r in thresholds]
  def summ(rows,keys,field='model'):
   output={}
   for k in keys:
    a=np.array([r[k] for r in rows if r[field] in [NAMES[0],CONDS[0]]],float);b=np.array([r[k] for r in rows if r[field] in [NAMES[1],CONDS[1]]],float)
    assert len(a)==len(b)==10
    output[k]={name:dict(mean=v.mean(),sd=v.std(ddof=1)) for name,v in [('ce',a),('triplet',b),('paired_difference',b-a)]}
   return output
  ts={str(target):summ([dict(model=r['condition'],**r['test']) for r in result if r['target']==target],['recall','specificity','precision','f1','tp','fp','fn','tn']) for target in [.9,.925]}
  es=summ(errors,['A_to_B','B_to_A','B_to_C','C_to_B','A_to_C','C_to_A','AB_total','BC_total','AC_total'])
  write('RESULTS.json',dict(threshold_results=result,threshold_summary=ts,ranking_per_seed=rank,ranking_summary=summ(rank,['roc_auc','average_precision']),error_per_seed=errors,error_summary=es,source_hashes=hashes,notes='No t-tests. Same 300 images repeated across ten seeds, not 3000 independent samples. Counts are images, not biological replicates. OOF checkpoint selection optimism and OOF-to-fold-ensemble calibration differences remain. AP is not trapezoidal PR-AUC.'))
  write('CURVES.json',curves)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(1,2,figsize=(10,4))
  for name,color in zip(NAMES,['#2463aa','#d34532']):
   first=True
   for c in curves:
    if c['model']!=name:continue
    ax[0].plot(c['fpr'],c['tpr'],color=color,alpha=.45,lw=1,label=name if first else None)
    ax[1].step(c['recall'],c['precision'],where='post',color=color,alpha=.45,lw=1,label=name if first else None);first=False
  ax[0].plot([0,1],[0,1],'k--',lw=.7);ax[1].axhline(1/3,color='gray',ls='--',lw=.7)
  ax[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Category C ROC: ten fixed seed models')
  ax[1].set(xlabel='Recall',ylabel='Precision',title='Category C precision-recall: ten fixed seed models')
  for a in ax:a.set_xlim(0,1);a.set_ylim(0,1.02);a.legend()
  fig.tight_layout();fig.savefig(OUT/'C_ROC_PR_ten_seeds.png',dpi=300);plt.close(fig)
  lines=['Exploratory ten-seed CPU follow-up; no retraining, no manuscript edits.','Thresholds selected only on 1200-image OOF; applied unchanged to 300-image test.','Results: mean +/- sample SD; paired differences = Triplet minus CE.','']
  for target,summary in ts.items():
   lines.append('OOF specificity target '+target)
   for k in ['recall','specificity','precision','f1']:
    v=summary[k];lines.append(f"{k}: CE {100*v['ce']['mean']:.2f} +/- {100*v['ce']['sd']:.2f}%; Triplet {100*v['triplet']['mean']:.2f} +/- {100*v['triplet']['sd']:.2f}%; difference {100*v['paired_difference']['mean']:+.2f} pp")
  lines+=['','Directed error counts per 300-image test set']
  for k,v in es.items():lines.append(f"{k}: CE {v['ce']['mean']:.2f} +/- {v['ce']['sd']:.2f}; Triplet {v['triplet']['mean']:.2f} +/- {v['triplet']['sd']:.2f}; difference {v['paired_difference']['mean']:+.2f}")
  (OUT/'SUMMARY.txt').write_text('\n'.join(lines));status('complete');write('COMPLETE.flag',dict(time=time.time()))
 except BaseException:
  write('ERROR.json',dict(traceback=traceback.format_exc()));status('failed');raise
if __name__=='__main__':main()
