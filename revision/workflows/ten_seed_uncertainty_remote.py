"""Read-only analysis of frozen 10-seed original-resize test predictions."""
import json, hashlib
from pathlib import Path
import numpy as np
from scipy import stats

ROOT=Path('${PROJECT_ROOT}/runs')
OLD=ROOT/'reviewer1_multiseed_test_20261029'
NEW=ROOT/'reviewer_ten_seeds_extension_20261030'
SEEDS=[70384,12731,76313,98831,80324,104729,130363,155921,180749,205019]
KEYS=['accuracy','macro_f1','c_recall','c_precision','c_specificity','c_f1']
saved=json.loads((NEW/'TEN_SEED_RESULTS.json').read_text())
lookup={(r['seed'],r['model']):r['metrics'] for r in saved['rows']}
y=np.load(OLD/'test_labels.npy');assert np.array_equal(y,np.load(NEW/'test/test_labels.npy'))
assert np.array_equal(np.bincount(y),[100,100,100])
sources=[];pred=[]
for seed in SEEDS:
 base=OLD if seed in SEEDS[:5] else NEW/'test'
 pair=[]
 for lam,model in [('0','CE-only ViT'),('0p0025','Triplet-ViT')]:
  path=base/f'seed_{seed}/lambda_{lam}_probabilities.npy'
  prob=np.load(path);assert prob.shape==(300,3) and np.isfinite(prob).all()
  assert np.allclose(prob.sum(1),1,atol=1e-5)
  p=prob.argmax(1);cm=np.bincount(3*y+p,minlength=9).reshape(3,3)
  assert np.array_equal(cm,lookup[seed,model]['confusion_matrix'])
  pair.append(p);sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 pred.append(pair)
pred=np.array(pred)

def metrics(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1)
 actual=cm.sum(-1);called=cm.sum(-2)
 f1=np.divide(2*tp,actual+called,out=np.zeros_like(tp,dtype=float),where=(actual+called)!=0)
 cp=np.divide(tp[...,2],called[...,2],out=np.zeros_like(tp[...,2],dtype=float),where=called[...,2]!=0)
 return np.stack([tp.sum(-1)/cm.sum((-1,-2)),f1.mean(-1),tp[...,2]/actual[...,2],cp,1-cm[...,:2,2].sum(-1)/actual[...,:2].sum(-1),f1[...,2]],-1)*100

def holm(ps):
 order=np.argsort(ps);out=np.zeros(len(ps));last=0
 for j,i in enumerate(order):last=max(last,(len(ps)-j)*ps[i]);out[i]=min(last,1.)
 return out.tolist()

base_cm=np.array([[np.bincount(3*y+p,minlength=9).reshape(3,3) for p in pair] for pair in pred])
values=metrics(base_cm);delta=values[:,1]-values[:,0]
seed_summary={}
for k,key in enumerate(KEYS):
 d=delta[:,k];t=stats.ttest_rel(values[:,1,k],values[:,0,k]);ci=stats.t.interval(.95,9,loc=d.mean(),scale=stats.sem(d))
 assert np.isclose(d.mean(),saved['summary'][key]['difference_pp'])
 seed_summary[key]={'ce_mean':values[:,0,k].mean(),'ce_sd':values[:,0,k].std(ddof=1),'triplet_mean':values[:,1,k].mean(),'triplet_sd':values[:,1,k].std(ddof=1),'delta_pp':d.mean(),'seed_ci95_pp':list(ci),'paired_t_p':t.pvalue}
for key,p in zip(KEYS,holm([seed_summary[k]['paired_t_p'] for k in KEYS])):seed_summary[key]['paired_t_p_holm']=p

# Stratified paired image bootstrap, same 300 image IDs resampled across all seeds/models.
# Conditional on observed balanced class counts and the ten fitted model pairs.
B=20000;rng=np.random.default_rng(20261102)
boot=np.empty((B,10,6))
for start in range(0,B,500):
 n=min(500,B-start);weights=np.zeros((n,300),dtype=np.int32)
 for label in range(3):
  ids=np.flatnonzero(y==label);weights[:,ids]=rng.multinomial(100,np.full(100,.01),size=n)
 for s in range(10):
  vals=[]
  for m in range(2):
   codes=3*y+pred[s,m]
   cm=np.stack([weights[:,codes==j].sum(1) for j in range(9)],1).reshape(n,3,3)
   vals.append(metrics(cm))
  boot[start:start+n,s]=vals[1]-vals[0]
image_summary={key:{'delta_pp':delta[:,k].mean(),'paired_image_bootstrap_ci95_pp':np.quantile(boot[:,:,k].mean(1),[.025,.975]).tolist()} for k,key in enumerate(KEYS)}
per_seed=[]
for s,seed in enumerate(SEEDS):
 ce=pred[s,0]==y;tr=pred[s,1]==y
 b=int((ce&~tr).sum());c=int((~ce&tr).sum());p=stats.binomtest(b,b+c,.5).pvalue if b+c else 1.
 per_seed.append({'seed':seed,'ce_only_correct':b,'triplet_only_correct':c,'mcnemar_exact_two_sided_p':p,'metrics':{key:{'ce':values[s,0,k],'triplet':values[s,1,k],'delta_pp':delta[s,k],'paired_image_bootstrap_ci95_pp':np.quantile(boot[:,s,k],[.025,.975]).tolist()} for k,key in enumerate(KEYS)}})
for row,p in zip(per_seed,holm([r['mcnemar_exact_two_sided_p'] for r in per_seed])):row['mcnemar_p_holm_10_seeds']=p
result={'protocol':{'analysis':'Post hoc; original square-resize test predictions; no retraining, no threshold changes','seeds':SEEDS,'n_images':300,'class_counts':[100,100,100],'bootstrap_replicates':B,'bootstrap_rng_seed':20261102,'bootstrap':'Class-stratified paired percentile bootstrap over image IDs; identical resamples for all ten seeds and both models. Average per-seed metric differences, NOT predictions, over the ten fixed fitted pairs. CI conditional on class counts and fitted models.','seed_ci':'Student-t CI over ten paired seed differences; fixed test set; six two-sided paired t-tests with Holm correction','mcnemar':'Exact two-sided binomial McNemar on paired multiclass correctness separately for each fixed seed model pair; Holm correction across ten seeds','limits':'Image bootstrap assumes independent images; unavailable organoid/batch IDs prevent biological cluster resampling. Bootstrap intervals unadjusted across six metrics. Original five seeds inspected before extension; all inference exploratory. Repeated predictions are not 3000 independent samples.'},'seed_summary':seed_summary,'image_bootstrap_summary':image_summary,'per_seed':per_seed,'source_hashes':sources,'original_summary':saved['summary'],'source_protocols':{str(p):json.loads(p.read_text()) for p in [OLD/'protocol.json',NEW/'test/protocol.json']}}
print(json.dumps(result,indent=2))
