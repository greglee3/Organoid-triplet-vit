"""Fixed OOF-derived C thresholds applied to existing held-out test scores."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs/threshold_transfer_20261029'
TEST=BASE/'outputs/reviewer1_multiseed_test_20261029'
SEEDS=[70384,12731,76313,98831,80324]
CONDS=['lambda_0','lambda_0p0025']
TARGETS=[.9,.925]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(y,pred):
    tp=int(np.sum(y & pred)); fp=int(np.sum(~y & pred))
    fn=int(np.sum(y & ~pred)); tn=int(np.sum(~y & ~pred))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp) if tp+fp else 0.,f1=2*tp/(2*tp+fp+fn))

def curve(y,s):
    order=np.argsort(s,kind='mergesort')[::-1]
    ss=s[order]; yy=y[order]
    ix=np.r_[np.where(np.diff(ss))[0],len(s)-1]
    tp=np.cumsum(yy)[ix]; fp=1+ix-tp
    return tp,fp,ss[ix]

def choose(y,s,target):
    tp,fp,th=curve(y,s)
    # Match sklearn roc_curve's default drop_intermediate rule.
    if len(fp)>2:
        keep=np.where(np.r_[True,np.logical_or(np.diff(fp,2),np.diff(tp,2)),True])[0]
        tp,fp,th=tp[keep],fp[keep],th[keep]
    tp=np.r_[0,tp]; fp=np.r_[0,fp]; th=np.r_[np.inf,th]
    fpr=fp/(~y).sum(); tpr=tp/y.sum()
    eligible=np.where(fpr<=1-target+1e-12)[0]
    candidates=eligible[np.isclose(tpr[eligible],max(tpr[eligible]))]
    ix=candidates[np.argmax(fpr[candidates])]
    return float(th[ix])

def ranking(y,s):
    positive=s[y]; negative=s[~y]
    auc=float(((positive[:,None]>negative).sum()+.5*(positive[:,None]==negative).sum())/(len(positive)*len(negative)))
    tp,fp,_=curve(y,s)
    ap=float(np.sum(np.diff(np.r_[0,tp/y.sum()])*tp/(tp+fp)))
    return dict(roc_auc=auc,average_precision=ap)

def summary(rows,keys):
    out={}
    for key in keys:
        a=np.array([r[key] for r in rows if r['condition']==CONDS[0]])
        b=np.array([r[key] for r in rows if r['condition']==CONDS[1]])
        assert len(a)==len(b)==5
        out[key]={n:dict(mean=float(v.mean()),sd=float(v.std(ddof=1))) for n,v in [('ce',a),('triplet',b),('difference',b-a)]}
        out[key]['triplet_greater']=int(np.sum(b>a))
        out[key]['equal']=int(np.sum(b==a))
    return out

def main():
    assert not (OUT/'RESULTS.json').exists(),'Do not overwrite completed analysis'
    thresholds=[]; sources={}; baseline_identity=None
    for seed in SEEDS:
        for cond in CONDS:
            p=OUT/'oof'/f'{seed}_{cond}.csv'
            sources[str(p.relative_to(BASE))]=digest(p)
            with p.open(newline='',encoding='utf-8-sig') as f:
                rows=sorted(csv.DictReader(f),key=lambda r:int(r['train_index']))
            identity=[(int(r['train_index']),r['sample'],r['true_label']) for r in rows]
            assert len(identity)==1200 and len(set(x[1] for x in identity))==1200
            assert [x[0] for x in identity]==list(range(1200))
            if baseline_identity is None: baseline_identity=identity
            assert identity==baseline_identity
            assert [sum(r['true_label']==c for r in rows) for c in 'ABC']==[400,400,400]
            prob=np.array([[float(r['prob_'+c]) for c in 'ABC'] for r in rows])
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
            y=np.array([r['true_label']=='C' for r in rows]); s=prob[:,2]
            for target in TARGETS:
                t=choose(y,s,target)
                m=metrics(y,s>=t)
                assert m['specificity']>=target-1e-12
                # Exhaustive attainable thresholds verify the constrained maximum recall.
                candidate=np.unique(s)
                recalls=np.array([np.mean(s[y]>=v) for v in candidate])
                specs=np.array([np.mean(s[~y]<v) for v in candidate])
                assert np.isclose(m['recall'],max(recalls[specs>=target-1e-12]))
                thresholds.append(dict(seed=seed,condition=cond,target=target,threshold=t,oof=m))
    # Freeze every threshold before accessing test scores in this analysis.
    protocol=dict(seeds=SEEDS,targets=TARGETS,threshold_source='Pooled 1200-image OOF per seed and condition',decision='C if p_C >= threshold; binary C versus A/B, not three-class argmax',selection='Maximum OOF recall subject to specificity >= target; tie: maximum eligible FPR, matching historical roc_curve rule',test_threshold_adjustment=False,analysis_status='Exploratory reviewer follow-up after test argmax results were inspected',sd='Sample SD across seeds (ddof=1)',sources=sources,thresholds=thresholds)
    (OUT/'FROZEN_OOF_THRESHOLDS.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    yl=np.load(TEST/'test_labels.npy'); assert np.bincount(yl).tolist()==[100,100,100]
    y=yl==2
    scores={}; rank=[]; testsources={}
    for seed in SEEDS:
        for cond in CONDS:
            p=TEST/f'seed_{seed}'/f'{cond}_probabilities.npy'
            prob=np.load(p); assert prob.shape==(300,3) and np.isfinite(prob).all()
            assert np.allclose(prob.sum(1),1,atol=1e-5)
            testsources[str(p.relative_to(BASE))]=digest(p)
            scores[seed,cond]=prob[:,2]
            rank.append(dict(seed=seed,condition=cond,**ranking(y,prob[:,2])))
    results=[]
    for r in thresholds:
        m=metrics(y,scores[r['seed'],r['condition']]>=r['threshold'])
        assert m['tp']+m['fn']==100 and m['tn']+m['fp']==200
        results.append(dict(**r,test=m))
    summ={str(t):summary([dict(condition=r['condition'],**r['test']) for r in results if r['target']==t],['recall','specificity','precision','f1','tp','fp','fn','tn']) for t in TARGETS}
    ranking_summary=summary(rank,['roc_auc','average_precision'])
    payload=dict(protocol=protocol,test_source_hashes=testsources,results=results,summary=summ,ranking_per_seed=rank,ranking_summary=ranking_summary)
    (OUT/'RESULTS.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    lines=['OOF-derived thresholds applied unchanged to held-out test scores','Values: mean +/- sample SD over five seeds. All binary metrics refer to Category C versus A/B.','']
    for t in TARGETS:
        lines+= [f'OOF specificity target: {100*t:.1f}%', 'Metric | CE-only ViT | Triplet-ViT | Mean paired difference (pp)']
        for k in ['recall','specificity','precision','f1']:
            v=summ[str(t)][k]
            lines.append(f"{k} | {100*v['ce']['mean']:.2f} +/- {100*v['ce']['sd']:.2f} | {100*v['triplet']['mean']:.2f} +/- {100*v['triplet']['sd']:.2f} | {100*v['difference']['mean']:+.2f}")
        lines.append('')
    lines.append('Threshold-independent test discrimination (0 to 1 scale)')
    for k,v in ranking_summary.items():
        lines.append(f"{k} | {v['ce']['mean']:.4f} +/- {v['ce']['sd']:.4f} | {v['triplet']['mean']:.4f} +/- {v['triplet']['sd']:.4f} | {v['difference']['mean']:+.4f}")
    lines+=['','Per-seed results','Target | Seed | Model | OOF threshold | OOF specificity | Test recall | Test specificity | Test precision | Test F1 | TP | FN | FP | TN']
    for r in results:
        m=r['test']; name='CE-only ViT' if r['condition']==CONDS[0] else 'Triplet-ViT'
        lines.append(f"{100*r['target']:.1f} | {r['seed']} | {name} | {r['threshold']:.8f} | {100*r['oof']['specificity']:.2f} | "+' | '.join(f'{100*m[k]:.2f}' for k in ['recall','specificity','precision','f1'])+' | '+' | '.join(str(m[k]) for k in ['tp','fn','fp','tn']))
    lines+=['','Per-seed ranking metrics','Seed | Condition | C ROC-AUC | C average precision']
    lines += [f"{r['seed']} | {r['condition']} | {r['roc_auc']:.6f} | {r['average_precision']:.6f}" for r in rank]
    lines+=['','Legend: Thresholds were selected separately for each model and seed using only pooled OOF predictions for 1,200 training images, to maximize C recall subject to OOF specificity of at least 90.0% or 92.5%. The frozen thresholds were applied to the existing five-fold-averaged probabilities for the same 300 test images (100 per category). Actual test specificity is reported and was not forced to equal the OOF target. Values are mean +/- sample standard deviation across five seeds. Differences are paired within seed (Triplet minus CE). C ROC-AUC and average precision were computed from continuous test probabilities. Average precision is not trapezoidal PR-AUC. No retraining, additional inference, or test-based threshold adjustment was performed.','Limitations: This is exploratory follow-up after inspection of the original test results. Single-fold OOF scores and five-fold-averaged test scores may differ in calibration. Across-seed SD describes training variability on a fixed test set, not uncertainty across biological datasets. No test of statistical significance is claimed.']
    (OUT/'Threshold_transfer_results.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__': main()
