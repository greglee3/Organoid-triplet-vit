"""Recompute validation-selected thresholds and frozen test operating points."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

ROOT=Path(__file__).resolve().parents[1]

def metrics(y,p):
    tp=int(np.sum(y&p));fn=int(np.sum(y&~p));fp=int(np.sum(~y&p));tn=int(np.sum(~y&~p))
    return dict(recall=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp) if tp+fp else 0.,tp=tp,fn=fn,fp=fp,tn=tn)

def choose(y,s,target):
    fpr,tpr,thresholds=roc_curve(y,s)
    eligible=np.flatnonzero(fpr<=1-target+1e-12)
    best=eligible[np.isclose(tpr[eligible],tpr[eligible].max())]
    return float(thresholds[best[np.argmax(fpr[best])]])

def plot_pr(test,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid=np.linspace(0,1,1001);export=[]
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for name,color in [('CE-only ViT','#64748b'),('Triplet-ViT','#2563eb')]:
        curves=[];aps=[]
        for seed in sorted({int(r['seed']) for r in test}):
            rows=[r for r in test if int(r['seed'])==seed and r['model']==name]
            y=np.array([r['true_label']=='C' for r in rows]);s=np.array([float(r['prob_C']) for r in rows])
            precision,recall,_=precision_recall_curve(y,s)
            curves.append(np.interp(grid,recall[::-1],precision[::-1]));aps.append(average_precision_score(y,s))
        curves=np.array(curves);mean=curves.mean(0);sd=curves.std(0,ddof=1)
        ax.plot(grid,mean,color=color,label=f'{name} (AP {np.mean(aps):.3f} ± {np.std(aps,ddof=1):.3f})')
        ax.fill_between(grid,np.maximum(mean-sd,0),np.minimum(mean+sd,1),color=color,alpha=.12)
        export.extend(dict(model=name,recall=float(x),mean_precision=float(y),sd_precision=float(z)) for x,y,z in zip(grid,mean,sd))
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Category C mean seed-specific precision–recall curves')
    ax.legend(loc='lower left',fontsize=8)
    fig.savefig(output/'category_c_mean_pr.png',dpi=200);fig.savefig(output/'category_c_mean_pr.svg');plt.close(fig)
    with (output/'category_c_mean_pr.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'generated/revision_thresholds')
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Output already exists; choose a fresh --output directory.')
    source=ROOT/'revision';frozen=json.loads((source/'results/frozen_oof_thresholds.json').read_text())
    with (source/'results/test_predictions.csv').open(newline='',encoding='utf-8') as f:test=list(csv.DictReader(f))
    results=[]
    for row in frozen['thresholds']:
        seed,cond,target=row['seed'],row['condition'],row['target']
        with (source/f'oof/seed_{seed}/{cond}.csv').open(newline='',encoding='utf-8') as f:oof=sorted(csv.DictReader(f),key=lambda r:int(r['train_index']))
        assert len(oof)==1200 and [int(r['train_index']) for r in oof]==list(range(1200))
        assert len({r['sample'] for r in oof})==1200
        y=np.array([r['true_label']=='C' for r in oof]);score=np.array([float(r['prob_C']) for r in oof])
        threshold=choose(y,score,target)
        assert np.isclose(threshold,row['threshold'],rtol=0,atol=1e-12),(seed,cond,target)
        name='CE-only ViT' if cond=='lambda_0' else 'Triplet-ViT'
        rows=[r for r in test if int(r['seed'])==seed and r['model']==name]
        assert len(rows)==300 and not {r['sample'] for r in rows}&{r['sample'] for r in oof}
        yt=np.array([r['true_label']=='C' for r in rows]);st=np.array([float(r['prob_C']) for r in rows])
        results.append(dict(seed=seed,model=name,target_specificity=target,threshold=threshold,oof=metrics(y,score>=threshold),test=metrics(yt,st>=threshold)))
    summary=[]
    for target in [.9,.925]:
        for name in ['CE-only ViT','Triplet-ViT']:
            rows=[r for r in results if r['target_specificity']==target and r['model']==name]
            d=dict(model=name,target_specificity=target)
            for key in ['recall','specificity','precision']:
                a=np.array([r['test'][key] for r in rows])*100
                d[key+'_mean']=float(a.mean());d[key+'_sd']=float(a.std(ddof=1))
            summary.append(d)
    # Validate every frozen-threshold test count against the archived analysis.
    reference=json.loads((source/'results/threshold_and_curves.json').read_text())
    for r in results:
        cond='lambda_0' if r['model']=='CE-only ViT' else 'lambda_0p0025'
        z=next(x for x in reference['threshold_results'] if x['seed']==r['seed'] and x['condition']==cond and x['target']==r['target_specificity'])
        for key in ['tp','fn','fp','tn']:assert r['test'][key]==z['test'][key],(r,key,z)
    args.output.mkdir(parents=True)
    (args.output/'threshold_results.json').write_text(json.dumps(dict(rows=results,summary=summary,scope='OOF-selected thresholds fixed before application to test probabilities; exploratory analysis. Realized test specificity is not forced to equal the OOF target.'),indent=2),encoding='utf-8')
    with (args.output/'threshold_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    plot_pr(test,args.output)
    print('PASS: all 40 thresholds and held-out test confusion counts matched the archived analysis.')
    print(args.output)

if __name__=='__main__':main()
