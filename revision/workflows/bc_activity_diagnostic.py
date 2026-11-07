"""Read-only mining instrumentation; separate seed-70384 training run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, hashlib, types, inspect, csv, time, traceback
from pathlib import Path, PosixPath
import torch
import numpy as np

ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'runs/reviewer_bc_activity_seed70384_20261101_r1'
SOURCE=ROOT/'05_final_tuning/02_ordinal_aware_triplet/train_pilot.py'
OLD=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/runs/seed_70384'
EXPECTED='440224281d06e42f4138b81068e2013bddac11eaf5f2f6db35127d1de8629287'
ARGS='--ordinal-lambda 0 --lambdas 0.0025 --mining-mode semi-hard --adjacent-margin 0.10 --far-margin 0.10 --embedding-head shared --projection-dim 384 --distance-metric cosine --embedding-normalization l2 --epochs 20 --folds 5 --classes-per-batch 3 --samples-per-class 8 --workers 8 --lr 3e-5 --head-lr-multiplier 10 --optimizer adamw --weight-decay 0.01 --warmup-epochs 2 --scheduler cosine --scheduler-milestones 10,15 --scheduler-gamma 0.1 --triplet-ce-only-epochs 0 --triplet-ramp-epochs 0 --augmentation paper --seed 70384 --min-c-specificity 0.90 --max-c-recall-drop 0.02 --max-c-precision-drop 0.03 --device cuda'.split()

def load(name,text):
    m=types.ModuleType(name);m.__file__=str(SOURCE);sys.modules[name]=m
    exec(compile(text,str(SOURCE),'exec'),m.__dict__)
    return m

def collect(labels, positive_mask, candidates, valid_pairs, selected_indices):
    # No random draws, detached tensors only, one CPU synchronization per batch.
    with torch.no_grad():
        selected_labels=labels[selected_indices]
        counts={'all_ap':positive_mask.sum(),'all_selected':valid_pairs.sum()}
        for a,n,key in [(1,2,'B_to_C'),(2,1,'C_to_B')]:
            ap=positive_mask & (labels[:,None]==a)
            eligible=ap & (candidates & (labels[None,None,:]==n)).any(dim=2)
            selected=ap & valid_pairs & (selected_labels==n)
            counts[key+'_ap']=ap.sum()
            counts[key+'_eligible']=eligible.sum()
            counts[key+'_selected']=selected.sum()
            counts[key+'_selected_any']= (ap & valid_pairs).sum()
        vals=torch.stack(list(counts.values())).detach().cpu().tolist()
        return dict(zip(counts,[int(v) for v in vals]))

def main():
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(SOURCE.parent))
    raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
    text=raw.decode()
    needle='    valid_pairs = torch.isfinite(selected_negative)\n'
    assert text.count(needle)==1
    patched=text.replace(needle,needle+'    if globals().get("BC_OBSERVER") is not None:\n        BC_OBSERVER(labels.detach(), positive_mask.detach(), valid_candidates.detach(), valid_pairs.detach(), selected_indices.detach())\n')
    original=load('bc_original_trainer',text);trainer=load('bc_instrumented_trainer',patched)
    assert list(trainer.CLASS_NAMES)==['A','B','C'],trainer.CLASS_NAMES
    # Prove instrumentation leaves the scalar loss and gradients unchanged.
    for probe_seed in [1,2,3]:
        torch.manual_seed(probe_seed)
        x=torch.nn.functional.normalize(torch.randn(24,384),dim=1).requires_grad_(); y=torch.arange(3).repeat_interleave(8)
        expected=original.ordinal_semi_hard_triplet_loss(x,y,.1,.1)
        g1=torch.autograd.grad(expected[0],x)[0]
        events=[]
        trainer.BC_OBSERVER=lambda *args:events.append(collect(*args))
        actual=trainer.ordinal_semi_hard_triplet_loss(x,y,.1,.1)
        g2=torch.autograd.grad(actual[0],x)[0]
        assert all(torch.equal(a,b) if torch.is_tensor(a) else a==b for a,b in zip(expected,actual))
        assert torch.equal(g1,g2)
        assert events[0]['all_ap']==168 and events[0]['all_selected']==actual[3]
        for key in ['B_to_C','C_to_B']:
            assert events[0][key+'_ap']==56
            assert 0<=events[0][key+'_selected']<=events[0][key+'_eligible']<=56
    trainer.BC_OBSERVER=None
    print('PREFLIGHT_PASS: loss, gradients, denominators, original source hash',flush=True)
    if '--check' in sys.argv:return
    OUT.mkdir(exist_ok=False)
    (OUT/'instrumented_train_pilot.py').write_text(patched)
    (OUT/'PROTOCOL.json').write_text(json.dumps({'seed':70384,'folds':5,'epochs':20,'args':ARGS,'original_source_sha256':EXPECTED,'instrumented_source_sha256':hashlib.sha256(patched.encode()).hexdigest(),'preprocessing':'original square resize, not aspect-ratio-preserving','test_policy':'No test inference. Training-set diagnostics only.','measurement':'B->C: B-B anchor-positive pairs with eligible C; C->B: C-C pairs with eligible B. Selected count uses actual global nearest semi-hard negative. Eligible and selected fractions both divided by directional AP count; selected share also divided by directional all-class selected count.','validation':'Only final full-fold mining call; discard per-minibatch validation diagnostics.','preflight':'CPU loss and gradient equality verified on three synthetic batches; no RNG in instrumentation.'},indent=2))
    base_epoch=trainer.run_epoch; sig=inspect.signature(base_epoch);calls={'train':0,'val':0}
    def logged_epoch(*args,**kwargs):
        bound=sig.bind(*args,**kwargs); training=bound.arguments['optimizer'] is not None
        phase='train' if training else 'val'; events=[]
        trainer.BC_OBSERVER=lambda *aa:events.append(collect(*aa))
        start=time.time()
        try:metrics=base_epoch(*args,**kwargs)
        finally:trainer.BC_OBSERVER=None
        calls[phase]+=1;index=calls[phase]-1
        assert events
        relevant=events if training else events[-1:]
        counts={k:sum(e[k] for e in relevant) for k in relevant[0]}
        assert counts['all_selected']==metrics['valid_triplet_pairs']
        if training:assert len(events)==40 and counts['all_ap']==6720
        row={'seed':70384,'fold':index//20+1,'epoch':index%20+1,'phase':phase,'seconds':time.time()-start,**counts}
        for key in ['B_to_C','C_to_B']:
            row[key+'_eligible_fraction']=counts[key+'_eligible']/counts[key+'_ap']
            row[key+'_selected_fraction']=counts[key+'_selected']/counts[key+'_ap']
            den=counts[key+'_selected_any']
            row[key+'_selected_share']=counts[key+'_selected']/den if den else None
        for key in ['ce_loss','triplet_loss','loss','accuracy','macro_f1']:row[key]=metrics[key]
        path=OUT/'bc_activity.csv';exists=path.exists()
        with path.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if not exists:w.writeheader()
            w.writerow(row)
        (OUT/'STATUS.json').write_text(json.dumps(row,indent=2))
        print('BC_ACTIVITY',json.dumps(row),flush=True)
        return metrics
    trainer.run_epoch=logged_epoch
    sys.argv=['bc_activity_diagnostic.py','--data',str(ROOT/'data/Asano_Full_Data.zip'),'--output',str(OUT/'training'),'--reference-baseline-metrics',str(OLD/'lambda_0/oof_metrics.json')]+ARGS
    trainer.main()
    assert calls=={'train':100,'val':100},calls
    with torch.serialization.safe_globals([PosixPath]):
        for fold in range(1,6):
            relative=Path('lambda_0p0025/folds')/f'fold_{fold}_best.pt'
            new=torch.load(OUT/'training'/relative,map_location='cpu',weights_only=True)
            old=torch.load(OLD/relative,map_location='cpu',weights_only=True)
            for key in ['train_indices','validation_indices','class_names']:assert np.array_equal(new[key],old[key]),(fold,key)
    (OUT/'COMPLETE.flag').write_text('100 training and 100 validation epoch records complete; historical fold assignments verified; no test inference.\n')
    print('COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'FAILED.txt').write_text(traceback.format_exc())
        raise
