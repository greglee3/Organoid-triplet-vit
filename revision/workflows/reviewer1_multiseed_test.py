"""Reviewer-requested inference only; preserves all historical checkpoints/results."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import importlib.util
import json
from pathlib import Path, PosixPath
from datetime import datetime, timezone
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
BASE = ROOT/'05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test'
RUNS = ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/runs'
OUT = ROOT/'runs/reviewer1_multiseed_test_20261029'
SEEDS = [70384,12731,76313,98831,80324]

def main():
    OUT.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('original_evaluator', BASE/'evaluate_locked_test_once.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    assert ev.sha256(BASE/'evaluate_locked_test_once.py') == '5bcef3de5da587b69c5221415467671cc7b091abd622f2c79e551412fc51d146'
    trainer = ev.load_trainer()
    original_config = json.loads((BASE/'LOCKED_CONFIG.json').read_text())
    manifest=[]
    # Validate all settings and paired splits before any test inference.
    with torch.serialization.safe_globals([PosixPath]):
        for seed in SEEDS:
            config=dict(original_config,seed=seed)
            for fold in range(1,6):
                pair=[]
                for cond,weight in [('lambda_0',0.),('lambda_0p0025',.0025)]:
                    p=RUNS/f'seed_{seed}'/cond/'folds'/f'fold_{fold}_best.pt'
                    ck=torch.load(p,map_location='cpu',weights_only=True)
                    ev.validate_checkpoint(ck,config,weight,fold)
                    pair.append(ck)
                    manifest.append(dict(seed=seed,condition=cond,fold=fold,path=str(p),sha256=ev.sha256(p),best_epoch=ck['best_epoch']))
                for key in ['train_indices','validation_indices','class_names']:
                    assert np.array_equal(pair[0][key],pair[1][key]),(seed,fold,key)
                del pair
    protocol=dict(purpose='Reviewer 1 requested multi-seed held-out test evaluation; inference only',seeds=SEEDS,original_seed42_lock=original_config,checkpoint_manifest=manifest,started_utc=datetime.now(timezone.utc).isoformat(),aggregation='Within each seed average five fold softmax vectors, then argmax; summarize metrics across seeds with sample SD ddof=1',test_used_for_tuning=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    transform=ev.transforms.Compose([ev.transforms.Resize((224,224),antialias=True),ev.transforms.ToTensor()])
    dataset=ev.AsanoDataset(ROOT/'data/Asano_Full_Data.zip','test',transform=transform)
    assert len(dataset)==300
    loader=DataLoader(dataset,batch_size=25,shuffle=False,num_workers=4)
    device=torch.device('cuda')
    rows=[]
    reference=None
    for seed in SEEDS:
        trainer.set_deterministic(seed)
        config=dict(original_config,seed=seed)
        folder=OUT/f'seed_{seed}'
        folder.mkdir()
        for name,cond,weight in [('CE-only ViT','lambda_0',0.),('Triplet-ViT','lambda_0p0025',.0025)]:
            print('START',seed,name,flush=True)
            metrics,prob,labels=ev.evaluate_condition(trainer,RUNS/f'seed_{seed}',cond,weight,config,loader,device)
            assert np.bincount(labels,minlength=3).tolist()==[100,100,100]
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
            if reference is None: reference=labels; np.save(OUT/'test_labels.npy',labels)
            assert np.array_equal(reference,labels)
            np.save(folder/f'{cond}_probabilities.npy',prob)
            metrics['c_f1']=metrics['per_class_f1']['C']
            row=dict(seed=seed,model=name,metrics=metrics)
            rows.append(row)
            (folder/f'{cond}_metrics.json').write_text(json.dumps(row,indent=2))
            print('DONE',seed,name,json.dumps({k:metrics[k] for k in ['accuracy','macro_f1','c_recall','c_precision','c_specificity','c_f1','c_to_a','c_to_b']}),flush=True)
    keys=['accuracy','macro_f1','c_recall','c_precision','c_specificity','c_f1','c_to_a','c_to_b']
    summary={}
    for key in keys:
        ce=np.array([r['metrics'][key] for r in rows if r['model']=='CE-only ViT'])
        tr=np.array([r['metrics'][key] for r in rows if r['model']=='Triplet-ViT'])
        summary[key]={name:dict(mean=float(v.mean()),sd=float(v.std(ddof=1))) for name,v in [('ce',ce),('triplet',tr),('paired_difference',tr-ce)]}
        summary[key]['triplet_greater_seeds']=int((tr>ce).sum())
        summary[key]['equal_seeds']=int((tr==ce).sum())
    result=dict(rows=rows,summary=summary,completed_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__': main()
