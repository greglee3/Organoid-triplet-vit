"""Single-seed normalization sensitivity; original trainers and outputs are read-only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, time, subprocess, hashlib, importlib.util
from pathlib import Path

ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'runs/reviewer_normalization_seed42_20261101'
PF=ROOT/'05_final_tuning/PROFESSOR_FEEDBACK'
RF=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK'
OLDV=RF/'07_locked_triplet_final_test/paired_seed_42_five_fold'
OLDE=RF/'11_locked_triplet_notebook_eff_final_test/seed_42_notebook_efficientnetv2_s'
VTRAIN=ROOT/'05_final_tuning/02_ordinal_aware_triplet/train_pilot.py'
ETRAIN=PF/'03_vit_triplet_efficientnet/train_efficientnet_oof.py'
PY=ROOT/'.venv/bin/python'

def write(p,obj):p.write_text(json.dumps(obj,indent=2,default=str))
def status(s,**kw):
    obj=dict(status=s,time=time.time(),**kw);write(OUT/'STATUS.json',obj);print(json.dumps(obj),flush=True)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def capacity(stage):
    good=0
    while good<3:
        line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
        free,util=map(int,line.split(','))
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
        good=good+1 if free>=16000 and util<=85 and len(procs)<=2 else 0
        status('checking_capacity' if good else 'waiting_for_capacity',stage=stage,free_mib=free,utilization=util,other_compute_processes=len(procs))
        time.sleep(5 if good else 60)

def patch_normalization():
    import torch
    from torchvision import transforms
    from PIL import Image
    original=transforms.ToTensor
    class NormalizedToTensor(original):
        def __call__(self,image):
            return transforms.functional.normalize(super().__call__(image),[0.5]*3,[0.5]*3)
    for value,expected in [(0,-1.),(255,1.)]:
        probe=NormalizedToTensor()(Image.new('RGB',(224,224),(value,)*3))
        assert probe.shape==(3,224,224) and torch.all(probe==expected)
    transforms.ToTensor=NormalizedToTensor
    print('NORMALIZATION_ACTIVE: RGB mean=std=(0.5,0.5,0.5); unchanged direct bilinear 224x224 resize',flush=True)

def train_vit():
    ev=module(PF/'07_locked_triplet_final_test/evaluate_locked_test_once.py','norm_vit_eval')
    trainer=ev.load_trainer()
    args='--ordinal-lambda 0 --lambdas 0,0.0025 --mining-mode semi-hard --adjacent-margin 0.1 --far-margin 0.1 --embedding-head shared --projection-dim 128 --epochs 20 --folds 5 --classes-per-batch 3 --samples-per-class 8 --workers 8 --lr 3e-5 --head-lr-multiplier 10 --optimizer adamw --weight-decay 0.01 --warmup-epochs 2 --scheduler cosine --scheduler-milestones 10,15 --scheduler-gamma 0.1 --triplet-ce-only-epochs 0 --triplet-ramp-epochs 0 --augmentation paper --seed 42 --min-c-specificity 0.90 --max-c-recall-drop 0.02 --max-c-precision-drop 0.03 --device cuda'.split()
    sys.argv=[str(VTRAIN),'--data',str(ROOT/'data/Asano_Full_Data.zip'),'--output',str(OUT/'vit')]+args
    trainer.main()
    verify_vit(ev)

def verify_vit(ev):
    import torch, numpy as np
    config=json.loads((PF/'07_locked_triplet_final_test/LOCKED_CONFIG.json').read_text())
    for cond,weight in [('lambda_0',0.),('lambda_0p0025',.0025)]:
        for fold in range(1,6):
            name=f'folds/fold_{fold}_best.pt'
            ck=torch.load(OUT/'vit'/cond/name,map_location='cpu',weights_only=False)
            old=torch.load(OLDV/cond/name,map_location='cpu',weights_only=False)
            ev.validate_checkpoint(ck,config,weight,fold)
            for key in ['train_indices','validation_indices','class_names']:
                assert np.array_equal(ck[key],old[key]),(cond,fold,key)
    print('VIT_ORIGINAL_FOLDS_AND_SETTINGS_VERIFIED',flush=True)

def train_eff():
    trainer=module(ETRAIN,'norm_eff_trainer')
    args='--seed 42 --epochs 50 --batch-size 35 --workers 16 --lr 0.00015 --backbone-lr 0.00015 --weight-decay 0.01 --trainable-mode all --scheduler step --step-size 3 --gamma 0.97 --gridmask-probability 0.3 --gridmask-implementation notebook --horizontal-flip-probability 0.5 --vertical-flip-probability 0.5 --device cuda'.split()
    sys.argv=[str(ETRAIN),'--data',str(ROOT/'data/Asano_Full_Data.zip'),'--folds-json',str(OLDV/'folds.json'),'--output',str(OUT/'efficient')]+args
    trainer.main()

def evaluate():
    import numpy as np, torch
    from torchvision import transforms
    from torch.utils.data import DataLoader
    ev=module(PF/'07_locked_triplet_final_test/evaluate_locked_test_once.py','norm_vit_eval')
    ee=module(PF/'11_locked_triplet_notebook_eff_final_test/evaluate_ensemble_test_once.py','norm_eff_eval')
    trainer=ev.load_trainer();eff=module(ETRAIN,'norm_eff_trainer')
    verify_vit(ev)
    lock=json.loads((RF/'11_locked_triplet_notebook_eff_final_test/ENSEMBLE_LOCK.json').read_text())
    # Freeze hashes of the NEW checkpoints, without changing the historical lock.
    lock['efficientnet']['checkpoints']=[{'sha256':ee.sha256(OUT/'efficient/folds'/f'fold_{fold}_best.pt')} for fold in range(1,6)]
    write(OUT/'NORMALIZED_CHECKPOINT_LOCK.json',lock)
    for fold in range(1,6):
        path=OUT/'efficient/folds'/f'fold_{fold}_best.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        ee.validate_checkpoint(ck,lock,fold,path)
    ds=ev.AsanoDataset(ROOT/'data/Asano_Full_Data.zip','test',transform=transforms.Compose([transforms.Resize((224,224),antialias=True),transforms.ToTensor()]))
    loader=DataLoader(ds,batch_size=25,shuffle=False,num_workers=4)
    config=json.loads((PF/'07_locked_triplet_final_test/LOCKED_CONFIG.json').read_text())
    oldlabels=np.load(RF/'07_locked_triplet_final_test/final_public_test/test_labels.npy')
    rows=[]
    for name,cond,weight in [('ce_vit','lambda_0',0.),('triplet_vit','lambda_0p0025',.0025)]:
        metrics,probs,labels=ev.evaluate_condition(trainer,OUT/'vit',cond,weight,config,loader,torch.device('cuda'))
        assert np.array_equal(labels,oldlabels)
        np.save(OUT/f'{name}_test_probabilities.npy',probs)
        oldname='ce_test_probabilities.npy' if weight==0 else 'triplet_test_probabilities.npy'
        oldprobs=np.load(RF/'07_locked_triplet_final_test/final_public_test'/oldname)
        rows.append(dict(model=name,normalized_test=metrics,original_test=ev.extended_metrics(trainer,labels,oldprobs),normalized_oof=json.loads((OUT/'vit'/cond/'oof_metrics.json').read_text()),original_oof=json.loads((OLDV/cond/'oof_metrics.json').read_text())))
    probs_list=[]
    for fold in range(1,6):
        ck=torch.load(OUT/'efficient/folds'/f'fold_{fold}_best.pt',map_location='cpu',weights_only=False)
        model=eff.create_model(pretrained=False,trainable_mode='all').to('cuda');model.load_state_dict(ck['model'])
        probs,labels=ee.predict(model,loader,torch.device('cuda'));assert np.array_equal(labels,oldlabels)
        probs_list.append(probs);del model;torch.cuda.empty_cache()
    probs=np.stack(probs_list).mean(axis=0);np.save(OUT/'efficient_test_probabilities.npy',probs);np.save(OUT/'test_labels.npy',oldlabels)
    oldprobs=np.load(RF/'11_locked_triplet_notebook_eff_final_test/final_public_test/efficient_test_probabilities.npy')
    rows.append(dict(model='efficientnetv2_s',normalized_test=ev.extended_metrics(trainer,labels,probs),original_test=ev.extended_metrics(trainer,labels,oldprobs),normalized_oof=json.loads((OUT/'efficient/oof_metrics.json').read_text()),original_oof=json.loads((OLDE/'oof_metrics.json').read_text())))
    write(OUT/'RESULTS.json',dict(seed=42,scope='Post-hoc single-seed normalization sensitivity; no test-driven selection',models=rows))

def main():
    sys.path.insert(0,str(ROOT))
    if '--check' in sys.argv:
        patch_normalization()
        from torchvision import transforms
        from PIL import Image
        import torch
        from paper_reproduction import PaperTrainTransform
        ev=module(PF/'07_locked_triplet_final_test/evaluate_locked_test_once.py','check_ev')
        ev.load_trainer()
        eff=module(ETRAIN,'check_eff')
        for tf in [PaperTrainTransform(224),eff.EfficientPaperTransform(gridmask_implementation='notebook'),transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])]:
            x=tf(Image.new('RGB',(1920,1440),(128,128,128)))
            assert x.shape==(3,224,224) and torch.isfinite(x).all() and x.min()>=-1 and x.max()<=1
        for p in [OLDV/'folds.json',OLDV/'lambda_0/oof_metrics.json',OLDV/'lambda_0p0025/oof_metrics.json',OLDE/'oof_metrics.json']:
            assert p.is_file(),p
        print('CPU_PREFLIGHT_OK: all three transforms, original outputs and trainer imports verified');return
    if '--stage' in sys.argv:
        stage=sys.argv[sys.argv.index('--stage')+1];patch_normalization()
        {'vit':train_vit,'efficient':train_eff,'test':evaluate}[stage]()
        (OUT/f'{stage}.done').write_text('completed\n');return
    OUT.mkdir(exist_ok=False)
    write(OUT/'PROTOCOL.json',dict(seed=42,folds=5,mean=[.5]*3,std=[.5]*3,resize='Direct 224x224 bilinear unchanged',only_change='Normalize after augmentation and ToTensor; applied identically during training, validation and test',vit_epochs=20,efficient_epochs=50,triplet_lambda=.0025,margin=.1,test_policy='All 15 checkpoints fixed before evaluation; post-hoc sensitivity, no test-based tuning',historical_vit=str(OLDV),historical_efficient=str(OLDE),trainer_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [VTRAIN,ETRAIN]},wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        for stage in ['vit','efficient','test']:
            capacity(stage);status('running',stage=stage)
            with (OUT/f'{stage}.log').open('x') as log:
                subprocess.run([str(PY),'-u',str(Path(__file__).resolve()),'--stage',stage],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        status('complete');(OUT/'COMPLETE.flag').write_text('15 folds and normalized test comparisons completed\n')
    except Exception as exc:
        status('failed',error=repr(exc));raise

if __name__=='__main__':main()
