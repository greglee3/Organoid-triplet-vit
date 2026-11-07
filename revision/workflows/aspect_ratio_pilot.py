"""Single-seed paired preprocessing experiment; preserves historical runs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, hashlib, importlib.util
from pathlib import Path, PosixPath
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'runs/reviewer_aspect_ratio_seed70384_20261029'
BASE=ROOT/'05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test'
OLD=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/runs/seed_70384'
OriginalResize=transforms.Resize

class LetterboxResize:
    def __init__(self,size,interpolation=transforms.InterpolationMode.BILINEAR,max_size=None,antialias=True):
        assert tuple(size)==(224,224), size
        self.interpolation=interpolation
        self.antialias=antialias
    def __call__(self,image):
        w,h=image.size
        scale=min(224/w,224/h)
        nw,nh=round(w*scale),round(h*scale)
        image=TF.resize(image,[nh,nw],interpolation=self.interpolation,antialias=self.antialias)
        left=(224-nw)//2; top=(224-nh)//2
        return TF.pad(image,[left,top,224-nw-left,224-nh-top],fill=0)

def main():
    OUT.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT))
    spec=importlib.util.spec_from_file_location('original_evaluator',BASE/'evaluate_locked_test_once.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    trainer=ev.load_trainer()
    # Both the original training augmentation and validation transform resolve
    # torchvision.transforms.Resize at construction time. Change only resizing.
    transforms.Resize=LetterboxResize
    from PIL import Image
    probe=LetterboxResize((224,224))(Image.new('RGB',(1920,1440),(255,255,255)))
    assert probe.size==(224,224)
    assert probe.getpixel((0,27))==(0,0,0) and probe.getpixel((0,28))==(255,255,255)
    args='--ordinal-lambda 0 --lambdas 0,0.0025 --mining-mode semi-hard --adjacent-margin 0.10 --far-margin 0.10 --embedding-head shared --projection-dim 384 --distance-metric cosine --embedding-normalization l2 --epochs 20 --folds 5 --classes-per-batch 3 --samples-per-class 8 --workers 8 --lr 3e-5 --head-lr-multiplier 10 --optimizer adamw --weight-decay 0.01 --warmup-epochs 2 --scheduler cosine --scheduler-milestones 10,15 --scheduler-gamma 0.1 --triplet-ce-only-epochs 0 --triplet-ramp-epochs 0 --augmentation paper --seed 70384 --min-c-specificity 0.90 --max-c-recall-drop 0.02 --max-c-precision-drop 0.03 --device cuda'.split()
    sys.argv=['aspect_ratio_pilot.py','--data',str(ROOT/'data/Asano_Full_Data.zip'),'--output',str(OUT/'training')]+args
    protocol={'seed':70384,'change':'Replace square resize with bilinear aspect-preserving resize and centered black padding; no normalization change. Augmentation remains before resizing.','original_1920x1440':'224x168 plus top/bottom 28 pixels','argv':sys.argv,'trainer_sha256':ev.sha256(ev.TRAINER_PATH),'wrapper_sha256':ev.sha256(Path(__file__)),'test_policy':'Evaluate after all training checkpoints fixed; no tuning on test. Single-seed exploratory reviewer experiment.'}
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    trainer.main()
    config=json.loads((BASE/'LOCKED_CONFIG.json').read_text());config['seed']=70384
    with torch.serialization.safe_globals([PosixPath]):
        for fold in range(1,6):
            for cond,weight in [('lambda_0',0.),('lambda_0p0025',.0025)]:
                ck=torch.load(OUT/'training'/cond/'folds'/f'fold_{fold}_best.pt',map_location='cpu',weights_only=True)
                old=torch.load(OLD/cond/'folds'/f'fold_{fold}_best.pt',map_location='cpu',weights_only=True)
                ev.validate_checkpoint(ck,config,weight,fold)
                for key in ['train_indices','validation_indices','class_names']:
                    assert np.array_equal(ck[key],old[key]),(fold,cond,key)
    print('CHECKPOINTS_AND_HISTORICAL_SPLITS_VERIFIED',flush=True)
    ds=ev.AsanoDataset(ROOT/'data/Asano_Full_Data.zip','test',transform=transforms.Compose([LetterboxResize((224,224)),transforms.ToTensor()]))
    loader=DataLoader(ds,batch_size=25,shuffle=False,num_workers=4)
    rows=[]
    for cond,weight in [('lambda_0',0.),('lambda_0p0025',.0025)]:
        trainer.set_deterministic(70384)
        metrics,prob,labels=ev.evaluate_condition(trainer,OUT/'training',cond,weight,config,loader,torch.device('cuda'))
        assert np.bincount(labels,minlength=3).tolist()==[100,100,100]
        np.save(OUT/f'{cond}_test_probabilities.npy',prob)
        np.save(OUT/'test_labels.npy',labels)
        rows.append({'condition':cond,'metrics':metrics})
    (OUT/'TEST_RESULTS.json').write_text(json.dumps(rows,indent=2))
    (OUT/'COMPLETE.flag').write_text('Completed training, historical split verification, and held-out test inference.\n')
    print('COMPLETE',json.dumps(rows),flush=True)

if __name__=='__main__':main()
