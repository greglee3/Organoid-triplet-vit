"""Additional common-recipe EfficientNet experiment; original runs are read-only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, time, hashlib, importlib.util, argparse, traceback, subprocess
from pathlib import Path
ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'runs/reviewer1_comment3_matched_efficientnet_20261103_v2'
TRAIN=ROOT/'05_final_tuning/02_ordinal_aware_triplet/train_pilot.py'
EFF=ROOT/'05_final_tuning/PROFESSOR_FEEDBACK/03_vit_triplet_efficientnet/train_efficientnet_oof.py'
OLD=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK/14_ce_vs_triplet_confirmatory/runs'
SEEDS=[70384,12731,76313]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(TRAIN.parent))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(p)
def status(state,**kw):write('STATUS.json',dict(state=state,time=time.time(),**kw));print(state,kw,flush=True)
def setup():
    import torch
    from torch import nn
    m=load('matched_original_vit_trainer',TRAIN);e=load('matched_original_eff_trainer',EFF)
    class EfficientMetricAdapter(nn.Module):
        def __init__(self,pretrained,embedding_head='shared',projection_dim=384,embedding_normalization='l2'):
            super().__init__();assert embedding_head=='shared'
            self.vit=e.create_model(pretrained=pretrained,trainable_mode='all')
            # Alias the unchanged historical EfficientNet classifier to the trainer's head interface.
            head=self.vit.classifier;del self.vit.classifier;self.vit.head=head
            self.embedding_dim=int(self.vit.num_features);self.embedding_source='efficientnet_pooled_pre_classifier'
            self.embedding_head='shared';self.metric_projection=None;self.embedding_normalization=embedding_normalization
        def metric_head_parameters(self):return []
        def forward(self,x):
            f=self.vit.global_pool(self.vit.forward_features(x))
            logits=self.vit.head(f)
            emb=torch.nn.functional.normalize(f,p=2,dim=1) if self.embedding_normalization=='l2' else f
            return logits,emb
    m.ViTSharedMetricModel=EfficientMetricAdapter
    return m,e,EfficientMetricAdapter
def check(m,e,Model):
    import torch,numpy as np
    settings={}; keys=['epochs','folds','classes_per_batch','samples_per_class','lr','head_lr_multiplier','optimizer','weight_decay','warmup_epochs','scheduler','augmentation','seed','distance_metric','embedding_normalization']
    ds=m.AsanoDataset(ROOT/'data/Asano_Full_Data.zip','train')
    from sklearn.model_selection import KFold
    for seed in SEEDS:
        manifest=json.loads((OLD/f'seed_{seed}/folds.json').read_text())
        assert [(x['sample'],x['label']) for x in manifest['ordered_samples']]==ds.samples
        for f,(a,b) in zip(manifest['folds'],KFold(5,shuffle=True,random_state=seed).split(np.arange(1200))):
            assert f['train_indices']==a.tolist() and f['validation_indices']==b.tolist()
        ce=torch.load(OLD/f'seed_{seed}/lambda_0/folds/fold_1_best.pt',map_location='cpu',weights_only=False)['args']
        tr=torch.load(OLD/f'seed_{seed}/lambda_0p0025/folds/fold_1_best.pt',map_location='cpu',weights_only=False)['args']
        assert all(ce[k]==tr[k] for k in keys)
        assert ce['epochs']==20 and ce['samples_per_class']==8 and ce['lr']==3e-5 and ce['augmentation']=='paper'
        assert ce.get('max_batches') is None and ce.get('max_folds') is None and not ce.get('no_pretrained',False)
        settings[str(seed)]=ce
    torch.manual_seed(123); original=e.create_model(False,'all').eval()
    torch.manual_seed(123); adapted=Model(False).eval()
    with torch.no_grad():
        x=torch.randn(2,3,224,224); a=original(x); b,emb=adapted(x)
        assert torch.allclose(a,b,atol=1e-6),(a-b).abs().max()
        assert emb.shape==(2,1280)
    del original,adapted
    print('PREFLIGHT_OK: matched archived settings, all fold identities, unchanged EfficientNet logits/head',flush=True)
    return settings
def evaluate(m,Model):
    import numpy as np, torch
    from torch.utils.data import DataLoader
    locks=[]
    for seed in SEEDS:
        for fold in range(1,6):
            p=OUT/f'training/seed_{seed}/lambda_0/folds/fold_{fold}_best.pt';locks.append(dict(seed=seed,fold=fold,sha256=sha(p)))
    write('CHECKPOINT_LOCK.json',locks)
    tf=m.transforms.Compose([m.transforms.Resize((224,224),antialias=True),m.transforms.ToTensor()])
    ds=m.AsanoDataset(ROOT/'data/Asano_Full_Data.zip','test',transform=tf);labels=np.array(ds.targets)
    loader=DataLoader(ds,batch_size=24,shuffle=False,num_workers=8)
    np.save(OUT/'test_labels.npy',labels);write('TEST_SAMPLE_MANIFEST.json',[dict(index=i,path=p,label=y) for i,(p,y) in enumerate(ds.samples)])
    rows=[]
    for seed in SEEDS:
        status('testing',seed=seed);arr=[]
        for fold in range(1,6):
            ck=torch.load(OUT/f'training/seed_{seed}/lambda_0/folds/fold_{fold}_best.pt',map_location='cpu',weights_only=False)
            model=Model(False).cuda();model.load_state_dict(ck['model'])
            probs,y,_=m.predict_probabilities(model,loader,torch.device('cuda'),None);assert np.array_equal(y,labels)
            arr.append(probs);del model,ck;torch.cuda.empty_cache()
        ep=np.mean(arr,axis=0);np.save(OUT/f'efficientnet_seed_{seed}_probabilities.npy',ep)
        base=ROOT/'runs/reviewer1_multiseed_test_20261029'
        assert np.array_equal(labels,np.load(base/'test_labels.npy'))
        for name,prob in [('EfficientNetV2-S matched recipe',ep),('CE-only ViT',np.load(base/f'seed_{seed}/lambda_0_probabilities.npy')),('Triplet-ViT',np.load(base/f'seed_{seed}/lambda_0p0025_probabilities.npy'))]:
            metric=m.calculate_metrics(labels,prob);p=metric['c_precision'];r=metric['c_recall'];metric['c_f1']=2*p*r/(p+r) if p+r else 0.;rows.append(dict(seed=seed,model=name,metrics=metric))
    summary={name:{k:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1))) for k in ['accuracy','macro_f1','c_recall','c_precision','c_specificity','c_f1'] for v in [[r['metrics'][k]*100 for r in rows if r['model']==name]]} for name in sorted({r['model'] for r in rows})}
    write('THREE_SEED_RESULTS.json',dict(rows=rows,summary_percent=summary,notes='Original architecture-specific heads retained. Shared training recipe. Selection: max macro-F1, lower objective-specific total validation loss on ties; total loss equals CE for CE-only ViT and EfficientNet, CE+weighted triplet for Triplet-ViT. No test tuning or t-tests.'))
def main():
    import torch
    torch.set_num_threads(4)
    m,e,Model=setup();settings=check(m,e,Model)
    if '--check' in sys.argv:return
    OUT.mkdir(exist_ok=False)
    try:
        write('PROTOCOL.json',dict(seeds=SEEDS,arch='tf_efficientnetv2_s',head='unchanged historical 1280-512-SiLU-Dropout(0.3)-3',initialization='timm pretrained=True; backbone-specific default weights',settings=settings,selection='Original trainer rule: maximum macro-F1 then lower total loss. EfficientNet total loss=CE. Triplet tie-break retains historical CE+lambda*triplet; this residual difference must be disclosed.',source_sha256={str(p):sha(p) for p in [TRAIN,EFF,Path(__file__)]},fold_hashes={str(s):sha(OLD/f'seed_{s}/folds.json') for s in SEEDS},scope='Only 15 new EfficientNet fold models. Historical CE and Triplet outputs reused unchanged. Test only after all training.'))
        for p in [TRAIN,EFF,Path(__file__)]: (OUT/p.name).write_bytes(p.read_bytes())
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])
            if free>=18000:break
            status('waiting_for_gpu',free_mib=free);time.sleep(60)
        for seed in SEEDS:
            status('training',seed=seed)
            args=dict(settings[str(seed)]);args.update(data=ROOT/'data/Asano_Full_Data.zip',output=OUT/f'training/seed_{seed}',lambdas='0',seed=seed,device='cuda')
            m.parse_args=lambda:argparse.Namespace(**args)
            m.main()
            generated=json.loads((args['output']/'folds.json').read_text());old=json.loads((OLD/f'seed_{seed}/folds.json').read_text());assert generated==old
        evaluate(m,Model);write('COMPLETE.flag',dict(time=time.time()));status('complete')
    except BaseException:
        write('ERROR.json',dict(time=time.time(),traceback=traceback.format_exc()));status('failed');raise
if __name__=='__main__':main()
