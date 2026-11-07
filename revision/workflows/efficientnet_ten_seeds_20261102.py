"""Frozen original EfficientNet recipe, ten matched seeds, no test tuning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,hashlib,subprocess,importlib.util,traceback
from pathlib import Path
ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT=ROOT/'runs/reviewer_efficientnet_ten_seeds_20261102'
PF=ROOT/'05_final_tuning/PROFESSOR_FEEDBACK'
RF=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK'
TRAINER=PF/'03_vit_triplet_efficientnet/train_efficientnet_oof.py'
OLD=RF/'11_locked_triplet_notebook_eff_final_test/seed_42_notebook_efficientnetv2_s'
SEEDS=[70384,12731,76313,98831,80324,104729,130363,155921,180749,205019]
SETTINGS={'epochs':50,'batch_size':35,'workers':16,'lr':.00015,'backbone_lr':.00015,'weight_decay':.01,'trainable_mode':'all','scheduler':'step','step_size':3,'gamma':.97,'gridmask_probability':.3,'gridmask_implementation':'notebook','horizontal_flip_probability':.5,'vertical_flip_probability':.5,'device':'cuda'}
def foldpath(seed):
 base=RF/'14_ce_vs_triplet_confirmatory/runs' if seed in SEEDS[:5] else ROOT/'runs/reviewer_ten_seeds_extension_20261030/training'
 return base/f'seed_{seed}/folds.json'
def probsbase(seed):
 return ROOT/'runs/reviewer1_multiseed_test_20261029' if seed in SEEDS[:5] else ROOT/'runs/reviewer_ten_seeds_extension_20261030/test'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):
 p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(p)
def status(state,**kw):write('STATUS.json',dict(state=state,time=time.time(),**kw));print(state,kw,flush=True)
def load():
 sys.path.insert(0,str(ROOT));spec=importlib.util.spec_from_file_location('original_eff_trainer',TRAINER)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def preflight():
 import torch,numpy as np
 m=load();ck=torch.load(OLD/'folds/fold_1_best.pt',map_location='cpu',weights_only=False)
 for k,v in SETTINGS.items():assert ck['args'][k]==v,(k,ck['args'][k],v)
 ds=m.AsanoDataset(ROOT/'data/Asano_Full_Data.zip','train')
 for seed in SEEDS:
  f=json.loads(foldpath(seed).read_text());assert f['seed']==seed and len(f['folds'])==5
  assert [(x['sample'],x['label']) for x in f['ordered_samples']]==ds.samples
  val=[]
  for split in f['folds']:
   assert not set(split['train_indices'])&set(split['validation_indices'])
   assert len(split['train_indices'])==960 and len(split['validation_indices'])==240
   val+=split['validation_indices']
  assert sorted(val)==list(range(1200))
  for cond in ['lambda_0','lambda_0p0025']:
   p=np.load(probsbase(seed)/f'seed_{seed}/{cond}_probabilities.npy');assert p.shape==(300,3)
 print('PREFLIGHT_OK: historical settings, ten shared fold manifests, and saved ViT predictions verified',flush=True)
def evaluate():
 import numpy as np,torch
 from torch.utils.data import DataLoader
 m=load();tf=m.transforms.Compose([m.transforms.Resize((224,224),antialias=True),m.transforms.ToTensor()])
 ds=m.AsanoDataset(ROOT/'data/Asano_Full_Data.zip','test',transform=tf)
 loader=DataLoader(ds,batch_size=25,shuffle=False,num_workers=4)
 reference=np.array(ds.targets);assert np.bincount(reference).tolist()==[100,100,100]
 np.save(OUT/'test_labels.npy',reference)
 write('TEST_SAMPLE_MANIFEST.json',[dict(index=i,path=p,label=int(y)) for i,(p,y) in enumerate(ds.samples)])
 manifests=[]
 # Freeze all checkpoints before accessing model test predictions.
 for seed in SEEDS:
  for fold in range(1,6):
   p=OUT/f'training/seed_{seed}/folds/fold_{fold}_best.pt';manifests.append(dict(seed=seed,fold=fold,path=str(p),sha256=sha(p)))
 write('CHECKPOINT_LOCK.json',manifests)
 existing=json.loads((ROOT/'runs/reviewer_ten_seeds_extension_20261030/TEN_SEED_RESULTS.json').read_text())['rows']
 lookup={(r['seed'],r['model']):r['metrics'] for r in existing};rows=[]
 for seed in SEEDS:
  status('testing',seed=seed);folds=json.loads(foldpath(seed).read_text())['folds'];plist=[]
  for fold in range(1,6):
   ck=torch.load(OUT/f'training/seed_{seed}/folds/fold_{fold}_best.pt',map_location='cpu',weights_only=False)
   assert ck['seed']==seed and ck['fold']==fold
   for key in ['train_indices','validation_indices']:assert ck[key]==folds[fold-1][key]
   for key,v in SETTINGS.items():assert ck['args'][key]==v
   model=m.create_model(pretrained=False,trainable_mode='all').to('cuda');model.load_state_dict(ck['model'])
   p,y=m.predict(model,loader,torch.device('cuda'),None);assert np.array_equal(y,reference)
   plist.append(p);del model,ck;torch.cuda.empty_cache()
  eff=np.stack(plist).mean(0);base=probsbase(seed)
  assert np.array_equal(reference,np.load(base/'test_labels.npy'))
  ce=np.load(base/f'seed_{seed}/lambda_0_probabilities.npy');tri=np.load(base/f'seed_{seed}/lambda_0p0025_probabilities.npy')
  folder=OUT/f'test/seed_{seed}';folder.mkdir(parents=True)
  for name,p,slug in [('CE-only ViT',ce,'ce'),('Triplet-ViT',tri,'triplet'),('EfficientNetV2-S',eff,'efficientnet'),('50:50 ensemble',(tri+eff)/2,'ensemble')]:
   assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
   metric=m.calculate_metrics(reference,p);metric['c_f1']=metric['per_class_f1']['C']
   if name in ['CE-only ViT','Triplet-ViT']:assert np.array_equal(metric['confusion_matrix'],lookup[seed,name]['confusion_matrix'])
   np.save(folder/f'{slug}_probabilities.npy',p);rows.append(dict(seed=seed,model=name,metrics=metric))
  write('PARTIAL_RESULTS.json',{'rows':rows})
 summary={}
 for name in ['CE-only ViT','Triplet-ViT','EfficientNetV2-S','50:50 ensemble']:
  summary[name]={}
  for k in ['accuracy','macro_f1','c_recall','c_precision','c_specificity','c_f1']:
   values=np.array([r['metrics'][k] for r in rows if r['model']==name])*100
   assert len(values)==10;summary[name][k]=dict(mean=values.mean(),sd=values.std(ddof=1))
 write('FOUR_MODEL_TEN_SEED_RESULTS.json',dict(rows=rows,summary=summary,notes='Same ten seeds and test images; EfficientNet original model-specific recipe differs from ViT. Ensemble = 0.5 Triplet probabilities + 0.5 EfficientNet probabilities within seed, each already five-fold averaged. No t-tests.'))
 write('COMPLETE.flag',{'completed':time.time()});status('complete')
def main():
 if '--check' in sys.argv:preflight();return
 OUT.mkdir(exist_ok=False)
 try:
  preflight()
  write('PROTOCOL.json',dict(seeds=SEEDS,settings=SETTINGS,resize='Original direct 224x224 bilinear; RGB; ToTensor only, no channel normalization',selection='minimum validation CE, same original EfficientNet trainer',ensemble='Fixed equal-weight Triplet plus EfficientNet per seed; no test optimization',scope='50 new fold models; existing CE and Triplet results unchanged',trainer_sha256=sha(TRAINER),wrapper_sha256=sha(Path(__file__)),fold_hashes={str(s):sha(foldpath(s)) for s in SEEDS},wait_for_existing_pid=653677))
  # Do not contend with or modify the authorized crop experiment.
  def active(pid):
   try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
   except FileNotFoundError:return False
  while active(653677):status('waiting_for_crop',pid=653677);time.sleep(60)
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])
  while free<10000:
   status('waiting_for_gpu_memory',free_mib=free);time.sleep(60)
   free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])
  for seed in SEEDS:
   status('training',seed=seed)
   args=[sys.executable,'-u',str(TRAINER),'--data',str(ROOT/'data/Asano_Full_Data.zip'),'--folds-json',str(foldpath(seed)),'--seed',str(seed),'--output',str(OUT/f'training/seed_{seed}')]
   for k,v in SETTINGS.items():args+=['--'+k.replace('_','-'),str(v)]
   subprocess.run(args,cwd=ROOT,check=True)
  evaluate()
 except BaseException:
  write('ERROR.json',dict(time=time.time(),traceback=traceback.format_exc()));status('failed');raise
if __name__=='__main__':main()
