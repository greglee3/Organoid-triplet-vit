"""Post-hoc simple baselines. Fixed protocol; no test-driven selection.

Uses the existing manuscript's exact seed-42 folds. All five models of both
architectures are trained before the public test partition is loaded.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, random, sys, time
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

ROOT=Path(__import__('os').environ.get('ORGANOID_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(ROOT))
from asano_data import AsanoDataset
FOLDS=ROOT/'runs/05_final_tuning/PROFESSOR_FEEDBACK/07_locked_triplet_final_test/paired_seed_42_five_fold/folds.json'
DATA=ROOT/'data/Asano_Full_Data.zip'
PROTOCOL={
 'models':['pixel_mlp','small_cnn'], 'seed':42, 'epochs':50,'batch_size':32,
 'optimizer':'AdamW','lr':0.001,'weight_decay':0.0001,
 'scheduler':'CosineAnnealingLR, T_max=50, eta_min=1e-6',
 'criterion':'mean cross entropy; no weighting, triplet loss, ordinal penalty or label smoothing',
 'checkpoint_selection':'minimum validation CE; ties retain earlier epoch',
 'input':'RGB resized to 224x224 with torchvision PIL bilinear; fixed ImageNet mean/std',
 'augmentation':'independent per-image horizontal and vertical flips, each p=0.5, training only',
 'sampling':'random shuffle without replacement each epoch; no balanced sampler',
 'initialization':'random, no pretraining for either model',
 'pixel_mlp':'bilinear antialiased reduction to 64x64; flatten 12288; Linear 256, ReLU, Dropout .3; Linear 64, ReLU; Linear 3',
 'small_cnn':'3 blocks (Conv3x3 pad1 bias=False, BatchNorm, ReLU, MaxPool2); channels16,32,64; AdaptiveAvgPool4x4; Flatten1024; Linear128, ReLU, Dropout .3; Linear3',
 'folds':'exact existing seed42 image-level shuffled KFold, 960 train/240 validation',
 'public_test':'300 original test images, once after all ten checkpoints selected; mean of 5 fold softmax vectors; argmax',
 'scope':'Post-hoc reviewer-requested baselines; NOT a matched architecture-only ablation. Prior manuscript test results were already known. No hyperparameter search.',
 'reporting':'report both prespecified models regardless of performance; train-only pooled OOF and public test reported separately',
 'precision':'float32; deterministic algorithms; TF32 disabled on CUDA',
}
DEVICE=torch.device('cpu')

def now():return datetime.now(timezone.utc).isoformat()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

class PixelMLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(3*64*64,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,3))
 def forward(self,x):return self.net(F.interpolate(x,size=(64,64),mode='bilinear',align_corners=False,antialias=True))

class SmallCNN(nn.Module):
 def __init__(self):
  super().__init__();blocks=[];inc=3
  for outc in (16,32,64):
   blocks += [nn.Conv2d(inc,outc,3,padding=1,bias=False),nn.BatchNorm2d(outc),nn.ReLU(),nn.MaxPool2d(2)];inc=outc
  self.net=nn.Sequential(*blocks,nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1024,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,3))
 def forward(self,x):return self.net(x)

def model(name):return PixelMLP() if name=='pixel_mlp' else SmallCNN()
def metrics(y,p):
 pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=[0,1,2]);prec,rec,f1,sup=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
 result={'n':len(y),'accuracy':float((pred==y).mean()),'macro_f1':float(f1.mean()),'c_recall':float(rec[2]),'c_precision':float(prec[2]),'c_specificity':float(((pred!=2)&(y!=2)).sum()/(y!=2).sum()),'c_f1':float(f1[2]),'confusion_matrix':cm.tolist(),'per_class':{k:{'precision':float(prec[i]),'recall':float(rec[i]),'f1':float(f1[i]),'support':int(sup[i]),'auc':float(roc_auc_score(y==i,p[:,i]))} for i,k in enumerate('ABC')}}
 assert cm.sum()==len(y) and np.allclose(p.sum(1),1,atol=1e-5)
 assert np.isclose(result['macro_f1'],f1_score(y,pred,average='macro'))
 return result

def load_images(split):
 ds=AsanoDataset(DATA,split,transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
 loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=4)
 xs=[];ys=[]
 for x,y in loader:xs.append(x);ys.append(y)
 return ds,torch.cat(xs).to(DEVICE),torch.cat(ys).to(DEVICE)

MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1)
STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)
def prep(x,augment=False):
 if augment:
  h=torch.rand(len(x),1,1,1,device=x.device)<.5;v=torch.rand(len(x),1,1,1,device=x.device)<.5
  x=torch.where(h,x.flip(-1),x);x=torch.where(v,x.flip(-2),x)
 return (x-MEAN.to(x.device))/STD.to(x.device)

@torch.no_grad()
def predict(m,x,y,indices):
 m.eval();ps=[];total=0.
 for ix in indices.split(32):
  logits=m(prep(x[ix]));total+=F.cross_entropy(logits,y[ix],reduction='sum').item();ps.append(logits.softmax(1).cpu().numpy())
 return total/len(indices),np.concatenate(ps)

def save_predictions(path,ds,labels,probs,fold_ids=None):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['sample','label','prediction','p_A','p_B','p_C','heldout_fold'])
  for i,(p,y) in enumerate(zip(probs,labels)):
   w.writerow([ds.samples[i][0],int(y),int(p.argmax()),*map(float,p),'' if fold_ids is None else int(fold_ids[i])])

def main():
 global DEVICE
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=ap.parse_args()
 DEVICE=torch.device(args.device)
 torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 if DEVICE.type=='cuda':assert torch.cuda.is_available(),'CUDA unavailable'
 if args.smoke:
  for name in PROTOCOL['models']:
   seed(42);m=model(name).to(DEVICE);x=torch.rand(6,3,224,224,device=DEVICE);y=torch.arange(6,device=DEVICE)%3
   l=F.cross_entropy(m(prep(x,True)),y);l.backward();m.eval();p=m(prep(x)).softmax(1).detach().cpu().numpy();metrics(y.cpu().numpy(),p)
   print('SMOKE PASS',name,'parameters',sum(p.numel() for p in m.parameters()),'loss',l.item(),flush=True)
   m.train();x=torch.rand(32,3,224,224,device=DEVICE);y=torch.arange(32,device=DEVICE)%3;opt=torch.optim.AdamW(m.parameters(),lr=.001)
   start=time.monotonic()
   for _ in range(5):
    opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(prep(x,True)),y);loss.backward();opt.step()
   print('BENCHMARK',name,'seconds_per_training_batch',(time.monotonic()-start)/5,flush=True)
  return
 out=args.output
 if out is None:raise ValueError('--output required')
 out.mkdir(parents=True,exist_ok=False)
 lock={'created_utc':now(),'protocol':PROTOCOL,'trainer_sha256':sha(Path(__file__)),'folds_sha256':sha(FOLDS),'data_path':str(DATA),'data_size_bytes':DATA.stat().st_size,'versions':{'torch':torch.__version__,'numpy':np.__version__},'device':str(DEVICE),'cpu_threads':8,'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,'cpu_reason':'GPU failed CUDA initialization during smoke check' if DEVICE.type=='cpu' else None}
 dump(out/'LOCKED_PROTOCOL.json',lock)
 foldspec=json.loads(FOLDS.read_text());dump(out/'folds.json',foldspec)
 assert len(foldspec['ordered_samples'])==1200 and len(foldspec['folds'])==5
 print('Loading TRAIN ONLY',now(),flush=True)
 ds,x,y=load_images('train');labels=y.cpu().numpy();assert len(ds)==1200;assert np.bincount(labels).tolist()==[400,400,400]
 assert [(s['sample'],s['label']) for s in foldspec['ordered_samples']]==ds.samples,'Dataset ordering mismatch'
 seen=np.zeros(1200,int)
 for fold in foldspec['folds']:
  ti=fold['train_indices'];vi=fold['validation_indices']
  assert len(ti)==960 and len(vi)==240 and len(set(ti)&set(vi))==0 and set(ti)|set(vi)==set(range(1200))
  seen[vi]+=1
 assert (seen==1).all()
 summaries={}
 for name in PROTOCOL['models']:
  md=out/name;md.mkdir();oof=np.full((1200,3),np.nan);foldids=np.zeros(1200,int);summaries[name]={};best_info=[]
  for fold in foldspec['folds']:
   k=int(fold['fold']);seed(42+k);m=model(name).to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
   sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-6)
   ti=torch.tensor(fold['train_indices'],device=DEVICE);vi=torch.tensor(fold['validation_indices'],device=DEVICE);best=float('inf');bestep=0
   with (md/f'fold_{k}_history.csv').open('w',newline='') as hf:
    writer=csv.writer(hf);writer.writerow(['epoch','train_loss','val_loss','val_accuracy','val_macro_f1','lr','seconds'])
    for ep in range(1,51):
     start=time.monotonic();m.train();order=ti[torch.randperm(len(ti),device=DEVICE)];total=0.
     for ix in order.split(32):
      opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(prep(x[ix],True)),y[ix]);loss.backward();opt.step();total+=loss.item()*len(ix)
     vl,p=predict(m,x,y,vi);vm=metrics(labels[fold['validation_indices']],p)
     assert np.isfinite(vl),'nonfinite validation loss'
     if vl<best:
      best=vl;bestep=ep
      torch.save({'model':{k:v.cpu() for k,v in m.state_dict().items()},'architecture':name,'fold':k,'epoch':ep,'validation_loss':vl,'protocol':PROTOCOL},md/f'fold_{k}_best.pt')
     writer.writerow([ep,total/len(ti),vl,vm['accuracy'],vm['macro_f1'],opt.param_groups[0]['lr'],time.monotonic()-start]);hf.flush();sched.step()
     if ep==1 or ep%10==0:print(name,'fold',k,'epoch',ep,'valCE',round(vl,4),'best_epoch',bestep,'seconds',round(time.monotonic()-start,1),flush=True)
     dump(out/'STATUS.json',{'phase':'training','model':name,'fold':k,'epoch':ep,'updated_utc':now()})
   ck=torch.load(md/f'fold_{k}_best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);_,p=predict(m,x,y,vi)
   oof[fold['validation_indices']]=p;foldids[fold['validation_indices']]=k;best_info.append({'fold':k,'best_epoch':bestep,'validation_loss':best,'checkpoint_sha256':sha(md/f'fold_{k}_best.pt')})
   del m,opt,sched;torch.cuda.empty_cache()
  assert np.isfinite(oof).all();save_predictions(md/'oof_predictions.csv',ds,labels,oof,foldids);np.save(md/'oof_probabilities.npy',oof)
  summaries[name]={'oof':metrics(labels,oof),'checkpoints':best_info,'parameters':sum(p.numel() for p in model(name).parameters())};dump(md/'oof_metrics.json',summaries[name]);print('OOF COMPLETE',name,json.dumps(summaries[name]['oof']),flush=True)
 dump(out/'TRAINING_COMPLETE.json',{'utc':now(),'models':summaries})
 del x,y;torch.cuda.empty_cache()
 # No test image or test label has been loaded above this boundary.
 assert all((out/n/f'fold_{k}_best.pt').is_file() for n in PROTOCOL['models'] for k in range(1,6))
 print('All settings/checkpoints fixed. Loading PUBLIC TEST ONCE.',now(),flush=True)
 td,tx,ty=load_images('test');yl=ty.cpu().numpy();assert len(td)==300 and np.bincount(yl).tolist()==[100,100,100]
 assert not (set(s[0] for s in ds.samples)&set(s[0] for s in td.samples))
 for name in PROTOCOL['models']:
  foldps=[]
  for k in range(1,6):
   ck=torch.load(out/name/f'fold_{k}_best.pt',map_location='cpu',weights_only=False);m=model(name).to(DEVICE);m.load_state_dict(ck['model']);_,p=predict(m,tx,ty,torch.arange(300,device=DEVICE));foldps.append(p);del m
  stacked=np.stack(foldps);p=stacked.mean(0);np.save(out/name/'public_test_fold_probabilities.npy',stacked);save_predictions(out/name/'public_test_predictions.csv',td,yl,p)
  summaries[name]['public_test']=metrics(yl,p);dump(out/name/'public_test_metrics.json',summaries[name]['public_test'])
 dump(out/'BASELINE_RESULTS.json',{'completed_utc':now(),'protocol':lock,'models':summaries})
 dump(out/'STATUS.json',{'phase':'complete','updated_utc':now()});print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
