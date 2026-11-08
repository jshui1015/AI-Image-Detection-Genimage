import csv, json, os, sys, io, hashlib, math, inspect
from pathlib import Path
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT))
OLD=ROOT/'outputs/completion-batch-2218770'
SHARED=Path('/project/mscaiehpc42026/AI-Image-Detection/data')
SEEDS=(20261031,20261101,20261102)
SPLITS=('val','test_seen','test_glide_unseen','test_vqdm_unseen')
CONDITIONS=[('clean',None),('jpeg_q75',('jpeg',75)),('jpeg_q50',('jpeg',50)),('resize_075',('resize',.75)),('resize_050',('resize',.5)),('blur_r05',('blur',.5)),('blur_r10',('blur',1.))]
def read(p): return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def models():
 for v in read(OLD/'provenance.json')['models']: yield v['name'],Path(v['path'])
 for s in SEEDS: yield f'clip-{s}',OLD/f'clip-{s}'
def prepare(out):
 import importlib.util
 from huggingface_hub import HfApi,snapshot_download
 spec=importlib.util.spec_from_file_location('previous_runner',OLD/'runner/run_batch.py')
 r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r); r.preflight()
 assert read(OLD/'status.json')=={'failed_tasks':[],'finished':True}
 for d in read(OLD/'cache-audit.json').values(): assert sha(d['file'])==d['sha256']
 repo='openai/clip-vit-base-patch32'; revision=HfApi().model_info(repo).sha
 # Pin this download and record it; the GPU stage additionally tests compatibility with old features.
 local=snapshot_download(repo,revision=revision,cache_dir=str(ROOT/'.cache/huggingface'),allow_patterns=['*.json','model.safetensors','pytorch_model.bin'])
 save(out/'encoder.json',{'model_id':repo,'revision':revision,'local_path':local})
 save(out/'inputs.json',{'models':{n:{'path':str(p),'sha256':sha(p/'best.pt'),'threshold':read(p/'validation_report.json')['validation_threshold_max_f1']} for n,p in models()},'manifests':{p.name:sha(p) for p in (ROOT/'manifests').glob('*.csv')}})
 print('PREPARE SUCCESS',flush=True)

def dump_scores(p,rows,z):
 import numpy as np
 p.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(p,logits=np.asarray(z,dtype=np.float64),labels=np.array([int(r['label']) for r in rows]),paths=np.array([r['path'] for r in rows]))
def manifest_rows(split):
 name='all_splits.csv' if split in ('val','test_seen') else split+'.csv'
 with (ROOT/'manifests'/name).open() as f: return [r for r in csv.DictReader(f) if r['split']==split]

def gpu(out):
 import numpy as np
 import torch
 from torch.utils.data import Dataset,DataLoader
 from PIL import Image,ImageOps,ImageFilter
 from train import Images,ResNet50
 import clip_detector as c
 assert torch.cuda.is_available()
 inputs=read(out/'inputs.json')
 for n,p in models(): assert sha(p/'best.pt')==inputs['models'][n]['sha256']
 for name,h in inputs['manifests'].items(): assert sha(ROOT/'manifests'/name)==h
 encoder,processor=c.model_and_processor(read(out/'encoder.json')['local_path'],None)
 encoder.to('cuda').eval()
 # Validate newly downloaded encoder/preprocessing against existing frozen validation features.
 cache=torch.load(SHARED/'outputs/clip-vit-b32-cap128-jpeg95/features/val.pt',map_location='cpu',weights_only=False)
 ds=c.ManifestImages(ROOT/'manifests/all_splits.csv','val',processor,'cap128_jpeg95')
 ix=np.linspace(0,len(ds)-1,32,dtype=int).tolist()
 x=torch.stack([ds[i][0] for i in ix]).to('cuda')
 with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
  result=encoder.get_image_features(pixel_values=x)
 f=torch.nn.functional.normalize(result.pooler_output.float(),dim=1).cpu()
 old=cache['features'][ix]; cosine=(f*old).sum(dim=1)
 error=float((f-old).abs().max())
 save(out/'encoder_compatibility.json',{'indices':ix,'minimum_cosine':float(cosine.min()),'max_absolute_difference':error,'policy':'cosine >= 0.9999 and maximum element difference <= 0.002'})
 assert float(cosine.min())>=.9999 and error<=.002,'Encoder/cache mismatch: stop without changing the frozen encoder'
 del x,f,cache
 # Extract each holdout condition once, reuse for all three heads.
 class HoldoutClip(Dataset):
  def __init__(self,rows,pert): self.rows,self.pert=rows,pert
  def __len__(self): return len(self.rows)
  def __getitem__(self,i):
   with Image.open(ROOT/self.rows[i]['path']) as src: im=ImageOps.exif_transpose(src).convert('RGB')
   w,h=im.size; scale=min(1.,128/max(w,h))
   if scale<1: im=im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS)
   def jpeg(im,q):
    b=io.BytesIO(); im.save(b,format='JPEG',quality=q); b.seek(0)
    with Image.open(b) as enc: return enc.convert('RGB')
   im=jpeg(im,95)
   if self.pert:
    kind,v=self.pert
    if kind=='jpeg': im=jpeg(im,int(v))
    elif kind=='resize':
     w,h=im.size; small=(max(1,round(w*v)),max(1,round(h*v)))
     im=im.resize(small,Image.Resampling.BILINEAR).resize((w,h),Image.Resampling.BILINEAR)
    elif kind=='blur': im=im.filter(ImageFilter.GaussianBlur(radius=float(v)))
   return processor(images=im,return_tensors='pt')['pixel_values'][0]
 with (ROOT/'manifests/test_robustness_new.csv').open() as f: rows=list(csv.DictReader(f))
 assert len(rows)==6000
 for condition,pert in CONDITIONS:
  loader=DataLoader(HoldoutClip(rows,pert),batch_size=128,shuffle=False,num_workers=4,pin_memory=True)
  feats=[]
  with torch.inference_mode():
   for x in loader:
    with torch.autocast(device_type='cuda',dtype=torch.float16): result=encoder.get_image_features(pixel_values=x.to('cuda',non_blocking=True))
    feats.append(torch.nn.functional.normalize(result.pooler_output.float(),dim=1).cpu())
  features=torch.cat(feats)
  for seed in SEEDS:
   name=f'clip-{seed}'; ck=torch.load(OLD/name/'best.pt',map_location='cpu',weights_only=False)
   head=torch.nn.Linear(512,1).cuda(); head.load_state_dict(ck['state_dict']); head.eval()
   with torch.inference_mode(): z=head(features.cuda()).squeeze(1).cpu().numpy()
   dump_scores(out/name/f'holdout-{condition}.npz',rows,z)
  print('CLIP HOLDOUT',condition,flush=True)
 del encoder,features,head; torch.cuda.empty_cache()
 for name,p in models():
  ck=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
  if name.startswith('clip'):
   model=torch.nn.Linear(512,1).cuda(); model.load_state_dict(ck['state_dict']); model.eval()
   for split in SPLITS:
    path=SHARED/'outputs/clip-vit-b32-cap128-jpeg95/features'/f'{split}.pt'
    d=torch.load(path,map_location='cpu',weights_only=False); rows=manifest_rows(split)
    assert d['paths']==[r['path'] for r in rows]
    assert d['labels'].tolist()==[float(r['label']) for r in rows]
    with torch.inference_mode(): z=model(d['features'].cuda()).squeeze(1).cpu().numpy()
    dump_scores(out/name/f'{split}.npz',rows,z)
  else:
   model=ResNet50().cuda(); model.load_state_dict(ck['state_dict']); model.eval()
   for split in SPLITS:
    mf=ROOT/'manifests'/('all_splits.csv' if split in ('val','test_seen') else split+'.csv')
    ds=Images(mf,split,input_control='cap128_jpeg95')
    loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=4,pin_memory=True); z=[]
    with torch.inference_mode():
     for x,y,g,pth in loader:
      with torch.autocast(device_type='cuda',dtype=torch.float16): logits=model(x.to('cuda',non_blocking=True))
      z.extend(logits.float().cpu().tolist())
    dump_scores(out/name/f'{split}.npz',ds.rows,z)
  del model; torch.cuda.empty_cache()
  print('LOGITS COMPLETE',name,flush=True)
 save(out/'gpu_status.json',{'finished':True})

def analyze(out):
 import numpy as np
 from scipy.special import expit
 from scipy.optimize import minimize_scalar
 from sklearn.metrics import roc_auc_score,average_precision_score
 inputs=read(out/'inputs.json'); results=[]; temperatures={}
 assert read(out/'gpu_status.json')['finished']
 def stats(z,y,t):
  prob=expit(z); pred=prob>=t
  bins=np.minimum((prob*15).astype(int),14); ece=0.; reliability=[]
  for b in range(15):
   sel=bins==b
   if sel.any():
    conf=float(prob[sel].mean()); freq=float(y[sel].mean())
    ece+=sel.mean()*abs(conf-freq)
    reliability.append({'bin':b,'count':int(sel.sum()),'mean_p_ai':conf,'fraction_ai':freq})
  tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum())
  return {'roc_auc':float(roc_auc_score(y,prob)),'average_precision':float(average_precision_score(y,prob)),
   'brier':float(np.mean((prob-y)**2)),'nll':float(np.mean(np.logaddexp(0,z)-y*z)),
   'ece_15_equal_width_p_ai':float(ece),'accuracy':float(np.mean(pred==y)),
   'f1':2*tp/max(1,2*tp+fp+fn),'reliability':reliability}
 for name,entry in inputs['models'].items():
  val=np.load(out/name/'val.npz'); z,y=val['logits'],val['labels']
  assert len(y)==8000 and np.isfinite(z).all()
  def loss(logt):
   scaled=z/np.exp(logt); return float(np.mean(np.logaddexp(0,scaled)-y*scaled))
  opt=minimize_scalar(loss,bounds=(-4,4),method='bounded',options={'xatol':1e-8})
  assert opt.success
  logt=min([0.,float(opt.x)],key=loss); T=float(np.exp(logt))
  temperatures[name]={'temperature':T,'val_nll_before':loss(0.),'val_nll_after':loss(logt),
    'near_search_boundary':abs(logt)>3.99,'fit_split':'val only',
    'limitation':'Same validation split was also used for checkpoint/threshold selection; validation calibration is in-sample.'}
  threshold=float(entry['threshold']); assert 0<threshold<1
  for file in sorted((out/name).glob('*.npz')):
   d=np.load(file); z,y=d['logits'],d['labels']; dataset=file.stem
   expected=8000 if dataset=='val' else 10000 if dataset=='test_seen' else 6000 if dataset.startswith('holdout') else 4000
   assert len(y)==expected and np.isfinite(z).all() and len(set(d['paths']))==expected
   if dataset in SPLITS and dataset!='val':
    original=list(csv.DictReader((OLD/name/dataset/'per_image.csv').open()))
    assert d['paths'].tolist()==[r['path'] for r in original]
    assert y.tolist()==[int(r['label']) for r in original]
    maxdiff=float(np.max(np.abs(expit(z)-np.array([float(r['score_ai']) for r in original]))))
    assert maxdiff<.002,(name,dataset,'prediction mismatch',maxdiff)
   for state,temp in [('uncalibrated',1.),('temperature_scaled',T)]:
    t=float(expit(math.log(threshold/(1-threshold))/temp))
    m=stats(z/temp,y,t); reliability=m.pop('reliability')
    save(out/name/f'{dataset}-{state}-reliability.json',reliability)
    results.append({'model':name,'dataset':dataset,'state':state,'count':len(y),'temperature':temp,'threshold':t,**m})
 save(out/'temperatures.json',temperatures)
 with (out/'metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
 # Sample SD across all three seeds, never select the best test seed.
 import statistics
 summary=[]
 for family in ('resnet','v2','clip'):
  for dataset in sorted(set(r['dataset'] for r in results)):
   for state in ('uncalibrated','temperature_scaled'):
    group=[r for r in results if r['model'].startswith(family+'-') and r['dataset']==dataset and r['state']==state]
    if not group: continue
    assert len(group)==3
    row={'family':family,'dataset':dataset,'state':state,'n_seeds':3}
    for metric in ('roc_auc','average_precision','brier','nll','ece_15_equal_width_p_ai','accuracy','f1'):
     values=[r[metric] for r in group]; row[metric+'_mean']=statistics.mean(values); row[metric+'_sample_sd']=statistics.stdev(values)
    summary.append(row)
 with (out/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
 save(out/'status.json',{'finished':True,'models':9,'clip_holdout_conditions':7,
   'temperature_fitted_on_test':False,'note':'Threshold transformed with temperature to preserve the original decision rule. No claim of external-domain validation.'})
 print('SUCCESS: phase2 analysis complete',flush=True)

if __name__=='__main__':
 stage=sys.argv[1]; out=Path(sys.argv[2]).resolve(); out.mkdir(parents=True,exist_ok=True)
 {'prepare':prepare,'gpu':gpu,'analyze':analyze}[stage](out)
