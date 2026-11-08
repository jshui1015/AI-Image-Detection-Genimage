"""Locked CIFAKE external test. No training, selection, or calibration fitting."""
import csv,json,hashlib,os,sys,subprocess,shutil,math,statistics
from pathlib import Path
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT))
PHASE2=ROOT/'outputs/phase2-20261108-144532'
URL='https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images.git'

def read(p): return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def rows(p):
 with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def write_csv(p,rr):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
def git(*args):
 env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
 return subprocess.check_output(['git',*map(str,args)],env=env,text=True).strip()

def prepare(out):
 from PIL import Image,ImageOps
 assert read(PHASE2/'status.json')['finished']
 expected=read(out/'source_hashes.json')
 for name,h in expected.items():
  assert hashlib.sha256(((ROOT/name).read_text().rstrip()+'\n').encode()).hexdigest()==h,('source changed',name)
 info=read(PHASE2/'inputs.json'); temps=read(PHASE2/'temperatures.json')
 assert len(info['models'])==9 and set(temps)==set(info['models'])
 for n,d in info['models'].items():
  p=Path(d['path']); assert sha(p/'best.pt')==d['sha256'],n
  assert read(p/'validation_report.json')['validation_threshold_max_f1']==d['threshold']
  assert 0<d['threshold']<1 and math.isfinite(temps[n]['temperature']) and temps[n]['temperature']>0
 for name,h in info['manifests'].items(): assert sha(ROOT/'manifests'/name)==h
 encoder=read(PHASE2/'encoder.json'); assert Path(encoder['local_path']).is_dir()
 save(out/'locked_models.json',info['models']); save(out/'temperatures.json',temps); save(out/'encoder.json',encoder)
 # Sparse checkout downloads official test images, never trains on CIFAKE.
 repo=out/'dataset'
 git('clone','--depth','1','--filter=blob:none','--sparse',URL,repo)
 commit=git('-C',repo,'rev-parse','HEAD')
 git('-C',repo,'sparse-checkout','set','DATASET/test')
 git('-C',repo,'checkout','--detach',commit)
 manifest=[]; hashes={}; dims={}; duplicates=0
 for folder,label in [('REAL',0),('FAKE',1)]:
  files=sorted(p for p in (repo/'DATASET/test'/folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
  assert len(files)==10000,(folder,len(files),'expected 10000; no subsampling')
  for p in files:
   h=sha(p)
   if h in hashes:
    assert hashes[h]['label']==label,'Identical bytes with conflicting labels'
    duplicates+=1
   else: hashes[h]={'label':label,'path':str(p)}
   with Image.open(p) as im:
    rgb=ImageOps.exif_transpose(im).convert('RGB'); rgb.load(); wh=rgb.size
   dims[str(wh)]=dims.get(str(wh),0)+1
   manifest.append({'path':str(p.resolve()),'relative_path':p.relative_to(repo).as_posix(),'label':label,
       'split':'cifake_external','generator':'real' if label==0 else 'stable_diffusion_v_1_4',
       'class_index':-1,'sha256':h})
 assert len(manifest)==20000
 # Check exact-byte overlap with all previously used/reserved GenImage rows.
 prior=set()
 for name in ['all_splits.csv','test_glide_unseen.csv','test_vqdm_unseen.csv','test_robustness_new.csv']:
  prior.update(r['path'] for r in rows(ROOT/'manifests'/name))
 matches=[]
 for i,p in enumerate(sorted(prior)):
  h=sha(ROOT/p)
  if h in hashes: matches.append({'genimage':p,'cifake':hashes[h]['path']})
  if i%10000==0: print('OVERLAP CHECK',i,'/',len(prior),flush=True)
 save(out/'data_audit.json',{'count':len(manifest),'class_counts':{'real':10000,'fake':10000},'dimensions':dims,
   'same_label_duplicate_files':duplicates,'exact_byte_overlap':matches,'genimage_unique_files_checked':len(prior),
   'limitation':'Exact-byte check does not exclude re-encoding, near duplicates, or foundation-model pretraining exposure.'})
 assert not matches,'Exact overlap found; stop for review instead of silently filtering'
 write_csv(out/'manifest.csv',manifest)
 save(out/'protocol.json',{'source':URL,'commit':commit,'manifest_sha256':sha(out/'manifest.csv'),
  'input_control':'cap128_jpeg95','selection':'all official test/REAL and test/FAKE; 10000 each',
  'model_count':9,'fit_on_external_data':False,'threshold_source':'original GenImage validation',
  'temperature_source':'completed phase2 GenImage validation fitting',
  'interpretation':'Cross-dataset evaluation, not an unseen-generator benchmark; SD1.4 already seen in GenImage.',
  'resolution_note':'Low-resolution CIFAKE is a substantial domain shift; do not interpret as universal deployment accuracy.'})
 save(out/'prepare_status.json',{'finished':True})
 print('SUCCESS: CIFAKE prepared and frozen',flush=True)

def gpu(out):
 import numpy as np
 import torch
 from torch.utils.data import DataLoader
 from train import Images,ResNet50
 import clip_detector as c
 assert torch.cuda.is_available()
 assert read(out/'prepare_status.json')['finished']
 protocol=read(out/'protocol.json'); assert sha(out/'manifest.csv')==protocol['manifest_sha256']
 records=rows(out/'manifest.csv'); assert len(records)==20000
 for r in records: assert sha(r['path'])==r['sha256'],r['path']
 locked=read(out/'locked_models.json')
 for n,d in locked.items(): assert sha(Path(d['path'])/'best.pt')==d['sha256'],n
 def output(name,z):
  target=out/name; target.mkdir(exist_ok=True)
  np.savez_compressed(target/'logits.npz',logits=np.asarray(z,dtype=np.float64),labels=np.array([int(r['label']) for r in records]),paths=np.array([r['relative_path'] for r in records]))
 for name,d in locked.items():
  if name.startswith('clip-'): continue
  ck=torch.load(Path(d['path'])/'best.pt',map_location='cpu',weights_only=False)
  assert ck['input_control']=='cap128_jpeg95'
  model=ResNet50().cuda(); model.load_state_dict(ck['state_dict'],strict=True); model.eval()
  ds=Images(out/'manifest.csv','cifake_external',input_control='cap128_jpeg95')
  loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=4,pin_memory=True); z=[]
  with torch.inference_mode():
   for x,y,g,p in loader:
    with torch.autocast(device_type='cuda',dtype=torch.float16): logits=model(x.to('cuda',non_blocking=True))
    z.extend(logits.float().cpu().tolist())
  output(name,z); del model; torch.cuda.empty_cache(); print('DONE',name,flush=True)
 enc=read(out/'encoder.json'); model,processor=c.model_and_processor(enc['local_path'],None)
 model.to('cuda').eval()
 data=c.extract(model,processor,out/'manifest.csv','cifake_external',out,c.DEFAULT_MODEL,'cap128_jpeg95',torch.device('cuda'),128,4)
 assert data['paths']==[r['path'] for r in records]
 del model; torch.cuda.empty_cache()
 for name,d in locked.items():
  if not name.startswith('clip-'): continue
  ck=torch.load(Path(d['path'])/'best.pt',map_location='cpu',weights_only=False)
  assert ck['input_control']=='cap128_jpeg95' and ck['model_id']==c.DEFAULT_MODEL
  head=torch.nn.Linear(ck['feature_dim'],1).cuda(); head.load_state_dict(ck['state_dict'],strict=True); head.eval()
  with torch.inference_mode(): z=head(data['features'].cuda()).squeeze(1).cpu().numpy()
  output(name,z); del head; print('DONE',name,flush=True)
 save(out/'gpu_status.json',{'finished':True,'gpu':torch.cuda.get_device_name(0),'models':9})
 print('SUCCESS: all nine external predictions complete',flush=True)

def score_metrics(z,y,log_threshold):
 import numpy as np
 p=np.exp(-np.logaddexp(0,-z)); pred=z>=log_threshold
 tp=int(((y==1)&pred).sum()); tn=int(((y==0)&~pred).sum()); fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum())
 # Tied-score ranks for AUROC; use logits to avoid sigmoid saturation ties.
 order=np.argsort(z,kind='stable'); zs=z[order]; ys=y[order]; n=int(y.sum()); neg=len(y)-n
 starts=np.r_[0,np.flatnonzero(np.diff(zs))+1]; ends=np.r_[starts[1:],len(y)]
 ranksum=sum(float(ys[i:j].sum())*(i+1+j)/2 for i,j in zip(starts,ends))
 auc=(ranksum-n*(n+1)/2)/(n*neg)
 ap=0.; positives=0; total=0
 for i,j in zip(starts[::-1],ends[::-1]):
  k=int(ys[i:j].sum()); positives+=k; total+=j-i; ap+=(k/n)*(positives/total)
 bins=np.minimum((p*15).astype(int),14); ece=0.; reliability=[]
 for b in range(15):
  selected=bins==b
  if selected.any():
   conf=float(p[selected].mean()); freq=float(y[selected].mean()); count=int(selected.sum())
   ece+=count/len(y)*abs(conf-freq); reliability.append({'bin':b,'count':count,'mean_p_ai':conf,'fraction_ai':freq})
 m={'roc_auc':auc,'average_precision':ap,'accuracy':(tp+tn)/len(y),'precision':tp/max(1,tp+fp),
  'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),'tn':tn,'fp':fp,'fn':fn,'tp':tp,
  'brier':float(np.mean((p-y)**2)),'nll':float(np.mean(np.logaddexp(0,z)-y*z)),
  'ece_15_equal_width_p_ai':ece}
 return m,reliability,p,pred

def analyze(out):
 import numpy as np
 assert read(out/'gpu_status.json')['finished']
 locked=read(out/'locked_models.json'); temps=read(out/'temperatures.json'); results=[]
 manifest=rows(out/'manifest.csv')
 for name,d in locked.items():
  data=np.load(out/name/'logits.npz'); z,y=data['logits'],data['labels']
  assert len(y)==20000 and np.isfinite(z).all() and y.tolist()==[int(r['label']) for r in manifest]
  assert data['paths'].tolist()==[r['relative_path'] for r in manifest]
  t=d['threshold']; logth=math.log(t/(1-t)); T=temps[name]['temperature']
  values={}
  for state,temp in [('uncalibrated',1.),('temperature_scaled',T)]:
   m,reliability,p,pred=score_metrics(z/temp,y,logth/temp)
   results.append({'model':name,'state':state,'count':len(y),'temperature':temp,'decision_logit_threshold':logth/temp,**m})
   save(out/name/f'{state}-reliability.json',reliability); values[state]=(p,pred)
  assert np.array_equal(values['uncalibrated'][1],values['temperature_scaled'][1])
  write_csv(out/name/'per_image.csv',[{'path':path,'label':int(label),'logit':float(logit),
    'score_ai':float(values['uncalibrated'][0][i]),'score_ai_calibrated':float(values['temperature_scaled'][0][i]),
    'prediction_fixed_validation':int(values['uncalibrated'][1][i])} for i,(path,label,logit) in enumerate(zip(data['paths'],y,z))])
 write_csv(out/'metrics.csv',results)
 summary=[]
 for family in ['resnet','v2','clip']:
  for state in ['uncalibrated','temperature_scaled']:
   g=[r for r in results if r['model'].startswith(family+'-') and r['state']==state]; assert len(g)==3
   row={'family':family,'state':state,'n_seeds':3}
   for metric in ['roc_auc','average_precision','accuracy','f1','brier','nll','ece_15_equal_width_p_ai']:
    v=[r[metric] for r in g]; row[metric+'_mean']=statistics.mean(v); row[metric+'_sample_sd']=statistics.stdev(v)
   summary.append(row)
 write_csv(out/'summary.csv',summary)
 save(out/'status.json',{'finished':True,'models':9,'images_per_model':20000,'external_training_or_tuning':False,'decisions_preserved_after_calibration':True})
 import zipfile
 with zipfile.ZipFile(out/'results.zip','w',zipfile.ZIP_DEFLATED) as archive:
  for p in sorted(out.rglob('*')):
   rel=p.relative_to(out)
   if p.is_file() and rel.parts[0] not in ('dataset','features') and p.name!='results.zip':
    archive.write(p,rel.as_posix())
 print('SUCCESS: CIFAKE external validation complete',flush=True)

if __name__=='__main__':
 stage=sys.argv[1]; out=Path(sys.argv[2]).resolve()
 {'prepare':prepare,'gpu':gpu,'analyze':analyze}[stage](out)
