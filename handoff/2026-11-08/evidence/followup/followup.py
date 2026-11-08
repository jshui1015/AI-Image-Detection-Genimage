import os,sys,csv,json,hashlib,math,shutil,io,subprocess,zipfile,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from bootstrap import cluster_bootstrap,interval
from metrics_reference import score_metrics
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
P1=ROOT/'outputs/completion-batch-2218770'
P2=ROOT/'outputs/phase2-20261108-144532'
CF=ROOT/'outputs/cifake-external-20261108-154531'
SEEDS=[20261031,20261101,20261102]; FAMILIES=['resnet','v2','clip']

def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csvout(p,rr):
 assert rr
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def sigmoid(z):return np.exp(-np.logaddexp(0,-np.asarray(z)))
def verify(out):
 for p,h in read(out/'protected_files.json').items():assert sha(p)==h,('Protected input changed',p)
 for name,h in read(out/'source_hashes.json').items():
  assert hashlib.sha256(((ROOT/name).read_text().rstrip()+'\n').encode()).hexdigest()==h,('Source changed',name)

def prepare(out):
 from huggingface_hub import hf_hub_download
 from PIL import Image,ImageOps
 assert read(CF/'status.json')['finished'] and read(P2/'status.json')['finished']
 locked=read(CF/'locked_models.json');assert len(locked)==9
 protected={}
 for n,d in locked.items():
  assert sha(Path(d['path'])/'best.pt')==d['sha256']
  protected[str(Path(d['path'])/'best.pt')]=d['sha256']
  v=Path(d['path'])/'validation_report.json';assert read(v)['validation_threshold_max_f1']==d['threshold'];protected[str(v)]=sha(v)
 for p in list((ROOT/'manifests').glob('*.csv'))+[CF/'temperatures.json',CF/'encoder.json']+list(P2.glob('*/*.npz'))+list(CF.glob('*/logits.npz')):
  protected[str(p)]=sha(p)
 for p in P1.glob('*/holdout/*per_image.csv'):protected[str(p)]=sha(p)
 enc=read(CF/'encoder.json')
 for p in Path(enc['local_path']).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.bin','.safetensors']:protected[str(p)]=sha(p)
 save(out/'protected_files.json',protected);verify(out)
 for f in ['locked_models.json','temperatures.json','encoder.json']:shutil.copy2(CF/f,out/f)
 lock=read(out/'dataset_lock.json');repo=lock['repo'];rev=lock['revision']
 assert shutil.disk_usage(out).free>5*1024**3,'Need at least 5 GiB filesystem free space (also check personal quota)'
 dataset=out/'dataset';dataset.mkdir(exist_ok=True)
 def fetch(f):return Path(hf_hub_download(repo_id=repo,repo_type='dataset',revision=rev,filename=f,local_dir=dataset))
 for f in ['README.md','LICENSE','prompts.json','coco/coco_filenames.txt']:fetch(f)
 names=(dataset/'coco/coco_filenames.txt').read_text().splitlines();assert len(names)==1061 and len(set(names))==1061
 prompts=read(dataset/'prompts.json'); ids={str(int(Path(n).stem)) for n in names}
 synthetic=[f for f in lock['files'] if f.startswith(('fooocus/','sdxl/'))]
 assert len(synthetic)==4244
 expected={f'{g}/{i}-{p}-{g}{ext}' for i in ids for p in ['short','long'] for g,ext in [('fooocus','.png'),('sdxl','.jpg')]}
 assert set(synthetic)==expected,'Unexpected official layout; stop without adapting test selection'
 # All official synthetic variants, not a score-selected subset.
 with ThreadPoolExecutor(max_workers=6) as ex:
  for i,_ in enumerate(ex.map(fetch,sorted(synthetic))):
   if i%250==0:print('SYNTHETIC DOWNLOAD',i,'/4244',flush=True)
 real=dataset/'real';real.mkdir(exist_ok=True)
 def getreal(name):
  assert Path(name).name==name and name.endswith('.jpg')
  p=real/name
  if not p.exists():
   tmp=p.with_suffix('.part')
   # Official COCO S3 bucket, TLS validation remains enabled.
   url='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+name
   subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3','--connect-timeout','20','--max-time','180',url,'-o',str(tmp)],check=True)
   tmp.replace(p)
  return p
 with ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(getreal,names))
 manifest=[];byte_seen={};pixel_seen={};dimensions={};duplicates=[]
 for f in ['real/'+n for n in sorted(names)]+sorted(synthetic):
  p=dataset/f;label=0 if f.startswith('real/') else 1
  ident=str(int(p.stem.split('-')[0]));group='coco-'+ident
  generator='real' if label==0 else f.split('/')[0]
  prompt='real' if label==0 else p.stem.split('-')[1]
  with Image.open(p) as im:
   rgb=ImageOps.exif_transpose(im).convert('RGB');rgb.load();w,h=rgb.size
   pixel=hashlib.sha256(f'{w}x{h}:'.encode()+rgb.tobytes()).hexdigest()
  assert min(w,h)>32,('Unexpected low resolution',f,w,h)
  if label==0:
   meta=prompts[ident]['coco_information'];assert (w,h)==(meta['width'],meta['height'])
  digest=sha(p)
  for index,key in [(byte_seen,digest),(pixel_seen,pixel)]:
   if key in index:
    assert index[key]['label']==label,'Duplicate has conflicting labels'
    duplicates.append({'path':f,'previous':index[key]['path'],'kind':'bytes' if index is byte_seen else 'pixels'})
   else:index[key]={'label':label,'path':f}
  dimensions[f'{w}x{h}']=dimensions.get(f'{w}x{h}',0)+1
  manifest.append({'path':str(p.resolve()),'relative_path':f,'label':label,'split':'followup_external','generator':generator,
   'class_index':-1,'group':group,'prompt':prompt,'width':w,'height':h,'sha256':digest,'pixel_sha256':pixel})
 # Unite duplicate images' COCO groups for bootstrap without dropping test rows.
 parent={r['group']:r['group'] for r in manifest}
 def root(g):
  while parent[g]!=g:
   parent[g]=parent[parent[g]];g=parent[g]
  return g
 hash_groups={}
 for r in manifest:
  for kind in ['sha256','pixel_sha256']:
   key=(kind,r[kind])
   if key in hash_groups:parent[root(r['group'])]=root(hash_groups[key])
   else:hash_groups[key]=r['group']
 for r in manifest:r['group']=root(r['group'])
 csvout(out/'external_manifest.csv',manifest)
 # Hash all old GenImage images and CIFAKE originals; stop on detected leakage.
 prior=set()
 for f in ['all_splits.csv','test_glide_unseen.csv','test_vqdm_unseen.csv','test_robustness_new.csv']:
  prior.update(str((ROOT/r['path']).resolve()) for r in rows(ROOT/'manifests'/f))
 prior.update(r['path'] for r in rows(CF/'manifest.csv'))
 overlaps=[]
 for i,p in enumerate(sorted(prior)):
  digest=sha(p)
  if digest in byte_seen:overlaps.append({'old':p,'external':byte_seen[digest]['path'],'kind':'bytes'})
  with Image.open(p) as im:
   rgb=ImageOps.exif_transpose(im).convert('RGB');w,h=rgb.size
   pixel=hashlib.sha256(f'{w}x{h}:'.encode()+rgb.tobytes()).hexdigest()
  if pixel in pixel_seen:overlaps.append({'old':p,'external':pixel_seen[pixel]['path'],'kind':'decoded_pixels'})
  if i%5000==0:print('OVERLAP AUDIT',i,'/',len(prior),flush=True)
 save(out/'data_audit.json',{'images':len(manifest),'real':1061,'fake':4244,'source_groups':1061,'bootstrap_groups_after_duplicate_union':len(set(r['group'] for r in manifest)),'dimensions':dimensions,'duplicates':duplicates,
  'overlaps':overlaps,'prior_images_checked':len(prior),'limitations':'Exact bytes and decoded RGB only; does not exclude near duplicates or pretraining exposure.'})
 assert not overlaps,'Detected overlap: inspect before evaluating; do not silently drop images'
 # Predeclared fixed consistency sample:32 per label in each old domain.
 checks=[]
 for domain in ['test_seen','test_glide_unseen','test_vqdm_unseen','CIFAKE']:
  ref=np.load((CF/'resnet-20261031/logits.npz') if domain=='CIFAKE' else P2/'resnet-20261031'/f'{domain}.npz')
  paths=ref['paths'].tolist(); labels=ref['labels']
  chosen=[]
  for label in [0,1]:
   candidates=np.flatnonzero(labels==label).tolist()
   chosen+=sorted(candidates,key=lambda i:hashlib.sha256(('20261108:'+paths[i]).encode()).hexdigest())[:32]
  cfmap={r['relative_path']:r for r in rows(CF/'manifest.csv')} if domain=='CIFAKE' else None
  for i in chosen:
   original=paths[i];p=cfmap[original]['path'] if cfmap else str((ROOT/original).resolve())
   checks.append({'path':p,'relative_path':f'check/{domain}/{i}','label':int(labels[i]),'split':'followup_external',
    'generator':'check','class_index':-1,'group':domain,'prompt':'check','width':0,'height':0,'sha256':sha(p),
    'pixel_sha256':'','reference_domain':domain,'reference_index':i,'reference_path':original})
 save(out/'consistency_selection.json',checks)
 # Shared inference entry point for consistency rows and all new images.
 fields=list(manifest[0]);combined=[{k:r[k] for k in fields} for r in checks]+manifest
 csvout(out/'inference_manifest.csv',combined)
 save(out/'protocol.json',{'dataset':repo,'revision':rev,'selection':'All 1061 COCO groups; all four synthetic variants per group',
  'labels':'real=0; fake=1; assigned by official folder, NOT HF numeric imagefolder labels',
  'primary_metric':'Macro AUROC and balanced accuracy over four equally-sized real-vs-generator/prompt cells; mean across three fixed seeds',
  'cells':['fooocus-short','fooocus-long','sdxl-short','sdxl-long'],
  'raw_pooled_accuracy':'Secondary only: 4:1 fake/real imbalance',
  'fit_or_tune':False,'control':'cap128_jpeg95 unchanged','consistency_max_probability_error':.002,
  'consistency_decision_policy':'Mismatches allowed only within 0.002 of original threshold; all reported',
  'duplicate_policy':'Keep all official rows; unite COCO groups linked by byte/RGB-identical images for bootstrap',
  'bootstrap':'2000 paired cluster resamples, seed 20261108; fixed-model conditional intervals, not uncertainty over new training seeds',
  'manifest_sha256':sha(out/'inference_manifest.csv')})
 verify(out);save(out/'prepare_status.json',{'finished':True});print('SUCCESS: prepared and frozen',flush=True)

def gpu(out):
 import torch
 from torch.utils.data import DataLoader
 from train import Images,ResNet50
 import clip_detector as c
 assert torch.cuda.is_available() and read(out/'prepare_status.json')['finished'];verify(out)
 assert sha(out/'inference_manifest.csv')==read(out/'protocol.json')['manifest_sha256']
 records=rows(out/'inference_manifest.csv');checks=read(out/'consistency_selection.json');locked=read(out/'locked_models.json')
 for r in records:assert sha(r['path'])==r['sha256']
 audit=[]
 # Compare relative-path original manifests with merged absolute-path manifest tensors.
 merged_ds=Images(out/'inference_manifest.csv','followup_external',input_control='cap128_jpeg95')
 tensor_checks=[]
 for domain in ['test_seen','test_glide_unseen','test_vqdm_unseen']:
  mf=ROOT/'manifests'/('all_splits.csv' if domain=='test_seen' else domain+'.csv')
  original_ds=Images(mf,domain,input_control='cap128_jpeg95')
  chosen=[i for i,r in enumerate(checks) if r['reference_domain']==domain]
  for i in chosen:
   ri=checks[i]['reference_index'];assert original_ds.rows[ri]['path']==checks[i]['reference_path']
   assert torch.equal(original_ds[ri][0],merged_ds[i][0]),('Tensor preprocessing mismatch',domain,i)
  tensor_checks.append({'family':'resnet_and_v2','domain':domain,'count':len(chosen),'exact_tensor_match':True})
 save(out/'tensor_consistency.json',tensor_checks)
 def result(name,z):
  z=np.asarray(z,dtype=float).reshape(-1);assert len(z)==len(records) and np.isfinite(z).all()
  folder=out/name;folder.mkdir(exist_ok=True)
  np.savez_compressed(folder/'logits.npz',logits=z,labels=np.array([int(r['label']) for r in records]),paths=np.array([r['relative_path'] for r in records]))
  t=locked[name]['threshold'];allok=True
  for domain in ['test_seen','test_glide_unseen','test_vqdm_unseen','CIFAKE']:
   ref=np.load(CF/name/'logits.npz' if domain=='CIFAKE' else P2/name/f'{domain}.npz')
   indices=[i for i,r in enumerate(checks) if r['reference_domain']==domain]
   ri=[checks[i]['reference_index'] for i in indices]
   assert ref['paths'][ri].tolist()==[checks[i]['reference_path'] for i in indices]
   assert ref['labels'][ri].tolist()==[int(checks[i]['label']) for i in indices]
   a,b=sigmoid(z[indices]),sigmoid(ref['logits'][ri]);diff=np.abs(a-b);mismatch=(a>=t)!=(b>=t)
   far=mismatch & (np.abs(b-t)>.002);ok=bool(diff.max()<=.002 and not far.any());allok &=ok
   audit.append({'model':name,'domain':domain,'count':len(indices),'max_abs_probability_error':float(diff.max()),
    'decision_mismatches':int(mismatch.sum()),'mismatches_far_from_threshold':int(far.sum()),'passed':ok})
  csvout(out/'consistency.csv',audit)
  assert allok,('Consistency check failed: stop without changing tolerances',name)
 for name,d in locked.items():
  if name.startswith('clip-'):continue
  ck=torch.load(Path(d['path'])/'best.pt',map_location='cpu',weights_only=False);assert ck['input_control']=='cap128_jpeg95'
  model=ResNet50().cuda();model.load_state_dict(ck['state_dict'],strict=True);model.eval()
  ds=Images(out/'inference_manifest.csv','followup_external',input_control='cap128_jpeg95')
  loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=2,pin_memory=True);z=[];paths=[];labels=[]
  with torch.inference_mode():
   for x,y,g,p in loader:
    with torch.autocast(device_type='cuda',dtype=torch.float16):v=model(x.cuda())
    z.extend(v.float().cpu().reshape(-1).tolist());paths.extend(p);labels.extend(y.tolist())
  assert paths==[r['path'] for r in records] and labels==[int(r['label']) for r in records]
  result(name,z);del model;torch.cuda.empty_cache();print('PASSED',name,flush=True)
 enc=read(out/'encoder.json');model,proc=c.model_and_processor(enc['local_path'],None);model.to('cuda').eval()
 merged_clip=c.ManifestImages(out/'inference_manifest.csv','followup_external',proc,'cap128_jpeg95')
 for domain in ['test_seen','test_glide_unseen','test_vqdm_unseen']:
  mf=ROOT/'manifests'/('all_splits.csv' if domain=='test_seen' else domain+'.csv')
  original_clip=c.ManifestImages(mf,domain,proc,'cap128_jpeg95')
  chosen=[i for i,r in enumerate(checks) if r['reference_domain']==domain]
  for i in chosen:
   ri=checks[i]['reference_index'];assert original_clip.rows[ri]['path']==checks[i]['reference_path']
   assert torch.equal(original_clip[ri][0],merged_clip[i][0]),('CLIP tensor mismatch',domain,i)
  tensor_checks.append({'family':'clip','domain':domain,'count':len(chosen),'exact_tensor_match':True})
 save(out/'tensor_consistency.json',tensor_checks)
 data=c.extract(model,proc,out/'inference_manifest.csv','followup_external',out,c.DEFAULT_MODEL,'cap128_jpeg95',torch.device('cuda'),128,2)
 assert data['paths']==[r['path'] for r in records] and data['labels'].tolist()==[int(r['label']) for r in records]
 del model;torch.cuda.empty_cache()
 for name,d in locked.items():
  if not name.startswith('clip-'):continue
  ck=torch.load(Path(d['path'])/'best.pt',map_location='cpu',weights_only=False)
  assert ck['input_control']=='cap128_jpeg95' and ck['model_id']==c.DEFAULT_MODEL
  head=torch.nn.Linear(ck['feature_dim'],1).cuda();head.load_state_dict(ck['state_dict'],strict=True);head.eval()
  with torch.inference_mode():z=head(data['features'].cuda()).squeeze(1).cpu().numpy()
  result(name,z);del head;print('PASSED',name,flush=True)
 verify(out);save(out/'gpu_status.json',{'finished':True,'gpu':torch.cuda.get_device_name(0),'models':9,'checks_per_model':len(checks),'external_images':len(records)-len(checks)})
 print('SUCCESS: consistency and external inference complete',flush=True)

METRICS=['roc_auc','balanced_accuracy','fake_recall']
def cirows(names,point,draws,dataset):
 result=[]
 def add(name,p,boot):
  bounds=interval(boot)
  for k,metric in enumerate(METRICS):result.append({'dataset':dataset,'model':name,'metric':metric,'estimate':float(p[k]),
   'ci95_low':float(bounds[0,k]),'ci95_high':float(bounds[1,k]),'bootstrap_replicates':len(boot)})
 for i,n in enumerate(names):add(n,point[i],draws[:,i])
 for f in FAMILIES:
  idx=[i for i,n in enumerate(names) if n.startswith(f+'-')]
  add(f+'-fixed3mean',point[idx].mean(axis=0),draws[:,idx].mean(axis=1))
 return result

def confidence(out):
 # CPU job independent of downloads and GPU; no changes to existing predictions.
 assert read(P2/'status.json')['finished'] and read(CF/'status.json')['finished']
 locked=read(CF/'locked_models.json');names=[f'{f}-{s}' for f in FAMILIES for s in SEEDS]
 manifest=rows(ROOT/'manifests/test_robustness_new.csv');assert len(manifest)==6000
 paths=[r['path'] for r in manifest];y=np.array([int(r['label']) for r in manifest])
 # Class blocks retain all matched generators/labels and paired perturbations.
 groups=np.array([r['class_index'] for r in manifest]);assert len(set(groups))==1000
 output=[];contrasts=[]
 for condition in ['clean','resize_050','blur_r10']:
  zs=[]
  for name in names:
   if name.startswith('clip-'):
    d=np.load(P2/name/f'holdout-{condition}.npz');assert d['paths'].tolist()==paths and d['labels'].tolist()==y.tolist();z=d['logits']
   else:
    rr=rows(P1/name/'holdout'/f'{condition}-per_image.csv');assert [r['path'] for r in rr]==paths and [int(r['label']) for r in rr]==y.tolist()
    p=np.array([float(r['score_ai']) for r in rr]);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    # Original artifacts saved float probabilities; preserve their tie structure.
    p=np.clip(p,np.nextafter(0.,1.),np.nextafter(1.,0.));z=np.log(p)-np.log1p(-p)
   zs.append(z)
  point,boot=cluster_bootstrap(y,zs,[locked[n]['threshold'] for n in names],groups)
  output+=cirows(names,point,boot,'holdout-'+condition)
  # Same images, class blocks, seed identifiers, and resamples for the difference.
  delta=point[3:6].mean(axis=0)-point[:3].mean(axis=0)
  dboot=boot[:,3:6].mean(axis=1)-boot[:,:3].mean(axis=1)
  ci=interval(dboot);adjusted=np.quantile(dboot,[.05/(2*3),1-.05/(2*3)],axis=0)
  for k,m in enumerate(METRICS):contrasts.append({'dataset':'holdout-'+condition,'comparison':'v2_mean_minus_resnet_mean','metric':m,
   'estimate':float(delta[k]),'ci95_low':float(ci[0,k]),'ci95_high':float(ci[1,k]),
   'bonferroni_three_conditions_low':float(adjusted[0,k]),'bonferroni_three_conditions_high':float(adjusted[1,k])})
  print('CI COMPLETE',condition,flush=True)
 csvout(out/'holdout_confidence.csv',output);csvout(out/'paired_differences.csv',contrasts)
 save(out/'ci_status.json',{'finished':True,'resampling_unit':'1000 semantic-class blocks, preserving all labels/generators and perturbation pairing',
  'reps':2000,'seed':20261108,'interpretation':'Conditional uncertainty for these nine fixed fitted models; not training-population uncertainty. Three-condition adjusted intervals provided per metric; other comparisons exploratory.'})

def analyze(out):
 assert read(out/'gpu_status.json')['finished'] and read(out/'ci_status.json')['finished'];verify(out)
 names=[f'{f}-{s}' for f in FAMILIES for s in SEEDS];locked=read(out/'locked_models.json');temps=read(out/'temperatures.json')
 rr=rows(out/'external_manifest.csv');ncheck=len(read(out/'consistency_selection.json'));y=np.array([int(r['label']) for r in rr]);zall=[];full=[]
 groups=np.array([r['group'] for r in rr]);cellpoints=[];cellboots=[];allci=[]
 for name in names:
  data=np.load(out/name/'logits.npz');assert data['paths'][ncheck:].tolist()==[r['relative_path'] for r in rr]
  assert data['labels'][ncheck:].tolist()==y.tolist();zall.append(data['logits'][ncheck:])
 zall=np.array(zall);thresholds=[locked[n]['threshold'] for n in names]
 selections={'pooled_imbalanced':np.arange(len(y))}
 for gen in ['fooocus','sdxl']:
  for prompt in ['short','long']:
   indices=np.array([i for i,r in enumerate(rr) if int(r['label'])==0 or (r['generator']==gen and r['prompt']==prompt)])
   assert len(indices)==2122 and int(y[indices].sum())==1061
   selections[gen+'-'+prompt]=indices
 for cell,idx in selections.items():
  for j,name in enumerate(names):
   t=thresholds[j];lt=math.log(t/(1-t));T=temps[name]['temperature']
   for state,temp in [('uncalibrated',1.),('temperature_scaled',T)]:
    m,_,_,pred=score_metrics(zall[j,idx]/temp,y[idx],lt/temp)
    assert np.array_equal(pred,zall[j,idx]>=lt)
    m['balanced_accuracy']=(m['recall']+m['tn']/(m['tn']+m['fp']))/2
    full.append({'dataset':cell,'model':name,'state':state,'n':len(idx),**m})
  if cell=='pooled_imbalanced':continue
  point,boot=cluster_bootstrap(y[idx],zall[:,idx],thresholds,groups[idx])
  allci+=cirows(names,point,boot,cell);cellpoints.append(point);cellboots.append(boot)
  print('EXTERNAL CI',cell,flush=True)
 # Group IDs sorted identically in all four cells, so bootstrap draws are paired.
 macro=np.mean(cellpoints,axis=0);macroboot=np.mean(cellboots,axis=0)
 allci+=cirows(names,macro,macroboot,'macro_four_cells')
 csvout(out/'external_metrics.csv',full);csvout(out/'external_confidence.csv',allci)
 summary=[]
 for family in FAMILIES:
  indices=[i for i,n in enumerate(names) if n.startswith(family+'-')]
  for k,m in enumerate(METRICS):
   p=macro[indices,k];ci=interval(macroboot[:,indices,k].mean(axis=1))
   summary.append({'family':family,'metric':m,'mean':float(p.mean()),'seed_sample_sd':float(p.std(ddof=1)),
    'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'basis':'equal macro of four real-vs-generator/prompt cells'})
 csvout(out/'external_summary.csv',summary)
 # Small, reusable external error index; all predictions retained in NPZ.
 errors=[]
 for j,name in enumerate(names):
  p=sigmoid(zall[j]);pred=p>=thresholds[j]
  for kind,label,want in [('false_negative',1,False),('false_positive',0,True)]:
   inds=np.flatnonzero((y==label)&(pred==want)).tolist();inds.sort(key=lambda i:p[i] if label==1 else -p[i])
   for rank,i in enumerate(inds[:20],1):errors.append({'model':name,'kind':kind,'rank':rank,'relative_path':rr[i]['relative_path'],
    'group':rr[i]['group'],'label':label,'score_ai':float(p[i]),'threshold':thresholds[j]})
 if errors:csvout(out/'error_cases.csv',errors)
 save(out/'status.json',{'finished':True,'models':9,'external_images':len(rr),'consistency_checks_per_model':ncheck,
  'gpu_allocations_required':1,'training_or_threshold_tuning':False,'original_inputs_hash_verified':True})
 (out/'RESULTS_README.md').write_text('''# 补充实验结果

先看 consistency.csv：36项（九模型×四旧测试域）必须 passed=True。
最大概率误差上限预先固定为0.002；同时报告所有决策差异，阈值以外超过0.002的翻转即停止。
这验证的是样本上的数值一致性，不声称完整数据或浮点计算逐位相同。

external_summary.csv 是外部主要结果：四个平衡的真图/生成器-提示词测试单元等权平均，再对三个固定种子取均值。
external_metrics.csv 中 pooled_imbalanced 为次要全量结果，假图比例80%，不能把其普通准确率与CIFAKE平衡集直接比较。
外部置信区间按1061个COCO原图编号成组重采样，所有变体、所有模型与种子使用相同重采样。
holdout_confidence.csv 与 paired_differences.csv 按1000个语义类别成组配对重采样。
置信区间描述测试抽样不确定性，条件于已训练九模型；不代表新随机种子的训练不确定性。
外部图片从未用于训练、温度拟合、阈值优化或挑选种子。
数据来源： https://huggingface.co/datasets/heikeadel/cocoxgen
真图来源：官方COCO val2017存储桶；版本/清单/哈希见protocol.json与external_manifest.csv。
检测了旧数据与新增数据的完全字节/解码RGB重叠；未声称排除近重复或基础模型预训练接触。
本次实验是CIFAKE之后新增的独立来源验证，不声称仅隔离了分辨率因素。
''')
 # Pack results and frozen scripts; omit image corpus/cache/features/checkpoints.
 with zipfile.ZipFile(out/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(out.rglob('*')):
   if not p.is_file() or p.name=='results.zip':continue
   rel=p.relative_to(out)
   if any(part in ['dataset','features','__pycache__'] for part in rel.parts):continue
   if p.suffix in ['.json','.csv','.npz','.py','.sh','.md','.txt','.log']:z.write(p,rel)
 print('SUCCESS: followup complete; results.zip ready',flush=True)

if __name__=='__main__':
 action=sys.argv[1];out=Path(sys.argv[2]).resolve();assert out.is_dir()
 {'prepare':prepare,'gpu':gpu,'confidence':confidence,'analyze':analyze}[action](out)
