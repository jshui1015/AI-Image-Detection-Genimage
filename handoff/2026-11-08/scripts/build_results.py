from pathlib import Path
import csv,json,math,statistics,hashlib,importlib.util,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]; EXT=B/'evidence/cifake'
P1=B/'evidence/internal'
P2=B/'evidence/phase2'
def rows(p): return list(csv.DictReader(Path(p).open()))
def read(p): return json.loads(Path(p).read_text())
def write(p,r):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0])); w.writeheader(); w.writerows(r)
spec=importlib.util.spec_from_file_location('metrics_impl',EXT/'cifake.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
manifest=rows(EXT/'manifest.csv'); assert len(manifest)==20000
assert all(int(r['label'])==(1 if '/FAKE/' in r['relative_path'] else 0) for r in manifest)
seen=set(); keep=[]
for i,r in enumerate(manifest):
 if r['sha256'] not in seen: keep.append(i);seen.add(r['sha256'])
assert len(keep)==19962
met=rows(EXT/'metrics.csv'); models=read(EXT/'locked_models.json'); temps=read(EXT/'temperatures.json')
dedup=[]; errors=[]; images=[]; audit=[]
for name,entry in models.items():
 d=np.load(EXT/name/'logits.npz'); z,y=d['logits'],d['labels']; paths=d['paths'].tolist()
 assert paths==[r['relative_path'] for r in manifest] and y.tolist()==[int(r['label']) for r in manifest]
 assert np.isfinite(z).all(); t=entry['threshold']; lt=math.log(t/(1-t))
 for state,T in [('uncalibrated',1.),('temperature_scaled',temps[name]['temperature'])]:
  m,_,p,pred=mod.score_metrics(z/T,y,lt/T)
  saved=next(r for r in met if r['model']==name and r['state']==state)
  for k,v in m.items(): assert abs(float(saved[k])-v)<1e-10,(name,state,k)
  assert np.array_equal(pred,z>=lt)
  # Independent rank-sum AUROC calculation using sorted unique score indices.
  _,inv,cnt=np.unique(z,return_inverse=True,return_counts=True)
  avg_rank=np.cumsum(cnt)-(cnt-1)/2
  n=int(y.sum()); a=(avg_rank[inv][y==1].sum()-n*(n+1)/2)/(n*(len(y)-n))
  assert abs(m['roc_auc']-a)<1e-10
  dm,_,_,_=mod.score_metrics(z[keep]/T,y[keep],lt/T)
  dedup.append({'model':name,'state':state,'count':len(keep),**dm})
 p=np.exp(-np.logaddexp(0,-z)); pred=z>=lt
 for error,label in [('false_negative',1),('false_positive',0)]:
  ix=np.flatnonzero((y==label)&(pred!=y))
  ix=sorted(ix,key=lambda i: (float(p[i]) if label==1 else -float(p[i]),paths[i]))
  for rank,i in enumerate(ix[:20],1): errors.append({'dataset':'CIFAKE','model':name,'error':error,'rank':rank,'path':paths[i],'label':int(y[i]),'p_ai':float(p[i]),'threshold':t,'image_sha256':manifest[i]['sha256']})
  if name.endswith('20261031'):
   for i in ix[:3 if label==1 else 1]: images.append({'dataset':'CIFAKE','model':name,'error':error,'path':paths[i],'label':int(y[i]),'p_ai':float(p[i]),'threshold':t,'image_sha256':manifest[i]['sha256']})
 audit.append({'model':name,'fake_recall':float(np.mean(pred[y==1])),'predicted_fake_count':int(pred.sum())})
write(B/'tables/cifake_deduplicated_metrics.csv',dedup); write(B/'cases/cifake_error_cases.csv',errors)
(B/'cases/image_selection.json').write_text(json.dumps(images,indent=2))
write(B/'tables/cifake_decisions.csv',audit)
# Internal error index for every family/seed and test set; no fabricated images.
internal=[]
for family in ['resnet','v2','clip']:
 for seed in [20261031,20261101,20261102]:
  name=f'{family}-{seed}'
  for split in ['test_seen','test_glide_unseen','test_vqdm_unseen']:
   rr=rows(P1/name/split/'per_image.csv'); rep=read(P1/name/split/'report.json'); th=rep['threshold']
   for kind,label in [('false_negative',1),('false_positive',0)]:
    rr2=[r for r in rr if int(r['label'])==label and (float(r['score_ai'])>=th)!=bool(label)]
    rr2.sort(key=lambda r:(float(r['score_ai']) if label else -float(r['score_ai']),r['path']))
    for rank,r in enumerate(rr2[:10],1): internal.append({'dataset':split,'model':name,'error':kind,'rank':rank,'path':r['path'],'label':label,'p_ai':float(r['score_ai']),'threshold':th})
write(B/'cases/genimage_error_cases.csv',internal)
# Summaries from all available evidence.
allmetrics=[]
for family in ['resnet','v2','clip']:
 for seed in [20261031,20261101,20261102]:
  name=f'{family}-{seed}'
  for split in ['test_seen','test_glide_unseen','test_vqdm_unseen']:
   r=read(P1/name/split/'report.json'); m=next(m for m in r['metrics'] if m['group']=='all' and m['threshold_name']=='fixed_validation_f1')
   allmetrics.append({'family':family,'seed':seed,'dataset':split,'auc':m['roc_auc'],'accuracy':m['accuracy'],'f1':m['f1']})
  m=next(r for r in met if r['model']==name and r['state']=='uncalibrated')
  allmetrics.append({'family':family,'seed':seed,'dataset':'CIFAKE','auc':float(m['roc_auc']),'accuracy':float(m['accuracy']),'f1':float(m['f1'])})
summary=[]
for family in ['resnet','v2','clip']:
 for dataset in ['test_seen','test_glide_unseen','test_vqdm_unseen','CIFAKE']:
  g=[r for r in allmetrics if r['family']==family and r['dataset']==dataset]
  row={'family':family,'dataset':dataset,'n_seeds':3}
  for k in ['auc','accuracy','f1']: row[k+'_mean']=statistics.mean(r[k] for r in g);row[k+'_sd']=statistics.stdev(r[k] for r in g)
  summary.append(row)
write(B/'tables/main_results.csv',summary);write(B/'tables/per_seed_results.csv',allmetrics)
hold=[]
conditions=['clean','jpeg_q75','jpeg_q50','resize_075','resize_050','blur_r05','blur_r10']
for family in ['resnet','v2','clip']:
 for cond in conditions:
  v=[]
  for seed in [20261031,20261101,20261102]:
   if family=='clip': m=next(r for r in rows(P2/'metrics.csv') if r['model']==f'clip-{seed}' and r['dataset']=='holdout-'+cond and r['state']=='uncalibrated');v.append(float(m['roc_auc']))
   else: m=next(r for r in read(P1/f'{family}-{seed}'/'holdout/report.json')['results'] if r['group']=='all' and r['condition']==cond);v.append(m['roc_auc'])
  hold.append({'family':family,'condition':cond,'auc_mean':statistics.mean(v),'auc_sd':statistics.stdev(v)})
write(B/'tables/holdout_results.csv',hold)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
colors={'resnet':'#3268a8','v2':'#169983','clip':'#df8c38'}; labels={'resnet':'ResNet baseline','v2':'Robust V2','clip':'Frozen CLIP + head'}
def finish(fig,name):
 fig.savefig(B/'figures'/f'{name}.png');fig.savefig(B/'figures'/f'{name}.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.7)); xs=np.arange(4)
for j,f in enumerate(colors):
 g=[r for r in summary if r['family']==f];ax.bar(xs+(j-1)*.24,[r['auc_mean'] for r in g],.23,yerr=[r['auc_sd'] for r in g],capsize=3,label=labels[f],color=colors[f])
ax.axhline(.5,color='#777',ls='--',lw=1);ax.set(xticks=xs,xticklabels=['Seen','GLIDE','VQDM','CIFAKE external'],ylim=(0,1.06),ylabel='AUROC',title='High internal performance does not transfer to CIFAKE');ax.legend(loc='lower left');ax.text(0,-.17,'All three seeds; error bars = sample SD, not confidence intervals.',transform=ax.transAxes,fontsize=9);finish(fig,'01_generalization')
fig,ax=plt.subplots(figsize=(10,4.5))
for f in colors:
 g=[r for r in hold if r['family']==f];ax.errorbar(range(7),[r['auc_mean'] for r in g],yerr=[r['auc_sd'] for r in g],marker='o',capsize=3,color=colors[f],label=labels[f])
ax.set(xticks=range(7),xticklabels=['Clean','JPEG 75','JPEG 50','Resize .75','Resize .50','Blur .5','Blur 1.0'],ylim=(.5,1.02),ylabel='AUROC',title='V2 retains performance under the tested image degradations');ax.legend();finish(fig,'02_robustness')
# Calibration changes across datasets, sample mean of three per-model deltas.
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
cal=[]
for f in colors:
 for ds in ['test_seen','test_glide_unseen','test_vqdm_unseen','CIFAKE']:
  source=met if ds=='CIFAKE' else rows(P2/'metrics.csv')
  for metric in ['brier','ece_15_equal_width_p_ai']:
   before=[];after=[]
   for seed in [20261031,20261101,20261102]:
    for state,dst in [('uncalibrated',before),('temperature_scaled',after)]:
     r=next(r for r in source if r['model']==f'{f}-{seed}' and r['state']==state and (ds=='CIFAKE' or r['dataset']==ds));dst.append(float(r[metric]))
   cal.append({'family':f,'dataset':ds,'metric':metric,'before':statistics.mean(before),'after':statistics.mean(after),'delta':statistics.mean(np.array(after)-before)})
write(B/'tables/calibration_results.csv',cal)
for ax,metric,title in zip(axes,['brier','ece_15_equal_width_p_ai'],['Brier score change','ECE change (15 equal-width bins)']):
 for j,f in enumerate(colors):
  g=[r for r in cal if r['family']==f and r['metric']==metric];ax.bar(xs+(j-1)*.24,[r['delta'] for r in g],.23,color=colors[f],label=labels[f])
 ax.axhline(0,color='#555',lw=1);ax.set(xticks=xs,xticklabels=['Seen','GLIDE','VQDM','CIFAKE'],ylabel='After minus before (lower is better)',title=title)
axes[0].legend(fontsize=8);finish(fig,'03_calibration')
fig,axes=plt.subplots(1,3,figsize=(11,3.7))
for ax,f in zip(axes,colors):
 g=[r for r in met if r['model'].startswith(f+'-') and r['state']=='uncalibrated']
 matrix=np.array([[statistics.mean(float(r['tn']) for r in g),statistics.mean(float(r['fp']) for r in g)],[statistics.mean(float(r['fn']) for r in g),statistics.mean(float(r['tp']) for r in g)]])/10000
 ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
 for i in range(2):
  for j in range(2): ax.text(j,i,f'{matrix[i,j]:.3%}',ha='center',va='center',color='white' if matrix[i,j]>.6 else 'black')
 ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','AI'],yticklabels=['Real','AI'],xlabel='Predicted',ylabel='True',title=labels[f])
fig.suptitle('CIFAKE: most fake images are classified as real (three-seed mean)',y=1.04);finish(fig,'04_external_confusion')
fig,axes=plt.subplots(1,3,figsize=(11,3.5))
for ax,f in zip(axes,colors):
 d=np.load(EXT/f'{f}-20261031/logits.npz');z=d['logits'];y=d['labels']
 ax.hist(z[y==0],bins=50,density=True,alpha=.55,label='Real',color='#3268a8');ax.hist(z[y==1],bins=50,density=True,alpha=.55,label='AI',color='#df8c38')
 t=models[f'{f}-20261031']['threshold'];ax.axvline(math.log(t/(1-t)),ls='--',color='black',label='Locked threshold');ax.set(title=labels[f],xlabel='Raw AI logit',ylabel='Density')
axes[0].legend(fontsize=8);fig.suptitle('CIFAKE score distributions — fixed seed 20261031, no test-based selection',y=1.05);finish(fig,'05_external_scores')
# quantify dedup impact
impact=max(abs(float(next(r for r in met if r['model']==d['model'] and r['state']==d['state'])['roc_auc'])-d['roc_auc']) for d in dedup)
(B/'audit.json').write_text(json.dumps({'external_npz_checked':9,'external_metric_rows_checked':18,'label_mapping':'REAL=0; FAKE=1; verified against paths','original_images':20000,'unique_file_hashes':19962,'dedup_max_absolute_auc_change':impact,'fixed_decisions_preserved':True,'internal_csv_audit':'69 files verified earlier; phase2 57 NPZ and 114 metrics verified earlier'},indent=2))
print('PASS external metrics, labels, ranking and calibration decisions. Dedup max AUROC delta:',impact)
print('Image selections:',len(images))
