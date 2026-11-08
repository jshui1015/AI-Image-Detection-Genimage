from pathlib import Path
import sys,csv,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];F=B/'evidence/followup'
sys.path.insert(0,str(F))
from bootstrap import cluster_bootstrap,interval
from metrics_reference import score_metrics

def rows(p):return list(csv.DictReader(Path(p).open()))
def read(p):return json.loads(Path(p).read_text())
rr=rows(F/'external_manifest.csv');checks=read(F/'consistency_selection.json');models=read(F/'locked_models.json');temps=read(F/'temperatures.json');saved=rows(F/'external_metrics.csv')
assert len(rr)==5305 and len(checks)==256
assert sum(int(r['label']) for r in rr)==4244
assert len({r['group'] for r in rr})==1061
assert all(int(r['label'])==(0 if r['relative_path'].startswith('real/') else 1) for r in rr)
audit=rows(F/'consistency.csv');assert len(audit)==36
assert all(r['passed']=='True' and int(r['decision_mismatches'])==0 for r in audit)
assert all(r['exact_tensor_match'] for r in read(F/'tensor_consistency.json'))
y=np.array([int(r['label']) for r in rr]);names=[f'{f}-{s}' for f in ['resnet','v2','clip'] for s in [20261031,20261101,20261102]];zs=[]
for name in names:
 d=np.load(F/name/'logits.npz');assert d['paths'][256:].tolist()==[r['relative_path'] for r in rr]
 assert d['labels'][256:].tolist()==y.tolist() and np.isfinite(d['logits']).all()
 z=d['logits'][256:];zs.append(z)
 for dom in ['test_seen','test_glide_unseen','test_vqdm_unseen','CIFAKE']:
  ref=np.load(B/'evidence/cifake'/name/'logits.npz' if dom=='CIFAKE' else B/'evidence/phase2'/name/(dom+'.npz'))
  ii=[i for i,r in enumerate(checks) if r['reference_domain']==dom];jj=[checks[i]['reference_index'] for i in ii]
  assert ref['paths'][jj].tolist()==[checks[i]['reference_path'] for i in ii]
  a=np.exp(-np.logaddexp(0,-d['logits'][ii]));b=np.exp(-np.logaddexp(0,-ref['logits'][jj]));err=float(np.max(abs(a-b)))
  orig=next(r for r in audit if r['model']==name and r['domain']==dom)
  assert abs(err-float(orig['max_abs_probability_error']))<1e-12
 for cell in ['pooled_imbalanced','fooocus-short','fooocus-long','sdxl-short','sdxl-long']:
  idx=np.arange(len(y)) if cell=='pooled_imbalanced' else np.array([i for i,r in enumerate(rr) if int(r['label'])==0 or r['generator']+'-'+r['prompt']==cell])
  for state,t in [('uncalibrated',1.),('temperature_scaled',temps[name]['temperature'])]:
   threshold=models[name]['threshold'];lt=np.log(threshold/(1-threshold))
   m,_,_,pred=score_metrics(z[idx]/t,y[idx],lt/t);m['balanced_accuracy']=(m['recall']+m['tn']/(m['tn']+m['fp']))/2
   old=next(r for r in saved if r['model']==name and r['dataset']==cell and r['state']==state)
   for k,v in m.items():assert abs(v-float(old[k]))<1e-10,(name,cell,k)
   assert np.array_equal(pred,z[idx]>=lt)
   # Independent pair-rank AUROC calculation.
   _,inv,cnt=np.unique(z[idx],return_inverse=True,return_counts=True)
   rank=np.cumsum(cnt)-(cnt-1)/2;n=int(y[idx].sum());neg=len(idx)-n
   auc=(rank[inv][y[idx]==1].sum()-n*(n+1)/2)/(n*neg)
   assert abs(auc-m['roc_auc'])<1e-12
zs=np.array(zs);thresholds=[models[n]['threshold'] for n in names];points=[];boots=[]
# Reproduce all four external paired bootstrap streams and primary macro CIs.
for cell in ['fooocus-short','fooocus-long','sdxl-short','sdxl-long']:
 idx=np.array([i for i,r in enumerate(rr) if int(r['label'])==0 or r['generator']+'-'+r['prompt']==cell])
 p,b=cluster_bootstrap(y[idx],zs[:,idx],thresholds,np.array([r['group'] for r in rr])[idx]);points.append(p);boots.append(b)
 print('Verified bootstrap',cell,flush=True)
p=np.mean(points,axis=0);boot=np.mean(boots,axis=0);summary=rows(F/'external_summary.csv')
for j,family in enumerate(['resnet','v2','clip']):
 for k,metric in enumerate(['roc_auc','balanced_accuracy','fake_recall']):
  v=p[j*3:j*3+3,k];ci=interval(boot[:,j*3:j*3+3,k].mean(axis=1));r=next(r for r in summary if r['family']==family and r['metric']==metric)
  for key,val in [('mean',v.mean()),('seed_sample_sd',v.std(ddof=1)),('ci95_low',ci[0]),('ci95_high',ci[1])]:assert abs(float(r[key])-val)<1e-10
for filename in ['external_summary.csv','external_metrics.csv','external_confidence.csv','holdout_confidence.csv','paired_differences.csv','consistency.csv']:shutil.copy2(F/filename,B/'tables'/('followup_'+filename))
shutil.copy2(F/'error_cases.csv',B/'cases/cocoxgen_error_cases.csv')
colors=['#2878a5','#d98928','#5a9b72'];labels=['ResNet','Robust V2','CLIP']
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,metric,title in zip(axes,['roc_auc','balanced_accuracy','fake_recall'],['Ranking: AUROC','Fixed threshold: balanced accuracy','Fixed threshold: fake recall']):
 vals=[next(r for r in summary if r['family']==f and r['metric']==metric) for f in ['resnet','v2','clip']]
 mean=np.array([float(r['mean']) for r in vals]);lo=np.array([float(r['ci95_low']) for r in vals]);hi=np.array([float(r['ci95_high']) for r in vals])
 ax.bar(labels,mean,color=colors);ax.errorbar(range(3),mean,yerr=[mean-lo,hi-mean],fmt='none',color='#253243',capsize=4)
 ax.set_ylim(0,1);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 if metric!='fake_recall':ax.axhline(.5,color='gray',ls='--',lw=1)
 for i,v in enumerate(mean):ax.text(i,hi[i]+.025,f'{v:.3f}',ha='center',fontsize=10)
fig.suptitle('COCOXGEN: ranking improves, original thresholds still miss fake images',fontsize=13)
fig.text(.5,.01,'Equal macro over four balanced cells; mean of 3 fixed seeds; 95% COCO-group bootstrap intervals',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.94])
for ext in ['png','pdf']:fig.savefig(B/'figures'/('07_cocoxgen.'+ext),dpi=200)
plt.close(fig)
diff=rows(F/'paired_differences.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.2))
for ax,metric,title in zip(axes,['roc_auc','balanced_accuracy'],['AUROC difference','Balanced accuracy difference']):
 data=[r for r in diff if r['metric']==metric];v=np.array([float(r['estimate']) for r in data]);lo=np.array([float(r['bonferroni_three_conditions_low']) for r in data]);hi=np.array([float(r['bonferroni_three_conditions_high']) for r in data])
 ax.errorbar(v,range(3),xerr=[v-lo,hi-v],fmt='o',capsize=5,color=colors[1]);ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(3),['Clean','Resize 0.50','Blur 1.0']);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2)
 for i,x in enumerate(v):ax.annotate(f'{x:+.4f}',(x,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
 ax.margins(x=.2,y=.35)
fig.suptitle('Robust V2 minus ResNet: paired differences, mean of 3 fixed seeds')
fig.text(.5,.01,'Class-cluster bootstrap; Bonferroni-adjusted intervals across 3 conditions, separately for each metric',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.94])
for ext in ['png','pdf']:fig.savefig(B/'figures'/('08_paired_robustness.'+ext),dpi=200)
plt.close(fig)
result={'finished':True,'external_metric_rows_recomputed':90,'external_predictions_checked':9*5305,'consistency_comparisons_recomputed':36,'max_probability_difference':max(float(r['max_abs_probability_error']) for r in audit),'decision_mismatches':0,'external_bootstrap_replicates_per_cell_reproduced':2000,'external_primary_summary_and_intervals_reproduced':True,'data_audit':'Server records 92000 old images checked, zero exact-byte/decoded-RGB overlap; original images not downloaded for independent rehash locally','holdout_intervals':'Server computation; local weighted-AUROC and pairing tests passed; source class manifest remains on server'}
(B/'followup_audit.json').write_text(json.dumps(result,indent=2))
print('PASS: followup verified and figures generated')
