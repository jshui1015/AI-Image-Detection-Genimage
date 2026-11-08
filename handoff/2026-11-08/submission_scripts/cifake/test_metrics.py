"""Local numerical checks; no server, models or dataset needed."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('cifake',Path(__file__).with_name('cifake.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rng=np.random.default_rng(17)
for size in [20,51,200]:
 for ties in [False,True]:
  z=rng.normal(size=size); z=np.round(z) if ties else z
  y=np.arange(size)%2
  met,_,_,pred=m.score_metrics(z,y,.31)
  pos=z[y==1]; neg=z[y==0]
  auc=((pos[:,None]>neg).mean()+.5*(pos[:,None]==neg).mean())
  assert abs(met['roc_auc']-auc)<1e-12
  ap=0.; previous=0
  for t in sorted(set(z),reverse=True):
   mask=z>=t; count=int(y[mask].sum())
   ap+=(count-previous)/sum(y)*count/sum(mask); previous=count
  assert abs(met['average_precision']-ap)<1e-12
  for T in [.02,.55,1.,1.2,50.]:
   after,_,_,q=m.score_metrics(z/T,y,.31/T)
   assert np.array_equal(pred,q)
   assert abs(after['roc_auc']-met['roc_auc'])<1e-12
print('PASS: AUROC pairwise reference, tied-score AP reference, and decision/ranking invariance at five temperatures.')
