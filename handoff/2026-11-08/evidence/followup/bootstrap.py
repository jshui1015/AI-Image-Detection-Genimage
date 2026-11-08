"""Paired cluster bootstrap; seed models are fixed, never pooled as independent images."""
import numpy as np

class WeightedMetrics:
 def __init__(self,y,z,threshold=.5):
  self.y=np.asarray(y,dtype=int); self.z=np.asarray(z,dtype=float)
  assert set(self.y)=={0,1} and np.isfinite(self.z).all()
  self.order=np.argsort(self.z,kind='stable')
  self.starts=np.r_[0,np.flatnonzero(np.diff(self.z[self.order]))+1]
  self.pred=self.z>=np.log(threshold/(1-threshold))
 def __call__(self,w):
  yp=self.y[self.order]; ws=w[self.order]
  p=np.add.reduceat(ws*yp,self.starts); n=np.add.reduceat(ws*(1-yp),self.starts)
  auc=np.sum(p*(np.cumsum(n)-n+.5*n))/(p.sum()*n.sum())
  recall=np.sum(w*self.y*self.pred)/np.sum(w*self.y)
  specificity=np.sum(w*(1-self.y)*(~self.pred))/np.sum(w*(1-self.y))
  return np.array([auc,(recall+specificity)/2,recall])

def cluster_bootstrap(y,zs,thresholds,groups,reps=2000,seed=20261108):
 # Groups resampled once per replicate; identical weights for every seed/model.
 # Includes all seeds as fixed fitted models, reports each and their mean.
 groups=np.asarray(groups); unique,inv=np.unique(groups,return_inverse=True)
 assert len(unique)>1
 metrics=[WeightedMetrics(y,z,t) for z,t in zip(zs,thresholds)]
 point=np.array([m(np.ones(len(y))) for m in metrics]); draws=[]
 rng=np.random.default_rng(seed)
 for _ in range(reps):
  weights=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inv]
  if np.sum(weights*y)==0 or np.sum(weights*(1-y))==0: raise ValueError('Single-class bootstrap replicate')
  draws.append([m(weights) for m in metrics])
 return point,np.asarray(draws)

def interval(draws):return np.quantile(draws,[.025,.975],axis=0)
