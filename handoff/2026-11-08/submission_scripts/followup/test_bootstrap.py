import numpy as np
from bootstrap import WeightedMetrics,cluster_bootstrap,interval
# Independent pairwise AUROC oracle exercises ties, zero multiplicities, weighting.
rng=np.random.default_rng(14)
for _ in range(100):
 y=np.array([0]*8+[1]*8);z=rng.integers(-2,3,len(y)).astype(float);w=rng.integers(0,4,len(y)).astype(float)
 if not (sum(w[y==0]) and sum(w[y==1])):continue
 expected=sum(w[i]*w[j]*(float(z[i]>z[j])+.5*float(z[i]==z[j])) for i in range(8,16) for j in range(8))/(w[:8].sum()*w[8:].sum())
 got=WeightedMetrics(y,z)(w)[0];assert abs(got-expected)<1e-12
# Perfect paired predictors have zero difference in every cluster draw.
y=np.tile([0,1],12);groups=np.repeat(np.arange(12),2);z=np.where(y,2.,-2.)
p,b=cluster_bootstrap(y,[z,z],[.5,.5],groups,reps=100)
assert np.array_equal(b[:,0],b[:,1]) and np.all(b==1)
assert np.array_equal(p[0],p[1])
assert np.all(interval(b[:,0]-b[:,1])==0)
print('PASS: weighted tied AUROC, clustered pairing, fixed-seed reproducibility')
