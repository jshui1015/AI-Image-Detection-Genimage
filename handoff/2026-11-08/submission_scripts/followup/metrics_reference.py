"""CIFAKE metric implementation reused unchanged for comparability."""
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
