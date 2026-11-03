#!/usr/bin/env python3
"""ResNet-50 binary AI-image detector using the fixed GenImage manifests."""
import argparse,csv,json,math,random,shutil,time,urllib.request,io
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageOps,ImageFilter
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parent
WEIGHTS_URL='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'

class Images(Dataset):
 def __init__(self,manifest,split,train=False,limit=0,input_control='raw',robust_augment=False):
  self.rows=[r for r in csv.DictReader(Path(manifest).open()) if r['split']==split]
  if limit:self.rows=self.rows[:limit]
  self.train=train;self.input_control=input_control;self.robust_augment=robust_augment
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(ROOT/r['path']) as im:im=ImageOps.exif_transpose(im).convert('RGB')
  if self.input_control in ('cap128','cap128_jpeg95'):
   w,h=im.size;scale=min(1.0,128/max(w,h))
   if scale<1:im=im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS)
  if self.input_control in ('jpeg95','cap128_jpeg95'):
   buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=95);buffer.seek(0)
   with Image.open(buffer) as encoded:im=encoded.convert('RGB')
  if self.train and self.robust_augment:
   # Apply the same stochastic degradations to both labels to reduce reliance
   # on image resolution and compression traces. Validation remains clean.
   if random.random()<.8:
    w,h=im.size;factor=random.uniform(.5,1.0);small=(max(1,round(w*factor)),max(1,round(h*factor)))
    if small!=(w,h):im=im.resize(small,Image.Resampling.BILINEAR).resize((w,h),Image.Resampling.BILINEAR)
   if random.random()<.8:
    buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=random.randint(55,100));buffer.seek(0)
    with Image.open(buffer) as encoded:im=encoded.convert('RGB')
   if random.random()<.25:im=im.filter(ImageFilter.GaussianBlur(radius=random.uniform(.1,.7)))
  if self.train:
   w,h=im.size;scale=random.uniform(.8,1.0);ratio=math.exp(random.uniform(math.log(.75),math.log(4/3)))
   cw=min(w,int(math.sqrt(w*h*scale*scale/ratio)));ch=min(h,int(cw*ratio))
   if ch>h:ch=h;cw=min(w,int(ch/ratio))
   left=random.randint(0,w-cw);top=random.randint(0,h-ch);im=im.crop((left,top,left+cw,top+ch)).resize((224,224),Image.Resampling.BILINEAR)
   if random.random()<.5:im=ImageOps.mirror(im)
  else:
   w,h=im.size;scale=256/min(w,h);nw,nh=round(w*scale),round(h*scale)
   im=im.resize((nw,nh),Image.Resampling.BILINEAR);left=(nw-224)//2;top=(nh-224)//2;im=im.crop((left,top,left+224,top+224))
  x=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()).permute(2,0,1).div_(255)
  mean=x.new_tensor([.485,.456,.406])[:,None,None];std=x.new_tensor([.229,.224,.225])[:,None,None]
  return (x-mean)/std,torch.tensor(float(r['label'])),r['generator'],r['path']

class Bottleneck(nn.Module):
 expansion=4
 def __init__(self,inp,planes,stride=1,downsample=None):
  super().__init__();self.conv1=nn.Conv2d(inp,planes,1,bias=False);self.bn1=nn.BatchNorm2d(planes)
  self.conv2=nn.Conv2d(planes,planes,3,stride,1,bias=False);self.bn2=nn.BatchNorm2d(planes)
  self.conv3=nn.Conv2d(planes,planes*4,1,bias=False);self.bn3=nn.BatchNorm2d(planes*4)
  self.relu=nn.ReLU(inplace=True);self.downsample=downsample
 def forward(self,x):
  identity=x;out=self.relu(self.bn1(self.conv1(x)));out=self.relu(self.bn2(self.conv2(out)));out=self.bn3(self.conv3(out))
  if self.downsample is not None:identity=self.downsample(x)
  return self.relu(out+identity)

class ResNet50(nn.Module):
 def __init__(self):
  super().__init__();self.inplanes=64;self.conv1=nn.Conv2d(3,64,7,2,3,bias=False);self.bn1=nn.BatchNorm2d(64);self.relu=nn.ReLU(inplace=True);self.maxpool=nn.MaxPool2d(3,2,1)
  self.layer1=self._layer(64,3);self.layer2=self._layer(128,4,2);self.layer3=self._layer(256,6,2);self.layer4=self._layer(512,3,2);self.avgpool=nn.AdaptiveAvgPool2d((1,1));self.fc=nn.Linear(2048,1)
  for m in self.modules():
   if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
 def _layer(self,planes,n,stride=1):
  down=None
  if stride!=1 or self.inplanes!=planes*4:down=nn.Sequential(nn.Conv2d(self.inplanes,planes*4,1,stride,bias=False),nn.BatchNorm2d(planes*4))
  layers=[Bottleneck(self.inplanes,planes,stride,down)];self.inplanes=planes*4
  layers.extend(Bottleneck(self.inplanes,planes) for _ in range(n-1));return nn.Sequential(*layers)
 def forward(self,x):
  x=self.maxpool(self.relu(self.bn1(self.conv1(x))));x=self.layer1(x);x=self.layer2(x);x=self.layer3(x);x=self.layer4(x);return self.fc(torch.flatten(self.avgpool(x),1)).squeeze(1)

def load_weights(model,path,download=True):
 p=Path(path) if path else Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'
 if not p.exists() and download:
  p.parent.mkdir(parents=True,exist_ok=True);print('Downloading official ImageNet ResNet-50 weights:',WEIGHTS_URL,flush=True)
  # The HPC login shell may export a dead localhost proxy; use a direct
  # connection for this public checkpoint, matching the successful wheel install.
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));partial=p.with_suffix(p.suffix+'.download')
  try:
   with opener.open(WEIGHTS_URL,timeout=60) as response,partial.open('wb') as target:shutil.copyfileobj(response,target)
   partial.replace(p)
  finally:partial.unlink(missing_ok=True)
 if p.exists():
  state=torch.load(p,map_location='cpu',weights_only=True);state={k:v for k,v in state.items() if not k.startswith('fc.')};missing,unexpected=model.load_state_dict(state,strict=False)
  assert set(missing)=={'fc.weight','fc.bias'} and not unexpected,(missing,unexpected)
  print('Loaded ImageNet pretrained backbone:',p,flush=True);return True
 print('WARNING: no pretrained weights found; using random initialization.',flush=True);return False

def metrics(ys,ps,threshold=.5):
 y=np.asarray(ys,dtype=np.int64);p=np.asarray(ps,dtype=np.float64);pred=p>=threshold;tp=int(((y==1)&pred).sum());tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
 order=np.argsort(p,kind='mergesort');pos=int(y.sum());neg=len(y)-pos;rank_sum=0.0;i=0
 while i<len(order):
  j=i+1
  while j<len(order) and p[order[j]]==p[order[i]]:j+=1
  average_rank=((i+1)+j)/2;rank_sum+=average_rank*int(y[order[i:j]].sum());i=j
 auc=(rank_sum-pos*(pos+1)/2)/(pos*neg) if pos and neg else float('nan')
 precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);f1=2*precision*recall/max(1e-12,precision+recall)
 order_ap=np.argsort(-p,kind='mergesort');total_pos=int(y.sum());ap=0.0;cum_pos=0;cum_count=0;i=0
 while i<len(order_ap):
  j=i+1
  while j<len(order_ap) and p[order_ap[j]]==p[order_ap[i]]:j+=1
  group_pos=int(y[order_ap[i:j]].sum());cum_pos+=group_pos;cum_count+=j-i
  if total_pos:ap+=(group_pos/total_pos)*(cum_pos/cum_count)
  i=j
 average_precision=ap if total_pos else float('nan')
 return {'accuracy':(tp+tn)/max(1,len(y)),'precision':precision,'recall':recall,'f1':f1,'roc_auc':auc,'average_precision':average_precision,'tn':tn,'fp':fp,'fn':fn,'tp':tp}

def evaluate(model,loader,device,amp):
 model.eval();ys=[];ps=[];generators=[]
 with torch.inference_mode():
  for x,y,g,_ in loader:
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp and device.type=='cuda'):z=model(x.to(device,non_blocking=True))
   ys.extend(y.int().tolist());ps.extend(z.float().sigmoid().cpu().tolist());generators.extend(g)
 return ys,ps,generators

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',default=str(ROOT/'manifests/all_splits.csv'));ap.add_argument('--seed',type=int,default=20261031);ap.add_argument('--output-dir',default=str(ROOT/'outputs/resnet50'));ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--patience',type=int,default=3);ap.add_argument('--pretrained',action=argparse.BooleanOptionalAction,default=True);ap.add_argument('--weights',default='');ap.add_argument('--input-control',choices=['raw','jpeg95','cap128','cap128_jpeg95'],default='raw');ap.add_argument('--robust-augment',action='store_true',help='Apply stochastic low-resolution, JPEG, and mild blur augmentations to training images only.');ap.add_argument('--smoke-test',action='store_true');ap.add_argument('--max-steps',type=int,default=2);ap.add_argument('--device',default='auto');a=ap.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError(f'CUDA was requested but unavailable (PyTorch CUDA build: {torch.version.cuda}). Check the GPU allocation, driver/runtime compatibility, and CUDA_VISIBLE_DEVICES.')
 if not a.smoke_test and a.device=='auto' and device.type=='cpu':raise RuntimeError(f'No GPU is visible (PyTorch CUDA build: {torch.version.cuda}). Check driver/runtime compatibility in the GPU allocation; use a CUDA 12.x PyTorch build for R570 unless the cluster provides CUDA 13 forward compatibility.')
 manifest=Path(a.manifest);assert manifest.is_file(),manifest
 tr=Images(manifest,'train',True,32 if a.smoke_test else 0,a.input_control,a.robust_augment);va=Images(manifest,'val',False,32 if a.smoke_test else 0,a.input_control,False)
 assert len(tr)>0 and len(va)>0
 kwargs={'num_workers':0 if a.smoke_test else a.workers,'pin_memory':device.type=='cuda'}
 tl=DataLoader(tr,batch_size=2 if a.smoke_test else a.batch_size,shuffle=True,**kwargs);vl=DataLoader(va,batch_size=2 if a.smoke_test else a.batch_size,shuffle=False,**kwargs)
 model=ResNet50();pretrained=load_weights(model,a.weights,download=a.pretrained and not a.smoke_test);model=model.to(device)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay);lossfn=nn.BCEWithLogitsLoss();amp=device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=amp)
 best=-1.;bad_epochs=0;best_path=out/'best.pt';history=[];epochs=1 if a.smoke_test else a.epochs
 for epoch in range(epochs):
  model.train();losses=[];start=time.time()
  for step,(x,y,_,_) in enumerate(tl):
   x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):z=model(x);loss=lossfn(z,y)
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
   if a.smoke_test and step+1>=a.max_steps:break
  ys,ps,_=evaluate(model,vl,device,amp);vm=metrics(ys,ps);record={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_at_0_5':vm,'seconds':round(time.time()-start,2)};history.append(record);print(json.dumps(record),flush=True)
  if vm['roc_auc']>best:
   best=vm['roc_auc'];bad_epochs=0;torch.save({'model':'resnet50','state_dict':model.state_dict(),'seed':a.seed,'pretrained':pretrained,'input_control':a.input_control,'robust_augment':a.robust_augment,'epoch':epoch+1,'val_roc_auc':best},best_path)
  else:bad_epochs+=1
  if a.smoke_test:break
  if bad_epochs>=a.patience:
   print(f'Early stopping after {bad_epochs} consecutive epochs without validation AUROC improvement.',flush=True);break
 ckpt=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(ckpt['state_dict']);ys,ps,gens=evaluate(model,vl,device,amp)
 candidates=sorted(set(ps));threshold=max(candidates,key=lambda t:(metrics(ys,ps,t)['f1'],-abs(t-.5)));report={'seed':a.seed,'device':str(device),'pretrained':pretrained,'input_control':a.input_control,'robust_augment':a.robust_augment,'best_epoch':ckpt['epoch'],'best_val_roc_auc':ckpt['val_roc_auc'],'validation_threshold_max_f1':threshold,'validation_at_selected_threshold':metrics(ys,ps,threshold),'validation_at_0_5':metrics(ys,ps),'per_generator_validation_at_0_5':{g:metrics([y for y,gg in zip(ys,gens) if gg==g],[p for p,gg in zip(ps,gens) if gg==g]) for g in sorted(set(gens))},'history':history,'smoke_test':a.smoke_test}
 (out/'validation_report.json').write_text(json.dumps(report,indent=2));print('BEST_CHECKPOINT',best_path,flush=True);print('VALIDATION_REPORT',out/'validation_report.json',flush=True)
if __name__=='__main__':main()
