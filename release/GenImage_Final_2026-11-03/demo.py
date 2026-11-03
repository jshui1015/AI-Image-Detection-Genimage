#!/usr/bin/env python3
"""Streamlit demo for the locked cap128+JPEG95 ResNet-50 checkpoint."""
import csv,io,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import streamlit as st
import torch
from train import ROOT,ResNet50

CHECKPOINT=ROOT/'outputs/resnet50-cap128-jpeg95/best.pt'
REPORT=ROOT/'outputs/resnet50-cap128-jpeg95/validation_report.json'
SAMPLES=ROOT/'demo_examples/samples.csv'

@st.cache_resource
def load_detector():
 if not CHECKPOINT.is_file() or not REPORT.is_file():raise FileNotFoundError('ResNet-50 checkpoint or validation report is missing.')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
 report=json.loads(REPORT.read_text())
 control=ckpt.get('input_control','raw')
 model=ResNet50();model.load_state_dict(ckpt['state_dict']);model.to(device).eval()
 return model,device,float(report['validation_threshold_max_f1']),control

def prepare(image,control):
 im=ImageOps.exif_transpose(image).convert('RGB')
 if control in ('cap128','cap128_jpeg95'):
  w,h=im.size;scale=min(1.,128/max(w,h))
  if scale<1:im=im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS)
 if control in ('jpeg95','cap128_jpeg95'):
  buf=io.BytesIO();im.save(buf,format='JPEG',quality=95);buf.seek(0)
  with Image.open(buf) as encoded:im=encoded.convert('RGB')
 w,h=im.size;scale=256/min(w,h);nw,nh=round(w*scale),round(h*scale)
 im=im.resize((nw,nh),Image.Resampling.BILINEAR);left=(nw-224)//2;top=(nh-224)//2;im=im.crop((left,top,left+224,top+224))
 x=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()).permute(2,0,1).div_(255)
 mean=x.new_tensor([.485,.456,.406])[:,None,None];std=x.new_tensor([.229,.224,.225])[:,None,None]
 return (x-mean)/std

def main():
 st.set_page_config(page_title='GenImage 图像检测演示',page_icon='🔎',layout='centered')
 st.title('AI 生成图像检测')
 st.write('上传图片或选择内置样例，使用 ResNet-50 基线模型查看 AI 图像分数。')
 try:model,device,threshold,control=load_detector()
 except Exception as e:
  st.error(f'模型加载失败：{e}');st.stop()
 st.caption(f'模型：ImageNet 预训练 ResNet-50 · 设备：{device} · 判定阈值：{threshold:.4f}')
 mode=st.radio('输入方式',['上传图片','内置样例'],horizontal=True)
 image=None;expected=None;sample=None
 if mode=='上传图片':
  uploaded=st.file_uploader('选择图片',type=['jpg','jpeg','png','webp'])
  if uploaded is not None:
   try:
    with Image.open(uploaded) as source:image=ImageOps.exif_transpose(source).convert('RGB').copy()
   except Exception as e:st.error(f'图片读取失败：{e}')
 else:
  try:
   with SAMPLES.open(newline='') as f:samples=list(csv.DictReader(f))
   labels=[f"{r['sample_id']}（{r['split']}）" for r in samples]
   selected=st.selectbox('选择内置图片',range(len(samples)),format_func=lambda i:labels[i])
   sample=samples[selected]
   with Image.open(ROOT/'demo_examples'/sample['image_file']) as source:image=ImageOps.exif_transpose(source).convert('RGB').copy()
   expected=sample
  except Exception as e:st.error(f'内置样例读取失败：{e}')
 if image is not None:
  try:
   st.image(image,caption='输入图片',use_container_width=True)
   x=prepare(image,control).unsqueeze(0).to(device)
   with torch.inference_mode():
    with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):score=float(model(x).sigmoid().item())
   st.metric('AI 生成分数',f'{score:.1%}',help='模型输出分数；未经概率校准，不应视作真实概率。')
   if score>=threshold:st.warning('模型判断：更像 AI 生成图像')
   else:st.success('模型判断：更像真实图像')
   if expected is not None:
    st.info(f"样例标签：{expected['true_label']} · 来源：{expected['generator']} · split：{expected['split']}")
    if expected['generator'].endswith('漏检'):
     st.caption('这是从最终测试中挑出的失败样例，用于演示模型局限；不能用来重新调整模型或阈值。')
  except Exception as e:st.error(f'图片处理失败：{e}')
 st.caption('此演示使用 GenImage 子集训练。结果是模型判断，不是图像来源证明；JPEG 重编码和缩放可能影响结果。')

if __name__=='__main__':main()
