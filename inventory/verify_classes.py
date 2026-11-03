import csv,json,hashlib,collections
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
R=Path('/scratch/jshui/GenImage'); O=R/'inventory'; M=O/'metadata'
a=json.loads((M/'imagenet_class_index.json').read_text()); b={int(x.split(':')[0]):x.split(':')[1].strip() for x in (M/'index_synset.yaml').read_text().splitlines() if ':' in x}; val=(M/'imagenet2012_validation_labels.txt').read_text().splitlines(); assert len(a)==len(b)==1000; canonical=(M/'imagenet_synsets.txt').read_text().splitlines(); assert len(canonical)==1000 and all(a[str(i)][0]==canonical[i] for i in range(1000)); assert len(val)==50000 and set(val)<=set(canonical)
reverse={v[0]:int(k) for k,v in a.items()}; counts=collections.defaultdict(collections.Counter); unknown=[]; selected={}; targets=[0,207,281,417,817,999]
fields=['path','label','generator','official_split','class_index','content_class','class_name','mapping_evidence']
with (O/'class_mapping_manifest.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row in csv.DictReader((O/'files.csv').open()):
  name=Path(row['path']).name;label=int(row['kind']=='ai');idx=None;evidence=''
  try:
   if label:idx=int(name.split('_')[0]);evidence='filename_prefix_0based_inferred'
   elif row['official_split']=='train':idx=reverse[name.split('_')[0]];evidence='filename_synset_verified'
   else:idx=reverse[val[int(name.rsplit('_',1)[1].split('.')[0])-1]];evidence='TFDS_validation_labels_verified'
   syn,desc=a[str(idx)]
  except (KeyError,ValueError,IndexError):unknown.append(row['path']);idx='';syn='';desc='';evidence='unresolved'
  w.writerow(dict(path=row['path'],label=label,generator=row['generator'] if label else 'real',official_split=row['official_split'],class_index=idx,content_class=syn,class_name=desc,mapping_evidence=evidence))
  counts[(row['generator'],row['official_split'],row['kind'])][idx]+=1
  if label and idx in targets and row['official_split']=='train' and int(row['bytes'])>0:selected.setdefault((row['generator'],idx),row['path'])
with (O/'class_mapping.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['class_index','synset','class_name'])
 for i in range(1000):w.writerow([i,*a[str(i)]])
with (O/'verified_class_counts.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['dataset','official_split','kind','class_index','files'])
 for key,cc in counts.items():
  for i,n in sorted(cc.items(),key=lambda x:str(x[0])):w.writerow([*key,i,n])
canvas=Image.new('RGB',(6*180,3*210),'white');draw=ImageDraw.Draw(canvas);gens=['ADM','BigGAN','stable_diffusion_v_1_4']
for y,g in enumerate(gens):
 for x,i in enumerate(targets):
  p=selected[(g,i)]
  with Image.open(R/p) as im:canvas.paste(ImageOps.contain(im.convert('RGB'),(176,176)),(x*180,y*210))
  draw.text((x*180,y*210+177),f'{g[:8]} {i}: {a[str(i)][1]}',fill='black')
canvas.save(O/'class_mapping_spotcheck.jpg')
result={'mapping_records':sum(sum(x.values()) for x in counts.values()),'unresolved':unknown,'official_repo_mapping_comparison':{'classes':1000,'canonical_timm_mismatches':0,'sd_yaml_mismatches':sum(a[str(i)][0]!=b[i] for i in b),'sd_yaml_rejected':True},'validation_labels':{'source':'https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/imagenet2012/validation_labels.txt','rows':len(val)},'fake_mapping_status':'0-based filename prefix inferred from range and official generator class indexing; export naming script unavailable; visual spot check required','groups':[{'dataset':key[0],'split':key[1],'kind':key[2],'classes':len(cc),'missing_indices':sorted(set(range(1000))-set(cc)),'min_count':min(cc.values()),'max_count':max(cc.values())} for key,cc in counts.items()]}
(M/'sources.json').write_text(json.dumps({'genimage_class_index':'https://raw.githubusercontent.com/GenImage-Dataset/GenImage/main/generator_codes/VQ-Diffusion-main/image_synthesis/data/imagenet_class_index.json','genimage_sd_synsets':'https://raw.githubusercontent.com/GenImage-Dataset/GenImage/main/generator_codes/stable-diffusion-main/data/index_synset.yaml','validation_labels':result['validation_labels']['source'],'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in M.iterdir() if p.is_file()}},indent=2))
(O/'class_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
