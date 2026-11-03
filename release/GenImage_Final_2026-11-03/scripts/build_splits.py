"""Build fixed, class-balanced GenImage subsets without copying image data."""
import argparse,collections,csv,hashlib,heapq,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; INV=ROOT/'inventory'
GENERATORS=['BigGAN','stable_diffusion_v_1_4','ADM']
FIELDS=['image_id','path','label','generator','content_class','class_index','source_group','split','official_split','mapping_evidence']

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def build(seed,out):
 # Trace official boundaries using all exact-duplicate and near-duplicate members,
 # including members excluded from the canonical candidate manifest.
 repeated=set()
 for r in csv.DictReader((INV/'exact_duplicate_members.csv').open()):repeated.add('src_'+r['hash'])
 for r in csv.DictReader((INV/'near_duplicate_candidates.csv').open()):
  repeated.update(['src_'+r['pixel_sha_a'],'src_'+r['pixel_sha_b']])
 masks=collections.defaultdict(int)
 for r in csv.DictReader((INV/'content_manifest.csv').open()):
  if r['source_group'] in repeated:masks[r['source_group']]|={'train':1,'val':2}[r['official_split']]
 forbidden={s for s,m in masks.items() if m==3}
 pools=collections.defaultdict(list);available=collections.Counter();excluded=0
 for r in csv.DictReader((INV/'deduplicated_manifest.csv').open()):
  assert r['status']=='keep'
  if r['source_group'] in forbidden:excluded+=1;continue
  kind=r['generator'] if r['label']=='1' else 'real';c=int(r['class_index']);key=(r['official_split'],kind,c)
  score=int(hashlib.sha256(f"{seed}\0{r['source_group']}".encode()).hexdigest(),16)
  cap=(40 if key[0]=='train' else 16) if kind=='real' else (16 if key[0]=='train' else 8)
  item=(-score,r['path'],r);heap=pools[key];available[key]+=1
  if len(heap)<cap:heapq.heappush(heap,item)
  elif item>heap[0]:heapq.heapreplace(heap,item)
 print('candidate pools ready',len(pools),'excluded cross-boundary candidates',excluded,flush=True)
 ordered={k:[x[2] for x in sorted(v,key=lambda x:(-x[0],x[1]))] for k,v in pools.items()}
 used=set();splits=collections.defaultdict(list)
 def take(split,official,gen,c,n):
  candidates=ordered.get((official,gen,c),[]);chosen=[]
  for r in candidates:
   if r['source_group'] not in used:
    used.add(r['source_group']);chosen.append({**{k:r[k] for k in FIELDS if k!='split'},'split':split})
    if len(chosen)==n:break
  if len(chosen)!=n:raise RuntimeError(f'Insufficient independent images: {split}, {gen}, class {c}: {len(chosen)}/{n}')
  splits[split].extend(chosen)
 for c in range(1000):
  for split,official,n in [('train','train',20),('val','train',4),('test_seen','val',5),('reserved_real_glide','val',2),('reserved_real_vqdm','val',2)]:take(split,official,'real',c,n)
  for split,official,perclass,offset in [('train','train',20,0),('val','train',4,2),('test_seen','val',5,0)]:
   allocation=collections.Counter(GENERATORS[(c*perclass+j+offset)%3] for j in range(perclass))
   for g in GENERATORS:take(split,official,g,c,allocation[g])
 expected={'train':(20000,[6667,6667,6666]),'val':(4000,[1333,1333,1334]),'test_seen':(5000,[1667,1667,1666]),'reserved_real_glide':(2000,[0,0,0]),'reserved_real_vqdm':(2000,[0,0,0])}
 summary={}
 for name,rows in splits.items():
  counts=collections.Counter(r['generator'] for r in rows); real,gs=expected[name]
  assert counts['real']==real and [counts[g] for g in GENERATORS]==gs
  assert len({r['source_group'] for r in rows})==len(rows)
  labels=collections.Counter((r['label'],int(r['class_index'])) for r in rows)
  for c in range(1000):
   assert labels['0',c]==real//1000
   assert labels['1',c]==sum(gs)//1000
  summary[name]={'files':len(rows),'real':real,'fake':sum(gs),'generators':dict(counts),'real_classes':1000,'fake_classes':1000 if sum(gs) else 0,'real_per_class':real//1000,'fake_per_class':sum(gs)//1000,'status':'reserved_real_only_missing_fake_generator' if name.startswith('reserved') else 'ready'}
 assert len(used)==62000 and sum(map(len,splits.values()))==62000
 out.mkdir(parents=True,exist_ok=False)
 allrows=[];digests={}
 for name,rows in splits.items():
  # Shuffle export order separately so CSV order does not expose label/class blocks.
  rows.sort(key=lambda r:hashlib.sha256(f"{seed}\0{name}\0{r['path']}".encode()).digest());allrows.extend(rows)
  with (out/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
  digests[name+'.csv']=sha(out/(name+'.csv'))
 with (out/'all_splits.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(allrows)
 with (out/'class_distribution.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['split','label','generator','class_index','images'])
  for name,rows in splits.items():
   for key,n in sorted(collections.Counter((r['label'],r['generator'],int(r['class_index'])) for r in rows).items()):w.writerow([name,*key,n])
 cfg={'seed':seed,'dataset_root':str(ROOT),'label_definition':{'0':'real','1':'AI-generated'},'generators':GENERATORS,'summary':summary,'total_selected':62000,'experimental_train_val_test':58000,'reserved_real':4000,'cross_official_source_groups_excluded':len(forbidden),'cross_official_candidate_files_excluded':excluded,'source_group_policy':'At most one selected image per source group across all five manifests. Exclude any known group spanning official train/val.','official_boundary_policy':'Train/validation use official train; seen test and reserved reals use official val.','quota_policy':'Exactly equal real/fake class counts within each ready split; generator counts differ by at most one per class.','unseen_test_policy':'GLIDE/VQDM absent; files are reserved real halves only, not completed unseen tests. Never use them for model/threshold/calibration selection.','source_manifest_sha256':sha(INV/'deduplicated_manifest.csv'),'output_sha256':digests,'known_limitations':json.loads((INV/'dedup_report.json').read_text())['limitations']}
 (out/'split_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
 lines=['# 固定实验划分','',f'随机种子：{seed}。路径相对于数据根目录 {ROOT}；label=0为真图，1为生成图。原始图像未移动或复制。','', '| 清单 | 真图 | 假图 | 总量 |','|---|---:|---:|---:|']
 for name,s in summary.items():lines.append(f"| {name}.csv | {s['real']:,} | {s['fake']:,} | {s['files']:,} |")
 lines+=['','三个完整实验集合共58,000张，另预留4,000张独立真图，共62,000张。补入GLIDE与VQDM各2,000张独立假图后，总计66,000张。两个reserved文件目前不构成完整未见测试集。','', '训练/验证取自官方train，测试及预留真图取自官方val。每个完整集合覆盖1000类，真假每类数量分别为训练20/20、验证4/4、已见测试5/5；每个预留真图集每类2张。每类生成器配额相差不超过1。','', '已知跨官方train/val的同源组全部排除。五个清单之间及内部不重复使用image_id、path、source_group；每来源组最多一张。此保证限于当前审计已识别的同源关系，不代表穷尽裁剪与旋转等未知同源版本。','', '所有模型和增强实验复用这些清单，不重新抽样。GLIDE/VQDM及预留真图只供最终未见测试，不用于模型选择、阈值或概率校准。','', '复现命令（输出目录必须尚不存在）：','```bash',f'python scripts/build_splits.py --seed {seed} --output /tmp/genimage-splits-reproduction','```','', '配额、来源清单哈希和输出哈希见split_config.json；逐类分布见class_distribution.csv。']
 (out/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(cfg,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261031);p.add_argument('--output',type=Path,default=ROOT/'manifests');a=p.parse_args();build(a.seed,a.output)
