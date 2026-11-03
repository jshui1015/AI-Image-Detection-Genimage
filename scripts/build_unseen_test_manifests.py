#!/usr/bin/env python3
"""Build deterministic, class-balanced final GLIDE/VQDM test manifests."""
import csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEED=20261102
AUDIT=ROOT/'inventory/unseen_generator_hashes.csv'
AUDIT_REPORT=ROOT/'inventory/unseen_generator_audit.json'
CLASS_MAP=ROOT/'inventory/class_mapping.csv'

def read_csv(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
 audit=read_csv(AUDIT); report=json.loads(AUDIT_REPORT.read_text())
 if report['errors'] or report['exact_or_thumbnail_duplicates_vs_existing'] or report['glide_vqdm_pixel_duplicates'] or report['same_class_near_duplicate_pairs_vs_existing']:
  raise SystemExit('Audit has errors or duplicate candidates; resolve them before building final test manifests.')
 mapping={int(r['class_index']):r for r in read_csv(CLASS_MAP)}
 output_summary={}
 fields=['image_id','path','label','generator','content_class','class_index','source_group','split','official_split','mapping_evidence']
 for generator in ('glide','vqdm'):
  split=f'test_{generator}_unseen'
  candidates=defaultdict(list)
  for r in audit:
   if r['generator']==generator:
    c=int(r['filename_class_index']);candidates[c].append(r)
  if set(candidates)!=set(range(1000)) or any(len(v)!=6 for v in candidates.values()):
   raise SystemExit(f'{generator}: expected 1000 classes with 6 images each')
  reals=read_csv(ROOT/f'manifests/reserved_real_{generator}.csv')
  real_by_class=defaultdict(list)
  for r in reals:real_by_class[int(r['class_index'])].append(r)
  if set(real_by_class)!=set(range(1000)) or any(len(v)!=2 for v in real_by_class.values()):
   raise SystemExit(f'{generator}: expected 2 reserved real images per class')
  selected=[]
  for c in range(1000):
   ranked=sorted(candidates[c],key=lambda r:hashlib.sha256(f'{SEED}:{generator}:{c}:{r["file_sha256"]}'.encode()).hexdigest())
   selected.extend(ranked[:2])
  rows=[]
  for r in selected:
   c=int(r['filename_class_index']);m=mapping[c];rows.append({'image_id':r['pixel_sha256'],'path':r['path'],'label':1,'generator':generator,'content_class':m['synset'],'class_index':c,'source_group':'src_'+r['pixel_sha256'][:56],'split':split,'official_split':'val','mapping_evidence':'GenImage_val_ai_filename_component_4_zero_based_1000_class_coverage'})
  for c in range(1000):
   for r in real_by_class[c]:
    d=dict(r);d['split']=split;rows.append(d)
  rows.sort(key=lambda r:(int(r['class_index']),int(r['label']),r['path']))
  counts=Counter((int(r['class_index']),int(r['label'])) for r in rows)
  assert len(rows)==4000 and all(counts[(c,0)]==2 and counts[(c,1)]==2 for c in range(1000))
  out=ROOT/f'manifests/test_{generator}_unseen.csv'
  with out.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
  output_summary[generator]={'manifest':out.relative_to(ROOT).as_posix(),'files':len(rows),'real':sum(int(r['label'])==0 for r in rows),'fake':sum(int(r['label'])==1 for r in rows),'classes':len(set(int(r['class_index']) for r in rows)),'per_class_real':2,'per_class_fake':2,'fake_source':'GenImage official val/ai; 2 of 6 selected per class by stable SHA-256 ordering','real_source':f'manifests/reserved_real_{generator}.csv','selection_seed':SEED}
 result={'seed':SEED,'label_definition':{'0':'real','1':'AI-generated'},'sets':output_summary,'threshold_or_model_selection':'none; final test only'}
 (ROOT/'manifests/unseen_test_config.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
