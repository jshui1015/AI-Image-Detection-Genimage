#!/usr/bin/env python3
"""Read-only audit of GLIDE/VQDM official-val fake images against the audited corpus."""
import csv,hashlib,io,json,re,concurrent.futures,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

ROOT=Path('/scratch/jshui/GenImage')
CACHE=ROOT/'inventory/content_hashes.csv'
OUT=ROOT/'inventory/unseen_generator_hashes.csv'
REPORT=ROOT/'inventory/unseen_generator_audit.json'
GENERATORS={
 'glide':ROOT/'glide/extracted/imagenet_glide/val/ai',
 'vqdm':ROOT/'VQDM/extracted/imagenet_ai_0419_vqdm/val/ai',
}
FIELDS=['path','generator','filename_class_index','file_sha256','pixel_sha256','thumbnail_sha256','dhash64','width','height','error']

def audit(item):
 gen,p=item; row=dict.fromkeys(FIELDS,''); row.update(path=p.relative_to(ROOT).as_posix(),generator=gen)
 m=re.fullmatch(r'(?:GLIDE|VQDM)_1000_200_\d{2}_(\d{3})_(?:glide|vqdm)_\d{5}',p.stem,re.I)
 if not m:
  row['error']='unrecognized_filename';return row
 row['filename_class_index']=m.group(1)
 try:
  raw=p.read_bytes();row['file_sha256']=hashlib.sha256(raw).hexdigest()
  with Image.open(io.BytesIO(raw)) as im:
   im.load();rgb=ImageOps.exif_transpose(im).convert('RGB');row['width']=rgb.width;row['height']=rgb.height
   row['pixel_sha256']=hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
   a=np.asarray(rgb.resize((32,32),Image.Resampling.LANCZOS))
   row['thumbnail_sha256']=hashlib.sha256(a.tobytes()).hexdigest()
   gray=np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS));bits=(gray[:,1:]>gray[:,:-1]).ravel()
   row['dhash64']=f'{int.from_bytes(np.packbits(bits).tobytes(),"big"):016x}'
 except Exception as e:row['error']=f'{type(e).__name__}: {e}'
 return row

def main():
 old=[]
 with CACHE.open() as f:
  for r in csv.DictReader(f):
   if not r.get('error') and r.get('file_sha256'):
    old.append(r)
 old_file={r['file_sha256']:r['path'] for r in old}
 old_pixel={r['pixel_sha256']:r['path'] for r in old}
 old_thumb={r['thumbnail_sha256']:r['path'] for r in old}
 old_class={}
 with (ROOT/'inventory/class_mapping_manifest.csv').open() as f:
  for r in csv.DictReader(f):old_class[r['path']]=int(r['class_index'])
 items=[(g,p) for g,d in GENERATORS.items() for p in sorted(d.glob('*')) if p.is_file()]
 t=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool: rows=list(pool.map(audit,items))
 with OUT.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 errors=[r for r in rows if r['error']]
 duplicates=[]
 for r in rows:
  if r['file_sha256'] in old_file:duplicates.append({'kind':'file_sha256','new_path':r['path'],'old_path':old_file[r['file_sha256']]})
  elif r['pixel_sha256'] in old_pixel:duplicates.append({'kind':'pixel_sha256','new_path':r['path'],'old_path':old_pixel[r['pixel_sha256']]})
  elif r['thumbnail_sha256'] in old_thumb:duplicates.append({'kind':'thumbnail_sha256','new_path':r['path'],'old_path':old_thumb[r['thumbnail_sha256']]})
 bygen={}
 for gen in GENERATORS:
  rs=[r for r in rows if r['generator']==gen]
  cls={}
  for r in rs:cls[r['filename_class_index']]=cls.get(r['filename_class_index'],0)+1
  bygen[gen]={'files':len(rs),'classes':len(cls),'per_class_histogram':{str(n):sum(v==n for v in cls.values()) for n in sorted(set(cls.values()))},'errors':sum(bool(r['error']) for r in rs)}
 # Hash overlap between GLIDE and VQDM.
 glide={r['pixel_sha256']:r['path'] for r in rows if r['generator']=='glide' and r['pixel_sha256']}
 cross=[{'glide_path':glide[r['pixel_sha256']],'vqdm_path':r['path']} for r in rows if r['generator']=='vqdm' and r['pixel_sha256'] in glide]
 # Follow the existing audit policy: within an ImageNet class, dHash distance <=3
 # followed by an RGB 32x32 thumbnail RMSE <=3. This is a candidate screen only.
 old_by_class={}
 for r in old:
  c=old_class.get(r['path'])
  if c is not None and r.get('dhash64'):
   old_by_class.setdefault(c,[]).append((int(r['dhash64'],16),r['path']))
 lut=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
 candidates=[]
 for n in rows:
  if n['error'] or not n['dhash64']:continue
  c=int(n['filename_class_index']);arr=old_by_class.get(c,[])
  if not arr:continue
  vals=np.fromiter((v for v,_ in arr),dtype=np.uint64,count=len(arr));xor=np.bitwise_xor(vals,np.uint64(int(n['dhash64'],16)))
  distances=lut[(xor[:,None] >> np.arange(0,64,8,dtype=np.uint64)) & 255].sum(axis=1)
  for j in np.flatnonzero(distances<=3):candidates.append((n['path'],arr[int(j)][1],int(distances[j])))
 def thumb(path):
  with Image.open(ROOT/path) as im:return np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
 near=[]
 thumb_cache={}
 for a,b,distance in candidates:
  if a not in thumb_cache:thumb_cache[a]=thumb(a)
  if b not in thumb_cache:thumb_cache[b]=thumb(b)
  rmse=float(np.sqrt(np.mean((thumb_cache[a]-thumb_cache[b])**2)))
  if rmse<=3.0:near.append({'new_path':a,'existing_path':b,'class_index':int(next(r['filename_class_index'] for r in rows if r['path']==a)),'dhash_distance':distance,'rgb_rmse':round(rmse,6)})
 result={'scope':'official val/ai only; no files modified','existing_audit_rows':len(old),'new_images':len(rows),'by_generator':bygen,'exact_or_thumbnail_duplicates_vs_existing':duplicates,'glide_vqdm_pixel_duplicates':cross,'same_class_near_duplicate_pairs_vs_existing':near,'near_pair_candidates_dhash_le3':len(candidates),'near_policy':'same ImageNet class, 64-bit dHash Hamming distance <=3, then 32x32 RGB RMSE <=3.0; does not exhaustively match crops/rotations.','errors':errors,'seconds':round(time.time()-t,2),'hash_policy':'SHA-256 file bytes, EXIF-corrected RGB pixels, and 32x32 RGB thumbnail.'}
 REPORT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
