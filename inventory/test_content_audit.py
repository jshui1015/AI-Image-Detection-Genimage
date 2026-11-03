import tempfile,csv,importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image,PngImagePlugin
HERE=Path(__file__).resolve().parent

def load(name):
 s=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test():
 root=Path(tempfile.mkdtemp(prefix='genimage-audit-test-'));out=root/'inventory';out.mkdir(); rng=np.random.default_rng(7);a=rng.integers(20,220,(64,64,3),dtype=np.uint8)
 Image.fromarray(a).save(root/'a.png');(root/'b.png').write_bytes((root/'a.png').read_bytes());info=PngImagePlugin.PngInfo();info.add_text('test','different encoding');Image.fromarray(a).save(root/'c.png',pnginfo=info)
 b=rng.integers(20,220,(64,64,3),dtype=np.uint8);Image.fromarray(b).save(root/'d.png');Image.fromarray(b+1).save(root/'e.png');(root/'bad.png').write_bytes(b'')
 audit=load('deduplicate');audit.ROOT=root; hashes=[];mapping=[]
 for name,label,cls,split in [('a.png',0,0,'train'),('b.png',0,0,'val'),('c.png',1,0,'train'),('d.png',0,1,'train'),('e.png',0,1,'val'),('bad.png',1,1,'train')]:
  hashes.append(audit.audit({'path':name}));mapping.append({'path':name,'label':label,'generator':'fake' if label else 'real','official_split':split,'class_index':cls,'content_class':f'n{cls:08d}','mapping_evidence':'fixture'})
 with (out/'content_hashes.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=audit.FIELDS);w.writeheader();w.writerows(hashes)
 with (out/'class_mapping_manifest.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
 summary=load('summarize_content');summary.R=root;summary.O=out;summary.main();r=json.loads((out/'dedup_report.json').read_text()); assert r['status_counts']=={'mapping_or_label_conflict':3,'keep':2,'invalid':1},r
 assert r['byte_redundant_files']==1 and r['pixel_redundant_files']==2 and r['near_conservative_pairs']==1,r
 rows=list(csv.DictReader((out/'deduplicated_manifest.csv').open()));assert len(rows)==2 and rows[0]['source_group']==rows[1]['source_group'];print('PASS: byte/pixel duplicates, cross-label quarantine, invalid images, near-source grouping')
if __name__=='__main__':test()
