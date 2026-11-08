"""Optional CPU-only export of 18 indexed GenImage cases. No inference."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
source=Path('/project/mscaiehpc42026/AI-Image-Detection/data').resolve()
index=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=False)
with index.open() as f: rows=list(csv.DictReader(f))
selected=[r for r in rows if r['model'].endswith('20261031') and r['rank']=='1']
records=[]
for i,r in enumerate(selected):
 p=(source/r['path']).resolve(); assert p.is_relative_to(source) and p.is_file()
 target=out/f'{i+1:02d}{p.suffix.lower()}'; shutil.copy2(p,target)
 records.append({**r,'export_file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
(out/'case_index.json').write_text(json.dumps(records,indent=2))
print('Exported',len(records),'images; no GPU required.')
