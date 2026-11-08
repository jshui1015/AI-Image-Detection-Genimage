"""One allocation, independent cache audit, original experiment implementations."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
SHARED = Path('/project/mscaiehpc42026/AI-Image-Detection/data')
HERE = Path(__file__).resolve().parent
SPLITS = {'train':40000, 'val':8000, 'test_seen':10000,
          'test_glide_unseen':4000, 'test_vqdm_unseen':4000}

def read_json(p):
    return json.loads(Path(p).read_text())

def write_json(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def caches():
    import torch
    import clip_detector as clip
    result = {}
    for split, count in SPLITS.items():
        manifest = ROOT/'manifests'/clip.SPLITS[split][0]
        with manifest.open(newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['split']==split]
        p = SHARED/'outputs/clip-vit-b32-cap128-jpeg95/features'/f'{split}.pt'
        d = torch.load(p, map_location='cpu', weights_only=False)
        assert len(rows)==count==d['count'], (split, 'count')
        assert d['split']==split and d['model_id']==clip.DEFAULT_MODEL
        assert d['input_control']=='cap128_jpeg95'
        assert list(d['paths'])==[r['path'] for r in rows], (split,'path/order')
        assert d['labels'].tolist()==[float(r['label']) for r in rows]
        assert d['classes'].tolist()==[int(r['class_index']) for r in rows]
        assert list(d['generators'])==[r['generator'] for r in rows]
        x=d['features']
        assert x.shape==(count,512) and x.dtype==torch.float32
        assert torch.isfinite(x).all()
        assert torch.allclose(x.norm(dim=1), torch.ones(count), atol=1e-4)
        result[split]={'file':str(p),'sha256':digest(p),
                       'manifest_sha256':digest(manifest),'count':count,
                       'original_manifest_path':d['manifest']}
    return result

def model_paths():
    return [(f'resnet-{seed}',
             SHARED/'outputs/resnet50-cap128-jpeg95' if seed==20261031 else
             ROOT/f'outputs/completion-resnet-seed{seed}-2218236', seed)
            for seed in (20261031,20261101,20261102)] + [
            (f'v2-{seed}',SHARED/'outputs'/name,seed) for seed,name in [
                (20261031,'resnet50-robust-v2-complete'),
                (20261101,'resnet50-robust-v2-seed20261101'),
                (20261102,'resnet50-robust-v2-seed20261102')]]

def preflight():
    import torch
    expected=read_json(HERE/'source_hashes.json')
    for name,h in expected.items():
        assert hashlib.sha256(( (ROOT/name).read_text().rstrip()+'\n').encode()).hexdigest()==h, f'Code changed: {name}; review before running'
    for name,p,seed in model_paths()+[('clip-original',SHARED/'outputs/clip-vit-b32-cap128-jpeg95',20261031)]:
        c=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
        v=read_json(p/'validation_report.json')
        assert c['seed']==seed, (name,'seed mismatch')
        assert c['input_control']=='cap128_jpeg95', (name,'control mismatch')
        assert 0<=float(v['validation_threshold_max_f1'])<=1
    for name in ['all_splits.csv','test_glide_unseen.csv','test_vqdm_unseen.csv','test_robustness_new.csv']:
        assert digest(ROOT/'manifests'/name)==digest(SHARED/'manifests'/name), name
    print('PREFLIGHT PASSED',flush=True)

def holdout(model_dir,out):
    import torch
    from torch.utils.data import DataLoader
    import evaluate_robustness_holdout as h
    from train import ResNet50,metrics
    ck=torch.load(model_dir/'best.pt',map_location='cpu',weights_only=False)
    threshold=float(read_json(model_dir/'validation_report.json')['validation_threshold_max_f1'])
    model=ResNet50().to('cuda')
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    with h.MANIFEST.open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==6000
    results=[]
    out.mkdir(parents=True,exist_ok=True)
    for condition,perturbation in h.CONDITIONS:
        loader=DataLoader(h.HoldoutSet(rows,perturbation),batch_size=64,
                          shuffle=False,num_workers=4,pin_memory=True)
        ys,ps,gs=[],[],[]
        with torch.inference_mode():
            for x,y,g in loader:
                z=model(x.to('cuda',non_blocking=True))
                ys.extend(y.tolist()); ps.extend(z.sigmoid().cpu().tolist()); gs.extend(g)
        with (out/f'{condition}-per_image.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['path','label','matched_generator','score_ai'])
            w.writerows((r['path'],y,g,p) for r,y,g,p in zip(rows,ys,gs,ps))
        for group in ['all']+sorted(set(gs)):
            ix=[i for i,g in enumerate(gs) if group=='all' or g==group]
            results.append({'condition':condition,'group':group,'count':len(ix),
                'threshold':threshold,**metrics([ys[i] for i in ix],[ps[i] for i in ix],threshold)})
        print(condition,results[-len(set(gs))-1],flush=True)
    write_json(out/'report.json',{'checkpoint':str(model_dir/'best.pt'),
        'precision':'float32; original holdout protocol','results':results,
        'threshold_source':'saved validation report; no test retuning',
        'derived_images_counted_as_new_samples':False})

def clip_run(out,audit):
    import torch
    import numpy as np
    import clip_detector as c
    data={s:torch.load(p['file'],map_location='cpu',weights_only=False) for s,p in audit.items()}
    # Keep original cache metadata; the audit records verified current manifests.
    for seed in (20261031,20261101,20261102):
        target=out/f'clip-{seed}'
        target.mkdir()
        if seed==20261031:
            for name in ('best.pt','validation_report.json'):
                shutil.copy2(SHARED/'outputs/clip-vit-b32-cap128-jpeg95'/name,target/name)
        else:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            c.fit_head(data['train'],data['val'],torch.device('cuda'),target,
                       seed,500,20,1e-2,1e-4,False)
        ck=torch.load(target/'best.pt',map_location='cpu',weights_only=False)
        val=read_json(target/'validation_report.json')
        for split in ('test_seen','test_glide_unseen','test_vqdm_unseen'):
            c.evaluate_head(data[split],ck,val,torch.device('cuda'),target/split)

def main():
    if sys.argv[1]=='preflight':
        preflight(); return
    if sys.argv[1]=='audit':
        write_json(sys.argv[2],caches()); return
    import torch
    assert torch.cuda.is_available(), 'GPU not available'
    preflight()
    out=ROOT/'outputs'/f'completion-batch-{os.environ["SLURM_JOB_ID"]}'
    out.mkdir()
    shutil.copytree(HERE,out/'runner')
    write_json(out/'provenance.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'gpu':torch.cuda.get_device_name(0),'models':[
            {'name':n,'path':str(p),'seed':s,'checkpoint_sha256':digest(p/'best.pt')} for n,p,s in model_paths()],
        'note':'Supplementary evaluation of previously examined fixed tests; not a new untouched test.'})
    (out/'requirements.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    failures=[]
    def task(name,fn):
        print('START',name,flush=True)
        try:
            fn()
            print('DONE',name,flush=True)
        except Exception:
            failures.append(name); traceback.print_exc()
        write_json(out/'status.json',{'failed_tasks':failures,'last_task':name,'finished':False})
    with (out/'cache-audit.log').open('w') as log:
        audit=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'audit',str(out/'cache-audit.json')],stdout=log,stderr=subprocess.STDOUT)
        for name,p,seed in model_paths():
            for split in ('test_seen','test_glide_unseen','test_vqdm_unseen'):
                dest=out/name/split
                def run(p=p,split=split,dest=dest):
                    dest.mkdir(parents=True)
                    with (dest/'run.log').open('w') as f:
                        subprocess.run([sys.executable,'-u','evaluate.py','--checkpoint',str(p/'best.pt'),
                            '--split',split,'--output-dir',str(dest),'--batch-size','64','--workers','4'],stdout=f,stderr=subprocess.STDOUT,check=True)
                task(name+'/'+split,run)
        audit_rc=audit.wait()
    if audit_rc==0:
        task('clip-three-seeds',lambda:clip_run(out,read_json(out/'cache-audit.json')))
    else:
        failures.append('cache-audit: see cache-audit.log; CLIP skipped')
    for name,p,seed in model_paths():
        task(name+'/holdout',lambda p=p,name=name:holdout(p,out/name/'holdout'))
    write_json(out/'status.json',{'failed_tasks':failures,'finished':True})
    print('BATCH_OUTPUT',out,flush=True)
    if failures:
        raise SystemExit('PARTIAL FAILURE: '+str(failures))
    print('SUCCESS: internal three-seed batch finished.',flush=True)

if __name__=='__main__':
    main()
