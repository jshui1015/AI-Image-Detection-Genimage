"""Evaluate controlled resolution changes using local release checkpoints."""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import Images, ResNet50, metrics

FILES = {
    'resnet-20261031': 'resnet50_baseline_cap128_jpeg95.pt',
    'v2-20261031': 'resnet50_robust_v2_seed20261031.pt',
    'v2-20261101': 'resnet50_robust_v2_seed20261101.pt',
    'v2-20261102': 'resnet50_robust_v2_seed20261102_selected.pt',
}


class ResolutionImages(Images):
    """Reuse the original deterministic preprocessing after downsampling."""
    def __init__(self, manifest, split, resolution):
        super().__init__(manifest, split, input_control='cap128_jpeg95')
        self.resolution = resolution

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        original = image.size
        if self.resolution:
            w, h = image.size
            scale = min(1., self.resolution / max(w, h))
            image = image.resize((max(1, round(w*scale)), max(1, round(h*scale))), Image.Resampling.LANCZOS)
        reduced = image.size
        w, h = image.size
        scale = min(1., 128 / max(w, h))
        if scale < 1:
            image = image.resize((max(1, round(w*scale)), max(1, round(h*scale))), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            image = encoded.convert('RGB')
        w, h = image.size
        scale = 256 / min(w, h)
        nw, nh = round(w*scale), round(h*scale)
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        left, top = (nw-224)//2, (nh-224)//2
        image = image.crop((left, top, left+224, top+224))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2,0,1).div_(255)
        mean = x.new_tensor([.485,.456,.406])[:,None,None]
        std = x.new_tensor([.229,.224,.225])[:,None,None]
        return (x-mean)/std, int(row['label']), row['path'], torch.tensor([*original,*reduced])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', choices=FILES, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    locked = json.loads((ROOT/'handoff/2026-11-08/evidence/cifake/locked_models.json').read_text())
    threshold = locked[args.model]['threshold']
    checkpoint_path = ROOT/'release/GenImage_Final_2026-11-03/models'/FILES[args.model]
    import hashlib
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != locked[args.model]['sha256']:
        raise ValueError(f'Checkpoint hash differs from locked model: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if ckpt['input_control'] != 'cap128_jpeg95':
        raise ValueError('Unexpected checkpoint preprocessing')
    model = ResNet50().eval()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    out = args.output_dir/args.model
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for split in ['test_seen', 'test_glide_unseen', 'test_vqdm_unseen']:
        manifest = ROOT/'manifests'/('all_splits.csv' if split=='test_seen' else split+'.csv')
        for condition, resolution in [('original',0),('low64',64),('low32',32)]:
            ds = ResolutionImages(manifest, split, resolution)
            loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
            ys, ps, paths, dimensions = [], [], [], []
            with torch.inference_mode():
                for x,y,path,dims in loader:
                    ps.extend(model(x).sigmoid().tolist())
                    ys.extend(y.tolist())
                    paths.extend(path)
                    dimensions.extend(dims.tolist())
            np.savez_compressed(out/f'{split}-{condition}.npz', labels=ys, scores=ps, paths=paths, dimensions=dimensions)
            result = {'model':args.model,'split':split,'condition':condition,'count':len(ys),'threshold':threshold,**metrics(ys,ps,threshold)}
            results.append(result)
            print(json.dumps(result), flush=True)
            # Save the exact normalized-input inverse as lossless example images.
            for index in [0, next(i for i,r in enumerate(ds.rows) if r['label']=='1')]:
                x, label, _, _ = ds[index]
                rgb = x*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]
                pixels = rgb.permute(1,2,0).mul(255).round().clamp(0,255).byte().numpy()
                Image.fromarray(pixels).save(out/f'{split}-{condition}-label{label}.png')
            with (out/'metrics.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    (out/'protocol.json').write_text(json.dumps({'checkpoint_sha256':digest,'threshold':threshold,'interpolation':'Lanczos longest-side cap before cap128_jpeg95, then original bilinear resize and center crop','dimensions_columns':['original_width','original_height','reduced_width','reduced_height'],'device':'cpu','threads':args.threads},indent=2))


if __name__ == '__main__':
    main()
