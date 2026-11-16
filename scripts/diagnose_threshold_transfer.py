"""Post-hoc COCOXGEN threshold adaptation using saved predictions only."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'handoff/2026-11-08/evidence/followup'
CELLS = ['fooocus-short', 'fooocus-long', 'sdxl-short', 'sdxl-long']


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_threshold(z, y, original_logit):
    # Each group has one real and four fake images. Equal class weights give
    # exactly the four-cell macro balanced accuracy, including shared reals.
    candidates = np.unique(np.r_[z, np.nextafter(z.max(), np.inf), original_logit])
    real = np.sort(z[y == 0])
    fake = np.sort(z[y == 1])
    specificity = np.searchsorted(real, candidates, side='left') / len(real)
    recall = 1 - np.searchsorted(fake, candidates, side='left') / len(fake)
    objective = (specificity + recall) / 2
    best = np.flatnonzero(objective == objective.max())
    selected = best[np.argmin(np.abs(candidates[best] - original_logit))]
    return float(candidates[selected]), float(objective[selected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/threshold_transfer')
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE/'external_manifest.csv').open() as stream:
        manifest = list(csv.DictReader(stream))
    groups = sorted({r['group'] for r in manifest}, key=lambda g: hashlib.sha256(f'{args.seed}:{g}'.encode()).hexdigest())
    development = set(groups[:round(.3*len(groups))])
    for row in manifest:
        row['diagnostic_split'] = 'development' if row['group'] in development else 'holdout'
    for group in groups:
        members = [r for r in manifest if r['group'] == group]
        assert len(members) == 5 and sum(int(r['label']) for r in members) == 4
    write_csv(args.output_dir/'split_manifest.csv', manifest)
    spec = importlib.util.spec_from_file_location('reference', EVIDENCE/'metrics_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    locked = json.loads((EVIDENCE/'locked_models.json').read_text())
    y = np.array([int(r['label']) for r in manifest])
    dev = np.array([r['diagnostic_split'] == 'development' for r in manifest])
    cells = np.array([r['generator']+'-'+r['prompt'] for r in manifest])
    results, selections = [], []
    for name, info in locked.items():
        data = np.load(EVIDENCE/name/'logits.npz')
        index = {p: i for i,p in enumerate(data['paths'].tolist())}
        indices = [index[r['relative_path']] for r in manifest]
        assert data['labels'][indices].tolist() == y.tolist()
        z = data['logits'][indices]
        assert np.isfinite(z).all()
        original = float(np.log(info['threshold']/(1-info['threshold'])))
        adapted, objective = choose_threshold(z[dev], y[dev], original)
        selections.append({'model':name,'original_logit_threshold':original,'adapted_logit_threshold':adapted,'adapted_score_threshold':float(np.exp(-np.logaddexp(0,-adapted))),'development_macro_balanced_accuracy':objective})
        for setting, threshold in [('original',original),('adapted',adapted)]:
            cell_results = []
            for cell in CELLS:
                mask = ~dev & ((y == 0) | (cells == cell))
                metrics, _, _, _ = reference.score_metrics(z[mask],y[mask],threshold)
                metrics['real_fpr'] = metrics['fp']/(metrics['fp']+metrics['tn'])
                metrics['balanced_accuracy'] = (metrics['recall']+1-metrics['real_fpr'])/2
                row = {'model':name,'setting':setting,'cell':cell,'count':int(mask.sum()),'logit_threshold':threshold,**metrics}
                results.append(row)
                cell_results.append(row)
            results.append({'model':name,'setting':setting,'cell':'macro_four_cells','count':int((~dev).sum()),'logit_threshold':threshold,**{k:float(np.mean([r[k] for r in cell_results])) for k in metrics}})
    write_csv(args.output_dir/'thresholds.csv',selections)
    write_csv(args.output_dir/'holdout_metrics.csv',results)
    summary = []
    for family in ['resnet','v2','clip']:
        for setting in ['original','adapted']:
            rows = [r for r in results if r['model'].startswith(family+'-') and r['setting']==setting and r['cell']=='macro_four_cells']
            summary.append({'family':family,'setting':setting,**{k:float(np.mean([r[k] for r in rows])) for k in ['roc_auc','balanced_accuracy','recall','real_fpr','f1']}})
    write_csv(args.output_dir/'summary.csv',summary)
    protocol = {'seed':args.seed,'development_groups':len(development),'holdout_groups':len(groups)-len(development),'selection':'maximize development four-cell macro balanced accuracy; ties nearest original logit threshold','interpretation':'Post-hoc target-domain adaptation. Entire COCOXGEN was previously inspected; this holdout is not a new independent external benchmark.','macro_confusion_counts':'Macro rows average cell confusion counts; shared reals must not be summed as independent images.','source_manifest_sha256':hashlib.sha256((EVIDENCE/'external_manifest.csv').read_bytes()).hexdigest()}
    (args.output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(protocol,indent=2))
    for row in summary:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
