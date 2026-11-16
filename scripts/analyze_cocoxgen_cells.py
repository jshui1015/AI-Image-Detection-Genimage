"""Summarize generator and prompt cells from the threshold-transfer holdout."""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/threshold_transfer'
CELLS = ['fooocus-short', 'fooocus-long', 'sdxl-short', 'sdxl-long']
METRICS = ['roc_auc', 'average_precision', 'balanced_accuracy', 'recall', 'real_fpr']


def main():
    with (OUT/'holdout_metrics.csv').open() as stream:
        source = list(csv.DictReader(stream))
    summary = []
    for family in ['resnet', 'v2', 'clip']:
        for cell in CELLS:
            for setting in ['original', 'adapted']:
                rows = [r for r in source if r['model'].startswith(family+'-') and r['cell']==cell and r['setting']==setting]
                assert len(rows)==3
                row = {'family':family, 'cell':cell, 'setting':setting, 'seeds':3, 'images_per_seed':int(rows[0]['count'])}
                for metric in METRICS:
                    values = [float(r[metric]) for r in rows]
                    row[metric+'_mean'] = statistics.mean(values)
                    row[metric+'_sample_sd'] = statistics.stdev(values)
                summary.append(row)
    with (OUT/'cell_summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    contrasts = []
    for family in ['resnet','v2','clip']:
        for generator in ['fooocus','sdxl']:
            rows = [r for r in source if r['model'].startswith(family+'-') and r['setting']=='original']
            differences = []
            for seed in [20261031,20261101,20261102]:
                short = next(r for r in rows if r['model']==f'{family}-{seed}' and r['cell']==generator+'-short')
                long = next(r for r in rows if r['model']==f'{family}-{seed}' and r['cell']==generator+'-long')
                differences.append(float(long['roc_auc'])-float(short['roc_auc']))
            contrasts.append({'family':family,'generator':generator,'long_minus_short_auc_mean':statistics.mean(differences),'seed_sample_sd':statistics.stdev(differences)})
    with (OUT/'prompt_contrasts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    lines=['# COCOXGEN configuration analysis','','Holdout only: 743 COCO groups, 743 real and 743 fake images per cell. Thresholds were selected on separate development groups. All data had previously been inspected. Values average three fitted seeds. Shared reals across cells remain paired. Differences are descriptive, without significance testing.','','| Model | Cell | AUROC | Original recall | Adapted recall | Adapted real FPR |','|---|---|---|---|---|---|']
    for family in ['resnet','v2','clip']:
        for cell in CELLS:
            original=next(r for r in summary if r['family']==family and r['cell']==cell and r['setting']=='original')
            adapted=next(r for r in summary if r['family']==family and r['cell']==cell and r['setting']=='adapted')
            lines.append(f"| {family} | {cell} | {original['roc_auc_mean']:.4f} | {original['recall_mean']:.2%} | {adapted['recall_mean']:.2%} | {adapted['real_fpr_mean']:.2%} |")
    lines+=['','Generator comparisons also change generation and file-processing pipelines. Prompt comparisons describe paired generated variants and do not isolate semantic content or a causal prompt-length effect. A shared threshold per model is reused for all cells.']
    (OUT/'CELL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
