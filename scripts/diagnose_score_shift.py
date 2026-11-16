"""Diagnose saved scores without fitting models, temperatures, or thresholds."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'handoff/2026-11-08/evidence'


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/score_diagnosis')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('reference', EVIDENCE / 'followup/metrics_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    locked = json.loads((EVIDENCE / 'cifake/locked_models.json').read_text())
    with (EVIDENCE / 'followup/external_manifest.csv').open() as stream:
        manifest = list(csv.DictReader(stream))
    stats, scores, metric_rows = [], [], []
    plot_data = {}
    for name, config in locked.items():
        datasets = {}
        for split in ['val', 'test_seen', 'test_glide_unseen', 'test_vqdm_unseen']:
            data = np.load(EVIDENCE / 'phase2' / name / f'{split}.npz')
            datasets[split] = (data['logits'], data['labels'])
        data = np.load(EVIDENCE / 'cifake' / name / 'logits.npz')
        assert len(data['labels']) == 20000
        datasets['CIFAKE'] = (data['logits'], data['labels'])
        data = np.load(EVIDENCE / 'followup' / name / 'logits.npz')
        index = {path: i for i, path in enumerate(data['paths'].tolist())}
        indices = np.array([index[row['relative_path']] for row in manifest])
        z, y = data['logits'][indices], data['labels'][indices]
        assert y.tolist() == [int(row['label']) for row in manifest]
        assert len(y) == 5305
        datasets['COCOXGEN_pooled'] = (z, y)
        for generator in ['fooocus', 'sdxl']:
            for prompt in ['short', 'long']:
                mask = np.array([row['label'] == '0' or (row['generator'] == generator and row['prompt'] == prompt) for row in manifest])
                datasets[f'COCOXGEN_{generator}_{prompt}'] = (z[mask], y[mask])
        threshold = config['threshold']
        for domain, (logits, labels) in datasets.items():
            assert np.isfinite(logits).all() and set(labels.tolist()) == {0, 1}
            metrics, _, probabilities, _ = reference.score_metrics(logits, labels, np.log(threshold / (1 - threshold)))
            row = {'model': name, 'domain': domain, 'count': len(labels), 'threshold': threshold, **metrics}
            row['real_fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn'])
            row['balanced_accuracy'] = (metrics['recall'] + 1 - row['real_fpr']) / 2
            metric_rows.append(row)
            for label in [0, 1]:
                selected = probabilities[labels == label]
                quantiles = np.quantile(selected, [.05, .25, .5, .75, .95])
                stats.append({'model': name, 'domain': domain, 'label': label, 'count': len(selected), 'mean': selected.mean(), **dict(zip(['q05', 'q25', 'median', 'q75', 'q95'], quantiles)), 'fraction_above_threshold': float(np.mean(selected >= threshold))})
            if domain in ['val', 'test_seen', 'CIFAKE', 'COCOXGEN_pooled']:
                plot_data[name, domain] = (logits, labels, threshold)
                for label, logit, probability in zip(labels, logits, probabilities):
                    scores.append({'model': name, 'domain': domain, 'label': int(label), 'logit': float(logit), 'score_ai': float(probability)})
    write_csv(args.output_dir / 'metrics.csv', metric_rows)
    write_csv(args.output_dir / 'score_quantiles.csv', stats)
    # Equal class weight in each ECDF makes the imbalanced external set comparable.
    domains = ['val', 'test_seen', 'CIFAKE', 'COCOXGEN_pooled']
    for family in ['resnet', 'v2', 'clip']:
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
        for i, seed in enumerate([20261031, 20261101, 20261102]):
            name = f'{family}-{seed}'
            for j, domain in enumerate(domains):
                ax = axes[i, j]
                logits, labels, threshold = plot_data[name, domain]
                for label, color, title in [(0, '#2878b5', 'Real'), (1, '#d95319', 'AI')]:
                    values = np.sort(logits[labels == label])
                    ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, label=title)
                ax.axvline(np.log(threshold / (1 - threshold)), color='black', linestyle='--', label='Validation threshold')
                ax.set_title(f'{domain} / {seed}', fontsize=10)
                ax.set_xlabel('AI logit')
                if j == 0:
                    ax.set_ylabel('Cumulative fraction within class')
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'{family}: saved score distributions (raw logits)')
        fig.tight_layout()
        fig.savefig(args.output_dir / f'{family}_score_ecdf.png', dpi=160)
        plt.close(fig)
    summary = []
    for family in ['resnet', 'v2', 'clip']:
        for domain in domains:
            rows = [r for r in metric_rows if r['model'].startswith(family + '-') and r['domain'] == domain]
            row = {'family': family, 'domain': domain}
            for key in ['roc_auc', 'average_precision', 'balanced_accuracy', 'recall', 'real_fpr', 'accuracy', 'f1']:
                values = [r[key] for r in rows]
                row[key + '_mean'] = float(np.mean(values))
                row[key + '_sample_sd'] = float(np.std(values, ddof=1))
            for label, title in [(0, 'real'), (1, 'fake')]:
                medians = [r['median'] for r in stats if r['model'].startswith(family + '-') and r['domain'] == domain and r['label'] == label]
                row[title + '_median_mean'] = float(np.mean(medians))
            summary.append(row)
    write_csv(args.output_dir / 'family_summary.csv', summary)
    lines = ['# Saved-score diagnosis', '', 'Scores use original logits and each checkpoint\'s original validation threshold. All three seeds are retained. No parameters are fitted. The median columns average the three per-seed class medians.', '', '| Family | Domain | AUROC | AI recall | Real FPR | Real median | AI median |', '|---|---|---|---|---|---|---|']
    for row in summary:
        lines.append(f"| {row['family']} | {row['domain']} | {row['roc_auc_mean']:.5f} | {row['recall_mean']:.5f} | {row['real_fpr_mean']:.5f} | {row['real_median_mean']:.6g} | {row['fake_median_mean']:.6g} |")
    lines += ['', 'COCOXGEN pooled AP and accuracy reflect 80% fake prevalence. Use the four balanced cell rows in metrics.csv for comparisons. Reused real images across these cells are not independent samples.', '', 'The val results reuse data previously used for checkpoint and threshold selection. These are descriptive, post-hoc diagnostics. ECDF plots use logits to retain resolution near zero and one. A leftward shift in both classes and AUROC below 0.5 may coexist. These measurements do not identify the causal contribution of resolution, source, or processing.', '', 'Confusion counts are per seed in metrics.csv. Quantiles are per seed and class in score_quantiles.csv. All original labels and score directions are preserved.']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
