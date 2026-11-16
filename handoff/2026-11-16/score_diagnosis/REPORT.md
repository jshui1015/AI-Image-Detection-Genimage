# Saved-score diagnosis

Scores use original logits and each checkpoint's original validation threshold. All three seeds are retained. No parameters are fitted. The median columns average the three per-seed class medians.

| Family | Domain | AUROC | AI recall | Real FPR | Real median | AI median |
|---|---|---|---|---|---|---|
| resnet | val | 0.99695 | 0.99000 | 0.03117 | 5.32226e-06 | 0.9986 |
| resnet | test_seen | 0.99533 | 0.98987 | 0.04780 | 1.00078e-05 | 0.998598 |
| resnet | CIFAKE | 0.34534 | 0.05137 | 0.08820 | 0.109033 | 0.0423429 |
| resnet | COCOXGEN_pooled | 0.73601 | 0.00943 | 0.00063 | 2.7019e-06 | 8.76041e-05 |
| v2 | val | 0.99612 | 0.98542 | 0.03117 | 4.56646e-05 | 0.982609 |
| v2 | test_seen | 0.99360 | 0.98507 | 0.04593 | 6.25936e-05 | 0.982036 |
| v2 | CIFAKE | 0.43781 | 0.00083 | 0.00003 | 0.00180586 | 0.00103284 |
| v2 | COCOXGEN_pooled | 0.69718 | 0.00628 | 0.00000 | 1.30958e-05 | 0.000138149 |
| clip | val | 0.98098 | 0.95283 | 0.08125 | 0.0780807 | 0.898389 |
| clip | test_seen | 0.97196 | 0.95020 | 0.10767 | 0.103231 | 0.895428 |
| clip | CIFAKE | 0.45244 | 0.00850 | 0.01067 | 0.10132 | 0.0849919 |
| clip | COCOXGEN_pooled | 0.82968 | 0.20963 | 0.01194 | 0.0638029 | 0.266791 |

COCOXGEN pooled AP and accuracy reflect 80% fake prevalence. Use the four balanced cell rows in metrics.csv for comparisons. Reused real images across these cells are not independent samples.

The val results reuse data previously used for checkpoint and threshold selection. These are descriptive, post-hoc diagnostics. ECDF plots use logits to retain resolution near zero and one. A leftward shift in both classes and AUROC below 0.5 may coexist. These measurements do not identify the causal contribution of resolution, source, or processing.

Confusion counts are per seed in metrics.csv. Quantiles are per seed and class in score_quantiles.csv. All original labels and score directions are preserved.
