# CIFAKE external validation

Official source: https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images

Run submit.sh from /scratch/$USER/AI-Image-Detection. It submits CPU preparation -> one GPU job -> CPU analysis, each afterok-dependent. A new timestamped directory contains all data and outputs. No original manifests, weights, calibration files, or results are overwritten. No dependencies are installed.

Preparation uses a shallow sparse checkout of DATASET/test, freezes the resolved commit, keeps the author's root README/license, verifies 10,000 REAL and 10,000 FAKE images and decodes all images. SHA256 hashes are recorded per image. Exact file-byte overlap is checked against the unique paths in all_splits and both unseen and robustness manifests. Matches stop the pipeline rather than silently change the test. Same-label duplicates inside CIFAKE are reported and retained to preserve the official test; conflicting-label identical bytes stop preparation. This does not establish absence of near-duplicates or pretraining exposure.

The phase2 nine checkpoints, thresholds and temperatures are locked before inference. No training, temperature fitting, threshold tuning, or seed selection occurs on CIFAKE. Primary outputs include uncalibrated and previously calibrated metrics. Three-seed means use sample standard deviations, not confidence intervals. Ranking metrics use logits to avoid numerical sigmoid saturation ties. ECE uses 15 equal-width p(AI) bins. Predictions use the fixed validation logit threshold and its temperature-transformed counterpart.

ResNet/V2 reuse train.Images and AMP evaluation (batch64); CLIP reuses clip_detector extraction (batch128), computes one feature set for all three heads, and uses the phase2 local encoder snapshot offline. Original cap128/JPEG95 and model-specific transforms are retained. CIFAKE is cross-dataset, not an unseen-generator test: fake images use Stable Diffusion 1.4, already represented in GenImage training. Low input resolution is an important domain shift. Do not infer general deployment accuracy from this one dataset.

The CPU final step creates results.zip excluding the dataset and feature cache. This is the package to download for review; keep the full server directory for reproducibility.

Checks completed locally: Python and shell syntax, numerical AUROC/AP including ties against independent reference implementations, decision/ranking invariance at five temperatures. GPU and official dataset end-to-end execution is pending on HPC.
