# AI Image Detection with GenImage

English · [简体中文](README-CN.md)

This project studies AI-generated image detection using an ImageNet-pretrained ResNet-50, a frozen CLIP ViT-B/32 encoder with a linear classifier, and a ResNet-50 trained with robust augmentation (V2). Experiments cover generator transfer, image degradation, probability calibration, and cross-dataset failure analysis.

Project status updated on **November 19, 2026**. Labels are **Real = 0** and **AI-generated = 1**. The detector returns an AI score. Its interpretation depends on the evaluation domain and calibration procedure.

## Completed experiments

| Work | Completed scope | Evidence |
|---|---|---|
| Data preparation | GenImage/ImageNet inventory, class mapping verification, duplicate/source-group audits, fixed train/validation/test manifests, and an additional 6,000-image holdout | [Manifests](manifests/) and [data audit](inventory/README.md) |
| Baseline models | ImageNet-pretrained ResNet-50 and frozen CLIP ViT-B/32 with a linear classification head | [Training record](BASELINE.md) |
| Robust training | Initial augmentation study and V2 training with clean/degraded image mixtures | [Methods and development report](FINAL_REPORT.md) |
| Three-seed evaluation | ResNet, V2, and CLIP, each with seeds 20261031, 20261101, and 20261102 | [Main experimental report](handoff/2026-11-08/FINAL_REPORT.md) |
| Generator transfer | Seen BigGAN/SD1.4/ADM and separate unseen GLIDE/VQDM tests | [Per-seed results](handoff/2026-11-08/tables/per_seed_results.csv) |
| Perturbation robustness | Seven conditions on the fixed 6,000-image holdout, including JPEG, resizing, and blur | [Holdout results](handoff/2026-11-08/tables/holdout_results.csv) |
| Calibration | Validation-fitted temperature scaling and Brier/NLL/ECE evaluation for nine models | [Calibration results](handoff/2026-11-08/tables/calibration_results.csv) |
| External evaluation | All 20,000 official CIFAKE test images and all 5,305 COCOXGEN images | [External results and limitations](handoff/2026-11-08/FINAL_REPORT.md) |
| Statistical analysis | Seed sample standard deviations, paired cluster bootstrap intervals, and inference consistency checks | [Main evidence package](handoff/2026-11-08/evidence/) |
| Score diagnosis | Classwise score quantiles, ECDF plots, ranking metrics, and fixed-threshold confusion counts | [Score analysis](handoff/2026-11-16/score_diagnosis/REPORT.md) |
| Resolution control | One ResNet and three V2 checkpoints, three test sets, and original/64/32 conditions, totaling 36 evaluations | [Resolution outputs](handoff/2026-11-16/resolution_control/) |
| Threshold adaptation | COCO-group development/holdout split and one development-selected threshold per model | [Adaptation results](handoff/2026-11-16/threshold_transfer/summary.csv) |
| Generation configuration analysis | Fooocus/SDXL × short/long prompt comparisons on the same holdout groups | [Configuration analysis](handoff/2026-11-16/threshold_transfer/CELL_REPORT.md) |

A Streamlit upload demo and example images are available. Its current runtime acceptance remains pending. Final course-paper/PPT integration and long-term backup of the additional model checkpoints remain team deliverables.

## Main results

The following AUROC values are means across three training seeds. Full reports provide seed sample standard deviations and the applicable confidence intervals.

| Model | Seen generators | GLIDE | VQDM | CIFAKE | COCOXGEN |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 0.99533 | 0.99210 | 0.97419 | 0.34534 | 0.73601 |
| Robust V2 | 0.99360 | 0.98501 | 0.97690 | 0.43781 | 0.69718 |
| Frozen CLIP + linear head | 0.97196 | 0.91804 | 0.92852 | 0.45244 | 0.82968 |

COCOXGEN uses an equal macro average over four balanced real-versus-generation-configuration cells. CIFAKE includes SD1.4, which also appears in GenImage training. These external evaluations measure transfer across data sources and generation pipelines. AUROC describes ranking, while Accuracy and Recall additionally depend on the selected threshold.

V2 improves robustness under the evaluated degradation conditions. On the additional holdout, resize-to-half-and-restore AUROC increases from **0.85638** for ResNet to **0.97682** for V2. Under blur radius 1.0, it increases from **0.92977** to **0.97993**. Clean performance shows a small tradeoff. [Robustness analysis](handoff/2026-11-08/FINAL_REPORT.md)

## Findings from the latest diagnostics

- **Low-resolution processing causes substantial degradation.** On the seen test set, ResNet AUROC falls from 0.9951 to 0.5311 at low64 and 0.3555 at low32. V2 three-seed means are 0.9936, 0.7852, and 0.4277. These conditions preserve aspect ratio, cap the longest side, then apply the original `cap128_jpeg95` preprocessing. The experiment provides partial evidence relevant to CIFAKE's 32×32 inputs. [Resolution study](handoff/2026-11-16/README.md)
- **Threshold transfer accounts for part of the COCOXGEN failure.** On the diagnostic holdout, adapting thresholds raises ResNet/V2/CLIP balanced accuracy to 66.11%/63.71%/74.76%, with real-image false-positive rates of 34.50%/31.99%/21.09%. Score overlap continues to limit performance. [Threshold study](handoff/2026-11-16/threshold_transfer/summary.csv)
- **Generation configurations differ in difficulty.** Long-prompt variants have lower mean AUROC across the evaluated models. These descriptive comparisons also involve changes in generated content and processing. [Configuration study](handoff/2026-11-16/threshold_transfer/CELL_REPORT.md)

The latest diagnostics were conducted after the external results had been inspected. Threshold adaptation uses 318 COCO groups for development and 743 for holdout evaluation. The complete data had already been viewed, so these results should be presented as post-hoc target-domain analysis. The causal contributions of resolution, content, source, and compression history remain partly unresolved.

## Reports and archived assets

- [Experiment handoff](HANDOFF.md) connects the original project and subsequent contributions.
- [Main report, November 8](handoff/2026-11-08/FINAL_REPORT.md) contains nine-model evaluations, calibration, external tests, and confidence intervals.
- [Diagnostic report and data, November 16](handoff/2026-11-16/README.md) contains the latest failure analyses and 36 resolution evaluations.
- [Original methods and development report](FINAL_REPORT.md) preserves the initial methodology and development history with current status annotations.
- [Figures](handoff/2026-11-08/figures/), [tables](handoff/2026-11-08/tables/), and [writing guide](handoff/2026-11-08/WRITING_GUIDE.md) support the paper and presentation.
- [Release directory](release/GenImage_Final_2026-11-03/) includes original code, fixed manifests, and Git LFS model checkpoints. Additional nine-model checkpoint availability is documented in the [asset index](handoff/2026-11-08/ASSETS_AND_REPRODUCTION.md).

Raw image datasets require separate download or access to the shared cluster storage. The archived CLIP `.pt` stores the classifier head and requires the corresponding CLIP encoder. Exact-hash audits cover the documented checks, while near duplicates and pretraining exposure remain limitations.

## Reproduction

Follow the [reproduction guide](REPRODUCIBILITY.md) for environment setup, Hugging Face data downloads, training, evaluation, and remote demo access. Its detailed instructions are currently in Chinese.

On the school cluster, load the module before activating an existing environment. Replace the environment name and project directory with your own setup when needed.

```bash
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
cd "/scratch/$USER/GenImage"
git lfs pull
```

Recompute the saved-prediction diagnostics on CPU.

```bash
python scripts/diagnose_score_shift.py
python scripts/diagnose_threshold_transfer.py
python scripts/analyze_cocoxgen_cells.py
```

Submit the resolution experiment to the school's AMD CPU partition. This step requires the original images and the four complete release checkpoints.

```bash
mkdir -p logs
sbatch --partition=amd scripts/run_resolution_cpu.sbatch
squeue -u "$USER"
```

The completed reference run was array job **2311461**. All four tasks exited successfully in approximately 26–28 minutes. New runs write to `outputs/resolution_control/<array-job-id>/`. Archived reference predictions, input examples, dimensions, metrics, logs, and checksums are available in the [diagnostic package](handoff/2026-11-16/).

## Repository structure

```text
GenImage/
├── README.md                         English project overview
├── README-CN.md                      Chinese project overview
├── HANDOFF.md                        Experiment handoff and current status
├── FINAL_REPORT.md                   Original methodology and development history
├── REPRODUCIBILITY.md                Setup and reproduction guide
├── BASELINE.md                       Training and evaluation record
├── train.py                          ResNet-50 training
├── train_robust_v2.py                 Robust V2 training
├── clip_detector.py                  Frozen CLIP detector
├── evaluate.py                       Fixed-manifest evaluation
├── evaluate_robustness_holdout.py     Holdout perturbation evaluation
├── scripts/                          Data preparation and diagnostic scripts
├── manifests/                        Fixed splits and audit configuration
├── inventory/                        Data inventory and audit documentation
├── handoff/
│   ├── 2026-11-08/                   Main reports, figures, tables, predictions
│   └── 2026-11-16/                   Diagnostic report, data, and examples
├── demo.py                           Streamlit upload demo
├── demo_examples/                    Example inputs
├── outputs/                          Local generated results (Git-ignored)
└── release/GenImage_Final_2026-11-03/ Original reproducibility archive
```
