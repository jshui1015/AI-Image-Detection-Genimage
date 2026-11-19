# AI Image Detection with GenImage

English · [简体中文](README-CN.md)

This project studies AI-generated image detection using an ImageNet-pretrained ResNet-50, a frozen CLIP ViT-B/32 encoder with a linear classifier, and a ResNet-50 trained with robust augmentation (V2). Experiments cover generator transfer, image degradation, probability calibration, and cross-dataset failure analysis.

Project status updated on **November 19, 2026**. Labels are **Real = 0** and **AI-generated = 1**. The detector returns an AI score. Its interpretation depends on the evaluation domain and calibration procedure.

## Completed experiments

| Work | Purpose | Completed scope | Results and evidence |
|---|---|---|---|
| Data preparation | Verify labels, classes, and sources, reduce duplicates and leakage across splits, and support fair model comparisons | GenImage/ImageNet inventory, class mapping verification, duplicate and source-group audits, fixed training/validation/test manifests, and an additional 6,000-image holdout | [Fixed manifests](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/manifests) · [Data audit](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/inventory/README.md) |
| Baseline models | Establish reference performance and compare learned CNN detection features with pretrained CLIP features | ImageNet-pretrained ResNet-50 and a frozen CLIP ViT-B/32 encoder with a linear classification head | [Training record](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/BASELINE.md) |
| Robust training | Assess whether augmentation improves resistance to image degradation and measure the tradeoff in clean-image performance | Initial augmentation experiments and V2 training with a mixture of clean and degraded images | [Methods and development report](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/FINAL_REPORT.md) |
| Three-seed evaluation | Measure training variability and assess the stability of results and improvements | ResNet, V2, and CLIP evaluated with seeds 20261031, 20261101, and 20261102 | [Main experimental report](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/FINAL_REPORT.md) |
| Cross-generator detection | Assess detection on independent images and transfer to generators absent from training | Seen BigGAN/SD1.4/ADM and separately evaluated unseen GLIDE/VQDM | [Per-seed results](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/per_seed_results.csv) |
| Image perturbation robustness | Measure performance degradation under compression, resizing, and blur, and evaluate the benefits of robust training | Seven conditions on a fixed 6,000-image holdout, including clean inputs, JPEG, resizing, and blur | [Holdout results](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/holdout_results.csv) |
| Probability calibration | Assess correspondence between scores and observed class frequencies, including calibration transfer across domains | Validation-fitted temperature scaling for nine models, evaluated using Brier score, NLL, and ECE | [Calibration results](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/calibration_results.csv) |
| External dataset evaluation | Assess generalization under changes in image sources, content, and generation pipelines | All 20,000 official CIFAKE test images and all 5,305 COCOXGEN images | [External results and limitations](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/FINAL_REPORT.md) |
| Statistical analysis | Quantify training variability and test-sampling uncertainty, and check inference consistency | Seed sample standard deviations, paired cluster bootstrap intervals, and inference consistency checks | [Main experiment evidence](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/handoff/2026-11-08/evidence) |
| Score diagnosis | Identify ranking degradation, score-distribution shifts, and fixed-threshold failures | Classwise score quantiles, ECDF plots, ranking metrics, and fixed-threshold confusion counts | [Score analysis](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/score_diagnosis/REPORT.md) |
| Resolution control | Hold images, models, and thresholds fixed to assess degradation caused by the specified low-resolution processing and investigate CIFAKE failure | One ResNet and three V2 checkpoints; three test sets × original/64/32, totaling 36 evaluations; 64/32 denote aspect-ratio-preserving longest-side caps | [Resolution experiment outputs](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/handoff/2026-11-16/resolution_control) |
| Threshold adaptation | Quantify how much target-domain threshold adjustment reduces missed detections and increases real-image false positives | COCO-source-group split into 318 development groups and 743 holdout groups; one shared threshold selected per model on development data | [Threshold adaptation results](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/threshold_transfer/summary.csv) |
| Generation configuration analysis | Assess whether detection difficulty varies with generation pipelines and prompt configurations | Holdout comparisons of four configurations combining Fooocus/SDXL with short/long prompts | [Generation configuration analysis](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/threshold_transfer/CELL_REPORT.md) |


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
