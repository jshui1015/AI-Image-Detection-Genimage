> **历史阶段文档（2026-11-03）。** 2026-11-08已补齐三种子评估、校准、两个外部来源与置信区间。当前交接入口见 [HANDOFF.md](HANDOFF.md)，最新结果见 [补充实验报告](handoff/2026-11-08/FINAL_REPORT.md)。以下原文保留作历史方法/结果记录；其中“未完成”状态、单种子数字和演示说明不能当作本轮最新验收结论。

# 复现说明

## 1. 复现范围

本说明覆盖固定数据划分、ResNet-50 基线、冻结 CLIP 线性头、鲁棒 V2、验证集扰动分析及最终评估。原始图片不在最终归档中；清单中的 `path` 相对于项目根目录。推荐在各自可访问的项目根目录（下文记为 `<PROJECT_ROOT>`）中运行。

归档快照包含：模型检查点、所有固定清单、评估 JSON、代码和审计报告。归档不包含原始图像；CLIP 检查点只包含训练好的线性分类头，冻结编码器 `openai/clip-vit-base-patch32` 需要另行缓存或下载。

## 2. 计算环境

已记录的正式训练环境：学校 GPU 集群，NVIDIA L20，Python 3.11，PyTorch `2.11.0+cu128`；ResNet 运行于 CUDA。申请 GPU 交互节点后执行：

先在登录节点使用MAIE的 Slurm account 申请 GPU：

```bash
srun --account=mscaiehpc42026 --partition=gpu-l20 \
  --gpus-per-node=1 --nodes=1 --time=01:00:00 --pty bash
```

进入分配到的 GPU 节点后，再配置环境和项目路径。以下模块名与分区名适用于本校集群；其他集群请换成管理员提供的模块和分区。`genimage` 是示例 Conda 环境名，可替换为队友自己选定的名称。创建环境和安装依赖只需做一次；后续作业直接激活已有环境。

```bash
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n genimage python=3.11 -y
conda activate genimage

# 与记录环境匹配的 CUDA 12.8 PyTorch 组合。
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy pillow transformers streamlit

PROJECT_ROOT="/scratch/$USER/GenImage"  # 如果项目存放在别处，改成实际路径
cd "$PROJECT_ROOT"
```

运行前确认 CUDA 可见：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print("CUDA:", torch.cuda.is_available(), "GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("No GPU visible; do not start full training/evaluation")
print(torch.cuda.get_device_name(0))
PY
```

记录环境快照，便于报告版本差异：

```bash
python --version
python -m pip freeze > outputs/reproduction-pip-freeze.txt
nvidia-smi
```

## 3. 数据与固定清单

### 从 Hugging Face 下载实验数据

本项目使用 Hugging Face 上的 `jzousz/GenImage` 镜像。需在项目根目录下载，并保留镜像中的目录层级；清单中的 `path` 是相对于项目根目录的路径。下面按生成器目录逐个下载本实验需要的数据：

```bash
PROJECT_ROOT="/scratch/$USER/GenImage"  # 换成项目实际存放位置
cd "$PROJECT_ROOT"

hf download jzousz/GenImage --repo-type dataset --include "BigGAN/*" --local-dir "$PROJECT_ROOT"
hf download jzousz/GenImage --repo-type dataset --include "stable_diffusion_v_1_4/*" --local-dir "$PROJECT_ROOT"
hf download jzousz/GenImage --repo-type dataset --include "ADM/*" --local-dir "$PROJECT_ROOT"
hf download jzousz/GenImage --repo-type dataset --include "glide/*" --local-dir "$PROJECT_ROOT"
hf download jzousz/GenImage --repo-type dataset --include "VQDM/*" --local-dir "$PROJECT_ROOT"
```

如果 `hf` 命令不存在，先在当前 Conda 环境安装 Hugging Face Hub CLI：

```bash
python -m pip install -U huggingface_hub
```

完整镜像很大；以上 `--include` 会按生成器目录筛选文件，不必下载本实验未使用的 Midjourney、SD 1.5 或 Wukong。下载完成后，检查固定清单引用的图片都存在：

```bash
python - <<'PY'
import csv
from pathlib import Path

root = Path.cwd()
manifest_names = [
    "manifests/all_splits.csv",
    "manifests/test_glide_unseen.csv",
    "manifests/test_vqdm_unseen.csv",
    "manifests/test_robustness_new.csv",
]
missing = set()
for name in manifest_names:
    with (root / name).open(newline="") as f:
        for row in csv.DictReader(f):
            path = root / row["path"]
            if not path.is_file():
                missing.add(row["path"])
print(f"missing files: {len(missing)}")
for path in sorted(missing)[:20]:
    print(path)
if missing:
    raise SystemExit("Some manifest images are missing; do not start training yet.")
PY
```

`jzousz/GenImage` 是 Hugging Face 上的镜像，不是 GenImage 官方仓库本身。镜像更新或文件布局变化时，先确认以上路径检查通过；不要为了让训练启动而改写固定清单。GenImage 官方项目及其原始数据说明见[官方仓库](https://github.com/GenImage-Dataset/GenImage)，镜像文件见[Hugging Face 数据集页](https://huggingface.co/datasets/jzousz/GenImage)。

使用归档提供的清单，不要重新抽样：

- `manifests/all_splits.csv`：train、val、test_seen 和预留真图；
- `manifests/test_glide_unseen.csv`、`manifests/test_vqdm_unseen.csv`：各自未见生成器测试；
- `manifests/test_robustness_new.csv`：6,000 张图片级独立 holdout；
- `manifests/*config.json`：随机种子、配额和构建策略。

固定基本规模为 train 40,000、val 8,000、test_seen 10,000；GLIDE/VQDM 测试各4,000。新 holdout 每个类别、每个已见生成器各1真1假，共6,000张。新 holdout 与前面清单在已识别 image_id、path、source_group 上无重叠，但来源仍是 GenImage/ImageNet。

如果要单独验证基础划分脚本，应输出到全新临时目录，不能覆盖正式清单：

```bash
python scripts/build_splits.py --seed 20261031 \
  --output /tmp/genimage-splits-reproduction
```

该命令要求项目的候选清单/审计输入存在。正式实验直接使用归档中的固定 CSV。

## 4. ResNet-50 基线

预训练 ResNet-50、BCEWithLogitsLoss、AdamW；默认最多10轮，batch size 64、学习率 `1e-4`、weight decay `1e-4`，验证 AUROC patience 3。所有正式模型使用 `cap128_jpeg95`：先将最长边限制到128像素，再 JPEG Q95，然后执行确定性验证/测试预处理。输出目录必须新建：

```bash
python train.py \
  --input-control cap128_jpeg95 \
  --seed 20261031 \
  --output-dir outputs/reproduction/resnet50-baseline
```

训练脚本会在固定 val 上按 AUROC 保存最佳检查点，并在该检查点上选 F1 阈值；检查点和 `validation_report.json` 都写在输出目录。

## 5. 冻结 CLIP 基线

冻结 `openai/clip-vit-base-patch32` 图像编码器、缓存训练/验证 embedding、只训练线性头。默认 batch size 128、最多500轮、学习率 `1e-2`、patience 20、seed 20261031；使用 CLIP 自己的 image processor。

```bash
python clip_detector.py --stage train \
  --input-control cap128_jpeg95 \
  --seed 20261031 \
  --output-dir outputs/reproduction/clip-vit-b32
```

首次运行可能下载模型权重；若集群不能联网，请先在可联网环境缓存 Hugging Face 模型，并用 `--cache-dir` 指定缓存目录。

## 6. 鲁棒 V2 与验证扰动

V2 训练时每张训练图片有50%概率保持额外退化处理干净；否则随机选择降采样后恢复、JPEG重压缩或轻微模糊中的一种。增强对真/假图使用相同策略。验证集不使用随机增强。

三个固定 seed 的复现实验命令：

```bash
python train_robust_v2.py --input-control cap128_jpeg95 --robust-augment \
  --seed 20261031 --output-dir outputs/reproduction/robust-v2-20261031
python train_robust_v2.py --input-control cap128_jpeg95 --robust-augment \
  --seed 20261101 --output-dir outputs/reproduction/robust-v2-20261101
python train_robust_v2.py --input-control cap128_jpeg95 --robust-augment \
  --seed 20261102 --output-dir outputs/reproduction/robust-v2-20261102
```

完成训练后，对每个输出目录仅评估固定 val 的干净图和预先定义扰动：

```bash
python evaluate_validation_robustness.py outputs/reproduction/robust-v2-20261031
python evaluate_validation_robustness.py outputs/reproduction/robust-v2-20261101
python evaluate_validation_robustness.py outputs/reproduction/robust-v2-20261102
```

这些验证扰动结果用于开发阶段的方案比较；不要把它们写成独立测试结果。

## 7. 最终测试与阈值规则

常规最终测试必须使用训练目录中对应的验证阈值；不在测试集上重选阈值。以下命令是**一次性最终评估入口**。基线与 V2 结果分别保存；V2 的 GLIDE/VQDM 最终结果已完成并记录。

基线 ResNet：

```bash
python evaluate.py --checkpoint outputs/resnet50-cap128-jpeg95/best.pt --split test_seen
python evaluate.py --checkpoint outputs/resnet50-cap128-jpeg95/best.pt --split test_glide_unseen
python evaluate.py --checkpoint outputs/resnet50-cap128-jpeg95/best.pt --split test_vqdm_unseen
```

选定的鲁棒 V2（seed 20261102）在 GLIDE 和 VQDM 未见生成器清单上的评估命令如下。`evaluate.py` 会从检查点同目录读取 `validation_report.json`，沿用该 V2 模型在 val 上选出的阈值 **0.5053975582**；不要用测试集另选阈值。该评估已完成，结果存于 `outputs/final_test/robust-v2-seed20261102/`，重复运行仅用于复核固定结果，不应用于继续调参。

```bash
python evaluate.py \
  --checkpoint outputs/resnet50-robust-v2-seed20261102/best.pt \
  --split test_glide_unseen \
  --output-dir outputs/final_test/robust-v2-seed20261102/test_glide_unseen

python evaluate.py \
  --checkpoint outputs/resnet50-robust-v2-seed20261102/best.pt \
  --split test_vqdm_unseen \
  --output-dir outputs/final_test/robust-v2-seed20261102/test_vqdm_unseen
```

V2 最终测试摘要：

| 未见生成器 | n | Accuracy | F1 | AUROC | AP | 验证阈值 |
|---|---:|---:|---:|---:|---:|---:|
| GLIDE | 4,000 | 0.94225 | 0.94206 | 0.98287 | 0.97643 | 0.505398 |
| VQDM | 4,000 | 0.88375 | 0.87530 | 0.97230 | 0.95598 | 0.505398 |

完整报告及逐图分数分别写入每个输出目录的 `report.json`、`metrics.csv`、`per_class.csv`、`per_image.csv`。V2 未见生成器等权宏平均 AUROC/AP 为 **0.97759/0.96621**。完整方法、混淆矩阵及解释见 `FINAL_REPORT.md` 第4.4节。

冻结 CLIP：

```bash
python clip_detector.py --stage evaluate --split test_seen \
  --output-dir outputs/clip-vit-b32-cap128-jpeg95
python clip_detector.py --stage evaluate --split test_glide_unseen \
  --output-dir outputs/clip-vit-b32-cap128-jpeg95
python clip_detector.py --stage evaluate --split test_vqdm_unseen \
  --output-dir outputs/clip-vit-b32-cap128-jpeg95
```

新 holdout 的最终基线/V2比较使用同一清单、固定各自验证阈值、相同扰动条件；不要看完结果后调整模型或阈值：

```bash
python evaluate_robustness_holdout.py outputs/resnet50-cap128-jpeg95
python evaluate_robustness_holdout.py outputs/resnet50-robust-v2-seed20261102
```

结果写入 `outputs/robustness_new_holdout/`。该 holdout 是 GenImage/ImageNet 内未用过的新图片，不是新数据源；它只覆盖 BigGAN、SD 1.4、ADM。V2 对 GLIDE/VQDM 的独立结果见上表。若在其他环境复核最终测试，应将测试结果视为固定报告，不要据此做新一轮模型或阈值选择。

## 8. 从个人电脑访问 Streamlit 演示

需要两个终端窗口：一个在集群 GPU 节点运行 Streamlit，另一个在个人电脑保持 SSH 隧道。`gpu21` 是示例节点名；每次 Slurm 分配可能不同，应使用当前 `hostname` 输出替换。

**GPU 节点终端：** 在项目目录和已激活的 Conda 环境中启动：

```bash
cd "$PROJECT_ROOT"
streamlit run demo.py --server.address 0.0.0.0 --server.port 8501
```

保持这个 Slurm 交互作业和 Streamlit 进程运行。

**个人电脑终端：** 执行本地端口转发。下面沿用用户名 `jshui`、登录节点 `login2` 和 GPU 节点 `gpu21` 作为示例；队友需把用户名、SSH 登录主机/别名和当前 GPU 节点替换成自己的实际值。

```bash
ssh -N -L 8501:gpu21:8501 jshui@login2
```

SSH 窗口保持打开，然后在个人电脑浏览器访问：

```text
http://localhost:8501
```

如果个人电脑的8501端口已被占用，可以换本地端口，例如 `-L 8502:gpu21:8501`，再打开 `http://localhost:8502`；冒号右侧的GPU节点端口仍是8501。结束时关闭SSH窗口并在GPU节点停止Streamlit。

若 SSH 登录需使用学校提供的完整主机名或 `~/.ssh/config` 别名，请将 `jshui@login2` 换成平时成功登录集群时使用的目标。如果集群策略不允许通过login节点转发到计算节点，使用学校支持的Web代理/隧道服务或按集群文档申请对应访问方式。不要把GPU节点地址直接当作公网地址。

## 9. 指标与预处理解释

报告 Accuracy、Precision、Recall、F1、ROC-AUC、Average Precision 与混淆矩阵。阈值只从验证集选出；不同模型各自阈值可以不同。模型分数未经概率校准，不等于可靠概率。

扰动在基线输入控制（cap128 + JPEG Q95）之后应用：JPEG Q75/Q50、缩放至0.75/0.50后恢复原尺寸、Gaussian blur 半径0.5/1.0。每个扰动作用于真图和假图；派生图是原测试图的条件版本，不作为新增独立样本计数。

## 10. 结果位置与交付

- 完整方法、汇总结果和限制：`FINAL_REPORT.md`；
- 模型权重、清单、报告和代码快照：`release/GenImage_Final_2026-11-03/`；
- 归档：`release/GenImage_Final_2026-11-03.tar`；
- 归档 SHA-256 索引：`release/GenImage_Final_2026-11-03/PACKAGE_MANIFEST.json`；
- 上传图片演示：`streamlit run demo.py --server.address 0.0.0.0 --server.port 8501`。

训练可能因 Slurm walltime 中断。正式训练输出目录里的最佳检查点可能已写入，但验证报告要等训练循环结束后才生成。若中断，应申请新的 GPU 时段并使用新输出目录完整重跑；不要将不完整运行误当成完整实验。
