# ResNet-50 基线

`train.py` 从固定清单读取数据，按224×224处理，使用 ImageNet 预训练 ResNet-50 做二分类。完整运行默认训练10轮，依据验证集 AUROC 保存最佳模型，随后在验证集选F1阈值。该阈值只用于后续锁定评估规则，不能在 `test_seen.csv` 上调整。

在已申请到 GPU 的作业节点中运行：

```bash
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
cd /scratch/jshui/GenImage
python train.py
```

先做 CPU 小批次检查：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --smoke-test --no-pretrained --device cpu --max-steps 1 --workers 0 --output-dir outputs/resnet50-smoke
```

完整训练检查点和验证报告写入 `outputs/resnet50/`。正式训练默认要求可见GPU，避免意外在CPU上运行长作业；GPU不可见时会立即报错。冒烟模式可显式用CPU。`--weights /path/to/resnet50.pth` 可以指定本地 ImageNet 预训练权重。训练参数可通过 `--epochs`、`--batch-size`、`--workers`、`--lr` 设置。

CPU冒烟试跑分别在 PyTorch 2.14.1 和 2.11.0+cu128 上通过，均使用随机初始化权重；这不是正式基线结果。产物分别在 `outputs/resnet50-smoke/` 与 `outputs/resnet50-smoke-cu128/`。gpu21上的 CUDA 可用性仍需在交互作业中确认。

## GPU节点预检

在GPU节点先运行：

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("GPU unavailable; do not start the full run")
PY
```

你贴出的gpu21信息显示驱动570.211、`myenv`中的PyTorch构建为CUDA13.0。NVIDIA兼容表中CUDA13.x通常要求R580或更新驱动；R570属于CUDA12.x兼容范围。若预检失败，请让集群管理员提供CUDA13 forward-compat支持，或在`myenv`安装与R570兼容的CUDA12.x PyTorch构建，再运行完整训练。
## R570 节点上的 PyTorch 安装

如果 GPU 预检显示 `torch.version.cuda == '13.0'` 且 CUDA 不可用，在GPU节点的 `myenv` 执行官方 CUDA 12.8 wheel 组合：

```bash
python -m pip install --user --force-reinstall \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

PyTorch官方版本页列出了 `torch 2.11.0`、`torchvision 0.26.0` 的 CUDA 12.8 组合。安装完成后，重新运行前面的 GPU 预检；确认 CUDA 可用后执行 `python /scratch/jshui/GenImage/train.py`。本会话没有附加到你的 `gpu21` 终端，因此需由该终端执行安装和训练命令。

在你当前的 `gpu21` 交互作业中，CUDA 12.8 wheel 已安装成功。确认 GPU 可见后即可启动正式基线：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" && python /scratch/jshui/GenImage/train.py
```


## 已见生成器测试

训练结束并锁定检查点后，在同一个GPU作业中运行一次：

```bash
python /scratch/jshui/GenImage/evaluate.py
```

评估脚本读取最佳检查点和验证报告中的固定阈值，不在测试集重新选阈值。指标写入 `outputs/resnet50-rerun/test_seen/`，并按全部测试数据、按类别匹配真图的各生成器子集、ImageNet类别分别报告。
## 已见测试输入偏差对照

在GPU节点运行以下三个固定扰动。它们复用同一个最佳检查点和验证集阈值，不改写原图或数据清单：

```bash
cd /scratch/jshui/GenImage
for control in jpeg95 cap128 cap128_jpeg95; do
  python evaluate.py --control "$control" \
    --output-dir "outputs/resnet50-rerun/test_seen_bias_control/$control" || exit
 done
```

- `jpeg95`：每张解码后的RGB图按原尺寸重编码为JPEG质量95。
- `cap128`：保持宽高比，把长边缩至最多128像素，再按相同模型预处理输入。
- `cap128_jpeg95`：同时执行以上两项。

每组会与未扰动的 `test_seen/report.json` 对比AUROC、AP、F1和准确率变化。它们是测试集鲁棒性诊断，不用于改阈值或选模型；只是测试时干预，不能替代使用相同处理重新训练模型的后续对照实验。

## 一致输入控制重训：cap128 + JPEG Q95

2026-10-31，使用与基线相同的固定划分和随机种子，训练一份独立 ResNet-50。训练、验证和测试图像都先将最长边缩至最多128像素（保持宽高比），再以JPEG质量95编码；模型训练增强和224×224模型输入照常执行。该实验的输出目录为 `outputs/resnet50-cap128-jpeg95/`，最佳检查点为 `best.pt`，验证报告为 `validation_report.json`。测试使用独立的 `test_seen`，阈值只由对应验证集确定。

| test_seen 分组 | AUROC | AP | F1（验证集阈值） | 准确率（验证集阈值） |
|---|---:|---:|---:|---:|
| 全部 | 0.99507 | 0.99406 | 0.97128 | 0.97070 |
| ADM vs 类别匹配真图 | 0.99248 | 0.99095 | 0.96977 | 0.96939 |
| BigGAN vs 类别匹配真图 | 0.99881 | 0.99871 | 0.97769 | 0.97720 |
| Stable Diffusion 1.4 vs 类别匹配真图 | 0.99629 | 0.99584 | 0.97437 | 0.97391 |

与原始预处理基线在同一 `test_seen` 上的全部数据相比，AUROC 增加0.01325、AP增加0.00879；ADM子集AUROC增加0.04301。BigGAN和Stable Diffusion 1.4子集AUROC分别变化-0.00113和-0.00265。以上数字来自一次训练运行；不能据此断言输入控制普遍提升性能，也不能代替GLIDE、VQDM未见生成器测试。

已见测试结果：`outputs/resnet50-cap128-jpeg95/test_seen/report.json`。GLIDE/VQDM各自保留的2000张独立真图见 `manifests/reserved_real_glide.csv` 和 `manifests/reserved_real_vqdm.csv`。

## GLIDE/VQDM 未见生成器最终测试

2026-11-02，两套数据已解压至 `glide/extracted/imagenet_glide/` 和 `VQDM/extracted/imagenet_ai_0419_vqdm/`。只使用其 `val/ai` 假图：各6000张，覆盖1000类、每类6张；`train/ai` 不进入最终测试。文件名第四段按0-based类别号解析，覆盖0–999。ImageNet预训练ResNet-50在各生成器100个固定抽样类别上的top-1/top-5命中率分别为GLIDE 47%/69%、VQDM 56%/75%；对应保留真图为82%/96%、78%/98%。这是类别字段抽查，不是逐张语义认证。

新图片全部可读。与既有1,003,440张可读图片比较，没有SHA-256文件重复、EXIF校正像素重复或32×32缩略图重复；GLIDE和VQDM之间没有像素重复。同类别近重复筛查按dHash距离≤3候选，再以32×32 RGB RMSE≤3.0复核：22个候选对最终均未达到RMSE阈值。该筛查不能穷尽裁剪或旋转版本。

使用固定种子20261102，从每个类别的6张假图中按SHA-256稳定排序选2张，与相应reserved清单中每类2张真图配对。两个独立测试集各4,000张（2,000真+2,000假），每类每标签各2张；与train、val、test_seen不重叠，GLIDE/VQDM两份测试集也互不重叠。清单分别为 `manifests/test_glide_unseen.csv` 和 `manifests/test_vqdm_unseen.csv`，构建配置为 `manifests/unseen_test_config.json`。重建命令：

```bash
cd /scratch/jshui/GenImage
python scripts/build_unseen_test_manifests.py
```

只运行最终测试，不重新训练或调阈值；使用受控模型和验证阶段锁定的阈值：

```bash
python evaluate.py \
  --checkpoint outputs/resnet50-cap128-jpeg95/best.pt \
  --split test_glide_unseen

python evaluate.py \
  --checkpoint outputs/resnet50-cap128-jpeg95/best.pt \
  --split test_vqdm_unseen
```

`evaluate.py` 会依checkpoint自动使用训练时的`cap128_jpeg95`，并从对应checkpoint目录读取已锁定的验证阈值。结果分别写入`outputs/resnet50-cap128-jpeg95/test_glide_unseen/`和`test_vqdm_unseen/`。数据哈希审计和类别抽查结果见`inventory/unseen_generator_audit.json`、`inventory/unseen_generator_label_spotcheck.json`。

### 最终未见测试结果（2026-11-02）

同一个最佳checkpoint（epoch 3）、同一预处理和验证集阈值0.372619，在每个未见测试集上评估一次。每行全量数据同时就是相应生成器与同类别真图比较；未在测试集调阈值。

| 测试集 | AUROC | AP | 准确率 | 精确率 | 召回率 | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Seen：ADM + BigGAN + SD1.4 | 0.99507 | 0.99406 | 0.97070 | 0.95234 | 0.99100 | 0.97128 |
| Unseen：GLIDE | 0.98921 | 0.98525 | 0.95900 | 0.94135 | 0.97900 | 0.95980 |
| Unseen：VQDM | 0.97233 | 0.95569 | 0.89825 | 0.94079 | 0.85000 | 0.89309 |

相对seen，GLIDE的AUROC/AP分别低0.00585/0.00881，VQDM低0.02273/0.03837。VQDM的主要错误是漏检：在锁定阈值下2,000张假图中有300张判为真图；GLIDE对应42张漏检。阈值0.5时VQDM召回进一步降至77.85%，故主结果应报告锁定的验证阈值0.372619。结论限于当前GenImage子集与这个ResNet-50运行；不能据此推断所有生成器或实际场景的表现。

报告文件：`outputs/resnet50-cap128-jpeg95/test_glide_unseen/report.json` 与 `outputs/resnet50-cap128-jpeg95/test_vqdm_unseen/report.json`。由于最终测试指标已查看，后续开发应只使用train/val作模型或阈值选择；测试集只保留为最终报告结果。

## 冻结 CLIP 特征基线

`clip_detector.py` 使用 Transformers 中的 `openai/clip-vit-base-patch32`，冻结图像编码器并缓存归一化图像 embedding，只训练一个线性二分类头。train/val/test统一使用`cap128_jpeg95`和CLIP官方图像预处理；线性头的epoch与阈值只根据val确定。初次运行需要从Hugging Face下载CLIP权重并缓存；正式提取要求GPU。

先在GPU节点运行冒烟检查（会下载模型权重，如缓存中没有）：

```bash
cd /scratch/jshui/GenImage
python clip_detector.py --smoke-test \
  --output-dir outputs/clip-vit-b32-cap128-jpeg95-smoke
```

成功后做完整训练：

```bash
python clip_detector.py --stage train \
  --input-control cap128_jpeg95 \
  --output-dir outputs/clip-vit-b32-cap128-jpeg95
```

train/val embedding保存在`outputs/clip-vit-b32-cap128-jpeg95/features/`，checkpoint和验证阈值保存在同目录。完整训练完成后，依次在每个锁定测试集评估一次：

```bash
python clip_detector.py --stage evaluate --split test_seen \
  --output-dir outputs/clip-vit-b32-cap128-jpeg95
python clip_detector.py --stage evaluate --split test_glide_unseen \
  --output-dir outputs/clip-vit-b32-cap128-jpeg95
python clip_detector.py --stage evaluate --split test_vqdm_unseen \
  --output-dir outputs/clip-vit-b32-cap128-jpeg95
```

注意：测试结果已用于ResNet-50最终报告；CLIP的模型选择与阈值只看train/val，三个测试结果不用于进一步调整。

### CLIP 验证阶段记录（2026-11-02）

训练和验证CLIP特征已缓存，后续重训分类头可复用。冻结编码器使用`openai/clip-vit-base-patch32`，特征维度512，预处理`cap128_jpeg95`。线性头训练到预设500轮上限，最佳点为第500轮（不是patience早停）。验证AUROC 0.98097、AP 0.97702；验证F1阈值0.509909，对应准确率0.93588、精确率0.92205、召回率0.95225、F1 0.93691。此checkpoint和阈值已锁定，下一步对三个测试清单各评估一次。

### CLIP 最终测试（2026-11-02）

同一checkpoint及验证阈值0.509909用于所有测试，没有测试后校准。下表F1与准确率均使用该阈值。

| 测试子集 | AUROC | AP | 准确率 | F1 | 假图召回率 |
|---|---:|---:|---:|---:|---:|
| Seen：全部 | 0.97198 | 0.96362 | 0.92160 | 0.92372 | 0.94940 |
| Seen：ADM vs matched real | 0.97956 | 0.97416 | 0.93247 | 0.93434 | 0.96098 |
| Seen：BigGAN vs matched real | 0.97002 | 0.95703 | 0.92801 | 0.92954 | 0.94961 |
| Seen：SD1.4 vs matched real | 0.96766 | 0.95239 | 0.91932 | 0.92077 | 0.93761 |
| Unseen：GLIDE | 0.91810 | 0.89130 | 0.83400 | 0.82274 | 0.77050 |
| Unseen：VQDM | 0.92860 | 0.89940 | 0.85300 | 0.84783 | 0.81900 |

ResNet-50的同设置AUROC为seen 0.99507、GLIDE 0.98921、VQDM 0.97233。CLIP线性头的seen结果低于ResNet；未见测试中CLIP的VQDM AUROC（0.92860）高于其GLIDE AUROC（0.91810），但两个未见集的表现都低于ResNet，VQDM下的阈值召回/F1也低于ResNet。当前这两种模型都显示VQDM更难检测，CLIP线性检测器总体未超过ResNet-50。报告见`outputs/clip-vit-b32-cap128-jpeg95/test_seen/report.json`、`outputs/clip-vit-b32-cap128-jpeg95/test_glide_unseen/report.json`、`outputs/clip-vit-b32-cap128-jpeg95/test_vqdm_unseen/report.json`。

## VQDM 双模型错误分析

对已报告的ResNet和CLIP逐图预测做配对汇总：VQDM假图中ResNet漏检300张、CLIP漏检362张；双方同时漏检73张，仅ResNet漏检227张，仅CLIP漏检289张。每类假图只有2张，不足以支持稳定的类别难度结论。该分析仅作描述，不据此调整模型或阈值。详细表见`outputs/error_analysis_vqdm/`。

## 演示界面

启动Streamlit应用（GPU节点需有可见GPU，模型使用受控ResNet checkpoint）：

```bash
cd /scratch/jshui/GenImage
streamlit run demo.py --server.address 0.0.0.0 --server.port 8501
```

页面支持上传图片，也支持“内置样例”下拉选择。内置样例包括seen真图、ADM/BigGAN/SD1.4、GLIDE/VQDM正例和漏检例。页面显示AI图像分数、验证集阈值及二分类判断；对内置案例会在判定后显示来源标签。分数未经校准，页面明确提示它只是模型判断。样例清单与说明在`demo_examples/`。需要从个人电脑访问时，可使用学校集群支持的SSH端口转发或Web代理访问GPU节点的8501端口。
