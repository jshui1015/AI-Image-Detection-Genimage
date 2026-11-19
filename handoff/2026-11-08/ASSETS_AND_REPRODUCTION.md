
## 最新诊断复现（2026-11-19）

在项目根目录使用已有环境运行。学校模块应先加载，再获取 Conda 的 base 路径。`myenv` 为本项目现有环境名，队友可替换为自己安装依赖的环境。

```bash
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
cd "/scratch/$USER/GenImage"
python scripts/diagnose_score_shift.py
python scripts/diagnose_threshold_transfer.py
python scripts/analyze_cocoxgen_cells.py
```

以上离线分析读取 `handoff/2026-11-08/evidence` 的逐图预测，输出位于 `outputs/score_diagnosis` 与 `outputs/threshold_transfer`。分数图需要 Matplotlib，其余诊断主要依赖 NumPy。已归档结果见 `handoff/2026-11-16/`。

分辨率评估使用完整原图、PyTorch 与 release 的四份完整权重，CPU 数组作业覆盖一个 ResNet 基线及三个 V2，各评估三测试集的 original、low64、low32。提交前通过 Git LFS 取得模型权重。

```bash
git lfs pull
mkdir -p logs
sbatch --partition=amd scripts/run_resolution_cpu.sbatch
squeue -u "$USER"
```

本校CPU分区为amd、intel。脚本通过Slurm分配8核、16GB内存，6线程推理与2个数据加载worker。线程变量在作业内设置。输出为 `outputs/resolution_control/数组作业编号/模型名/`。作业2311461四任务均COMPLETED、退出码0:0，耗时26–28分钟。每模型9行指标、9个逐图NPZ、18张输入样例PNG和协议JSON已归档。每个NPZ的dimensions依次为原始宽高、降采样宽高。

重新运行会更新outputs的同名分析产物，归档材料有独立SHA256SUMS。CPU分辨率original应作为本轮处理差值的参照，历史GPU混合精度结果可能有轻微数值差异。

> 文档状态更新于 2026-11-19。分数诊断、36 组分辨率评估、COCOXGEN 阈值适配与生成配置分析已完成，见[最新实验补充报告](../2026-11-16/README.md)。九模型主实验结果见 [三种子主报告](FINAL_REPORT.md)。历史章节中的待办以当前交接说明为准。

# 资产与复现索引

## 写PPT和论文无需服务器

本目录包含完整四批机器可读证据、8张PNG/PDF、CSV、可离线打开的FINAL_REPORT.html、方法及结果草稿、真实CIFAKE错误图。仓库根目录保留原代码、固定清单、原报告和11月3日归档。只需克隆/下载这些材料即可完成报告写作，不必下载全量图像或重新运行模型。

## 权重、阈值、编码器

- 九模型准确服务器路径、SHA256与验证阈值见 [MODEL_INDEX.csv](MODEL_INDEX.csv)，原机器锁定记录见 evidence/cifake/locked_models.json。
- 原阶段权重在仓库 `release/GenImage_Final_2026-11-03/models/`，由Git LFS管理；完整下载需Git LFS。它们不是全部新增模型的备份。
- 新ResNet两seed及CLIP补充头仍在tyangbj服务器scratch路径。本次没有取得/上传这些权重；它们的预测与哈希已提供。写作不依赖权重，重新推理/长期存档前须备份；不能宣称GitHub已保存全部九模型checkpoint。
- CLIP的.pt只含分类头，还需 `openai/clip-vit-base-patch32` 编码器；本轮记录revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，详见evidence/followup/encoder.json。历史特征来源仅有兼容性抽查证据。
- 温度见evidence/phase2/temperatures.json和evidence/followup/temperatures.json；阈值须随温度作相同单调变换，不能直接沿用数值后宣称决策保持不变。

## 数据与运行位置

| 资产 | 位置/来源 |
|---|---|
| 共享GenImage原图和固定清单 | `/project/mscaiehpc42026/AI-Image-Detection/data`（需学校组读取权限） |
| 队友环境记录 | `/project/mscaiehpc42026/AI-Image-Detection/environment` |
| tyangbj代码工作目录 | `/scratch/tyangbj/AI-Image-Detection` |
| tyangbj环境 | `/home/tyangbj/.conda/envs/genimage` |
| 第一批结果 | 工作目录`outputs/completion-batch-2218770` |
| 第二批结果 | 工作目录`outputs/phase2-20261108-144532` |
| CIFAKE结果 | 工作目录`outputs/cifake-external-20261108-154531` |
| 补充CI与COCOXGEN | 工作目录`outputs/followup-20261108-DVbELg` |

服务器路径是追溯信息，不会自动授予队友访问权限；不要把别人用户名路径直接替换后假定文件存在。原始全量图片未上传GitHub。CIFAKE固定commit、COCOXGEN固定revision、采集URL与哈希见对应evidence清单/lock文件。新COCOXGEN和内部GenImage错误例提供索引，只有CIFAKE11张错误图实际包含于本包。

## 本地重建分析（不使用GPU）

使用独立Python环境，安装NumPy、Matplotlib、Pillow。环境安装以团队已有环境为准，完整服务器版本保留在evidence各批requirements/environment文件；不是要求在Mac安装Linux CUDA依赖。

在本交接目录运行：

```bash
python scripts/build_results.py
python scripts/make_cases.py
python scripts/add_followup.py
python scripts/write_report.py
```

这些命令重写本目录生成的表格/图和实验报告，不改变服务器旧实验。make_cases.py在图片缺失时可能需要网络。新文件生成后SHA256SUMS须重新计算。核验结果分别见audit.json与followup_audit.json。

## 服务器原始运行脚本与版本

`submission_scripts/`保留四批提交脚本；`evidence/`保留实际作业结果侧代码、配置、环境和日志。以结果侧锁定配置/记录为证据，提交脚本不是一键跨账户复现工具：其中包含原项目路径、依赖批次及输出目录。重跑前需检查路径、空间、数据权限及缓存，并使用新输出目录；当前不需重跑。

基础代码commit为 `8919f28e5b679f3a322cb7896175afcb56b485a8`。已逐一核验train.py、clip_detector.py、evaluate.py、evaluate_robustness_holdout.py与服务器锁定SHA256一致，见SOURCE_VERIFICATION.json；没有声称每个历史开发版本都与本提交相同。

## 完成状态与边界

实验指标、图表与写作证据齐备。用户明确取消本轮demo验收，demo保留原状；不存在本轮页面通过截图或视频。尚待团队确认：正式作者/课程格式/引用、PPT和论文、全量权重长期备份。不要把scratch当永久归档。本包不含密码、私钥或访问令牌。
