# 共享数据与权重导航

学习请先读[一条真实对话到batch/labels](../学习手册/02_数据到Batch.md)，再做[样本解码练习](../学习手册/10_实操练习和答案.md)。下载了某个数据集不代表每个模型都训练过它；本次已完成Hybrid只用了表中的官方mini SFT。

数据仅在此公共目录保存一份。各模型配置引用它，不复制到每个模型文件夹。下载都固定远端 revision，并校验官方文件大小与 SHA256；清单在 `manifests/`。

| 用途 | 原始文件规模 | 当前训练读取位置 | 审计 |
|---|---:|---|---|
| 文本预训练 | 8.275GB JSONL，8,468,827条 | `processed/pretrain_t2t_full_v1/` | `../reports/pretrain_data_audit.md` |
| 官方mini SFT（当前Hybrid） | 1.739GB JSONL，905,718条 | `processed/sft_t2t_mini_full_v1/` | `../reports/sft_official_mini_T768_audit_v1.md` |
| 扩展文本 SFT | 14.096GB JSONL，5,109,432条 | `processed/sft_t2t_full_v1/` | `../reports/sft_data_audit.md` |
| Omni T2A | 5.353GB Parquet，1,248,923条 | `processed/sft_t2a_shards_v1/` | `../reports/omni_t2a_full_data_audit.json` |
| Omni A2A | 5.754GB Parquet，414,024条 | `processed/sft_a2a_shards_v1/` | `../reports/omni_a2a_audio_audit_v1/` |
| Omni I2T | 4.935GB Parquet，2,904,511条 | `processed/sft_i2t_shards_v1/` | `../reports/omni_sft_i2t_full_data_audit.json` |

大小使用十进制GB。原始下载保留在 `raw/`；训练缓存的大小不等于原始文件大小。文本缓存保存未截断token，读取时按阶段构造与mask。Omni分片保存有界Arrow表，音频code无损压到uint16；I2T原图像字典展开后占21.67GB，不能据4.93GB下载大小估算全部磁盘需求。

Omni每个分片目录的 `index.json` 标记数据来源、schema、分片行数；`audit_v1/` 保存原始按conversation hash划分。I2T正式训练另用 `split_image_v2/`，按图像和相同对话的连通分量划分，避免同一图像跨训练/验证。`visual_train.npy`、`visual_val.npy` 供投影对齐；完整 `train.npy`、`val.npy` 还保留纯文本混合记录。图像与占位符逐行审计数组分别在 `image_audit_v1/`、`placeholder_audit_v1/`。

A2A另有 `evaluation_audio_unseen_v2/val.npy`：原451条验证中，排除31条使用训练侧相同音频资产的记录和61条无音频记录，留下359条、237个未见输入资产。它是额外的语音评估分组，不重分训练集，也不删除旧验证。跨语言SFT→T2A→A2A→I2T视觉的完整conversation hash重叠检查在 `../reports/omni_multistage_overlap_v1/`；已检查组合均为0，仍不代表语义去污染。

冻结外部模型在 `frozen_models/`，来源及校验见 `manifests/omni_frozen_models.json`。官方已训练语言权重在 `reference_models/`，用于独立质量校准以及明确标注来源的Hybrid迁移分支；我们198M正式预训练仍从随机初始化开始。自己的正式/实验checkpoint保存在 `../models/<模型>/runs/<运行>/checkpoints/`。

`../artifacts/pipeline_validation/` 的均匀抽样视图只改行索引，大型不可变缓存通过hardlink共享；它们是硬件/训练器短测数据，不能代替正式全量训练预算。

当前Hybrid使用官方mini原文件完整两轮，未用10万条自选子集。`processed/sft_complete_100k_T768_v1/`只保留被否决提案的证据，不是正式数据预算。

官方mini的完整cache文件SHA和8条零目标位置清单：`manifests/sft_mini_token_cache_v1.json`。原始对话及未截断token仍全部保留，训练数据中的重复对话没有为了提速擅自删去。
