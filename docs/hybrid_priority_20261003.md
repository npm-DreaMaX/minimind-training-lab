# 205M Hybrid 优先执行：完整预算与今天的截止时间

> 状态更新14:33：本文保留旧扩展方案的决策历史；控制器已暂停，不代表当前仍在执行。当前官方原版两轮SFT路线和子集撤回记录见根README。

> 13:55口径纠正：这里的5000步适应＋全量续训＋全量SFT是自有研究预算，不是作者规定的必需流程。[官方3.23小时的条件与作者Linear训练路线](official_hours_and_our_budget.md)已重新核对。文中多天估算仅对本配置成立，不能推广成205M必须训练这么久。

2026-10-03 用户要求优先训练完整 205M Hybrid，18:00 前需要结果，同时明确「不能偷工减料，该训练的一点不能少」。这两个要求不能被解释成缩减数据后宣称训练完成。今天保存真实的阶段权重、验证、生成和资源记录；正式预算没有因为截止时间而减少，到 18:00 不自动停止。

## 调整与代价

此前本机正在从零训练 198M AR，Hybrid 排队等待其完整预训练和 SFT。13:33 左右给本项目 AR 主进程发送 SIGTERM，在 optimizer 边界第 9,768 步保存 FP32 参数、AdamW、RNG、epoch/cursor：累计 319,137,649 个有效监督 token，完整恢复文件 2,381,236,747 字节。原等待式 Hybrid 控制器也已暂停。没有终止外部任务，没有删除训练证据。

新分支改用已经下载并核验的官方 198M **预训练**权重作为初始化，避免等待尚未训练完的自有 AR。源文件 `data/reference_models/minimind_3_moe/pretrain_768_moe.pth`，revision `edba70ec15e06bc4280fbb96ac3383d73a7eab91`，SHA256 `ccc0542230cb0dd936521bf18b38940fa3dbd3ed4d3f1b1ecb8b8dac71bbeedf`。本轮再次读取全部字节校验。它不是我们的从零训练结果，也不是把官方 SFT 权重改名交付。

**实验含义改变的是初始化来源和执行顺序。** 自有 AR → Hybrid 的原方案保留，新分支不能直接与旧初始化的 AR 对照归因比较。公平比较需要相同官方基座和额外训练预算，后续另设对照；当前优先把 Hybrid 正式训练推进。完整模型结构、数据、损失、数值精度与三个阶段预算不变。

## 执行预算

| 阶段 | 更新的参数 | batch × 累积 / sequence | optimizer steps | 数据预算 |
|---|---|---|---:|---|
| 迁移与初始化 | 保留兼容 AR，初始化 6 层线性注意力 | CPU 严格键/shape 校验 | 无 | 不算训练 |
| 线性模块适应 | 新增的 17,824,416 参数 | 4 × 6 / 512 | 5,000 | 完整预训练集合的固定 seed 随机顺序，120,000 次样本读取 |
| 全参数续训 | 完整 205,623,072 参数 | 4 × 6 / 512 | 352,511 | 8,460,241 条训练记录完整一轮，约 19.20 亿监督 token |
| 全参数 SFT | 完整 205,623,072 参数 | 4 × 4 / 1536，activation checkpointing | 318,877 | 5,102,028 条有效训练记录完整一轮，约 30.50 亿监督 token |

保持 6 层 Gated DeltaNet＋2 层 Full Attention，各层 4 专家 Top-1 FFN。FP32 参数及 AdamW，BF16 autocast；线性模块保持通过数值检验的 FP32 路径与 Triton TF32x3。不启用未通过门梯度检查的 BF16 线性投影候选。适应阶段保留原 B4×6；改变 microbatch 会改变 MoE aux 的统计，不能仅凭相同 effective batch 称为严格等价。

此前真实完整模型短测分别约 20.75、16.96、5.13 样本/s。纯更新外推：适应约 1.6 小时、全量续训约 5.8 天、全量 SFT 约 11.5 天；验证、checkpoint、温控和故障另计。这些是估计而非期限保证，正式日志将提供更可靠的吞吐。18:00 前无法完整完成这套预算；即使某份 checkpoint 能生成文字，也不等于预算完成或质量达标。

## 实时证据与恢复

- [新执行计划与哈希锁定](../plans/formal_hybrid_official_priority_v1.json)：只包含本轮优先的三个 Hybrid 阶段，后阶段自动检查前阶段预算和验证。
- [控制器原始日志](../logs/hybrid_official_priority_v1.log)与[控制器状态](../runs/formal_hybrid_official_priority_v1/status.json)。失败会留下原始日志并停住，不会自行减模型/减预算。
- [迁移逐键报告](../models/02_hybrid_moe/runs/transfer_official_pretrain_v1/transfer.json)。新模块不是从 AR 获得的训练权重，必须经历适应。
- [适应运行](../models/02_hybrid_moe/runs/hybrid_adapt_official_full_v1/)、[全参数续训](../models/02_hybrid_moe/runs/hybrid_continue_official_full_v1/)、[SFT](../models/02_hybrid_moe/runs/hybrid_sft_official_full_v1/)；后两目录在对应阶段真正启动后创建。
- 每个运行保存 config、源码快照、provenance、metrics.jsonl、gpu.csv、evaluation、plots 和 checkpoints。`latest_resume.pt` 是精确恢复状态；`best_validation.pth` 是固定验证集选出的 FP16 推理/阶段初始化权重；两者用途不同。step 0 的 best 还没有训练，不能当作已训练交付。
- [暂停与切换证据](../runs/hybrid_priority_20261003/priority_change.json)。原 AR 第 9,768 步的恢复状态仍在原目录；GPU 让回后用原配置加 `--resume`，不可同时起第二份本机 GPU 训练。
- 18:00 自动捕获脚本已启动，日志在 `logs/hybrid_deadline_1800_v1.log`，计划保存到 `models/02_hybrid_moe/deliveries/20261003_1800/`。目录会在实际捕获时创建。它将固定当时最新完整恢复文件的inode，再导出全模型参数和对应步数的已有评估；记录SHA，不暂停训练、不减少预算、不宣称完成。源码为 `tools/capture_hybrid_deadline.py`。

新控制器意外退出后，先检查状态与原始失败原因，再恢复：

```bash
cd /mnt/d/minimind
source env/activate.sh
python -u -B tools/run_formal_pipeline.py --plan plans/formal_hybrid_official_priority_v1.json --resume
```

不要重复启动已存活的控制器。阶段 budget complete 仍要单独检查生成与能力；不是最终质量批准。

## 这次应学会判断的边界

迁移保留兼容参数但替换 6 层注意力，因此初始 loss 和生成可能明显变差。观察固定验证 CE 如何恢复、新模块 grad norm、非有限数、路由偏置及生成重复；不能用训练前基座质量代替迁移后质量。线性模块适应结束也不能替代全参数训练。按时提供中间结果与宣称完整训练结束是不同的承诺。

训练阶段与数据预算是可检验的实验设计，并不存在一个脱离数据质量和验证曲线、保证「训练够了」的固定轮数。这里先兑现既定完整预算，再依据验证、生成和错误分析决定是否需要后续调整；不能把跑满一轮直接解释成模型充分收敛。

本轮已遇到的具体学习案例：[迁移后初始CE接近10及能力恢复诊断](debug/013_hybrid_transfer_recovery.md)。不能用“权重加载成功”代替模型能力验证。
