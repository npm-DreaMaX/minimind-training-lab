> 历史快照：2026-10-03总计划，含后来暂停/更正的路线；当前计划见 [experiment_plan.md](experiment_plan.md)。

# 三个正式目标与学习路线

更新时间：2026-10-03。当前任务是持续完成前三个模型，不在硬件小测或流程冒烟后宣告完成。

**已落地的当前配置与接续方式以[正式执行v1](formal_execution_v1.md)为准**：Hybrid及同预算对照已排队；Omni七阶段配置已生成，独立服务器环境与完整数据部署有专门的前置检查及回传桥。下文的早期候选/预检过程保留为历史背景，不表示还要无限重复预检。实时状态见根目录`reports/project_status.md`。

## 完成的口径

完整模型结构、计划的数据预算、正确的训练/恢复、验证与生成评估、可回溯的记录，缺一不可。跑完一个预算不等于达到最佳收敛，更不等于达到主流大模型能力。训练失败、显著不收敛或不能解释的异常必须先诊断，不能自动进入下一阶段。

## 主线与分支

1. **198M Full Attention+MoE**：官方完整预训练语料的一轮从零训练，随后完整 SFT 语料的一轮。完整统计完成后报告精确记录数与有效 token 数。正式预训练采用已通过真实数据验证的 sequence 512、microbatch 6、累积 24、BF16 autocast/FP32 参数、AdamW；最初384候选及其测试结果仍保留。一轮后根据验证与生成结果决定是否需要继续。SFT 首先评估1536长度，避免默认768大量截掉回答；batch与累积另测。
2. **205M Hybrid+MoE**：从①保存的兼容权重分支。对照作者 Discussion #704 与现主线，补齐已修复的路由梯度等版本差异，先验证 Gated DeltaNet 的参考实现与加速实现，再做迁移、新增模块适应、全参数续训与 SFT。恢复预算通过实际 loss 恢复曲线决定，不把固定几步微调算完整复现。保留①在同等额外 token 预算上的 Full Attention 对照，记录吞吐、显存、损失及生成差异。
3. **315M Omni-MoE**：从①语言主干分支，遵循 Thinker–Talker 路线。固定版本 `trainer/train.sh` 的 full MoE 建议包含七阶段：T2A全参数→音频投影→A2A全参数→视觉投影→I2T全参数→A2A低学习率回训→视觉投影再对齐。此前只根据README概述把后三项排成“自有消融”不准确，现补回正式整体计划；来源、epoch及纠错在 `docs/recipe_decisions.md`。冻结外部编码器属于官方设定，不能以只训投影替代主体训练。各阶段保留权重并反复做语言/语音/视觉回归，以观察遗忘和恢复。先测本机，如果无法合理执行，才使用服务器空闲16GB卡。

Hybrid 与 Omni 分别研究语言主干结构与多模态系统，不把它们强行拼成一个未经验证的模型。Dense AR/dLM、Preference、GRPO/CISPO、Agentic RL 保留独立目录，作为后续对照和训练范式研究。

## 当前自有 recipe 与官方脚本的区别

- 官方模型结构不变。正式训练器是 `lab/train.py`，不是冒充原始 CLI 的输出。
- CE 按 accumulation group 的有效标签数加权；原脚本平均每个 microbatch 的均值，变长样本时二者并不严格相同。我们的目标是让每个受监督 token 权重一致。
- aux loss 仍按 microbatch 平均。MoE 的负载统计依赖 microbatch，因此梯度累积不等于所有算法意义上的更大物理 batch。
- 采用 100 optimizer steps 线性 warmup＋cosine decay，预训练初始候选峰值 3e-4。原脚本没有这个 warmup；这是稳定性导向的 recipe 选择，需要从梯度范数与验证曲线验证，不标为纯工程等价。
- token cache 保留未截断文本。预训练读取时按官方 BOS/text/EOS 和前缀截断规则；SFT 采用官方模板与标签解析。SFT 随机系统提示/空 think 处理离线抽样一次，不在每一 epoch 重新抽样，属于明确的数据增强策略差异。
- 内容 hash 固定 0.1% 验证集，相同内容落在同一侧，避免完全重复文本跨 split。尚未做近重复语义去重，不能把这称为完全无污染的能力评测。
- 保存 FP32 恢复状态，另存 FP16 里程碑导出。官方 helper 的恢复模型保存为 FP16，不能假设与连续 FP32 训练逐位一致。

## 记录与保留

每个 `runs/<run_id>/` 包含 config、源码副本、版本与 hash、metrics.jsonl、gpu.csv、evaluation、checkpoint_progress、status 与 failure。完整 `latest_resume.pt` 原子替换用于恢复，里程碑 FP16 权重另外保留，不能以“每一步都留一份完整 AdamW”耗尽磁盘。每一次保存的时间、位置和训练 token 数都记录。

正式预训练新增第10、100步早期保存；每次保存更新本地 PNG 曲线。固定验证子集上的最佳模型保存在 `best_validation.pth`，不含优化器；全验证集用于最终报告。SFT pipeline 已用1536长度真实数据通过，但正式 SFT 配置还要结合完整语料长度统计，不能根据文件开头的一小段推断全量分布。

SFT衔接任务已排队，见 `tools/continue_language_pipeline.py`。当前配置采用microbatch2、累积8，有效样本batch16、峰值LR1e-5与固定版本官方SFT默认相同；1536上下文、100步warmup和本轮先训练一epoch是本项目recipe（官方默认长度768、epoch2）。正式启动前检查全量数据、零标签截断比例、预训练完成预算和初始化权重SHA。新阶段加载FP16导出到FP32参数，并重建AdamW，这不是声称与预训练无缝resume。后续是否加epoch由实际验证和生成决定。

数据下载、预处理与正式训练分别记录状态。下载采用固定 revision、大小和 SHA256 三重约束。已校验的分段缓存可以清理；失败模型实验不删除。

## 学习顺序

真实一条样本 → tokenizer → `[B,T]` ids/labels → mask/shift → `[B,T,V]` logits → CE/aux → backward → gradient clipping → AdamW → scheduler → validation → checkpoint/resume → 显存与吞吐 → 对照实验 → 多模态。

每一个关键知识点都结合此次实际源码、张量形状、数学目标、资源消耗、观测指标与故障记录讲解；文档随着真实实验补充，不先制造大量空章节。

## 已执行检查后的衔接

`runs/preflight_window_v5/` 的七项短验证全部完成，主线从3502步恢复。第六组资源预检：Hybrid从完整语料均匀抽样比较TF32x3、重计算/物理batch和冻结新增模块；Omni比较较少DataLoader预取的A2A、B2 T2A及B16视觉投影。它们只决定recipe，不计入前三模型正式完成。当前实际窗口为 `runs/preflight_window_v6c/`，调度在 `runs/preflight_window_v6c_scheduler/`。原v6因挂载盘状态读取窗口失败（debug010）；修复后的v6b原订5001步，已在尚未触发时提前为4710步之后的完整更新边界，以尽快验证新增的逐记录评估和后续配置。原调度及替换原因保留，正式样本/token预算、优化器与数据顺序不变。

正式Pretrain与SFT完成后，`tools/evaluate_completed_stages.py` 会独立用CPU保存固定生成探针，和官方参考使用同一协议。它不会因为输出文件成功产生就宣布质量合格；原始输出仍须分析。该进程不占GPU，SFT衔接仍由 `tools/continue_language_pipeline.py` 管理。
