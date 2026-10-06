# 205M Hybrid Attention＋MoE

**学习入口：**[十章实操手册](../../学习手册/README.md) · [只读使用最终模型](../../学习手册/10_实操练习和答案.md) · [图表与逐步记录](../../学习手册/06_曲线和状态字典.md)。以下含历史试验记录；本轮正式两轮SFT已结束，没有Hybrid预训练，也没有启动下一轮。

6层 Gated DeltaNet＋2层 Full Attention，每层保留4专家Top-1 FFN。与 Full Attention 分支比较时控制额外数据预算。

## 当前状态

**2026-10-06：完整两轮已结束，113,082步、735,833,934监督token；评估结果与生成样例已记录。** [最终结果与基座对照](../../reports/hybrid_final_review_20261006.md)、[最终权重/恢复状态](deliveries/20261006_final_integrity_v1/README.md)。CPU/GPU及加长生成仍存在重复和实质错误；后续改进独立研究，未机械追加第三轮。

2026-10-03 14:39：作者SFT转换路线已正式开始，完整205M、官方mini原文件、完整两轮全参数SFT。全文件SHA/分词/标签审计完成，固定113,082次更新、735,833,934监督token。**10万条自选子集已撤回，未用于正式训练。** [预算和实现差异](../../docs/hybrid_official_mini_2ep_v1.md)。

当前：[训练状态](runs/hybrid_sft_official_mini_2ep_v1/status.json)、[曲线](runs/hybrid_sft_official_mini_2ep_v1/plots/training.png)、[评估](runs/hybrid_sft_official_mini_2ep_v1/evaluation/)、[checkpoint](runs/hybrid_sft_official_mini_2ep_v1/checkpoints/)、[正式配置](configs/hybrid_sft_official_mini_2ep_v1.json)。18:00只捕获中间研究快照，不停止两轮预算。

新初始化位于 `runs/transfer_official_sft_today_v1/`，来源是官方198M SFT权重；严格保留187,798,656参数，新建17,824,416线性模块参数。资源检查选B16、T768、累积1、激活重计算，完整模型短测峰值allocated约4.50GiB。它们是工程预检，不是正式质量结果。

旧官方预训练基座适应分支已在1,369步、7,446,025有效token处保存暂停；5,000步适应→全量预训练→14GB SFT是此前自加研究路线，不能说成作者强制流程。旧checkpoint、配置、日志与原自训AR分支均保留。[原决策历史](../../docs/hybrid_priority_20261003.md)。

作者代码已归档；已同步主线 Top-1 路由任务梯度与 repetition penalty 修复。真实初始化检查发现FP32 FLA衰减门梯度偏差，现已定位默认TF32 dot精度为主要来源；IEEE/TF32x3独立参考检查通过，完整205M模型更新已通过，IEEE精确恢复也已通过。早期成功不能覆盖新的失败输入，全部证据保留。新增 `require_fla` 防止正式训练静默回退；旧适应分支已经保存暂停；当前执行官方SFT基座的新分支。

当前新增**未选定的精度候选**：`linear_precision=bfloat16`只将qkv、z、输出投影与卷积结果改为BF16，控制门、归一化、主参数保持FP32；FP32默认路径不变且已与修改前代码逐项比较输出/全部梯度，完全一致。BF16候选已进行GPU数值检查：输出误差约0.97%，但dt_bias/A_log梯度差异约69%～72%，未通过预设检查，暂不进入长训练；[失败与待检验假设](../../docs/debug/007_hybrid_bf16_gradient.md)已保留。不能把旧FP32测试当作它的证据。变更链是`author_compatibility.patch`之后再应用`bf16_projection_experiment.patch`。

学习入口：[真实首批input/labels/右移](learning/official_mini_first_batch_v1/README.md)、[Hybrid 的数据流与状态递推](../../docs/learning/02_hybrid_attention.md)、[版本漂移与数值验证](../../docs/debug/003_hybrid_router_and_kernels.md)。原始测试在本目录 `runs/implementation_validation/`。

完整权重迁移的CPU预检已通过，结果在`runs/transfer_preflight_early_ar/transfer.json`：保留187,798,656个参数，替换6层Full Attention的10,617,984个参数，新建17,824,416个Gated DeltaNet参数。此目录使用的是早期AR权重，只验证迁移，不是正式Hybrid训练。原自有AR分支仍等待其完整预算；本次官方基座分支独立迁移到`runs/transfer_official_pretrain_v1/`。不能从预检报告模型质量。


## 从哪里开始读

- 源码：`models/02_hybrid_moe/src/model_hybrid.py`。来源与commit见根目录 `sources/provenance.json`。
- 配置：本目录 `configs/`。每次运行在 `runs/<run_id>/` 独立保存记录。
- 公共数据：根目录 `data/raw/` 和 `data/processed/`，不在模型目录重复复制。
- 先读根目录 `docs/learning/01_training_step.md`，再对照该模型 forward、loss、优化器与生成路径。
- 全部实验的阶段、控制变量和完成口径见根目录 `docs/experiment_plan.md`。

完整精度预检：`runs/full_pretrain_ieee_v4/`、`full_pretrain_tf32x3_v4/`、`full_sft_ieee_v4/` 和 `resume_ieee_v4/`。同一短预训练样本与batch下，预热后每步中位数IEEE约0.453秒、TF32x3约0.344秒；只可比较此短测的内核成本，不可推算完整语料的有效token吞吐或质量。汇总在根 `reports/hybrid_full_precision_v4.json`。

TF32x3的完整205M精确恢复也已通过（`runs/resume_tf32x3_v5/`）。两组全语料随机视图资源检查完成：后续全参数预训练选B4×累积6、T512、不重计算；新增线性模块适应选B8×3；SFT选B4×4、T1536、重计算。根据完整模型实测选择，见根 `reports/preflight_window_v7b_review.md`；当前实际正式配置与预算见上方优先分支，学习率和恢复情况用正式验证曲线持续检查。

BF16诊断的解释也已更新：最初约69%～72%的门梯度差异用了后来发现有误的FP32参考，不能全部归因于BF16。独立参考下BF16投影仍有约7%～9.5%的门梯度差异，未通过原3%门槛，因此仍不采用。完整因果链见debug007。

旧分支第一项正式里程碑（当前已暂停）：官方基座分支适应500步，2,712,906个有效token，相同512条验证CE从9.997485降至6.078842。生成仍重复、答非所问，未达可用标准。[初次独立快照](deliveries/20261003_first_500/README.md)已核验完整205M形状与真实新模块更新；该分支现已暂停，等待后续研究决策。[诊断记录](../../docs/debug/013_hybrid_transfer_recovery.md)。
