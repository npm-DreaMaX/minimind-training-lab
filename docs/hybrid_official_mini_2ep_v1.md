# 完整官方mini数据的205M Hybrid训练

2026-10-03。当前主目标是作者Linear扩展的SFT转换路线；旧的大数据实验另行保留。

## 本次锁定的预算

- 架构：205,623,072参数，6层Gated DeltaNet、2层Full Attention，8层均保留4专家Top-1 FFN MoE；全部参数训练。
- 初始化：官方198M `full_sft_768_moe.pth`，固定revision `edba70ec15e06bc4280fbb96ac3383d73a7eab91`。严格迁移187,798,656参数，新增17,824,416参数随机初始化。此分支不是本项目从零完成的基座。
- 数据：官方 `sft_t2t_mini.jsonl` 原文件，1,739,201,170字节，SHA256 `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518`，固定数据revision `312afb4f76391145c6902f765bb51691c09a12f5`。905,718条全部分词，无limit、无短对话筛选、无为了截止时间的去重删减。
- 按内容哈希固定留出1,066条验证，其余904,652条；官方长度768下8条没有assistant监督标签，作为已审计的零目标异常不进入优化。最终训练904,644条，完整两轮。
- 一轮56,541次optimizer更新、367,916,967监督token；两轮**113,082次更新、735,833,934监督token**。无max_steps提前封顶，不在18:00停止。
- 物理batch16、梯度累积1、长度768、peak LR1e-5、AdamW、clip norm1、cosine衰减到0.1倍，warmup0；对应官方SFT CLI默认。每epoch最后4条也训练，没有drop_last。
- BF16外层autocast、FP32主参数/AdamW状态；线性模块保持FP32并用经独立梯度检查的Triton TF32x3。采用激活重计算以适配8GB。PyTorch TF32候选未采用。

## 官方实现与自己的实验改动

作者Linear命令从已有SFT权重继续全参数SFT，未规定额外冻结适应5000步或重跑全量预训练。脚本默认两轮，本次保留两轮。205M是官方MoE结构与作者Hybrid结构的组合；作者贴出的71M Dense日志不是205M MoE的现成效果保证。

### 2026-10-06复核：作者确实做过转换SFT，但示例质量也存在明显问题

[作者Discussion #704的训练与评估](https://github.com/jingyaogong/minimind/discussions/704)给出的单卡命令是：

```bash
python run_linear.py trainer/train_full_sft.py --save_weight sft_linear --from_weight full_sft
```

`run_linear.py`先把`model.model_minimind`替换为线性模型模块，再执行SFT训练器。固定主线`upstream/trainer/trainer_utils.py:init_model`构造目标模型，以`strict=False`加载源权重；对应线性模块没有同名源参数，保留构造时初始化。我们的显式迁移器另外审计允许的新键、丢弃键和shape，血缘见`models/02_hybrid_moe/runs/transfer_official_sft_today_v1/transfer.json`。共享参数已经受过官方训练；新Hybrid整网并没有因此自动完成预训练。

作者公开日志写明71.12M、两轮，示例命令未开启MoE，日志aux为0。公开生成例子中，光合作用回答大量循环且概念错误；Python例子在函数外引用函数局部变量，并伴有错误说明。它展示了实验可以训练和生成，没有提供205M Hybrid MoE超过198M普通MoE的配对证据。我们开启MoE并进行兼容性修复，属于有作者代码依据的扩展实验，不能将其称为官方已经验证效果的205M配方。

本次权重只经历了“官方SFT基座→六层随机注意力替换→两轮全参数SFT”；旧分支1,369步适应没有接入，完整Hybrid预训练也没有执行。“全参数”表示所有参数更新，不表示预训练、SFT等全流程都已经完成。此前反复强调完整两轮，仍不足以清楚传达这个范围，更不能满足用户完整训练且质量合理的最终目标。

当前比基座差的直接证据是同集CE和配对生成；结构替换后的能力损失是有依据的解释，但没有证明“缺一轮预训练”就是唯一根因或补上就一定超过基座。应纠正效果预期并继续受控诊断，不能以作者示例也差为由降低本项目质量要求。此复核不改变已经封存的训练配方，也未启动新的GPU任务。

本项目已有兼容性修复同步主线Top-1路由任务梯度、生成repetition penalty，强制FLA避免静默慢回退；源码/patch均在模型目录。自有训练器补充token计数、验证/生成、梯度/路由/资源指标、原子checkpoint、RNG/cursor恢复。这不是声称逐字节复刻作者训练器。

数据使用官方chat template和assistant labels。缓存保留未截断token，训练时按官方默认取768前缀。5.258%的记录有尾部截断，保留98.797%的监督token；这是官方上下文配置的后果，不是筛掉所有长记录。离线增强只采样一次、各epoch不重新采样，是明确的recipe差异；内容哈希留出集和零目标修复也是差异。完整cache重复记录46,862条未擅自删除。

验证集与本次Hybrid更新隔离，但官方初始化可能已经见过这些语料；不能把该CE称为对官方基座的严格未见测试。还需要独立题目和原始生成判断迁移恢复。

## 已有证据与耗时

相同完整205M、长度768、有效batch16，八步资源检查：B2不重算约10.52条/s，B16重算约10.15条/s，后者峰值allocated4.50GiB。选B16以保留官方物理batch的MoE aux统计。TF32候选约10.65条/s，单层独立输出/梯度通过，但收益小，未引入正式路径；实验patch和报告仍保留。

两轮纯参数更新按短测约49.5小时，额外需要验证、保存、故障/温控余量。不是18:00完成承诺，也不把短测速当长时间稳定性保证。官方3.23小时指3090上198M普通MoE的mini预训练/SFT各一轮，见[耗时口径核对](official_hours_and_our_budget.md)。

完整模型profile显示FLA backward、矩阵乘法和逐元素算子是重要开销；原生depthwise卷积约2%的self-device时间，不能未经证据宣称安装causal-conv1d就提速十倍。profile包含嵌套事件，百分比不能任意相加。原始trace保留在模型run目录。

## 为什么撤回10万条提案

错误：为赶18:00，提出自选10万条完整短对话；它改变数据规模和长度分布，用户明确拒绝。该提案未执行正式训练，仅128条fixture用于八步资源检查。保留配置、视图、日志和撤回记录，不计入本次正式token预算，不从测试权重继续。

旧官方预训练基座的适应实验已在1369步/7,446,025token保存暂停，非本次SFT的前置完成证明。本次从独立、SHA核验的官方SFT迁移重新开始。原14GB数据/计划和自训AR checkpoint仍保留。

## 完成与交付检查

计划 `plans/formal_hybrid_official_mini_2ep_v1.json` 固定源码/配置/来源SHA，校验全部113,082步和至少735,833,934监督token；结束必须有全验证集CE、实际更新后的选定权重和生成套件。预算完成仍不等于质量合格。失败保留现场，不自动降预算。

每500步评估/保存，早期10/100步另存恢复状态，每5000步导出里程碑；完整恢复点包含FP32模型、AdamW、RNG、epoch/cursor。未到结束时只能交付标注进度的研究快照。

## 导航

- [配置](../models/02_hybrid_moe/configs/hybrid_sft_official_mini_2ep_v1.json)
- [数据审计](../reports/sft_official_mini_T768_audit_v1.md)
- [模型与训练记录](../models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/)
- [源码/张量/数学学习](learning/02_hybrid_attention.md)
- [计划控制器](../runs/formal_hybrid_official_mini_2ep_v1/)


## 第500步实际观察

14:53，已完成500/113,082步、8,000条训练读取、3,252,946监督token。逐步计数连续、累计标签数复核一致；固定512条/191,220标签、同一selection SHA下，验证CE从10.703402降到6.103529。grad norm从最初约46降到该步约4.44，均为clip前值；没有NaN或OOM。

三个greedy题目仍重复、答非所问，全部达到64-token上限且没有EOS；原始输出保留，不判为可用。该步只占更新预算约0.44%，支持继续观察恢复，不足以证明收敛。没有据此提前停止、换小数据、加大LR或混入旧适应分支的分数。精确指标与telemetry见 `reports/hybrid_official_mini_step500_v1.json`。
