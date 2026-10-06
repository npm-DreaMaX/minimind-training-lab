# Hybrid 完整两轮结果、失败分析与后续训练

2026-10-06复核。**113,082次更新、735,833,934监督token已完整执行；质量尚未通过。** 这次是官方198M SFT权重转换后的205M Hybrid，全参数训练，不能称为我们的198M从零主线已经完成。

## 训练预算与权重

- 完整205,623,072参数，6层Gated DeltaNet＋2层Full Attention，8层4专家Top-1 FFN；没有缩层/缩专家。官方mini原文件全部预处理，训练904,644条、每条两轮。原始划分、8条零监督过滤和长度768截断统计见[锁定方案](../docs/hybrid_official_mini_2ep_v1.md)。
- 最后一次更新：10月6日14:10:32。完整留出验证、最终恢复点和16题生成套件随后完成；控制器明确记为`pipeline_budget_complete`、`quality_review_pending=true`。
- 从1至113,082步连续，逐步token累加严格等于735,833,934；最后一个batch为4条，也已更新。所有记录的CE/aux/梯度/LR有限。
- 最终恢复状态epoch=2、cursor=0；189个model state键、188组FP32 AdamW状态及四类RNG完整，全部张量有限。54个线性模块tensor键和135个继承tensor键相对初始化发生变化（键数不等于独立参数组数）。这项CPU审计没有声称再次证明逐位恢复等价。
- 第113,082步同时是固定子集最佳点。完整恢复文件约2.47GB，FP16导出约421MB；SHA、文件大小、只占一份数据的硬链接快照均在[最终交付目录](../models/02_hybrid_moe/deliveries/20261006_final_integrity_v1/README.md)。
- 更新的单调计时累计47.58小时；首末更新墙钟跨度71.52小时。后者包含睡眠及已记录时钟异常，不能当纯计算耗时；前者不含评估/保存。睡眠诊断及防空闲请求修补见[debug 015](../docs/debug/015_sleep_pause_and_eta.md)。

## 同一协议的对照

固定512条、191,220监督token，选择SHA完全一致；同一数据collator、B16、T768、assistant右移CE、CUDA BF16 autocast、FP32参数、PyTorch TF32关闭。Hybrid线性模块保持既定FP32/TF32x3路径。官方参考和Hybrid均从FP16导出严格加载。

| 权重与范围 | 验证CE |
|---|---:|
| 官方198M AR SFT，固定512条 | 1.307459 |
| 最终Hybrid FP16导出，固定512条 | 1.664561 |
| 最终Hybrid训练中FP32，固定512条 | 1.664181 |
| 最终Hybrid训练中FP32，全1,066条 | 1.638091 |

导出再加载相对训练中FP32固定CE差约0.000380，远小于和基座之间约0.3571的差距。全1,066条结果与固定512条样本范围不同，不能将1.6381和1.6642之间的差异解释成又发生一次学习。

基座可能已经见过我们的留出记录；该对照用于检查迁移后的能力恢复，不是无污染泛化基准，也不是额外两轮训练预算匹配的架构消融。当前证据支持“Hybrid在这些检查中尚未恢复到基座水平”，不支持“Hybrid架构必然更差”。[实际评估工具](../tools/evaluate_fixed_validation.py)只读权重/留出数据，不更新模型；原始协议和hash见JSON。

## 生成表现与评分限制

同一预注册16题，10题采用严格数字/JSON/名字格式评分，6题保留原文人工判断。标准预算为每题32–160个新增token；另将每题上限统一至少256，作为独立长输出对照。完整保留全部回答，没有只选成功例子。

| 配置 | 严格格式通过 | 自然EOS结束/16 | 到达输出上限/16 |
|---|---:|---:|---:|
| [AR CPU standard](../models/01_moe/runs/official_sft_reference_cpu_v1/outputs.jsonl) | 0/10 | 2 | 14 |
| [AR CPU 256](../models/01_moe/runs/official_sft_reference_cpu_long_v1/outputs.jsonl) | 0/10 | 7 | 9 |
| [AR GPU standard](../models/01_moe/runs/official_sft_reference_gpu_v1/outputs.jsonl) | 0/10 | 1 | 15 |
| [AR GPU 256](../models/01_moe/runs/official_sft_reference_gpu_long_v1/outputs.jsonl) | 0/10 | 8 | 8 |
| [Hybrid CPU standard](../models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/evaluation/final_selected_cpu_probes_v1/outputs.jsonl) | 0/10 | 2 | 14 |
| [Hybrid GPU standard](../models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/evaluation/final_selected_gpu_probes_v1/outputs.jsonl) | 0/10 | 2 | 14 |
| [Hybrid GPU 256](../models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/evaluation/final_selected_gpu_long_probes_v1/outputs.jsonl) | 0/10 | 7 | 9 |

**0/10不等于“所有能力准确率0%”**：即使算出正确数字，附加解释也会违反“只输出数字”的精确格式要求；定性题未混入这个分数。原始回答仍证实了实质性错误：

- 同一GPU标准题，官方参考写出用累加器求偶数和的函数；Hybrid收集并返回偶数列表，没有实现求和。没有执行生成的代码；此判断来自函数原文。
- 官方参考蓝天解释至少提到散射，但仍混淆光谱事实；Hybrid反复说“天空呈现蓝色所以是蓝色”。不能把官方参考也说成全部正确。
- Hybrid加长到256token后，17＋25仍在重复步骤，144÷12明确回答0并EOS结束，多轮猫名字题仍未按要求回答。错误不是统一由32/64token截断造成。
- CPU FP32也出现相同类别的重复、算术/代码错误，因而“只在当前GPU BF16推理坏掉”不足以解释全部问题。这不等于完整排除了CUDA训练数值误差。

## Debug推理与后续决策

现象：teacher-forcing CE持续下降，自由生成仍差。初始假设包括缓存/位置错误、低精度、输出预算、模板/数据格式、六层随机注意力替换后的能力恢复不足。

已完成：早期12,000步CPU cache/full-forward同输入对照已通过（范围与限制见[debug 013](../docs/debug/013_hybrid_transfer_recovery.md)）；本次补齐最终CPU/GPU生成、短/长预算、同提示官方参考、同留出CE、导出权重重载和完整checkpoint审计。错误在多种条件下仍存在，不能靠改解码长度或宣布“loss降了”消除。

尚未确定唯一根因。下一阶段Hybrid研究应聚焦训练数据/思考标签分布、转换后能力保持，以及适应/续训方案的受控对照。更多数据、新模块适应、蒸馏都只能先作为候选；本轮没有偷加第三轮、没有把探针答案加入训练，也没有为了提高分数改变原评分。

依照已经确定的总体计划，完成本轮完整预算和有区分力的诊断后，**恢复独立198M从零预训练主线**，不改变其B6×累积24、T512、全量数据一轮的recipe。从原第9,768步恢复FP32参数、AdamW、RNG和cursor，后续完整SFT接续器仍在等待预训练完成。该训练不是Hybrid追加训练，也不是用来证明Hybrid质量合格；Hybrid仍保留独立质量改进任务。Omni继续等待自训SFT，不直接拿不兼容的Hybrid做初始化。

[预算原始审计](hybrid_final_budget_20261006.json) · [完整评估JSON](hybrid_final_review_20261006.json) · [命令记录](hybrid_followup_commands_20261006.json) · [恢复前核验](ar_resume_precheck_20261006.json) · [198M恢复进程与命令](../runs/pretrain_resume_after_hybrid_20261006.json)
