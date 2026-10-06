# 第二课：Full Attention、Gated DeltaNet 与 MoE 如何组合

这份文档对应作者 Hybrid 实现与本项目的兼容性修复。先读 `models/02_hybrid_moe/src/model_hybrid.py` 的 `GatedDeltaNet`、`MiniMindBlock`、`MOEFeedForward`，再读 `docs/debug/003_hybrid_router_and_kernels.md` 看它们如何被验证。

## 一个 block 的输入与输出

输入 hidden states 是 `[B,T,768]`。RMSNorm → attention → residual → RMSNorm → FFN/MoE → residual，输出仍为 `[B,T,768]`。相同外部 shape 让两类 attention 可以出现在不同 block；MoE 替换的是 FFN，因此与 attention 的种类是两个维度。

本实验 8 层排列为 `Linear, Linear, Linear, Full, Linear, Linear, Linear, Full`。每层都有 4 个 FFN 专家，Top-1 路由。它是真实的跨层 Hybrid，同时仍然只是 FFN MoE；没有“attention 专家路由”。

## Full Attention 保存 token 历史

Full 层采用 Q/K norm、RoPE、GQA。8 个 Q heads、4 个 KV heads，head dimension 96。训练时每个 query 与此前 key 做相似度计算，经 softmax 加权 value；解码时每层 cache 的 K 和 V 分别是 `[B,T,4,96]`，随上下文长度增长。

Flash/SDPA 可减少显式 materialize 的注意力矩阵，但不使注意力算法变成线性复杂度。不要把 kernel 优化与更换注意力定义混为一谈。

## Linear 层保存压缩状态

输入投影得到 q/k/v，经过 kernel size 4 的 depthwise causal convolution 与激活；q、k 做 L2 normalization，8 heads 的 q/k/v shape 都是 `[B,T,8,96]`。另有 beta 更新门、g 遗忘门和输出门 z。

用列向量表示，每个 head 维护矩阵 `S`，shape `[96,96]`：

```
S_decay = exp(g_t) * S_previous
prediction = k_t^T * S_decay
innovation = beta_t * (v_t - prediction)
S_t = S_decay + k_t * innovation^T
o_t = (q_t / sqrt(96))^T * S_t
```

它学习的是对历史关联的压缩更新，不是把 softmax attention 直接删除。输出还要经过 gated RMSNorm 和 output projection。整层最后恢复 `[B,T,768]`。

单样本单 Linear 层的递推状态 `[1,8,96,96]` 是 73,728 个 FP32 数，约 288KiB；卷积状态另有 `[1,2304,3]`，约27KiB。这是推理状态，不等于训练显存：反向需要保存中间量，chunk 算法也有临时 buffer。训练仍要测 peak allocated，不能根据“小 cache”推断“大 batch 一定放得下”。

## 为什么保留 Full 层

Linear 状态大小不随上下文长度增长，但压缩会改变可表达的历史交互。Full 层保留直接检索 token 历史的途径。组合的效果不是定义上的必然优势，要用相同起点、相同数据与预算的 Full Attention 对照验证。2026-10-06完整两轮后的参考评估显示，本次Hybrid尚未恢复到官方AR基座水平；这不是同预算的架构优劣定论，详细证据见文末。

## 数学目标与最重要的观测

Hybrid 仍然是 AR next-token CE＋FFN MoE aux loss；它不是 dLM，也没有变成双向去噪。改 attention 会改变模型函数，移植旧权重后新增模块需要适应，不能把 missing keys 静默吞掉后直接称为无损转换。

具体说，Full层和Linear层都能接收、返回`[B,T,768]`，这使程序接口可以连接，却没有证明两者对相同输入产生相同输出。只复制名字和shape兼容的权重，不会让新状态递推自动学会旧attention的行为；下游FFN和路由器的输入也随之变化。因此这种做法可以作为结构迁移实验，但不是保持原模型功能的直接替换。本次没有用教师输出/隐藏表示蒸馏约束新模块，也没有证明转换保留能力；两轮SFT后的质量失败必须单独承认，不能用“作者也这样写”替代验证。

要记录：哪些参数加载成功、哪些 attention 被替换、新模块数值/梯度范围、验证 CE 恢复速度、router 分配、吞吐、显存，以及 cache 与 full-forward 是否一致。参数规模从 198,416,640 变为 205,623,072，不代表推理或训练一定更快。

## 本次真正遇到的问题

原作者 Top-1 gate 的归一化消除了任务梯度，已经通过与主线的数值对照修复。FLA 当前版本能执行 FP32、96 维的作者路径，但与独立递推有可测量浮点误差；已验证输出和梯度，没有只检查 import。完整模型前向反向与精确恢复检查通过；2026-10-03严格迁移官方SFT基座，10月6日完成完整mini两轮和参考评估，生成仍需分析具体评估与生成表现。

## 初始化也会改变这个系统如何学习

作者令 `dt_bias=1`，并令 `exp(A_log)` 从0到16均匀采样；衰减为 `g=-exp(A_log)*softplus(in_proj_a(x)+dt_bias)`。固定版FLA层源码采用另一种dt初始化：先在0.001到0.1间按log-uniform取dt，再用inverse-softplus得到bias。它不是一个可以无声同步的拼写修复，而是训练recipe选择。

`models/02_hybrid_moe/learning/initial_decay_v1/` 使用真实迁移预检checkpoint里的A与门投影，在相同单位尺度高斯输入上比较这两种bias。仅按直接衰减项计算的 `1/mean(-g)`，作者初始化48个head约0.047～0.974个位置，中位约0.089；另一dt分布约0.622～403，中位约14.35。图在 `direct_decay.png`。所有权重只读，没有将候选bias写回，也没有据此更改正式模型。

这个量**不是实测有效上下文长度**：delta更新、输入分布、学到的门、卷积和两层Full Attention都会影响实际记忆。它说明为什么某些初始遗忘门梯度很小，以及为什么“模型名字相同”仍可能对应不同学习动力学。若以后比较标准dt初始化，应独立保留同权重、同数据、同预算的作者初始化对照，检查数值梯度、CE恢复与长上下文任务；不能凭这张曲线宣布新初始化质量更好。


## 用当前真实run贯穿整条数据流

当前入口是 `models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/`。原始905,718条完整分词，经内容hash划分和零监督审计，形成904,644条训练记录、1,066条验证；不是抽取10万条。`lab/data.py`读取不截断的共享cache，在batch内按官方768长度取前缀并padding，`labels=-100`排除用户/padding位置。

一次更新输入 `[16,768]`，embedding后 `[16,768,768]`；6个线性层的q/k/v为 `[16,768,8,96]`、g/beta为 `[16,768,8]`。两类attention的输出都恢复hidden shape，因此之后四专家Top-1 FFN不需要另一种外部接口。最终logits `[16,768,6400]` 对齐右移labels，以assistant有效标签均值计算next-token CE，再加各层路由aux。

两轮共113,082次更新。源参数187,798,656个被继承，新线性模块17,824,416个随机初始化，但这一次**两者全部训练**；对比此前冻结适应实验，需要区分trainable范围，不能只看总参数量。权重迁移报告、实际provenance、metrics和checkpoint可交叉核对。初始CE很高和最初生成损坏没有被隐藏，完整训练结束与质量恢复是两个检查。

## 为什么loss下降，生成却仍可能失败

验证CE在给定真实前文时预测下一个token；自由生成会把自己的错误继续作为输入。较低CE不保证短问题答对，也不能证明推理缓存正确。实际第12,000步权重的[CPU缓存对照](../../models/02_hybrid_moe/runs/cache_diagnosis_trained_cpu_v1/report.json)在相同序列上比较缓存推理和完整forward的`[1,T,6400]` logits：3个问题和跨128位置边界样例的argmax全部一致，最大相对RMS约6.94e-7，但输出仍重复/错误。这说明应分别验证数值实现与生成质量；该CPU结果不能自动外推到GPU FLA/BF16内核。完整假设、证据和未排除因素见[debug记录](../debug/013_hybrid_transfer_recovery.md)。

## 为什么这次没有超过官方MoE：Hybrid、KDA与训练起点

2026-10-06根据真实源码、完整两轮评估和Kimi一手资料复核。选择Hybrid是因为它有重要学习价值；不能把这种选择解释成“同样训练几轮必然更强”。先前规划对结构替换后的能力恢复风险说明不够，后续方案必须把恢复到基座水平设为实测目标。

| 对比项 | 本项目205M Hybrid | Kimi Linear |
|---|---|---|
| 线性模块 | Gated DeltaNet | Kimi Delta Attention（KDA） |
| 遗忘门 | 每个head一个标量，g为`[B,T,H]` | 每个head的通道分别控制遗忘 |
| 全注意力模块 | GQA＋Q/K norm＋RoPE | MLA |
| 层间比例 | 3个Linear后接1个Full，共8层 | 同为3:1的交替设计 |

本地`GatedDeltaNet.in_proj_a`输出`num_v_heads`个值，调用`fla.ops.gated_delta_rule`；没有调用KDA算子。`Attention`保存普通K/V缓存，没有MLA的潜在表示压缩。因此“采用相似Hybrid布局”成立，“复现了同一个模型”不成立。Kimi论文把KDA、Gated DeltaNet Hybrid和Full MLA分别作对照，使用相同1.4T token预训练配方；其优势来自实际对照，不能只由Hybrid标签推出。[Kimi技术报告§3–5](https://arxiv.org/html/2510.26692v1) · [官方仓库](https://github.com/MoonshotAI/Kimi-Linear)。这些大模型训练量不是本项目必须照搬的门槛。

本次初始化的实际操作是：从已训练官方198M SFT模型继承187,798,656参数，把8层中的6层Full Attention替换为Gated DeltaNet，17,824,416个新参数随机初始化，再用mini SFT全参数训练两轮。没有先完成这套Hybrid的预训练或独立适应阶段；旧预训练基座适应分支的1,369步没有接入本次权重。约91.33%的参数数值被继承，不意味着91.33%的语言能力被保留：注意力改变后，原FFN和路由器接收到的hidden states也会改变。

作者确实公开过从`full_sft`直接转换并继续SFT的实验；但展示的是71.12M Dense Hybrid，其原文生成也有严重重复、概念和代码错误，不能当作205M MoE效果已验证的依据。[命令、加载语义、原文质量与本项目范围复核](../hybrid_official_mini_2ep_v1.md)。**全参数SFT**只说明全部参数参与更新，不等于**预训练→SFT完整流程**；继承已预训练参数也不等于新架构整网已经预训练完成。

| 同一固定512条验证 | CE，越低越好 |
|---|---:|
| 官方AR参考，FP16导出后加载为FP32 | 1.307459 |
| 刚替换后的Hybrid，FP32 | 10.703402 |
| 两轮后Hybrid，FP32训练状态 | 1.664181 |
| 两轮后Hybrid，FP16导出后加载为FP32 | 1.664561 |

初始与最终使用同一selection SHA及191,220标签。导出重载差约0.00038，远小于最终与参考的差距；但官方参考可能见过留出数据，本表不能当作干净的泛化benchmark。[初始评估](../../models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/evaluation/step_0000000.json) · [迁移报告](../../models/02_hybrid_moe/runs/transfer_official_sft_today_v1/transfer.json) · [最终完整比较](../../reports/hybrid_final_review_20261006.md)。

由此支持的判断是：转换明显破坏了原模型函数，后续训练恢复了大量预测能力，但当前配方尚未恢复到基座水平。两轮确实执行了735,833,934个监督token；这是同一批数据的两次遍历，不是同量新语料，也不能自动等价于新架构的预训练。SFT只监督assistant标签，与预训练的覆盖目标不同。

尚未证明的部分包括：随机新模块是否需要与继承模块不同的学习率/适应顺序；作者遗忘门初始化是否限制恢复；数据和思考标签分布造成多少重复；是否仍有未覆盖的训练数值问题。本轮统一peak LR为1e-5，是已执行的配方，不是已经证明的最优值。“提高LR”“再加轮数”“直接换KDA”目前都没有效果保证。

后续要分开两个问题。能力恢复实验固定基座、架构、数据、token预算与评估，只改变一个初始化或适应方案，观察固定CE、生成、门值和梯度；如引入教师logits/隐藏表示蒸馏，必须记为目标函数变化。架构优劣实验则让Full与Hybrid经过可比较的预训练/SFT，报告数据token、活跃参数/计算量和墙钟，不能把同token与同算力混为一谈。长上下文检索与cache收益另外评估，T768短对话结果不能代替长上下文结论。

本节是解释与后续实验原则；没有改模型、修改当前recipe或启动追加训练。独立198M预训练仍按原计划推进。
