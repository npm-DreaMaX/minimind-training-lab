# 03 网络怎样处理信息

![结构与数据流](图表/01_模型和训练数据流.png)

## 从embedding到logits

词表6400，hidden768。ID `[B,T]`查embedding矩阵`[6400,768]`得到`[B,T,768]`。每个block依次进行RMSNorm → attention → residual → RMSNorm → MoE → residual。8层结束后做末端RMSNorm与词表投影，logits为`[B,T,6400]`。embedding与lm_head共享同一参数，不要数两次。

RMSNorm按最后一维的均方根归一化，减少输入尺度波动；残差把模块增量加回原输入，提供信息和梯度传播路径。它们不自动让随机替换的模块保持原行为。

## Full Attention：按内容检索历史

第4、8层为Full。8个Q头、4个KV头，head dim96：Q为`[B,T,8,96]`，K/V为`[B,T,4,96]`，GQA复用KV；Q/K norm与RoPE之后计算因果softmax注意力。

每个query与可见历史key打分，除以sqrt(96)，softmax后加权value，再投影回768。RoPE作用于Q/K的位置关系，不是额外训练token。

训练可调用PyTorch SDPA；高效kernel减少中间量落显存，但数学上仍是softmax Full Attention。推理保存K/V，缓存随上下文长度增长。[源码Attention](../models/02_hybrid_moe/src/model_hybrid.py)

## Gated DeltaNet：维护可更新的压缩记忆

第1、2、3、5、6、7层为Linear。投影q/k/v → kernel4逐通道因果卷积 → SiLU → Q/K L2归一化。q/k/v都是`[B,T,8,96]`；g和beta是`[B,T,8]`，输出门z是`[B,T,8,96]`。

每个head的状态S为`[96,96]`。用列向量记：

```text
S_decay = exp(g_t) * S_previous
prediction = S_decay^T * k_t
innovation = beta_t * (v_t - prediction)
S_t = S_decay + k_t * innovation^T
o_t = S_t^T * (q_t / sqrt(96))
```

beta来自sigmoid，控制纠错写入；g为负的softplus缩放量，exp(g)控制遗忘；输出经过gated RMSNorm和投影回768。它不是简单累计KᵀV：必须减掉记忆对当前key的已有预测，这是delta rule的核心。

单样本单层状态8×96×96 FP32约288KiB，卷积历史另占约27KiB。状态大小不随T增长，但有限状态压缩可能损失检索细节。保留Full层给网络直接读取历史的路径。训练用FLA chunk算子并行，解码用递推，二者应在数值容差内表达同一运算。

这不是KDA。我们的遗忘门每head一个值；KDA细化到通道，Kimi还使用MLA Full层。相似3:1排列不等于相同模型。[已有KDA与迁移分析](../docs/learning/02_hybrid_attention.md)

## FFN MoE：每个token选一个专家

flatten到`[B*T,768]`，router线性投影输出`[B*T,4]`，softmax后Top-1；被分配的token进入对应专家。每个专家是SwiGLU：`down(SiLU(gate(x))*up(x))`，中间维2432，最后回到768。

这是FFN MoE，attention没有专家路由。4个专家都要存权重和optimizer状态，即使一个token只使用其中一个。总参数205M不等于每token都计算205M；一批内不同token可以覆盖所有专家。

aux用于负载均衡：本Top-1实现单层约为`alpha * E * sum(f_i*p_i)`，E=4，alpha=5e-4，f为硬分配比例、p为平均路由概率；8层相加。均衡时单层约0.0005，总约0.004。aux小不等于语言好，也不是越趋近0越好。

作者旧Top-1归一化使权重p/p恒为1，任务梯度消失。本项目同步主线的直通形式：`p - p.detach() + 1`，前向仍为1，反向保留router参数的梯度；该分支对x使用detach，不能声称所有梯度路径与普通softmax完全一样。[真实修复](../docs/debug/003_hybrid_router_and_kernels.md)

## 为什么直接换attention不能保证继承能力

两个模块的外部shape都为`[B,T,768]`，所以程序能运行；内部函数不同，后续专家的输入分布随之改变。我们没有用旧Q/K/V计算行为约束新模块，复制兼容参数仅提供初始化。这次CE先从基座1.31恶化到10.70，再降到1.66，说明转换损失没有充分弥补。

练习：令一个head的旧S为零，给定k、v和beta，手算第一次写入；随后给同一个k、不同v，说明delta项为什么不会简单把两份value相加。
