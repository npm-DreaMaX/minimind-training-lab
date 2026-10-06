# 第七课：相同参数如何变成不同训练目标——Dense AR 与 dLM

这是前三个正式目标推进期间的CPU源码检查，不占训练GPU，也没有开始第四、第五个模型的正式训练。实际执行源码与结果位于 `models/05_dlm/runs/author_contract_cpu_v1/`；作者模型和训练脚本保留在 `sources/experimental/dlm/`，当前主线来自 `upstream/model/model_minimind.py`。

## 保留什么，改变什么

768宽、8层Dense模型实际为63,912,192个参数。作者dLM继承当前主线的embedding、RMSNorm、GQA、RoPE与SwiGLU，把每层attention的 `is_causal` 改为False，并改变loss和generate。完整AR state_dict严格加载成功，没有缺key或shape变化。这说明结构权重兼容，不说明加载后已经拥有去噪生成能力。

作者复用token ID27，即tokenizer中的 `<|buffer1|>` 作为mask；它不是另行扩展词表添加的新参数。Dense对应无FFN专家；作者配置虽允许MoE，但forward没有把主干算出的aux并入返回loss，因此不能直接认定其MoE训练目标与我们的AR CE＋aux相同。前三条主线不依赖这段dLM代码。

实际修改第6个输入token时，AR第1位置的logits完全不变；dLM第1位置最大变化1.22668。将被修改位置用attention mask屏蔽后，dLM对应差异恢复为0。这个测试直接验证未来可见性与padding/key mask作用，不能用类名里的“Diffusion”代替此验证。

## 数据如何变成去噪目标

实际读取两条SFT held-out记录，原始行号298、372，裁到128位置：

| 张量 | shape | 含义 |
|---|---|---|
| X / Y | `[2,128]` | 原token及原SFT监督标记，Y的-100标记不监督的位置 |
| eligible | `[2,128]` | `Y != -100`，共157个可监督位置 |
| t | `[2]` | 每条序列的随机噪声程度；检查时固定为0.25、0.75 |
| p_mask | `[2,128]` | `epsilon+(1-epsilon)*t`，每行广播 |
| corruption_mask | `[2,128]` | 随机抽中的可监督位置，本次87个 |
| noisy_X | `[2,128]` | 仅这些位置换成mask ID；prompt仍保持原样 |
| logits | `[2,128,6400]` | 每个位置对原token的去噪预测 |

这里模型没有独立的t embedding；噪声程度通过mask后的输入体现，p还进入loss权重。原始prompt不是目标，PAD也不参与噪声和损失。官方SFT模板与监督边界依然重要，否则模型会被要求重建user提示而非回答。

AR目标用位置i预测位置i+1。dLM对位置i的原token做去噪，所以labels取原X、**同位置对齐，不再shift**：

\[
L=\frac{1}{N_{valid}}\sum_{b,i\in M}\frac{-\log p_\theta(X_{b,i}\mid \widetilde X_b)}{p_{mask,b,i}}.
\]

`M`是实际mask位置，`N_valid`是全部符合监督条件的位置数，而不是实际mask个数。本次p分别约0.25075和0.75025，算出的loss为9.00501，与手工逐项计算完全一致。非mask位置的logits梯度为0，第一层Q投影梯度norm为2.30180。非mask token仍可通过双向上下文影响其他位置的预测，因此不能把“其logits没有直接loss梯度”误解为“其hidden state/embedding完全不参与反向”。

`1/p`权重补偿不同mask采样概率，但输入条件也随mask变化，不能把它简单称为普通AR CE的无偏估计。该标量也不能直接取exp后与AR perplexity比较。epsilon避免p等于0；很小的p仍会带来较大的单token权重和梯度方差，需要看mask数、p分布、loss及grad norm。

## backward、显存与恢复

参数形状兼容意味着仍可用常规autograd和AdamW。按FP32参数、FP32梯度、两份FP32 Adam moment估算，63.912M模型约需1.023GB十进制基础存储，另加activations、logits、attention临时量与allocator；它不是实测VRAM峰值。此次CPU同时持有AR与dLM并执行反向，进程峰值RSS约1.67GiB，不能把它直接当作GPU预算。

双向训练的每次forward仍处理整段。屏蔽padding可能使当前Attention走显式attention矩阵分支，长序列尤其要profiling；不能仅因模型参数少就假设运行更快。正式预算与GPU配置尚未确定。

恢复必须保存随机t、随机mask对应的RNG状态，或者保证每个样本/step的确定性分配。只恢复模型、optimizer和行号，若噪声抽样序列不同，就不是同一更新轨迹。原作者trainer还有其自己的恢复/梯度累积逻辑，正式训练前要做连续与中断对照，而不是沿用AR测试结论。

## 生成与实际失败反例

生成先在prompt后放满mask，反复双向forward，选择高置信位置填入预测token；作者也支持分块处理。位置会被同时改写，因此普通AR的旧KV cache不能无论证地跨去噪迭代复用。

检查保留两个实际反例：

- 所有位置都无监督、显式传 `n_valid=0` 时，作者loss得到NaN。正式数据管线须识别并记录这类样本，明确skip/过滤规则，不能把NaN归罪于GPU精度。
- 用受控logits始终偏好mask token，4次迭代后仍剩4个mask。作者采样没有阻止“把mask重新填成mask”，也没有最终未填完的失败检查。这是代码边界反例，不是在报告一个已训练模型的发生率。正式生成需要明确控制token约束与结束条件，并保留修改diff。

后续Dense AR→dLM实验应共享来源权重、tokenizer和数据划分，分别记录AR next-token CE、去噪目标、实际mask比例、每个输出的迭代数、残留mask、生成长度、耗时与任务结果。Q/K-only是可选的结构适应消融，不能把它与全参数去噪续训混为同一预算。
