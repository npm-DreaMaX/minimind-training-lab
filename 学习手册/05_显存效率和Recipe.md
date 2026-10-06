# 05 用账本理解显存，用测量理解速度

## 先算必须驻留的训练状态

本次205,623,072参数采用FP32主参数、FP32梯度、FP32 AdamW一阶/二阶moment。理论近似：参数4P字节，梯度4P，m/v共8P，合计16P≈3.064GiB。没有另存一套完整BF16主模型；autocast产生的低精度临时量另算。

这只是下限，不是实际峰值。还要加activation、logits、attention/FLA临时buffer、CUDA context、allocator保留空间、显示程序和其他进程。MoE只激活一个专家也不代表其他专家可以从参数/moment账本消失。

`[16,768,6400]` logits含78,643,200元素，若BF16约150MiB，若FP32约300MiB；CE内部转换、连续化和反向可能额外分配。每层hidden `[16,768,768]` 若BF16约18MiB。不能只数一个hidden就估计所有activation。

## 三种显存数字

| 指标 | 含义 | 常见误判 |
|---|---|---|
| allocated | PyTorch当前活跃分配 | 低于8GB不等于还能随意分配，预算/context/其他进程另算 |
| reserved | allocator向设备保留的内存 | reserved-allocated不是一定可整理成所需连续块 |
| nvidia-smi memory.used | 整张卡或驱动层看到的占用 | 包含桌面等，不能全归因于模型 |

本训练器`cuda_peak_allocated`从每次进程启动时reset一次，是运行中的高水位，不是本步独立峰值。阶段内profiling需在各边界reset/同步，见[显存诊断](../docs/debug/004_omni_memory.md)。

## 参数怎么调，各改变什么

| 调整 | 主要效果 | 是否可能改变实验含义 |
|---|---|---|
| 降physical batch | 通常减少activation | 配合累积可维持样本batch，但MoE aux统计/舍入未必等价 |
| 增gradient accumulation | 较小microbatch形成较大有效batch | 不减少参数/moment；后续microbatch与已存在gradient叠加也会OOM |
| 降sequence length | 减少activation与attention开销 | 改变上下文/监督覆盖，明确是recipe变化 |
| BF16 autocast | 部分算子更省、更快 | 需验证关键梯度；不是把所有状态都变成2字节 |
| activation checkpointing | 保存更少activation，反向重算 | 理想数学目标不变；要检验输出、梯度和dropout RNG |
| optimizer state offload | moment放CPU，更新时搬回 | 本项目候选算法不变但传输/RAM成本高；其他offload实现可能不同 |
| 调LR、数据、轮次、损失权重 | 改变优化路径与目标分布 | 明确是训练配方变化 |
| 改层数/专家/attention/门初始化 | 改变模型或其学习起点 | 不能当作普通工程加速 |

有效样本batch≈B×G×GPU数，token batch取决于有效labels，不能固定等于这个乘积再乘T。当前单卡Hybrid B16×G1；停止的AR预训练B6×G24=144。

## 为什么Hybrid训练很慢

Full Attention有成熟的并行SDPA实现；Gated DeltaNet训练要处理chunk递推、门、矩阵状态和对应反向。本实现线性路径保留FP32，6个线性层还受activation重计算影响。参数数量相近，不等于算子、访存或kernel数量相近。

实际8步资源对照：B16重计算约10.15样本/s，B2累积8无重计算约10.52样本/s，差约3.7%；物理batch也改变了，因此不是纯重计算开关实验。不能据此承诺关掉重计算就快一倍，也没有实测B16不重计算必然OOM。[效率实测](../docs/debug/014_hybrid_budget_and_efficiency.md)

Profile中的kernel累计时间占比不能直接相加成墙钟占比，嵌套与并发会造成差异。GPU utilization高也不证明算子已经达到峰值算力；要看有效token/s、样本长度、kernel、数据等待、功耗温控和峰值显存。

## 真实recipe选择流程

先做数据长度/labels统计 → 理论状态账本 → 完整结构真实batch前后向 → 跨首次AdamW及后续累积峰值 → 数值检查 → 有代表性的短测 → 锁定config → 长训练监测。短测只能估计可执行性和吞吐，不能证明训练效果。

本机为RTX4060 Laptop8GB，WSL中的可用RAM与整机RAM是两回事，历史预检曾仅约7.6GiB WSL RAM。切勿只看Windows剩余内存就无限加worker或offload。环境入口[env/activate.sh](../env/activate.sh)，TMPDIR在Linux `/tmp/minimind-lab`，不要改回D盘。

练习：画出第二个累积microbatch时参数、m/v、第一批gradient和当前activation共存的图；解释为什么首次更新通过不代表后续不会OOM。
