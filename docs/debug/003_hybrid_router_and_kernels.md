# Hybrid 兼容性：路由梯度与加速核验证

日期：2026-10-03。源码来自作者 Discussion #704；保留原版于 `sources/experimental/linear/`，本地修复版和 diff 在 `models/02_hybrid_moe/src/`。本记录属于长训练前验证，不能当作 Hybrid 已经学会语言。

## 路由器为什么“前向正确”却学不到任务

**现象**：源码检查发现 Top-1 权重采用 `p / sum(p)`；当只选一个专家时就是 `p/p=1`。**初始假设**：语言模型损失对 router 的梯度被归一化消除了，路由只剩辅助负载目标。

**检查**：关掉 aux 系数，用完全相同的参数、输入和输出探针，比较作者版、本地同步修复版、当前官方主线的 MoE。测前向以及 router.weight 的任务梯度，而不是只看 loss 是否非零。

**证据**：三者前向输出逐位一致；路由梯度范数依次为 `2.9068e-7 / 3.449884 / 3.449884`，本地修复与主线梯度逐位一致。原版极小残差是浮点计算的结果，不是可用的学习信号。

**根因及修复**：同步主线的 straight-through 权重 `p - detach(p) + 1`，并与主线一致地在这条额外 gate 路径 detach 输入。前向仍只选一个 FFN 专家；反向保留对 gate 参数的任务信号。

**实验含义**：这是同步主线训练行为的兼容性修复，会改变原作者实验版的反向算法，不能说成单纯加速。没有增加专家、注意力层或参数量。以后看到 Top-1 概率归一化，要单独排查 task loss 和 aux loss 各自的 router 梯度。

## 加速核不能仅凭 import 成功来确认

**假设**：作者 GDN 显式使用 FP32、head dimension 96；当前 FLA 的 API 和数值行为需要实测。作者代码还会捕获异常尝试回退，其 `logging.Logger.warning_once` 在标准 Python logger 上并不存在。

**检查与证据**：`tools/validate_hybrid.py` 独立写逐 token 状态递推，与作者 chunk 参考实现、FLA chunk 的输出、最终状态及 q/k/v/g/beta 五组梯度比较。测试形状为 `[1,33,8,96]`，覆盖不整齐长度和多个参考 chunk。

- 作者 chunk 相对独立递推误差约 `1e-7` 量级。
- FLA FP32 输入路径的输出相对 RMS 误差 `0.001460`，五组梯度 `0.00147～0.00184`，低于预先设置的 `0.005` 检查界限。不能把这个结果描述为逐位相同；加速核内部运算精度与顺序不同，输入 FP32 并不保证每个乘法都按完整 FP32 实现。
- 完整 **205,623,072** 参数 Hybrid 在本机完成 BF16 外层 autocast 下的 forward/backward/AdamW；6 次 FLA chunk 调用得到确认，参考 fallback 被设置为直接报错。
- 全序列与 cache 单 token 推理的 logits 相对 RMS 误差 `0.002184`；6 次 recurrent 核调用得到确认。

**修复**：增加 `require_fla` 训练配置，要求加速核时发生错误就停止并保留 traceback；默认兼容模式保留参考计算，但用有效的 `logger.warning`。这项防护不改变成功路径的数学计算。

**限制**：这些检查验证指定形状和软件组合，不证明任意长度、左 padding、批量变长生成都正确。正式 recipe 仍须使用实际 batch/length 复测。当前缓存对照使用无 padding 单样本；不要外推为支持任意 padding 策略。

原始记录：`logs/hybrid_validation_v1.log`、`logs/hybrid_validation_v2.log`、`models/02_hybrid_moe/runs/implementation_validation/results.jsonl`。源码未修改 FLA kernel。
