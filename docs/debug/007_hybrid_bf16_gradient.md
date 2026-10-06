# Hybrid 精度候选：输出接近，衰减门梯度却没有通过

2026-10-03，`runs/preflight_window_v1` 中的真实失败。**更新：已定位默认TF32 dot精度是FP32 FLA门梯度偏差的主要来源；IEEE与TF32x3独立参考、完整模型更新及精确恢复均通过。两组随机全语料资源检查也已完成，后续选择TF32x3；见 `reports/preflight_window_v7b_review.md`。BF16投影候选尚未采用。** 早期“继续已验证 FP32”的判断不充分，原因和新证据见下文。Full Attention MoE 不使用此内核，不受影响。

## 现象与证据

候选保持完整模型结构、FP32 master parameters，将 Gated DeltaNet 的大投影、卷积输出、q/k/v 改为 BF16，衰减门投影、归一化计算和 recurrent state 仍为 FP32。CPU dtype/finite 检查通过，默认 FP32 路径与修改前实现的输出和全部梯度逐位相同。

GPU 检查采用单个完整宽度层，hidden=768、8 heads、head_dim=96，输入 `[2,257,768]`，跨越 chunk 边界；两条路径复制同一组参数、使用同一输入和标量目标。预先设定输出 relative RMS <2%、每个参数梯度 <3%，并检查缓存推理。没有在看到结果后放宽阈值。

| 对象 | relative RMS 差异 | 判断 |
|---|---:|---|
| 输出 | 0.974% | 通过 |
| 输入梯度 | 1.138% | 通过 |
| qkv/z/out/conv 参数梯度 | 约0.97%～1.10% | 通过 |
| `dt_bias` 梯度 | 72.39% | 未通过 |
| `A_log` 梯度 | 68.52% | 未通过 |
| `in_proj_a.weight` 梯度 | 14.54% | 未通过 |
| 缓存逐 token vs 并行输出 | 0.860% | 通过 |
| 缓存最终 state | 0.552% | 通过；state 为 FP32 |

原始记录：`models/02_hybrid_moe/runs/precision_gpu_v1/results.json`；traceback：`runs/preflight_window_v1/hybrid_precision.log`；检查源码：`tools/validate_hybrid_precision_gpu.py`。模型文件 SHA 为 `cf83fb3f054b3dc9fc1da055a68cbc21ec973a6f63b1615ebc66c761bee8c503`。

## 怎样分析

这里的 relative RMS 是 `RMS(candidate-reference) / RMS(reference)`，不是“每个参数都错了72%”。衰减梯度可能很小，必须同时看绝对误差、梯度尺度和不同种子；本次 `dt_bias` 最大绝对差约1.31e-5，`A_log` 约2.45e-5。即便绝对值小，也不能忽略，因为 AdamW 的 moment 归一化会影响梯度变化如何进入参数更新。

假设包括：低精度 q/k/v 的扰动经过衰减导数或抵消放大；BF16 加速核的 backward 与 FP32 路径差异；参考 FP32 路径本身的数值误差。这些还没有被区分。控制门保持 FP32，只能排除“我们直接把门投影也改成 BF16”这一解释，不能保证门的上游梯度是高精度的。

下一项有判别力的检查是：相同层/输入/目标下加入独立 FP32 递推参考，将“投影已经舍入，但 recurrence 输入转回 FP32”与“BF16 recurrence”分开比较，记录每个梯度的绝对尺度、相对误差和一次 AdamW 更新，再跨种子确认。不能单看 forward loss、finite 或训练不报错。

## 当前处理与实验含义

GPU 验证调度器跳过候选的完整训练/恢复测试，继续 FP32 基线和 Omni 检查，随后接回原预训练。候选源码和失败结果保留，未更改正式 MoE 模型或训练预算。代价是暂时没有获得预期的 Hybrid 精度加速；收益是避免把未解释的 backward 差异带入长训练。

本案仍在诊断中，没有声称“已修复”。以后遇到输出接近但训练表现不同，应逐参数检查梯度，尤其检查状态递推、衰减门和归一化，而不是只看整网平均误差掩盖少数关键参数。

## 第三轮：参考路径本身也要验证

`tools/diagnose_hybrid_precision_gpu.py` 对两个初始化种子、完整宽度单层 `[1,129,768]` 加入独立逐 token 状态递推。参数、输入、目标相同，比较结果在 `models/02_hybrid_moe/runs/precision_diagnosis_v2/results.json`。这次命令返回0仅表示诊断完成，不表示所有候选通过。

| 独立FP32递推 vs FP32 FLA | 种子20261007 | 种子20261008 |
|---|---:|---:|
| 输出 relative RMS | 0.0820% | 0.0838% |
| 输入梯度 relative RMS | 0.344% | 0.422% |
| dt_bias 梯度 relative RMS | 121.2% | 399.9% |
| A_log 梯度 relative RMS | 152.5% | 656.6% |
| in_proj_a 梯度 relative RMS | 18.43% | 17.00% |

种子08的 dt_bias 参考梯度RMS为2.90e-6，误差RMS为1.16e-5；A_log对应3.31e-6和2.17e-5。BF16投影＋独立FP32递推相比原FP32递推，两个门的误差分别为7.29%和9.52%，明显小于FP32 FLA路径的误差。因此第一次仅用FP32 FLA作为真值，不能把差异直接归因于BF16候选。

当前假设：Triton `tl.dot` 在NVIDIA上默认使用TF32，PyTorch的 `allow_tf32=False` 不能据此保证独立Triton算子的精度。实际FLA的 `ops/common/chunk_o.py` 多处dot未指定input_precision，而衰减梯度含两个求和结果相减，低精度误差可能被抵消放大。作者初始化产生较强负衰减，早期温和随机门的测试不足以覆盖它。**这仍是待验证假设，尚未宣布根因。**

下一步用FP64自动微分和有限差分核验独立递推；在隔离进程指定 `TRITON_F32_DEFAULT=ieee` / `tf32x3` 重做相同层对照。先验证数值，再测速度和全模型恢复，不改初始化、不放宽原阈值来制造通过。

依据：[Triton dot精度文档](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)、[Triton官方环境变量说明](https://github.com/triton-lang/triton)。本机安装版本的 `triton/knobs.py` 和 `backends/nvidia/compiler.py` 也已核对。显式指定TF32的其他算子不受默认值覆盖，因此设置环境变量后仍须实际比较。

## 第四轮：验证假设并修正内核精度

首先对独立递推做FP64自动微分、FP32对照和g的中心有限差分；既覆盖温和/强衰减合成输入，也逐head重放从真实完整宽度层捕获的kernel输入与上游adjoint。参考检查通过，结果在 `runs/reference_fp64_cpu_v1/`、`runs/captured_tf32_fp64_v1/`、`runs/captured_ieee_fp64_v1/`。这一步排除了“只因我们参考计算本身低精度”这一解释。

隔离进程只改变 `TRITON_F32_DEFAULT`，保持同一层、参数、输入、目标、PyTorch精度开关。两个种子都重测；下表为种子20261008，数值均为relative RMS百分比。

| Triton dot路径 | 输出 | dt_bias梯度 | A_log梯度 | in_proj_a梯度 |
|---|---:|---:|---:|---:|
| tf32 | 0.0950% | 399.85% | 656.57% | 17.002% |
| ieee | 0.000170% | 0.0971% | 0.1632% | 0.00543% |
| tf32x3 | 0.000171% | 0.1515% | 0.2477% | 0.00767% |

两个较高精度路径均满足原定输出<2%、各参数梯度<3%的门槛，没有事后放宽。`reports/hybrid_triton_precision_v3.json` 与三个 `runs/triton_*_v3/` 保留全部参数指标和源码。首次JIT/autotune分别耗时约100/295/166秒，不能当作每一步训练速度；单层一次暖运行也不足以判断哪个设置更快。

机制上，FLA backward里的 `sum(dq*q)-sum(dk*k)` 等操作可能抵消掉主要项。强负g下真实遗忘门梯度很小，TF32 dot舍入残差会占主导。解析检查也可辅助：初始state为0时，第一个时间步的g不影响输出，其真实梯度为0；默认TF32捕获到约1e-8数量级残差。故“tensor.dtype是float32”不等于所有内部乘法具有IEEE FP32精度。

修复候选只控制内核计算精度，不改模型结构、初始化或数学目标。自有训练器新增配置 `triton_f32_default`，运行记录中明确保存生效值；原始FLA checkout未改。准备完整205M真实数据更新、吞吐、cache与恢复验证后选择正式设置。BF16大投影候选仍未采用，其相对独立参考的舍入差异必须另行评估，不能把旧的错误FP32参考作为通过证据。

另发现作者 `dt_bias=1` 与FLA层的log-uniform dt及inverse-softplus初始化不同，使初始衰减较强。这不是本次悄悄修改的内容：默认初始化保留；若研究标准初始化，应作为独立recipe消融，而不是混进数值修复。

## 完整模型精度复测

IEEE与TF32x3各自完成完整205M模型10次真实数据更新；IEEE还通过1536长度SFT短测和完整模型/AdamW逐位恢复。汇总 `reports/hybrid_full_precision_v4.json`。同一短预训练fixture中稳态每步中位数0.453秒与0.344秒，TF32x3更快；尚不能把该短测提升直接视为正式全语料收益。短测初始化来自早期AR，验证CE约8，不是已经恢复语言能力。
