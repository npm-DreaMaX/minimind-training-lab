# RTX 4060 Laptop 与服务器 RTX A4000：小规模实测

## 结论

两台机器都能训练完整的 198M MiniMind MoE。服务器的 2、7 号卡已经实际完成 forward、backward、AdamW 更新，2 号卡还通过严格恢复复测。当前短测中，本机小 batch 更快；服务器显存和系统内存更充裕。

因此先让 198M 主线继续使用本机，服务器用于 Omni、较长序列、多模型后训练等更吃显存的实验。服务器这些更复杂模型还未逐项验证，不能视为已保证能跑。

## 控制条件与结果

使用原始 `upstream/model/model_minimind.py`，主线 commit `f659b55761b754d306bd140573493a6543cafd7f`，文件 SHA256 `548f1203282dccb3be43227b11463f469b976afbab614df52828bdd246c3786f`。

模型：8 层、hidden 768、4 个 FFN 专家、Top-1，198,416,640 参数。权重/梯度/AdamW 状态 FP32，forward 使用 BF16 autocast。AdamW lr=1e-4，foreach=False，grad clip=1。未编译、未 checkpointing、未 offload。

输入为固定 seed 的随机合法 token，`input_ids` 与 `labels` 形状 `[B,512]`；logits `[B,512,6400]`。模型内部 shift 一位，故每步监督 token 数为 `B*511`。两台的输入 hash、初始参数抽样 hash 和源码 hash 一致。

| 条件 | 本机 RTX 4060 Laptop 8GB | 服务器 RTX A4000 16GB，GPU 2 |
|---|---:|---:|
| B=1，预热 2 步、测量 5 步 | 3,608 input tokens/s | 2,698 input tokens/s |
| B=4，预热 3 步、测量 12 步 | 12,660 input tokens/s | 10,586 input tokens/s |
| B=4，中位每步时间 | 0.1610 秒 | 0.1940 秒 |
| B=4，PyTorch peak allocated | 约 3.69GiB | 约 3.70GiB |
| B=4，PyTorch peak reserved | 约 4.16GiB | 约 4.18GiB |
| 模型参数字节数 | 793,666,560 | 793,666,560 |
| 梯度字节数 | 793,666,560 | 793,666,560 |
| 优化器 tensor 字节数 | 1,587,333,800 | 1,587,333,800 |

服务器 GPU 7 的 B=1 测试也通过，约 2,647 input tokens/s。两张卡测试完成后释放，未停止其他人的任务。

这些显存数值是进程内 PyTorch 统计，不包含全部 CUDA context、显示程序及其他进程占用。`gpu.csv` 保存了 nvidia-smi 视角的显存、利用率、温度和功耗，不能与 allocated 直接混为一谈。

## 已验证与未验证

已验证：完整模型能上卡；loss/aux loss 有限；梯度有限；路由梯度非零；参数确实更新；BF16 可用；完整 FP32 模型及 AdamW 状态可以保存和重新加载。

未验证：真实数据管线吞吐、长期训练效果、最大 batch、OOM 边界、多卡 DDP、Hybrid/Omni/dLM 各自算子与训练过程、长时间散热和可用时段。本次没有启动正式长训练。

速度是当前整套环境的测量：本机 PyTorch 2.11.0+cu128/Python 3.10.12，服务器 2.6.0+cu124/Python 3.12.14；两边 transformers 4.57.6。CPU、驱动和 CUDA 不同，不能据此断言 4060 在所有任务都比 A4000 快。短测也不是持续负载吞吐保证。

## Debug：恢复后的 loss 相同，为什么参数 hash 不同？

**现象：**本机及 GPU 2 第一轮都完成了有限梯度和参数更新，但保存 checkpoint、继续一步，与重新加载再做同一步的 model/optimizer hash 不一致。下一步 loss 完全相同，不能仅据 loss 宣布严格恢复成功。

**初始假设：**漏保存优化器或 RNG；保存精度损失；模型加载不完整；GPU 算子非确定性。

**检查：**测试工具保存完整 FP32 模型、AdamW、CPU/CUDA RNG，strict=True 加载；增加“更新前保存/恢复所有 tensor hash 比较”，同时设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`torch.use_deterministic_algorithms(True)`，仅复测少量步骤。

**证据：**本机和 GPU 2 的确定性复测都通过五项比较：恢复前后模型逐位相同、优化器逐位相同、下一步 loss 相同、更新后的全部模型 tensor 相同、更新后的全部优化器 tensor 相同。

**判断：**结果支持 GPU 计算非确定性导致第一轮严格 hash 比较失败；复测没有发现模型/优化器保存遗漏。尚未逐算子定位，因此不把根因具体归咎于某一个 kernel。

**修复含义与代价：**模型结构、数学目标未改；调试时使用确定性计算可提高复现性，但可能降低速度、改变浮点运算路径。不能把确定性复测的速度混入普通模式速度表，也不据此断言正式训练必须总开启确定性。

**以后快速识别：**分别检查“刚加载是否一致”和“下一步计算是否一致”；保留优化器、RNG、数据位置、scheduler。loss 相同不是所有参数相同的证明。这里验证的是本项目测试工具的恢复，不是已验证官方训练器的完整 resume/data/scheduler 行为。

## 文件与复现

- [当前测试工具](../tools/hardware_smoke.py)
- [原始 v1 工具](../models/01_moe/runs/20261003_hardware/hardware_smoke_v1.py)
- [本机 B4 结果](../models/01_moe/runs/20261003_hardware/local_b4/result.json)
- [服务器 B4 结果](../models/01_moe/runs/20261003_hardware/server/gpu2_b4/result.json)
- [本机确定性复测](../models/01_moe/runs/20261003_hardware/local_deterministic/result.json)
- [服务器确定性复测](../models/01_moe/runs/20261003_hardware/server/gpu2_deterministic/result.json)
- 每个运行目录的 `stdout.log`、`steps.jsonl`、`gpu.csv` 是原始证据，`result.json` 包含配置、版本、源码 hash 与起止时间。
- 本机 checkpoint：`models/01_moe/runs/20261003_hardware/{local_b1,local_deterministic}/smoke_resume.pt`。
- 服务器 checkpoint：`/new_data/REMOTE_USER/minimind/hardware_test/results/{gpu2_b1,gpu2_deterministic}/smoke_resume.pt`，每份约 2.38GB，未作为正式训练成果。

服务器基础环境和网络检查另见 [配置审计](system/server_audit_20261003.json)。系统盘已满，测试环境和临时目录均放 `/new_data/REMOTE_USER/minimind`。临时环境只读复用已有 PyTorch，并在自己的 venv 固定 transformers/tokenizers/hub；正式训练应再建立独立可复现环境。
