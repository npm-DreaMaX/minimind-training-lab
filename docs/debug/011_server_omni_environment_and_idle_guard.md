# 服务器Omni环境与空闲卡检查的实际失败

本机正式198M继续训练，以下都是授权服务器的有界准备。原始日志分别在根目录 `logs/server_omni_readiness_environment_v*.log`、`server_omni_input_environment_v*.log`、`server_omni_real_inputs_gpu_v*.log`；模型、fixture、配置和运行证据回传到 `models/03_omni_moe/runs/server_readiness_t2a_v1/` 与 `server_readiness_inputs_v1/`。

## pip成功不等于运行环境可用

soundfile0.13.1安装成功，实际import却报 `libsndfile.so` 不存在。服务器是glibc2.27，所选pure-Python wheel不自带底层库。检查原始发布文件后改用包含glibc2.17 wheel的0.12.1，import和真实数据路径通过。没有使用sudo改系统，也没有改本机soundfile；音频库版本差异仍保留。[PyPI发布文件](https://pypi.org/project/soundfile/0.12.1/)支持这个兼容选择。

第二个失败来自把运行时显示值当成pip元数据版本：`torch.__version__`为`2.6.0+cu124`，但继承环境的dist-info是`2.6.0`。约束`torch==2.6.0+cu124`导致默认PyPI解析失败。改成元数据约束`torch==2.6.0`，另用运行时`torch.version.cuda==12.4`、实际GPU更新和恢复检查验证CUDA，配套torchaudio2.6.0+cu124、torchvision0.21.0+cu124来自PyTorch官方索引。[官方版本配对](https://docs.pytorch.org/get-started/previous-versions/)是依赖依据。

环境安装只写本账号 `/new_data/REMOTE_USER/minimind/envs/omni-readiness`，缓存、TMP、CUDA/Numba/Torch目录全部显式放在 `/new_data`。基础Python/Torch仍只读继承既有环境，所以这是readiness环境；正式长期运行前须固定可独立复现的基础运行时。没有把系统盘只剩33MiB的问题藏掉，也没有声称pip freeze能替代完整依赖验证。

## 前一任务结束，GPU utilization为什么还不是0

第一版顺序测试在T2A、A2A全参数、audio projector均通过后退出，准备视觉检查时看到：GPU2已占用3MiB、空闲16105MiB，但utilization仍为48%。控制器严格要求单次读数等于0，因此拒绝启动下一项。它没有终止任何其他人的任务，模型测试本身并未因此失败。

随后检查该卡无计算PID且utilization为0。初步解释是最近采样窗口仍覆盖了刚结束的训练；NVIDIA文档明确utilization反映过去一个采样周期内kernel活跃时间，而非“此刻是否有进程”。这与观察吻合，但原第一次快照未同时保存计算PID，所以不宣称已排除那一瞬间所有外部活动。[NVIDIA指标定义](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)

修复为最多20秒的有界观测：显存低占用、足够空闲、utilization=0、该UUID无计算PID，连续两次才进入下一项。每次观测记录在events中；仍不满足就保留失败并退出，不驱逐其他工作。判定分支验证涵盖空闲、残留利用率、已有进程、已占显存和剩余显存不足。剩余I2T全参数与vision projector在v2完成，前三项不重复；v1仍保留为中途退出的原始实验。

这种观测不是资源预约，启动前检查也无法彻底消除共享服务器的竞争。真正长期任务还要持续监测，并记录当时可用资源，不能把一次空闲当成永久独占。
