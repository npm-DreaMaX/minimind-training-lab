# 服务器完整Omni的短训练与恢复验证

结果 `result.json`：完整315M的真实T2A训练与确定性恢复通过；不是正式Omni权重，也不证明A2A/I2T或长期稳定性。

本目录从授权服务器 `/new_data/REMOTE_USER/minimind/omni_readiness_v1/project/runs/t2a_resume_and_resource_v1/` 回传。原始config中的远端run路径保持原样，便于重放；`continuous/`、`resumed/`和`resource_B4_recompute/`含源码快照、原始metrics、显存/主机监控、逐记录验证。三个run的随机初始化checkpoint保存在上述远端路径下的`checkpoints/`，没有把十余GB非正式权重重复复制到本机。完整模型状态、优化器、数据位置和Python/NumPy/Torch/CUDA RNG均参加恢复比较。

环境只在本账号 `/new_data/REMOTE_USER/minimind/envs/omni-readiness` 增加覆盖包，Torch2.6.0+cu124和Python基础运行时从已检查的既有环境只读继承；没有修改其他人的环境。`environment_freeze.txt`、根目录 `env/server_omni_readiness_*` 和 `logs/server_omni_readiness_environment_v*.log`保留依赖和失败。它是readiness环境；正式远程长训练前应固定独立运行时，避免外部基础环境后来改变。

第一次soundfile0.13.1安装成功但导入失败，因为本机缺系统libsndfile，pip选择的pure-Python wheel也未附带它。改为提供glibc2.17二进制wheel的0.12.1后导入及真实数据路径通过；[PyPI原始发布文件](https://pypi.org/project/soundfile/0.12.1/)是兼容性依据。该变化只在远端readiness环境，本机0.13.1不变。尚未验证两种音频库对所有A2A资产的解码等价性。基础环境中未使用的sentence-transformers依赖冲突警告保留，不把pip安装成功当作整个基础环境无冲突。

源码/fixture上传与结果压缩包的SHA分别见根目录 `reports/server_omni_readiness_bundle_v1.json`、`reports/server_omni_readiness_t2a_v1.json`。不含密码或其他凭据。全部操作只使用启动时空闲的2号卡；空闲状态是时间点观测，不是预约了服务器。
