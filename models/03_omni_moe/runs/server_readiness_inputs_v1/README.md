# 服务器真实音频/视觉输入的完整Omni预检

五条路径均已实际更新，模型315M保持完整，冻结SenseVoice/SigLIP使用本机已有、固定来源且逐文件SHA校验的权重。结果为根目录 `reports/server_omni_real_inputs_v1.json`。

| 路径 | microbatch × 累积 / 最大长度 | 更新数 | peak allocated GiB | 样本/s，排除前两更新 |
|---|---|---:|---:|---:|
| T2A all | 16 × 1 / 1536 | 12 | 10.27 | 9.21 |
| A2A all，含真实长语音 | 2 × 2 / 3072 | 5 | 7.74 | 1.71 |
| A2A audio_proj，特选长语音 | 2 × 2 / 3072 | 3 | 3.44 | 0.85 |
| I2T all | 4 × 4 / 768 | 3 | 6.49 | 5.22 |
| I2T vision_proj | 16 × 1 / 768 | 3 | 4.22 | 20.56 |

全部不用optimizer CPU offload，均启用block重计算。音频/视觉投影梯度均非零，真实A2A观察到2174帧fbank。所见训练draw无零监督、无某路audio STOP缺失；不代表全量无截断。后三更新的短测排除前两更新后仅剩一个计时点，尤其不能把特选长语音投影速度当全量平均或认为冻结一定更慢。

T2A B16相比此前本机B2/服务器B4更有吞吐潜力，但其累积为1；若正式改用更大的effective batch，需要检查累积期间常驻gradient的额外峰值。其他模式同样不能把表中配置直接当已验证的多天recipe。

`real_inputs_all_modes_v1/`保留前三项通过、随后空闲GPU检查退出的完整现场；它的controller总结果仍为false。`real_inputs_visual_v2/`仅运行剩余两项视觉检查。这个区分不能被汇总表掩盖。[环境与控制器诊断](../../../../docs/debug/011_server_omni_environment_and_idle_guard.md)记录失败、证据、修复及限制。

fixture从完整数据的原始train/val抽取，并包含明确的长语音样本；原始行号、split、逐scalar序列化对照和文件SHA在 `reports/server_omni_inputs_bundle_v1.json`。fixture重新编号，所以逐记录增强seed身份改变，不能与本机原始行号的训练做逐位对照。没有将这些fixture配置设为正式训练配置，也未上传替代用户自训的官方Omni成品权重。

实际远端目录 `/new_data/REMOTE_USER/minimind/omni_readiness_v1/project/runs/`。本地保存本页、两个完整run目录（除大型checkpoints）、原始日志、源码、config、metrics、评估和环境freeze；随机预检checkpoints仍留远端对应目录。正式训练的重要checkpoint必须回传本机。结束后2号卡已释放，未使用其他人的运行卡。
