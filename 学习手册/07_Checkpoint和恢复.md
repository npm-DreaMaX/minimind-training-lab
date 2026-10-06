# 07 权重、恢复与可复现

## 两类文件解决不同问题

| 文件 | 内容 | 能做什么 |
|---|---|---|
| `best_validation.pth` | 固定验证选择的FP16 state_dict | 加载结构后推理，或作为新阶段初始化 |
| `model_step_0113082.pth` | 最后一步FP16导出 | 保存最终里程碑 |
| `latest_resume.pt` | FP32模型、AdamW、RNG、cursor、step/config等 | 同一阶段完整恢复 |
| `deliveries/.../training_resume.pt` | 完整恢复点硬链接快照 | 固定这次交付，不被后续原子替换覆盖 |

Hybrid最终恢复文件约2.47GB，推理导出约421MB（十进制）。最终best恰好来自113082步，但best和last在一般实验中可能不同。两份导出文件的字节SHA不同，也不一定代表张量不同，序列化归档名称等会影响文件字节。

[最终文件大小与SHA](../models/02_hybrid_moe/deliveries/20261006_final_integrity_v1/files.json) · [完整状态审计](../models/02_hybrid_moe/deliveries/20261006_final_integrity_v1/report.json)

## 一个完整resume要保留什么

FP32参数；AdamW m/v/step；Python、NumPy、CPU Torch、CUDA RNG；当前epoch、shuffle seed与已消费cursor；当前optimizer step、config、LR调度公式；源码版本。这个训练器没有单独scheduler对象，位置由step和config重建，所以“没有scheduler state_dict”不等于漏了调度位置。

`EpochBatchSampler`用私有generator以seed+epoch重建顺序，从cursor取下一批；DataLoader另有私有generator，避免worker初始化消耗训练主RNG。预处理增强已经固定离线，这也是恢复语义的一部分。

评估前捕获RNG，eval后恢复RNG并回到train，减少评估对后续训练随机流的干扰。保存发生在optimizer边界，梯度已清空；如果在累积中间任意保存，就还需要未应用梯度和累积位置，本项目没有声称支持这种恢复。

## 精度与一致性分层检查

先检查刚load后的模型/moments是否相同，再比较下一步输入、loss、梯度和更新后参数。GPU非确定性和保存遗漏是两种不同问题；loss一样也不保证全部参数逐位一样。

本项目第一次非确定性恢复hash不一致，确定性复测通过；记录见[硬件与恢复诊断](../reports/hardware_comparison_20261003.md)。Hybrid最终审计核对189个state键、188组AdamW和四类RNG完整有限，但它不是在最终状态重新训练一步的逐位恢复证明。

FP16导出丢掉一部分FP32精度，并且没有AdamW。用导出重新启动一个阶段，应叫初始化或续训新实验，不要说成精确resume。实际导出重载CE差约0.00038，远小于相对官方基座的约0.357差距，说明导出误差不足以解释这次质量差距。

## 查看与推理

推荐第10章的CPU学习入口，模型源码、tokenizer、结构和权重自动对应，不需要猜文件名。核对字节可以在WSL执行：

```bash
sha256sum models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/checkpoints/best_validation.pth
```

预期为`1fc422dfac3b784e319676e251a58fdb1eb0eea83def7936f6be4578a4cac896`。本地学习工具推理前后也校验这个文件没有改变。

完整恢复命令存在于[原计划](../plans/formal_hybrid_official_mini_2ep_v1.json)和训练器参数中，但当前两轮已完成，普通MoE已由用户停止；学习练习不启动它们。学习“怎样恢复”不需要偷偷续跑长训练。

## 保留策略与磁盘

正式最终恢复点、全部正式里程碑、阶段转换初始化、关键恢复验证及失败现场都保留。清理仅针对审核通过的短资源预检冗余导出和缓存，并保留每次清理的路径/大小/原因。所有被删短测不再承诺能从当时的精确优化器状态resume，重新做同类预检需按保留的config/source运行。

硬链接的多个路径可以指向同一磁盘数据，不能将各路径文件大小简单求和成真实占用。删除其中一个链接也不一定释放空间；清理报告分别记录逻辑字节和最后链接释放估算。

练习：列出“拿best_validation.pth继续训练”与“加载latest_resume.pt”至少五项区别，并解释为什么模型参数相同仍可能在下一步走出不同轨迹。
