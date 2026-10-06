# 第四课：初始化、阶段衔接与断点恢复

三件事使用的文件可能都叫checkpoint，但解决的问题不同。

| 操作 | 输入 | 输出/目标 | 是否延续同一次优化 |
|---|---|---|---|
| 阶段衔接，如Pretrain→SFT | 已选模型权重、下一阶段数据/目标 | 同结构模型的新阶段 | 本项目重建AdamW与scheduler，否 |
| 结构迁移，如AR→Hybrid/Omni | 基座权重、明确的key/shape映射、新模块seed | 部分继承的新结构 | 否；须训练适应并重新验证 |
| 中断恢复 | FP32模型、AdamW、RNG、数据cursor、配置/源码 | 下一次本应执行的更新 | 目标是延续同一次优化 |

## 数学上要恢复什么

下一次更新不仅由参数θ决定。AdamW依赖一阶/二阶矩m、v和步数；scheduler依赖已完成的optimizer step；dropout与数据增强依赖随机数；样本顺序依赖epoch和shuffle种子。少恢复其中任何一个，loss仍可能看起来正常，但已经不是同一条优化轨迹。

本项目只在optimizer边界保存：该组microbatch已经完成backward、clip、step和zero_grad，不需要保存半个累积组的gradient。记录的cursor是已消费样本数，不是预取到worker里的样本数。保存临时文件后原子替换，避免写到一半覆盖最后一个完整恢复点。

FP16里程碑权重用于推理、分支与节省磁盘；加载到FP32不会恢复被舍入掉的低位。完整resume文件保留FP32状态。因此不能拿FP16模型导出＋新建AdamW称为精确resume。

## 本次为什么专门研究DataLoader随机数

AR token cache在读取时确定，增强已离线完成；Omni则每次读取会随机变速、加噪、改提示、截取对话轮次并扰动历史token。只保存主进程Python/Torch RNG无法恢复worker的prefetch状态。

`lab/omni_data.py:ReproducibleOmniDataset`按`seed:epoch:row`散列得到局部种子，在读取前保存Python/NumPy/Torch CPU状态，读取后恢复。训练sampler直接把epoch与样本位置传入worker，避免修改父进程dataset属性后以为worker副本也会更新。CPU种子通过局部Generator设置，不调用会顺带改变CUDA随机数的`torch.manual_seed`。

这种方法改变随机数分配顺序，但保留官方增强操作与分布；它使恢复不必倒放之前所有音频处理。代价是同epoch同row的重复读取得到同一增强，适合当前无放回epoch sampler；若将来改有放回采样，须增加draw index，不能直接照搬。

## 真正执行过的验证

`tests/test_omni_storage.py`验证0与2个worker、中途cursor恢复，输入和labels完全相同且主进程RNG不变。`tools/validate_omni_trainer_cpu.py`进一步调用真实`lab/omni_train.py`：真实T2A样本，小CPU模型，dropout、激活重计算、两worker、跨epoch、5次更新；一组连续运行，另一组第1步保存中断再恢复。最终所有模型tensor、AdamW状态、epoch/cursor、token计数、best选择和Torch RNG逐项相同。结果在`models/03_omni_moe/runs/trainer_cpu_resume_v2/result.json`。

第一版在载入官方collator时因独立Omni环境缺`datasets`失败，还没有进入forward。原始traceback在v1，安装固定datasets3.6.0后复测通过。没有把该依赖错误称为模型不收敛。

CPU确定性首先证明了控制逻辑；随后完整315M GPU在AdamW moment暂存方案下也通过了连续/中断对照，见 `models/03_omni_moe/runs/trainer_gpu_resume_v2_offload/result.json`。此前198M与205M完整模型的GPU恢复验证在各模型目录，非确定性失败与确定性复测分开保留。

## 如何诊断不一致

先比较刚加载的参数和optimizer tensor，排除保存精度/漏状态；再比较下一个样本的IDs/labels和所有随机数；再比较下一步logits、loss和梯度；最后比较optimizer更新。如果载入就不同，应先查serialization与dtype；若输入与状态相同但CUDA下一步略不同，再调查非确定性kernel，而不是归咎于数据。

恢复加载需要同时容纳模型/optimizer的CPU反序列化状态；训练本身能放入GPU，不代表CPU RAM足够加载全状态。必要时优化加载方式，但不能通过丢弃AdamW状态伪装成同一轨迹。

本项目后来为初始化/恢复采用`torch.load(..., mmap=True)`：tensor存储按需从文件映射，不一次全部复制到匿名CPU RAM。搬到GPU的参数与AdamW仍是完整FP32值，这不是optimizer CPU offload。第三版实际训练器CPU中断对照`trainer_cpu_resume_v3_mmap`的模型、AdamW、cursor、计数、best、RNG再次逐项一致。后续完整315M GPU恢复已通过，首次无卸载OOM与修复后结果分别保留。

文件映射需要保留底层文件内容稳定。这里checkpoint始终写临时文件再原子替换，旧映射仍指向旧inode，不能改成直接覆盖正在映射的文件。虚拟映射大小不等于物理RAM占用，但频繁缺页仍会带来I/O时间；两者都要观察。

阶段队列也不能只等最早一次训练的PID退出：resume会产生新PID。本项目SFT后继脚本改为等待真实GPU文件锁释放，以保证最终保存/清理完成后再开始新阶段。
