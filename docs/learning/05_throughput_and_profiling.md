# 第五课：吞吐、显存和 profiler 应该怎样一起看

本课来自一次完整198M模型、真实预训练样本的短时性能分析，原始结果在 `models/01_moe/runs/profile_real_update_v1/`，执行源码为 `tools/profile_training_step.py`。独立加载第2607步的FP16导出到FP32参数、初始化新的AdamW，仅用于分析；这些更新不计入正式训练。

## 输入、输出与测量边界

配置为microbatch6、sequence512、累积24，即每次optimizer更新处理144条记录。输入/labels均为 `[6,512]`；logits为 `[6,512,6400]`。一组固定的144条真实样本包含31,962个有效next-token标签；名义输入位置为73,728。目标仍是按有效标签加权的CE，加上按microbatch均值累积的MoE辅助损失。

第一次未开profiler的更新耗时6.010秒，包括AdamW首次分配moment；第二次3.597秒，约8,885个有效监督token/s。不能用第一次计时代表稳态，也不能把73,728个含padding的位置都称为“训练了这么多文本token”。这次只测一组样本，长训练均值以正式metrics为准。

profiler采集了接下来的一次完整更新，生成约306MiB trace。被分析的更新约4.870秒，比未采集时慢；trace整理和导出使整个命令耗时122秒，采集进程实测RSS一度约4.2GiB。**仪器也占RAM/CPU，并改变执行时序**，所以性能报告把未采集计时、采集区间和命令总耗时分开。

## 不能直接相加的数字

`operators.txt`同时有CPU算子、CUDA kernel及人工标注区间。同名的`microbatch_forward`可能出现在CPU与GPU两处；区间包含子操作，CUDA事件还可能在不同stream重叠。把表中所有时间或百分比相加，会重复计数。

`tools/summarize_profile_trace.py`流式读取原始trace，将类别分开。此次有95,597个kernel事件，kernel时长之和约2.667秒；这不是整个更新wall-clock，更不是硬件FLOP利用率。CPU标注区间分别约：数据传送调用50.8ms、forward1.813s、backward2.939s、clip+optimizer39.0ms。CPU区间不等于对应全部GPU工作的完成时刻，因为CUDA异步执行；不能据此把CPU与GPU时间直接相加。

这次证据说明，大量矩阵运算、逐元素运算、dtype复制和kernel调度都值得观察；仅优化每24个microbatch才运行一次的AdamW，不能直接解决全部更新耗时。是否采用fusion、编译或更大物理batch，要分别测速度、峰值内存、数值及恢复，不凭一个百分比拍板。

## 内存数字也有不同口径

此次peak allocated为5,232,313,344字节，约4.87GiB。模型198,416,640参数，FP32参数约0.739GiB；同量gradient，加上两份FP32 AdamW moment，完整训练状态约2.96GiB。其余来自activation、logits、转换和临时buffer，且不同阶段的存活范围不同。

例如一个BF16 `[6,512,6400]` logits本身约37.5MiB，CE路径可能产生其他dtype/连续内存副本。梯度累积24次不会保存24份完整计算图：每个microbatch完成backward后释放图，参数gradient累加；若把24个loss对象先留在Python列表最后一起backward，内存语义就不同了。

算子表中某些`CUDA Mem`超过70GiB，表示该算子在整个采集区间多次分配的累计量，**不是同一时刻占用了70GiB显存**。peak allocated、allocator reserved、nvidia-smi整卡占用分别回答活跃张量、分配器持有内存、包含Windows桌面/context的设备占用，不能互换。

## 训练效率与算法边界

提高物理batch可能减少调度开销并提高矩阵利用率，也会增加activation。降低累积、提高物理batch，即便有效样本batch不变，MoE每个microbatch的负载辅助损失也可能变化；不是所有意义上严格等价。改变sequence会改变文本覆盖、注意力成本和音频STOP保留率，不能只报告变快。

动态padding能减少空位置计算，但当前官方MoE的aux统计包括padding位置，因此它也会改变路由目标的实际统计。是否排除padding、pack多个样本、加入样本边界attention mask，都需要独立recipe和对照，而不是隐含在所谓“加速”里。

激活重计算、allocator策略、数据分片属于可以在保持模型结构下验证的系统措施；仍须检查loss/gradient、RNG与resume。BF16通常也是系统配置，但会改变数值误差：本项目Hybrid候选的[实际梯度失败](../debug/007_hybrid_bf16_gradient.md)说明不能仅以“不OOM”和“forward接近”作为采用依据。

## 以后如何判断慢在哪里

先在正式日志对齐有效token/s、step_seconds、验证/保存间隔和GPU telemetry，区分训练、首次初始化、加载、编译、评估、保存。再用少量代表性batch采集profile，记录相同模型/数据/precision、预热及同步边界。某次低GPU utilization不自动等于DataLoader慢，也可能是kernel启动、同步、很小的专家矩阵或CPU等待；结合trace定位后，每次只调整有明确假设的一项。

本次尚未做融合优化，没有声称某项修改已提升正式吞吐。重要学习成果是能区分真实训练工作量、采集开销、GPU活跃时间、FLOP利用率及各类内存口径，再设计有证据的后续实验。

## 同一优化在两个模型上产生不同结果

[实际batch对照](../../reports/preflight_window_v7b_review.md)给出反例：198M SFT将B2无重计算换成B4/B8重计算，虽然省显存，却更慢；Hybrid SFT在都开启重计算的前提下从B2增至B4略快。这两组比较不能混为“重计算必快/必慢”或“batch越大越快”。重计算增加重复forward开销；增大microbatch减少每次optimizer更新中的forward/backward调用次数，并可能改善矩阵利用率。最终取舍由两种开销的实测净变化决定。

A2A减少worker的目的则是RAM余量：保留完整SenseVoice、3072长度和AdamW状态暂存，比较相同记录的吞吐、父进程RSS、全局可用内存、swap变化与验证阶段峰值。CPU/RAM紧张时，更多worker反而可能增加共享内存、prefetch和分页压力。这里worker0与worker1速度接近，但前者的主机余量较好；这个判断不依赖“GPU利用率看起来更高”。


## 当前205M Hybrid的同一套读法

正式配置 `models/02_hybrid_moe/configs/hybrid_sft_official_mini_2ep_v1.json`：B16、T768、累积1、激活重计算。输入/labels `[16,768]`，logits `[16,768,6400]`；一个BF16 logits约150MiB，交叉熵的副本与反向另算。完整205,623,072个FP32参数约0.766GiB，参数＋gradient＋两份AdamW moment若同时存在约3.064GiB。不能再把全部reserved或累计分配量简单加上去。

八步资源测峰值allocated约4.50GiB；正式更新后清空gradient时allocated约2.34GiB，与参数＋两份moment的尺度一致。`nvidia-smi`还包括分配器保留、CUDA context和Windows桌面，观察到约7GiB并不与4.50GiB矛盾。GPU约93%～97%活跃、70～73摄氏度、约85W是此时telemetry，不是全程或FLOP利用率承诺。

独立完整模型profile在 `models/02_hybrid_moe/runs/profile_official_sft_B16_RC_v1/`：首步3.481s含moment初始化，第二步1.582s。`operators.txt`、`trace.json`及`trace_summary.json`分别保留聚合、原始事件和类别分离汇总。FLA状态反向、投影及MoE矩阵运算、转换和逐元素运算都有成本；原生卷积不是唯一或主要瓶颈。

以第100步为例，累计650,279监督token、1,600条记录。单步固定12,288输入位置，其中只有本批assistant标签参与CE；不能用100×12,288冒充实际训练token。第100步存储的完整恢复状态约2.47GB（十进制），保存约19s。吞吐要同时给纯update与包含保存/评估的wall-clock，否则预计结束时间会偏乐观。

当前源码没有开启PyTorch TF32；被测试后未采用的补丁保存在模型patches目录。Triton FP32 dot采用已验证TF32x3是独立设置。不要看到名称都含TF32，就以为两个开关控制同一处计算。

图表检查还发现学习率自动坐标偏移会把前500步约0.004%的变化放大成陡降。绘图已改为零起点、无数值offset的科学计数轴；只改变展示，不改变训练scheduler或metrics。完整113,082步cosine日程在最初500步理应几乎平坦。

## 21:45复核：参数差不多，为什么官方约3小时、这里约50小时

固定版本README的64M Dense在3090上，mini预训练/SFT各一轮估算2.31小时；198M普通MoE为3.23小时，其中SFT一轮1.54小时。因此不能把所有差距归因于MoE，也不能说3小时只对应Dense。当前205M与198M总参数确实只相差约3.6%，但六层注意力已经换成不同的运算。

首先统一阶段：我们当前是SFT两轮；若按作者普通MoE的SFT估算线性翻倍是3.08小时，与我们约50小时之间仍有约16倍的整套系统吞吐差。**轮次只能解释两倍工作量，不能解释剩下的差距。** README是经验估算，没有取得完全对应的数据版本、命令、软件环境和原始计时，不能把16倍再编造为精确的硬件/内核独立倍率。

目前能够确认的成本包括：本机4060 Laptop8GB与3090不同；当前线性模块按作者路径使用FP32投影/状态计算；普通主线使用BF16 autocast和SDPA注意力，而此Hybrid需要DeltaNet状态更新与其反向；8GB下B16需要激活重计算。线性随序列长度增长的复杂度，不等于在T768处比成熟Full Attention内核更快，更不保证训练反向更快。

已有完整205M profile中，单个`chunk_bwd_kernel_dqkwg`占CUDA kernel时长之和约22.4%，还有投影/专家矩阵乘法、dtype转换和逐元素运算；不是把嵌套算子表全部相加得出的百分比。六层DeltaNet forward出现12次、其backward出现6次，也直接体现了重计算。原生depthwise卷积forward只占kernel时长约1.9%，不支持“装一个卷积包即可十倍提速”的判断。

最近更新约1.5秒/步；904,644条/轮、B16，形成56,541步/轮。两轮仅更新约47小时，加周期性检查约50小时。每500步的验证、保存、绘图边界通常额外约40秒，而500次更新约750秒，因此这些记录工作的量级约5%，不能解释十几倍差距。这是近似估算，不是严格时长分区。固定输入位置两轮共1,389,551,616个，真实监督标签735,833,934个；近期约36%输入位置为padding。用户前文虽然不计CE也需要forward，不能简单把所有无监督位置都删掉来“等价加速”。

已测B2×8不重算约10.52条/s、B16×1重算约10.15条/s；后者保留官方物理batch的MoE aux统计。PyTorch TF32候选短测约快5%，BF16线性投影的门梯度未通过预设误差阈值。这些证据说明有工程代价及优化空间，**没有证明50小时是架构的最低耗时**，也不能承诺一个开关即可恢复官方速度。此次没有中断训练或改变预算。

本轮还发现`time.time()`时间戳存在回拨，不能将其窗口差减去所有`perf_counter()`更新耗时后，直接认作验证/I/O开销。原始样例和范围保存在[本轮计时证据](../../reports/hybrid_time_breakdown_20261003.json)，诊断见[预算与效率记录](../debug/014_hybrid_budget_and_efficiency.md)。

## 关闭激活重计算，会更快或改变结果吗

要分开两个问题。若模型、物理batch、数据、精度全部保持不变，只关闭激活重计算，就把原先丢弃再计算的中间激活留在显存中，通常减少重复forward、增加显存。数学目标不变，当前封装使用`use_reentrant=False, preserve_rng_state=True`，保留dropout随机状态；但CUDA非确定性和数值差异意味着不能无条件保证最终权重逐位一致。当前B16×T768未找到可引用的“同batch关闭后”完整计时/峰值测试，不能捏造显存需求或提速百分比；8GB上直接关掉有OOM风险。

已真正跑过的是两套组合配置：B16×累积1、重计算开，约10.15条/s、allocated峰值4.50GiB；B2×累积8、重计算关，约10.52条/s、峰值4.60GiB。完整205M、同初始化、同128条输入、T768、每次更新16条，两者都完成8步检查。后一方案短测净快约3.7%，这**不是重计算开关本身只有3.7%成本**：它同时缩小了物理batch，单次optimizer更新要执行8个microbatch，增加调用/调度并改变矩阵规模，抵消了一部分收益。未做长期质量对照，不能据此保证全程提速。

对结果也要区分：只切换正确实现的重计算，属于保存激活与重做激活的工程取舍；而B16×1改为B2×8时，CE可以按有效标签权重累积，但本模型MoE aux在每个microbatch内用专家负载均值和路由概率均值计算，再平均。八个小batch的aux均值不等于一个大batch的aux，因此这不是严格相同的训练目标统计。不能保证两者最终结果相同，也没有证据表明小batch方案必然变差。

当前决定：继续现有B16×1重计算配置，保留官方物理batch统计；没有为尚未确认的少量提速中途更换recipe。若后续研究选择性重计算或更大显存，另做固定batch的数值、显存和吞吐对照，不能从上述两组联合变化反推纯开关效果。[原始短测汇总](../../reports/hybrid_official_sft_resource_choice_v1.json)。
