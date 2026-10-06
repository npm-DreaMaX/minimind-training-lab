# 04 从一批输入到一次更新

本章沿[真实训练器](../lab/train.py)的`main`阅读。源码行号可查[源码地图](数据表/源码地图.csv)，不要把另一个项目的惯例当作本项目事实。

## Forward与CE

输入x/y分别是`[16,768]`的ID/labels；输出logits`[16,768,6400]`，每个位置给6400个未归一化分数。logits不是概率；softmax后才是归一化分布。

模型把logits去掉最后位置、labels去掉第一个位置，展开后用cross_entropy，忽略-100。对N个有效目标：

`L_CE = -(1/N) * Σ log softmax(z_t)[y_(t+1)]`。

它优化下一个token的条件概率。训练时前文是真实token；推理时前文包含自己生成的token，所以较低CE不能直接换算事实正确率。

模型返回`out.loss`为CE，`out.aux_loss`为路由辅助损失，训练器再相加；不要加两遍aux。验证只报告token加权CE，eval模式的aux为零，不能直接和训练总loss混比。

## 梯度累积的分母

设一组累积microbatch的有效标签分别为n₁…nG，总N。训练器对第i批执行：

`loss_i = CE_i * n_i/N + aux_i/G`，然后backward。

这样CE按有效token平均，aux按非空microbatch平均。不是简单每批loss/G：如果一批100个标签、另一批900个标签，简单平均会让前者每个token获得9倍权重。Hybrid正式G=1，普通MoE预训练G=24，所以这点在后者特别重要。

累积不保留所有microbatch的计算图：每批backward后释放图，累积的是`.grad`。有效batch相同，也不能保证MoE aux、dropout、数值舍入与大物理batch逐位等价。

## Backward不是更新参数

`loss.backward()`应用链式法则，把梯度累加到叶子参数的`.grad`。这一步没有调用optimizer，参数数值通常没有变化。`optimizer.zero_grad(set_to_none=True)`清掉上一组梯度；不清理会无意累积跨step梯度。

本次新增的[CPU实测](实操输出/完整Hybrid梯度检查_v2/trace.json)加载完整205M最终权重，取真实第一条记录前128个位置，验证手算CE与模型CE相等，计算全部参数梯度，记录每个token的embedding输出梯度。**模型结构完整，序列是教学前缀；这是只读梯度检查，不是额外训练，也不能把这个loss当正式T768质量分数。**

在报告中对照：用户位置label=-100，但embedding输出梯度可能非零；回答位置的预测来自前一个logit；最后一个logit不用于本次shift CE。共享embedding/lm_head的参数梯度包含输入和输出两条路径，不能只凭权重某一行非零推断它一定在输入出现过。

注意区分报告中的总loss梯度与`embedding_output_ce_only_gradient_norm`。aux本身也依赖hidden states，所以只看CE＋aux总梯度不足以证明上下文传播；第二版检查单独求CE梯度，确认23个未监督前缀位置仍非零，再正常计算总梯度。这是一次教学验证的改进，未改变模型或正式训练算法。

## Clip、AdamW与scheduler

训练器先计算全参数梯度L2范数，再以阈值1裁剪；记录的`grad_norm`是**裁剪前**。第一步约45.81不代表裁剪失效，也不自动等于NaN；它是需要结合学习曲线观察的高梯度。

AdamW维护一阶m、二阶v与计数t，做指数滑动平均及偏差校正，以`m_hat/(sqrt(v_hat)+eps)`调整参数，并进行解耦weight decay。默认betas=(0.9,0.999)、eps=1e-8；本训练器对所有可训练参数使用同一组，没有额外排除norm/bias衰减。moment在首次optimizer.step时建立，解释了部分“第一步能forward，optimizer时OOM”。

实际顺序：设置当前LR → 清梯度 → 各microbatch forward/backward → clip → optimizer.step → 清梯度 → CUDA同步 → step/cursor/token累加 → 日志 → 必要时评估/保存。

无warmup的Hybrid调度为`lr(s)=peak*(0.1+0.45*(1+cos(pi*s/S)))`，s是更新前step，S=113082；第一条记录step=1使用s=0的1e-5。最末使用s=S-1，接近1e-6。不要把日志step与调度输入差一位误判为恢复错误。

## 为什么会不收敛

先检查有效labels与shift，再查梯度是否存在/有限、参数有没有真的更新、LR是否合理、学习的是否是目标数据。loss变化也可能只是batch难度波动，需要固定验证。数据错误、错模板、过小/过大学习率、遗忘门初始化、专家偏置和实现数值问题都可能影响学习；仅增加epoch无法鉴别这些原因。

练习：解释“backward已经运行但参数没有变”可能是正常检查，也可能是漏掉optimizer.step。用完整模型trace的`optimizer_steps=0`和正式训练器调用位置区分两者。
