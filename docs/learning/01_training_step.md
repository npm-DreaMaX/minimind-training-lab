# 第一课：从一条文本到一次参数更新

先读 `models/01_moe/runs/recipe_batch8_probe_v2/`。这是正式训练前的真实语料验证，只有 640 条训练记录、129,577 个有效监督 token，不是已训练好的模型。

下文B8/T384/累积16用于解释这次早期probe。**当前正式训练已改为B6/T512/累积24**：输入`[6,512]`、logits`[6,512,6400]`、有效样本batch144，名义每次更新73,728个输入位置。原理相同，实测指标须按run区分；最终取舍见`docs/recipe_decisions.md`。

## 数据如何进入模型

原始 JSONL 每行 `text` → 官方 tokenizer（6400 词表）→ 不截断的 uint16 token cache。读取时构造 `[BOS] + text[:T-2] + [EOS]`，不足补 PAD。`input_ids` 和 `labels` 均为 `[B,T]`、int64，PAD 对应标签设为 -100。模型内部使用 logits 的前 T-1 个位置预测 labels 的后 T-1 个位置。

可以直接打开 `models/01_moe/learning/pretrain_sample/` 和 `sft_fixture_sample/` 的sample.json与token_trace.csv。真实预训练样例512位置中有167个非PAD输入、166个下一token标签；SFT测试样例1536位置中有555个非PAD输入、502个受监督标签。CSV逐位置标明`t`对应的输入与`t+1`的目标。SFT样例来自正式数据的下载前缀测试集，不能用它代表完整SFT分布。

本次 B=8、T=384，输入 `[8,384]`，logits `[8,384,6400]`。6400 个数是每个位置的词表打分，不是 token ID，也还不是最终生成的文字。数据源码见 `lab/data.py`，目标函数见 `upstream/model/model_minimind.py` 的 `MiniMindForCausalLM.forward`。

## 数学目标与梯度

CE 是受监督位置上 `-log p(下一个token)` 的平均值，-100 的位置不参与。MoE 额外输出 aux loss，约束专家负载分配。训练 loss 为 CE＋aux；验证主要报告 CE，不能直接把 eval 模式下没有 aux 的损失当成与训练总 loss 完全同一指标。

每 16 个 microbatch 更新一次，名义有效样本 batch=128。变长文本下每个 microbatch 的有效 token 数不同，因此我们的 CE 用 `当前有效token数/本组有效token总数` 加权；不是简单除以16。数学合同测试已证明 Dense 情况下这种加权梯度与合并 batch 一致。MoE aux 的统计依赖 microbatch，梯度累积不使它自动等价于一个大 batch。

`backward()` 把梯度累积到 parameter.grad；它不更新参数。gradient clipping 按全局梯度范数缩放梯度，然后 AdamW 根据梯度、一阶矩、二阶矩和 weight decay 更新参数。学习率由 optimizer step 决定，不能每个 microbatch 都错误推进一次 scheduler。

对一个受监督位置，CE对logits的梯度是`(softmax(logits)-one_hot(target))/有效标签数`。-100只取消该位置直接贡献的CE：SFT中被mask的用户问题仍会影响后面的assistant预测，梯度仍可经attention流向相应embedding与模型参数。输入ID本身是整数，不对ID求梯度；训练的是embedding表和后续参数。

AdamW保存`m = beta1*m + (1-beta1)*g`、`v = beta2*v + (1-beta2)*g²`，按step做偏差校正，然后执行`theta_new = (1-lr*weight_decay)*theta_old - lr*m_hat/(sqrt(v_hat)+eps)`。它与把L2惩罚直接加进Adam的梯度并不等价。当前betas沿用0.9/0.999，参数和m/v均FP32；BF16用于autocast计算路径，未使用FP16 GradScaler。梯度范数是在裁剪前记录的，所以大于1不表示裁剪失效。

## 显存花在哪里

198,416,640 个 FP32 参数约 0.739GiB，梯度约 0.739GiB，AdamW 两组矩约 1.478GiB，合计约 2.96GiB。这只是训练状态。BF16 autocast 不会自动把原始 FP32 参数与 AdamW 状态都变成 BF16。

真实 batch8 验证中 peak allocated 约 4.92GiB、reserved 约 5.25GiB。差额还涉及 activations、autocast 转换后的权重缓存、临时张量和 RoPE 等 buffer。reserved 是 PyTorch allocator 持有的缓存，不等于全部活跃 tensor；nvidia-smi 还包括 CUDA context、桌面和其他进程。不要把这三个数字混用。

真实 SFT 的 batch2×1536 测试再次说明了为什么不能只跑第一步：第1步 peak allocated 约3.46GiB，第2步达到4.92GiB。AdamW 的 moment state 在第一次 step 时才分配；下一次 forward 时这些状态已经常驻。只确认第一次 forward 不 OOM，不能证明长期训练能放下。原始每步记录见 `models/01_moe/runs/sft_pipeline_probe/metrics.jsonl`。

## 效率应该怎么看

batch8×384×16 的名义输入为 49,152 token，但样本有 padding，本次每组实际监督约 25,000～27,000 token。名义吞吐约 19,800 tokens/s，真正参与 CE 的吞吐约 10,000～10,800 tokens/s。仅报告前一个数字会掩盖 padding 开销。正式数据的长度分布可能不同，应重新观察。

关键参数是 microbatch、sequence length、gradient accumulation、precision、是否重计算激活、数据读取效率。累积增加每次更新包含的数据，但不会把每个 microbatch 的 activations 直接乘以累积次数；梯度在各 microbatch 之间保留。

指标定义也要核对。早期probe的 `padding_fraction` 实际是 `1-有效监督token/名义token`，在SFT中还包含被mask的用户问题，命名不准确。正式训练前已经改成两个独立字段：`input_padding_fraction` 只数输入PAD，`unsupervised_fraction` 数所有不参与CE的名义位置。旧日志保留并按旧定义解释，不能把SFT中用户问题占用的计算都说成padding浪费。

## 观测与故障诊断

看 CE、aux、grad norm、路由分配、有效 token 数、padding 比例、每步耗时、显存，以及独立验证和生成样例。loss 降低并不自动意味着模型会正确回答，当前少量步骤的生成仍有乱码和重复，原始样例已保留。

本次真实故障包括非确定性导致恢复后的参数 hash 不同，以及 Windows 文件系统上 DataLoader IPC socket 不可用。前者用“刚加载是否相同／再更新一步是否相同”拆开验证，后者用独立 socket.bind 实验定位。不能把它们误当成学习率、数据或显卡算力问题。

检查日志时先明确卡在取数据、forward、backward、optimizer、validation 还是保存 checkpoint。比如本机写入 2.38GB 恢复状态需要约20秒，此时 GPU 利用率下降有明确原因，不应直接认为 dataloader 或显卡故障。

## checkpoint 与恢复

完整恢复状态保存 FP32 模型、AdamW、CPU/CUDA/Python/NumPy RNG、epoch、样本 cursor、optimizer step、累计 token 数、配置与源码 hash。下一步学习率从 step 重建，数据顺序由固定 seed＋epoch 的私有随机生成器重建并直接跳到 cursor。

DataLoader创建iterator本身也会取一个worker base seed。后续训练器给它独立的generator，避免恢复时重新创建iterator额外推进模型的全局CPU RNG；单元测试覆盖了这一点。当前已经启动的预训练仍执行run/source中的启动时副本，其数据读取不含随机增强、模型运算用CUDA RNG，因此这个CPU RNG隔离补充不改变当前进程的参数更新。下次启动/恢复会保留新的源码副本，不悄悄覆盖旧记录。

官方模型源码保持不变；缓存、token 加权、warmup、恢复与指标工具是本项目自己的实现。对应区别见 `docs/experiment_plan.md`，不要把它们误记成官方默认行为。

正式预训练在第10、100步增加早期恢复点，此后每500步保存完整状态。`latest_resume.pt` 是继续训练的依据，`best_validation.pth` 是在固定验证子集上最好的模型导出，里程碑和最终模型另存。最佳权重没有 AdamW 状态，不能把它当作无缝续训 checkpoint；用于新阶段时会明确重置优化器。最终完整验证集与阶段内固定子集分开报告，不能混用两种样本集合的 CE 来选“最佳”。

## 正式MoE的路由观察

`models/01_moe/runs/pretrain_full_v1/plots/routing.png`按层展示四个专家接收到的token比例，虚线25%只是均匀分配参考，不是每个batch必须满足的约束。图来自每50步首个microbatch、排除padding的抽样，不能冒充全语料负载统计。原始计数和路由分布熵均在metrics中保留。

早期部分层的抽样确实出现接近单专家的偏载；到第3301步附近，各专家仍持续接收token。最近十个观测中第0层token加权比例约45.6%/20.7%/15.6%/18.0%，并非完全均匀，但也不是专家永久死亡。因此目前不因一张早期图就修改aux系数，继续观察持续性、路由熵、任务CE和各层差异。

本仓库Top-1 forward权重为1，借straight-through表达式给gate任务梯度；aux另外约束负载。它没有按容量丢弃过载专家的token，因此偏载不会自动等同于“部分训练标签被扔掉”。监控排除了padding，但官方aux计算仍包含padding；两者统计口径不同，切换动态padding时连aux也会变化，不能只称作速度优化。

## 逐位置实测：同一位置的CE梯度和aux梯度可以不同

现在可打开 `models/01_moe/learning/full_model_gradient_trace_v1/positions.csv`、`loss_and_embedding_gradients.png` 和 `report.json`。使用第3502步固定导出权重的完整198M模型，在CPU FP32上选择一条短验证样本演示padding；输入 `[1,512]`，实际91个非PAD位置、90个next-token标签，logits `[1,512,6400]`，embedding输出 `[1,512,768]`。它不是一轮正式训练，也不能用单例44.4%的top1预测率概括模型能力。

逐位置NLL平均值与模型返回CE相同：本例CE=2.33673，aux=0.004610。CSV的position t对应input[t]，target对应label[t+1]；-100不是词表中的一个token，所以没有“预测-100”的概率。被忽略的位置仍可显示logits，但不能把它们纳入CE准确率。

对输入embedding**输出向量**求梯度，本例尾部PAD位置来自CE的梯度精确为0，来自aux的最大范数却是8.31e-5。这与源码一致：因果CE的已监督位置看不到后面的PAD；路由辅助目标仍统计这些PAD位置。这里不是对embedding表中PAD那一行的梯度：输入embedding和输出LM head共享权重，即使某位置CE被mask，词表权重仍可能从输出softmax收到梯度。不要混淆位置向量与参数表的行。

第0层router矩阵shape `[4,768]`。其CE梯度范数0.19887、aux梯度0.006043，总梯度0.19903；总梯度与两部分相加的最大误差1.81e-8。这直接证明主线Top-1 router确实从任务CE学习，不能以为“选专家是离散的，所以gate只有aux梯度”。

最后只复制router矩阵做一次AdamW公式演示：新建零moment状态、LR3e-4、weight decay0.01；实现结果与首次更新公式最大误差3.73e-9。复制体发生变化，加载的模型权重逐位未变。此演示没有全模型梯度裁剪或累积，也未继承第3502步AdamW状态，不能当成正式下一步更新；它只把m、v、偏差校正和decoupled weight decay落到可核对的数字上。
