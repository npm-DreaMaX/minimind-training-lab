# 当前整体训练计划与 recipe

更新时间：2026-10-06 17:30。**用户要求停止当前普通MoE，不再继续此分支。已在13,209步、431,460,605监督token保存退出，自动SFT接续已终止；不按旧持续授权恢复。** 以下recipe保留为实验历史，未完成预算不能标为完成。本次没有切换到新训练，后续方向以用户新指示为准。[停止记录](../runs/stop_moe_by_user_20261006/request.json) · [核验](../runs/stop_moe_by_user_20261006/verification.json) · [配置指纹](../reports/recipe_inventory_20261004.json) · [旧总计划](experiment_plan_20261003_history.md)。

## 当前执行顺序与依赖

1. **先完成当前205M Hybrid**：官方198M SFT基座 → 迁移六层注意力 → 官方mini完整两轮全参数SFT → 全验证与生成评估。本轮预算、完整验证和生成已完成，评估结果与生成样例已记录；见[最终复核](../reports/hybrid_final_review_20261006.md)。
2. **198M从零主线已由用户停止**：原预训练→完整SFT计划不再自动执行；第13,209步完整恢复点与学习记录保留。
3. **启动315M Omni七阶段**：用上一步自训SFT的选定权重初始化Thinker，在已验证服务器环境运行。Omni使用普通Full Attention语言主干，不能把Hybrid权重直接塞进原Omni并声称兼容。
4. **继续独立研究分支**：Full Attention/Hybrid匹配对照、64M Dense AR/dLM、DPO、GRPO/CISPO、Agentic RL。保留模型和流程，不因难跑删除；未确定的recipe先完成有针对性的验证，不把候选数字冒充正式预算。

当前Hybrid从官方SFT初始化，和暂停的自训198M是不同血缘。它既不是“我们的198M已预训练完成”，也不是正在做一轮从零Hybrid预训练；本轮结束后不会再机械追加一遍同样SFT。

现有自动化的真实边界：Hybrid预算、最终验证和生成套件已经结束；198M训练与自动SFT接续进程均已终止。Omni的自训SFT依赖未满足，未替换初始化来绕过依赖。本机只保留状态记录等非训练进程，不会因本次停止而自动新开训练。历史控制器检查预算不等于自动判断模型质量。

## 如何读 recipe

下表 `B×G` 分别是物理microbatch与梯度累积次数，单卡每次optimizer更新的样本数约为B×G，末尾不足一组的样本也处理；不是token数。T是训练截断/补齐长度。显存峰值还受真实模态输入、激活、optimizer和临时buffer影响。

已配置的三条主线共同使用AdamW：betas=(0.9,0.999)、weight decay=0.01、梯度clip norm=1；FP32主参数与Adam状态，CUDA BF16 autocast。scheduler按optimizer step执行，warmup之后cosine降至峰值LR的0.1倍。Hybrid的线性模块另有FP32路径，不能仅凭外层autocast称作全BF16。

## ① 205M Hybrid：完整两轮预算已结束，质量待改进

| 项目 | 锁定值 |
|---|---|
| 模型 | 205,623,072参数；6层Gated DeltaNet＋2层Full Attention；8层均为4专家Top-1 FFN |
| 初始化 | 官方 `full_sft_768_moe.pth`；继承187,798,656参数，新增17,824,416参数随机初始化 |
| 训练范围 | 全部205M参数；本分支没有另加冻结5,000步适应阶段 |
| 数据 | 完整官方 `sft_t2t_mini.jsonl`，约1.739GB，905,718条全部预处理 |
| 划分/异常 | 1,066条held-out；904,652条剩余记录中8条在T768下零监督，审计后训练904,644条 |
| 预算 | **2 epochs，113,082 updates，735,833,934监督token** |
| B×G / T | **16×1 / 768** |
| LR / warmup | **1e-5 / 0步** |
| 精度/显存 | 外层BF16；线性模块FP32＋经独立梯度验证的Triton TF32x3；激活重计算开启；不做optimizer CPU offload |
| 训练目标 | assistant有效位置的next-token CE＋MoE aux；用户/padding标签为-100 |
| 设备 | 本机4060 Laptop8GB |
| 保存/评估 | 每500步固定512条验证、3个生成探针和恢复点；每5,000步模型里程碑；最终全1,066条验证和独立生成套件 |

原始数据/revision/hash、T768截断统计、源码修复分类在[本轮Hybrid详细方案](hybrid_official_mini_2ep_v1.md)。所有数据完整读取不代表所有文本无截断：5.258%记录有尾部截断，保留约98.797%监督token。内容hash划分避免完全相同记录跨侧，但官方初始化可能已见过这些数据。

[正式config](../models/02_hybrid_moe/configs/hybrid_sft_official_mini_2ep_v1.json) · [执行plan](../plans/formal_hybrid_official_mini_2ep_v1.json) · [模型/日志/权重目录](../models/02_hybrid_moe/)

## ② 198M Full Attention＋MoE：用户已停止，以下为原recipe

| 阶段 | 初始化与数据 | B×G / T | 轮次 / 更新数 | 峰值LR / warmup | 重计算 |
|---|---|---|---|---|---|
| Pretrain | 随机初始化；官方完整 `pretrain_t2t.jsonl`，约8.275GB，8,460,241训练记录 | **6×24 / 512** | **1轮 / 58,752步** | **3e-4 / 100步** | 关 |
| Full SFT | 完成预训练后的固定验证best；官方完整 `sft_t2t.jsonl`，约14.096GB，5,102,028有效训练记录 | **2×8 / 1536** | **1轮 / 318,877步** | **1e-5 / 100步** | 关 |

模型198,416,640参数，8层Full Attention、4专家Top-1 FFN，全部参数更新。本机优先，GPU常驻AdamW，不做optimizer offload。

预训练原完整预算约19.20亿监督token（审计估算1,919,567,915，literal PAD等口径限制见审计）；曾在第9,768步暂停，10月6日恢复后按用户新要求在13,209步、431,460,605监督token停止。完整FP32模型、AdamW、RNG、epoch/cursor及scheduler位置均保留，不会自行恢复。

SFT训练标签预算为3,050,341,350。T1536是根据全量长度/截断审计选定，不是官方768默认。原SFT训练split有2,003条在这一长度下零监督，均有审计。阶段预注册的一轮不是保证收敛；后续是否追加预算要根据验证和生成证据另立版本。

阶段之间从FP16导出加载到FP32并新建AdamW，属于新阶段初始化；阶段内部resume保留完整状态，二者不同。SFT采用预训练固定验证best，同时记录它的选中步数，不能声称best一定是最后一步。

预训练每500步评估/保存，5,000步里程碑；SFT每2,000步评估/保存，20,000步里程碑。二者周期验证512条，最终全held-out评估与生成。

[预训练config](../models/01_moe/configs/pretrain_full_v1.json) · [SFT config](../models/01_moe/configs/sft_full_v1.json) · [SFT数据审计](../reports/sft_data_audit.md) · [模型目录](../models/01_moe/)

## ③ 315M Omni-MoE：服务器七阶段，等待自训SFT基座

完整314,887,938参数，Thinker＋Talker＋模态投影＋8路音频head；SenseVoice/SigLIP等外部编码器保持冻结并单独计资源。正式数据是官方完整T2A、A2A、I2T，不是mini。部署、字节校验、独立环境及真实T2A B16×8/各输入模式短检查已通过，尚未正式训练。

以下沿用固定作者full MoE示例的阶段、epoch、LR；单卡微批、长度、划分、warmup等由我们的实测确定，不能称为四卡发布权重的逐项复刻。

| 阶段 | 更新范围 / 学什么 | epochs | B×G / T | 峰值LR | updates |
|---|---|---:|---|---:|---:|
| 1 T2A | 全参数：文本→语音 | 6 | **16×8 / 1536** | 5e-4 | 58,488 |
| 2 A2A | 仅audio projector：接入语音输入 | 1 | **2×64 / 3072** | 5e-4 | 3,232 |
| 3 A2A | 全参数：语音对话 | 3 | **2×64 / 3072** | 5e-5 | 9,696 |
| 4 I2T | 仅vision projector：接入图像 | 1 | **16×8 / 768** | 5e-5 | 20,873 |
| 5 I2T | 全参数：视觉问答/文字混合 | 1 | **4×32 / 768** | 5e-6 | 22,671 |
| 6 A2A | 全参数：低LR语音回训 | 1 | **2×64 / 3072** | 5e-6 | 3,232 |
| 7 I2T | 仅vision projector：视觉再对齐 | 1 | **16×8 / 768** | 5e-6 | 20,873 |

各阶段有效样本batch128，warmup100步，开启激活重计算，服务器GPU常驻AdamW、offload关闭。总139,065次更新、17,799,071次样本呈现。由于真实音频长度与每epoch增强不同，不能把样本数直接冒充固定音频token预算。

训练行：T2A 1,247,657；A2A 413,573；I2T投影使用2,671,732视觉行，all阶段使用2,901,800条视觉/文字混合记录。I2T按图像资产分组留出，并修复回答中占位符的错误展开；A2A除完整验证外额外报告未见输入录音分组。

目标是文本CE＋8路音频CE的均值＋MoE aux；音频STOP token的损失分子乘10，分母仍按有效位置计数，严格保留并单独观察。Omni的microbatch loss等权累积不同于语言训练器的有效标签加权，不能给三条主线统一套一个不真实的loss公式。音频历史扰动默认scheduled_sampling=0.05；验证关闭历史扰动，输入增强种子固定。

每250步验证/保存、每5,000步里程碑，最终完整held-out及逐条统计。每阶段接续固定验证best并重建optimizer，同时保留final与所选步数；每阶段重新测已学模态的文字/音频/图像，保存codes/WAV，观察遗忘。投影阶段不会把主体训练阶段偷换成只训投影。

服务器实际项目：`/new_data/REMOTE_USER/minimind/omni_formal_v1/project`；权重、config、日志、评估回传本机[Omni目录](../models/03_omni_moe/)。不终止服务器其他人的任务，卡状态在执行时检查。

[七阶段configs](../models/03_omni_moe/configs/) · [正式plan](../plans/formal_omni_server_v1.json) · [固定作者train.sh](../sources/minimind-o/trainer/train.sh) · [实际正式配置资源检查](../models/03_omni_moe/runs/server_formal_runtime_v1/result.json)

## ④ Hybrid与Full Attention的对照：保留，当前未运行

当前Hybrid与我们暂停的198M从零分支，初始化/数据/预算不同，不能直接比较loss后宣称Hybrid更好。10月6日已完成同模板、同问题、同GPU精度的官方AR基座/当前Hybrid推理与固定CE对照，另做CPU和256token预算诊断，确认当前能力恢复不足；如果要评价训练方案或架构收益，还需要相同来源、相同额外训练预算的AR续训对照。当前mini分支的匹配AR续训config尚未锁定，没有把已有官方基座伪装成接受了两轮额外训练的control。

另有已保存的自训AR基座研究计划 `formal_hybrid_and_control_v1`，当前暂停：

| 阶段 | Hybrid范围 | B×G / T | 预算 | LR |
|---|---|---|---|---|
| 新模块适应 | 仅新增线性模块 | 4×6 / 512 | 5,000步、120,000条样本呈现 | 1e-4 |
| 预训练续训 | 全205M | 4×6 / 512 | 完整预训练train split 1轮、352,511步 | 1e-4 |
| SFT | 全205M | 4×4 / 1536，重计算 | 完整SFT 1轮、318,877步 | 1e-5 |

对应198M control使用相同数据顺序、物理batch、LR、阶段预算与选择规则，但模型/新模块初始化及可训练范围不同，比较的是整套迁移程序。所有阶段warmup100；适应到续训用最后5,000步权重，避免提前best吃少了适应数据。两条支路不会自动把loss曲线拼成一个run。

这是保留的大数据研究分支，不是当前mini Hybrid完成的必要条件。另一条官方预训练基座适应分支已在1,369步保存暂停，和当前官方SFT基座分支也不同。恢复前要结合当前生成问题与完整运行成本复核，不悄悄删掉或自动把“几十小时”变成“额外几周”。

## ⑤ Dense AR/dLM与后训练：方向保留，正式recipe尚未锁定

| 实验 | 初始化/对照原则 | 目标与需要确定的参数 | 当前证据 |
|---|---|---|---|
| Dense AR，63.9M | 同tokenizer、固定数据split；保留AR基座 | next-token CE；正式数据预算、B/G/T/LR待GPU实测后锁定 | 独立目录，尚未训练 |
| Dense dLM，63.9M | 从配对Dense AR权重转换，保留同预算AR续训control | 双向同位置掩码去噪；噪声/重加权/采样步数、B/G/T/LR需要单独验证 | 完整Dense CPU结构/loss/backward检查完成，零监督NaN和残留mask反例保留 |
| DPO | 从质量经过检查的SFT权重分支；固定chosen/rejected数据 | 偏好目标、reference模型、beta、长度与预算待数据审计/显存测试 | 研究规划，未正式运行 |
| GRPO/CISPO | 从可评估的语言SFT权重分支，优先可验证任务 | rollout组大小/长度、reward、KL、更新规则与预算待验证 | 研究规划，未正式运行 |
| Agentic RL | 在可靠工具调用/语言基座后独立分支 | 多轮工具轨迹、环境反馈、奖励归因和评估；不与Omni强行拼接 | 研究规划，未正式运行 |

dLM不是普通AR换一种解码：attention方向、监督位置和生成算法都改变。AR CE与去噪loss不能直接数值排名。作者dLM的MoE aux返回问题尚待处理，因此先用Dense隔离训练目标；不会未经修复宣称MoE dLM已经正确。

这些阶段还没有“正式LR/epoch已经选好”的事实，不给出未经数据统计/显存/吞吐验证的伪精确数字。[Dense目录](../models/04_dense_ar/) · [dLM目录与实际检查](../models/05_dlm/) · [AR/dLM学习材料](learning/07_ar_and_masked_diffusion.md)

## 完成标准、学习产物与时间边界

每个正式阶段至少保留config、命令、源码快照与hash、diff、原始metrics/log、FP32恢复点、FP16里程碑、验证/生成和失败证据。恢复点在optimizer边界原子保存，包含model/AdamW/RNG/cursor与step；最新完整状态覆盖更新，重要里程碑另存，避免每一步复制数GB。

已运行模型持续看CE/aux、LR、裁剪前grad norm、有效监督token/s、显存、GPU温度/功率、MoE分层负载、数据覆盖和原始生成。数学/数据/模型/系统分别学习，不能把“文件产生了”当作模型训成。质量失败需要保留现象→假设→检查→证据→诊断→修复/尚未修复→再验证；不为好看的SUCCESS覆盖失败。

工程改变和算法改变分开记：正确的激活重计算以额外计算换显存；改变物理batch会改变本实现MoE aux统计；改变T/数据/轮次/LR/噪声目标则直接改变recipe。所有重要改变新建版本并说明证据、语义和代价。

本机当前Hybrid约50小时是这套实测配置的总预算估计，不是最优下限；余时看[实时检查](../reports/hybrid_official_mini_progress_latest.json)。198M全量SFT此前短测外推纯更新约4.5天，尚未正式长跑；Omni完整七阶段可能数周，不能把“今天Hybrid有结果”或“几天学习主流程”当成全部实验结束的承诺。

学习顺序从[根README](../README.md)进入：一条真实样本 → tokenizer/labels → forward/CE/aux → backward/AdamW/scheduler → resume → profiling/OOM/数值诊断 → attention/MoE对照 → 多模态 → dLM/偏好/RL。全部交付保留在本机 `D:\minimind`。
