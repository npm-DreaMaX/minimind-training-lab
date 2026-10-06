> 2026-10-06学习交付整理前的导航快照。正文保留历史状态；相对链接调整为从本目录指向原项目。当前入口见[根README](../../README.md)。

# MiniMind 大模型训练学习项目

本机目录 `/mnt/d/minimind`（Windows `D:\minimind`）是整个学习项目的入口。目标是理解数据、模型、优化、系统与 debug，并通过真实实验建立训练能力。

**整体计划与各模型recipe入口：[训练与学习计划及最新停止决定](../../docs/experiment_plan.md)。** 2026-10-06用户要求停止当前198M普通MoE分支，已保存停止并取消自动SFT接续；不再按旧队列自动推进。此前各分支recipe、[配置指纹清单](../../reports/recipe_inventory_20261004.json)和[旧计划快照](../../docs/experiment_plan_20261003_history.md)保留。Hybrid生成质量仍未合格。

## 当前进展：2026-10-06，普通MoE已按用户要求停止

**本机训练进程已经退出。** 17:30按用户要求停止198M普通MoE，在optimizer边界第13,209/58,752步保存，累计431,460,605监督token；FP32参数、AdamW、RNG、数据cursor和scheduler位置保留。自动SFT接续进程也已终止，不会继续训练；Omni所需自训SFT依赖未完成，没有切换新基座或启动新训练。训练器的`paused`表示可恢复保存状态，本项目执行决定是停止该分支，不能自行恢复。

[停止指令与信号记录](../../runs/stop_moe_by_user_20261006/request.json) · [CPU恢复文件核验](../../runs/stop_moe_by_user_20261006/verification.json) · [停止快照](../../models/01_moe/deliveries/20261006_user_stop/README.md)。

当前入口：[198M实时状态](../../models/01_moe/runs/pretrain_full_v1/status.json) · [恢复日志](../../logs/pretrain_resume_after_hybrid_20261006.log) · [198M曲线](../../models/01_moe/runs/pretrain_full_v1/plots/training.png) · [全项目实时表](../../reports/project_status.md)。恢复记录见[进程/命令](../../runs/pretrain_resume_after_hybrid_20261006.json)和[恢复前源码/config/状态检查](../../reports/ar_resume_precheck_20261006.json)。

**205M Hybrid的官方mini两轮已完整执行：113,082次更新、735,833,934监督token，未缩减模型和预算。** 最终完整1,066条验证CE=1.638091；固定512条CE=1.664181。最终第113,082步的完整FP32恢复状态、AdamW/RNG及FP16推理权重均已保存，CPU完整性和文件SHA核验通过。

**Hybrid仍需分析具体评估与生成表现，不能把预算完成等同于研究目标完成。** 同一512条验证、同样导出/加载和GPU精度下，官方AR参考CE=1.307459，Hybrid=1.664561。CPU/GPU、标准/256token生成仍有事实、算术、代码和重复错误；两者在10个严格格式题均未通过，不能将此当成全面准确率。Hybrid六层转换后的能力恢复仍需研究，当前没有机械追加第三轮，也没有用失败探针答案回填训练。

重点阅读：[完整结果与失败分析](../../reports/hybrid_final_review_20261006.md) · [最终权重与恢复点导航](../../models/02_hybrid_moe/deliveries/20261006_final_integrity_v1/README.md) · [Hybrid曲线](../../models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/plots/training.png) · [全部原始评估](../../models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1/evaluation/) · [真实预算审计](../../reports/hybrid_final_budget_20261006.json)。

本轮血缘是**官方198M SFT基座→6层注意力替换→完整205M全参数SFT**，不是我们的从零198M已训练完成。完整原文件、holdout、零标签过滤、T768截断和固定离线增强的差异见[锁定方案](../../docs/hybrid_official_mini_2ep_v1.md)。模型与配置在[独立目录](../../models/02_hybrid_moe/README.md)，[正式控制器](../../runs/formal_hybrid_official_mini_2ep_v1/)已记为`pipeline_budget_complete`、`quality_review_pending=true`。

**本次没有执行Hybrid预训练；全参数SFT不等于完整预训练＋SFT。** 作者确实公开过这种转换SFT方法，但其约71M Dense示例生成也存在明显质量问题；我们这次205M MoE不能被描述成复现了官方已验证的更强模型。[作者命令、质量证据与范围复核](../../docs/hybrid_official_mini_2ep_v1.md)。

**为什么Hybrid还不如官方MoE、它与KDA有什么不同：**[第二课新增解释](../../docs/learning/02_hybrid_attention.md#为什么这次没有超过官方moehybridkda与训练起点)对照Gated DeltaNet/KDA、GQA/MLA、六层随机替换与预训练的区别，以及CE从10.7034恢复到1.6642仍未追平基座的证据；架构学习价值不等于本次训练效果保证。

本轮更新计算累计47.58小时，首末更新墙钟跨度71.52小时（含休眠和时钟异常）；不能直接套用官方3090普通MoE mini各一轮的约3小时。[官方时长对齐](../../docs/official_hours_and_our_budget.md) · [吞吐与profile](../../docs/learning/05_throughput_and_profiling.md) · [睡眠、旧防睡眠机制失效及本次API心跳修补](../../docs/debug/015_sleep_pause_and_eta.md)。防空闲请求已验证重复刷新，尚不能保证主动睡眠/电池保护等情形。

历史实验没有删除：旧适应分支停在1,369步；10万条自选子集提案被撤回，未执行其正式训练；资源探针、18:00中间快照和失败均保留。[初次500步](../../models/02_hybrid_moe/deliveries/20261003_first_500/) · [18:00快照](../../models/02_hybrid_moe/deliveries/20261003_1800_official_mini_v1/) · [早期恢复检查](../../models/02_hybrid_moe/deliveries/20261003_mini_resume_integrity_v1/README.md) · [迁移质量诊断](../../docs/debug/013_hybrid_transfer_recovery.md) · [预算与效率诊断](../../docs/debug/014_hybrid_budget_and_efficiency.md)。

## 已有基础与原实验路线

**198M MoE曾于10月6日14:18恢复，17:30按用户要求再次停止。** 所有数据和实验记录保留。官方8.275GB预训练语料已通过SHA256与全量分词检查：846.9万条、22.17亿输入token。14.096GB完整SFT语料已完成校验、分词和审计：510万训练记录、1536长度下约30.50亿监督token；Omni完整T2A/A2A/I2T数据也已完成校验、分片和审计。历史预算见[实验与学习计划](../../docs/experiment_plan.md)。

**三个模型尚未全部训完。** [实时阶段表](../../reports/project_status.md)每60秒更新本机状态与最近回传的远端状态；[原正式预算与执行安排](../../docs/formal_execution_v1.md)保留初始设计。普通AR已停止；旧Hybrid/AR对照控制器仍暂停。Omni完整数据已在独立服务器通过部署校验，最近回传控制器状态为等待自训SFT基座；这次未重新检查远端存活。

已检查的正式里程碑：第9000步，累计294,091,490个有效token（约一轮预算的15.3%），固定512条验证CE=2.05964。生成仍有重复和事实错误：第8000步正确回答北京并结束，第8500步却出现“北京的首都是东京”；CE下降不能代替生成质量。原始样例全部保留。当前步数看实时表/status；这些是暂停之前的AR结果，不能作为Hybrid结果。

直接查看：[实时状态](../../models/01_moe/runs/pretrain_full_v1/status.json)、[训练曲线](../../models/01_moe/runs/pretrain_full_v1/plots/training.png)、[MoE路由曲线](../../models/01_moe/runs/pretrain_full_v1/plots/routing.png)、[显存/利用率/温度/功率](../../models/01_moe/runs/pretrain_full_v1/plots/system.png)、[正式日志](../../logs/pretrain_full_v1.log)、[评估与生成](../../models/01_moe/runs/pretrain_full_v1/evaluation/)、[checkpoint记录](../../models/01_moe/runs/pretrain_full_v1/checkpoint_progress.jsonl)、[全量数据报告](../../reports/pretrain_data_audit.md)。

- 本机训练环境已就绪；`source env/activate.sh` 使用项目缓存与 Linux IPC 临时目录。
- 真实数据的正式训练器冒烟已完成；batch=8、长度384、梯度累积16 的复测也已通过，约 1 万有效监督 tokens/s（含 padding 的名义吞吐约 2 万）。
- 随后的512长度候选也已通过真实数据复测。正式预训练采用 **batch6、长度512、累积24**，保持完整198M架构并减少前缀截断；取舍见[recipe决策](../../docs/recipe_decisions.md)。
- 数据/数学/采样器/scheduler 合同测试通过，新增 SFT 模板与 labels、DataLoader随机数隔离对照，原始结果在 `logs/training_contract_tests_v3.log`。
- SFT 的 batch2×1536 长度已通过真实数据验证，峰值 allocated 约 4.93GiB；短测不能代表长训练质量。
- 205M Hybrid 早期更新/cache测试通过；最新真实初始化对照发现的FP32 FLA门梯度偏差已定位到默认TF32 dot精度；IEEE/TF32x3独立参考及完整205M更新检查通过，IEEE精确恢复通过。[兼容性检查](../../docs/debug/003_hybrid_router_and_kernels.md)与[新数值失败](../../docs/debug/007_hybrid_bf16_gradient.md)分别保留。
- 完整315M Omni真实T2A＋AdamW moment卸载已通过GPU精确恢复对照；真实A2A随机与3072长度长音频更新、I2T全参数和视觉投影更新均通过。[显存诊断](../../docs/debug/004_omni_memory.md)保留OOM和修复。A2A在本机RAM余量较小，短测通过不等于长时间资源验证完成。
- Omni分片存储与官方Dataset逐项等价检查已通过，冻结编码器/codec已下载并校验；[原始int64数据超RAM及README音频时长口径差异](../../docs/debug/005_omni_storage_and_duration.md)均有实际证据。完整T2A/A2A均已分片与审计，I2T也已校验并完成分片与审计。
- Hybrid BF16精度候选的输出接近，但衰减门梯度未通过检查，暂不采用；[原始失败与诊断计划](../../docs/debug/007_hybrid_bf16_gradient.md)保留。
- I2T发现原验证图像大量与训练共享、回答里的占位符被错误展开；[数据修复与分组划分](../../docs/debug/008_i2t_split_and_placeholders.md)保留全量审计、原行为与修复对照，新验证包含606个未见图像资产。
- [早期生成重复与BOS对照](../../docs/debug/006_early_generation_and_bos.md)：验证CE下降不等于模型已经能可靠回答，失败样例保留。
- [Omni生成与音频评估](../../docs/debug/009_omni_generation_and_audio_io.md)：官方参考的文本/真实WAV、ASR一致性、生成预算对照及音频依赖故障；明确区分参考结果和自训模型。
- [WSL DataLoader 故障及修复](../../docs/debug/002_wsl_dataloader_socket.md)已保留。
- [状态文件并发读取故障](../../docs/debug/010_live_status_on_windows_mount.md)：D盘替换期间短暂缺失曾导致预检调度器退出，主训练持续正常；有界重试已复测，接续进程已恢复。
- [作业存活、独立环境、部署与完成口径](../../docs/debug/012_jobs_deployment_and_completion.md)：返回PID不代表后台存活，pip检查不代表项目import完整，3步complete也不代表正式模型训完；含本次实际检查与修复。
- [第六组资源检查](../../reports/preflight_window_v6c_review.md)与[第七组配置选择](../../reports/preflight_window_v7b_review.md)：两组已完成。198M SFT保留B2；Hybrid按实测选择较大batch；A2A worker0改善RAM余量。当前正式预训练配置不变，没有新的本机GPU预检排队。
- [服务器完整Omni验证](../../models/03_omni_moe/runs/server_readiness_t2a_v1/)：315M真实T2A更新、GPU常驻AdamW和逐位恢复通过，证据已回传；尚不代表语音/视觉输入已通过或模型已经训成。
- 后续[真实语音/视觉输入检查](../../models/03_omni_moe/runs/server_readiness_inputs_v1/)也已完成：五种更新模式保持315M完整模型，投影梯度非零。原始中断与修复见[服务器环境/空闲卡诊断](../../docs/debug/011_server_omni_environment_and_idle_guard.md)；这些仍是短检查。
- 第一项正式配置：[完整预训练配置](../../models/01_moe/configs/pretrain_full_v1.json)。完整SFT统计见 [SFT数据审计](../../reports/sft_data_audit.md)；预训练分词日志见 `logs/prepare_pretrain_full.log`；正式训练日志为 `logs/pretrain_full_v1.log`。

以下为已完成的硬件基础检查：

- 已完成本机与服务器的完整 **198,416,640 参数 MoE** 小规模训练测试；官方模型源码没有改动。
- 两边均通过 forward、loss＋aux loss、backward、非零路由梯度和 AdamW 参数更新检查。
- 服务器 2、7 号 RTX A4000 16GB 均通过训练测试；2 号卡与本机均通过确定性 checkpoint/resume 比较。
- 第一次非确定性测试的逐位恢复比较失败，原始失败日志保留；开启确定性算法后的复测通过，没有掩盖失败。
- 同样 batch=4、sequence=512：本机约 **12,660 input tokens/s**，服务器约 **10,586 input tokens/s**。这只是短测，软件版本不同，不是纯 GPU 性能排名。
- 本轮分工：**本机优先205M Hybrid；服务器用于更高显存需求的实验。**模型/配置仍须逐项验证，不能把 MoE 的成功外推为所有模型已通过。
- 硬件小测不计入正式模型训练完成；正式阶段以每个 run 的 `status.json` 和评估结果为准。

## 导航与学习顺序

1. [硬件小测报告](../../reports/hardware_comparison_20261003.md)：先理解“能训练”和“训练快”如何验证。
2. [测试源码](../../tools/hardware_smoke.py)：数据形状 → forward → loss → backward → optimizer → checkpoint。
3. [原始实验目录](../../models/01_moe/runs/20261003_hardware/)：每一步指标、显存、温度/功耗和失败记录。
4. [官方模型源码](../../upstream/model/model_minimind.py)与[官方训练器](../../upstream/trainer/train_pretrain.py)：对照本项目测试代码，区分模型算法与测试工具。
5. [第一课：从文本到参数更新](../../docs/learning/01_training_step.md)：数据、目标、显存、吞吐、恢复与故障诊断；附[真实198M逐位置梯度示例](../../models/01_moe/learning/full_model_gradient_trace_v1/)。
6. [第二课：Hybrid Attention 与 MoE](../../docs/learning/02_hybrid_attention.md)：张量形状、状态递推、训练目标与实验对照；用[真实第一个batch](../../models/02_hybrid_moe/learning/official_mini_first_batch_v1/README.md)核对tokenizer、labels和右移。
7. [第三课：Omni 数据与损失](../../docs/learning/03_omni_data_and_loss.md)：音频/图像如何进入Thinker、8路音频标签如何对齐、哪些冻结方式会切断梯度。
8. [第四课：checkpoint与恢复](../../docs/learning/04_checkpoint_lineage_and_resume.md)：阶段初始化与精确resume的区别、worker增强如何恢复、实际连续/中断对照。
9. [第五课：吞吐与profiling](../../docs/learning/05_throughput_and_profiling.md)：真实198M性能trace、首次AdamW开销、CPU/CUDA计时与显存统计误读。
10. [第六课：评估与数据质量](../../docs/learning/06_evaluation_and_data_quality.md)：CE与生成、checkpoint选择、重复数据、跨模态资产共享和截断的实验含义。
11. [第七课：AR与掩码扩散](../../docs/learning/07_ar_and_masked_diffusion.md)：完整64M Dense CPU检查、同位置去噪目标、双向attention，以及零监督NaN/残留mask反例；尚未正式训练。

| 模型/研究方向 | 独立目录 | 状态 |
|---|---|---|
| Full Attention＋MoE，198M | `models/01_moe/` | 用户已停止；13,209步完整保存，自动SFT已取消 |
| Hybrid Attention＋MoE，205M | `models/02_hybrid_moe/` | 完整205M、mini两轮预算及最终评估已完成；评估结果与生成样例已记录，待改进 |
| Omni-MoE，315M | `models/03_omni_moe/` | 七阶段正式预算已保存；完整数据部署校验通过，远端控制器等待正式SFT基座 |
| Dense AR 对照 | `models/04_dense_ar/` | 待实验 |
| dLM | `models/05_dlm/` | 完整Dense CPU结构/目标检查完成，作者边界问题保留；待后续正式实验 |
| Preference／RL／Agentic RL | `experiments/06_preference/`、`07_rl/`、`08_agentic_rl/` | 待基座与训练器验证 |

公共数据放 `data/`，见[数据导航](../../data/README.md)，避免每个模型重复复制。每个模型的配置放其 `configs/`，运行日志、评测和 checkpoint 放其 `runs/<run_id>/`。大型文件保存在磁盘，不纳入 Git；代码、配置、报告和指标纳入版本记录。

## 本机与服务器的分工

服务器工作目录为 `/new_data/REMOTE_USER/minimind`。本机保留代码、配置、命令、学习资料、指标、失败记录和评测；正式训练的重要权重也应回传本机，不能只留下远程链接。

本次已回传服务器的小测日志和结果。服务器测试产生的随机模型全状态 checkpoint 约 2.38GB/份，仍保留在远端并在报告列出路径；没有为短测重复传输这些非正式训练权重。本机自己的两份测试 checkpoint 已保留在本机实验目录。

最新执行决定：普通MoE已停止、自动SFT已取消；Hybrid本轮预算与评估已完成但评估结果与生成样例已记录，没有追加或新开训练。Omni的完整自训SFT依赖未完成。旧优先/对照控制器与全部实验资料保留，后续方向以用户新指示为准；[实时阶段表](../../reports/project_status.md)中的远端信息仍是最近回传状态。

以下是保留的旧扩展预算估算，不是正在执行的新官方mini路线：原预训练剩余约两天，随后原始SFT约4.5天；Hybrid完整续训约5.8天、SFT约11.5天，Omni七阶段也属于数周预算，均未含全部验证/故障开销。几天内不可能承诺完成这里全部完整语料实验，不能靠缩减预算把它写成成功。
