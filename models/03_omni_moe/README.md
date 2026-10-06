# 315M Omni-MoE

Thinker 198M、Talker约114M、视觉/语音投影；冻结的外部编码器单独计入资源预算。覆盖文本、图像、语音输入与文本/语音输出。

正式范围依据固定版本full MoE shell建议，包含T2A全参数、audio_proj、A2A全参数、vision_proj、I2T全参数、A2A回训、vision_proj再对齐七阶段。先前将后三项判为仅自有拓展的结论已纠正；[来源与epoch表](../../docs/recipe_decisions.md)说明依据和预算，不直接执行默认mini脚本。

## 当前状态

**2026-10-04：正式阶段尚未开始，正在等待自训198M SFT基座。** 七阶段完整配置为`configs/omni_0*_full_v1.json`，总17,799,071次样本呈现、139,065次更新。服务器独立环境、完整数据部署与字节校验已完成；完整315M真实T2A B16×累积8及各输入/更新模式短检查已通过，仍不代表正式训练质量。14:05实查远端控制器存活并等待依赖。启动桥/回传桥负责将记录和正式checkpoint保留到本模型目录；[总计划与recipe](../../docs/experiment_plan.md)、[实时表](../../reports/project_status.md)。以下本机offload测试属于历史设备选择证据，正式服务器候选使用GPU常驻AdamW。

官方源码已获取并固定commit，独立环境用 `source env/activate_omni.sh`。完整315M真实T2A已通过更新与精确恢复，真实A2A随机和长音频更新也已通过。当前需要AdamW moment CPU暂存；A2A的5.7GiB预算加冻结SenseVoice已实测，但RAM余量较小。I2T全参数与视觉投影更新已通过，语音持续资源验证仍待完成，不把短测当正式训练完成。

先读 [Omni 显存诊断](../../docs/debug/004_omni_memory.md)。原始记录在本目录 `runs/t2a_memory_*`。完整官方语料的固定版本与SHA在 `data/manifests/minimind_omni_full.json`，冻结外部模型来源在 `data/manifests/omni_frozen_models.json`，共享下载排在语言SFT数据之后。

SenseVoice、SigLIP2、Mimi、CAM++现已完整下载并校验。真实SenseVoice CPU编码短测在`runs/audio_encoder_cpu/`；Mimi真实codes解码结果在根`reports/mimi_codec_verification.json`。数据分片与官方Dataset等价测试通过，见[存储和音频时长诊断](../../docs/debug/005_omni_storage_and_duration.md)。这些均是正式训练准备，不是已训练出Omni。

真实SigLIP视觉编码器检查在`runs/vision_encoder_cpu/`，确认无缺失/不匹配权重和`[1,64,768]`输出。完整315M权重迁移预检在`runs/transfer_preflight_early_ar/`；实际训练器的CPU恢复检查在`runs/trainer_cpu_resume_v2/`，参数、AdamW、cursor、RNG逐项一致。`v1`缺依赖失败日志保留，独立环境已经补齐。恢复检查使用小模型验证控制逻辑，不算315M正式训练。

## 从哪里开始读

- 源码：`sources/minimind-o/model/model_omni.py`。来源与commit见根目录 `sources/provenance.json`。
- 配置：本目录 `configs/`。每次运行在 `runs/<run_id>/` 独立保存记录。
- 公共数据：根目录 `data/raw/` 和 `data/processed/`，不在模型目录重复复制。
- 先读根目录 `docs/learning/01_training_step.md`，再对照该模型 forward、loss、优化器与生成路径。
- 全部实验的阶段、控制变量和完成口径见根目录 `docs/experiment_plan.md`。

真实T2A首步通过、第二步OOM的旧记录保留在`runs/real_t2a_B1_T1536/`；完整GPU恢复的初次失败在`runs/trainer_gpu_resume_v1/`。累积1成功而累积2失败、分阶段显存证明：常驻gradient、AdamW moment与下一次forward叠加造成峰值。卸载修复后的完整315M真实T2A恢复对照在`runs/trainer_gpu_resume_v2_offload/`，连续训练与中断恢复的参数、AdamW、RNG、cursor全部逐位相同。5.7GiB不卸载仍OOM的反例在`runs/real_t2a_cap57_v3/`。

完整T2A已校验、分片和划分train/val。全语料随机抽2048条，经过官方轮次抽样/labels构造后，长度768在两种epoch增强下分别有13/16条缺少至少一个audio STOP；1536对应0/1条。原始逐样本记录在`reports/omni_t2a_label_audit_v1/`，这是样本估计，不能宣称全量无截断。

完整A2A也已校验/分片：414,024条，训练413,573、验证451。`reports/omni_a2a_audio_audit_v1/` 保存全量音频头部、时长和资产共享审计；相同音频资产有12个跨train/val，不能把conversation hash分割称为未见音频评估。`reports/omni_a2a_batches_v1/` 和 `omni_a2a_batches_L3072_v1/` 比较真实增强的截断：长尾混合样本L1536有7次零监督，L3072为0，后者已通过完整315M长样本更新与451条验证（`runs/real_a2a_long_L3072_v2/`），但主机RAM最低仅约170MiB。

真实A2A短测证据在 `runs/real_a2a_random_cap57_v1/` 与 `runs/real_a2a_long_cap57_v1/`。输入 `[1,9,1535]`、文本标签 `[1,1535]`、音频标签 `[1,8,1535]`，audio projector有非零梯度；fbank及marker对应关系保存于各run的`learning/`。本次仍从早期AR迁移预检权重开始，不能用这些loss报告已获得Omni能力。

完整I2T已校验并重排为710个有界Arrow分片（21.67GB，Parquet原始4.93GB；图像字典展开造成膨胀）。[图像划分与占位符诊断](../../docs/debug/008_i2t_split_and_placeholders.md)说明为什么改验证集、如何保留纯文本混合、如何修复19422条回答中的占位符。正式视觉训练采用显式user_only策略和split_image_v2；旧预检仍保持原配置，不能混用评估分数。

I2T完整GPU路径在 `runs/real_i2t_all_B2_L768_v1/` 与 `runs/real_i2t_vision_proj_B2_L768_v1/`：B2×L768、累积2，全参数/投影模式各3次更新并完成原2929条验证；vision projector梯度非零。全参数模式峰值5.14GiB并暂存moments，投影模式峰值1.95GiB、无需暂存。两者仍为早期权重短测，原划分含共享图像，不能用其CE证明视觉泛化。随后 `runs/real_i2t_proj_B16_grouped_role_v2/` 已使用分组划分和role-aware修复，通过B16×累积2的8更新与2513条完整新视觉验证。

[官方参考的生成与音频I/O诊断](../../docs/debug/009_omni_generation_and_audio_io.md)在 `reference_evaluation/` 保留原始文字、Mimi codes、FLOAT WAV和ASR检查。参考权重固定SHA，仅用于校准评估，不用于初始化我们的正式Omni。独立环境现有本地ffmpeg7.0.2，PyPI wheel与二进制hash已记录。

A2A评估新增未见输入录音分组：原451条验证中359条、237个资产，原训练与完整验证保留。跨阶段相同conversation检查也已完成，见根 `reports/omni_multistage_overlap_v1/`。新验证保存逐条原始行号与NLL分子/标签数，支持同一次forward的分组汇总；这项监控不改变训练loss，尚不能据此宣称语音泛化质量。

资源补测已完成：相同A2A配置采用worker0后，32更新及451条验证均通过，WSL最低可用约800MiB、速度约1.40样本/s，较worker1的297MiB有更好余量；非确定性运行的权重并不逐位相同，原始比较保留。服务器完整315M T2A更新及确定性恢复也通过，资料已回传到[独立实验目录](runs/server_readiness_t2a_v1/)。本机仍是语言主线与学习资料根目录；远端语音/视觉输入须另行验证后才能决定对应正式阶段的设备。

上述远端输入验证随后已完成：[五种真实更新模式](runs/server_readiness_inputs_v1/)均通过，含SenseVoice长语音、SigLIP真实图像及投影层梯度；没有使用optimizer offload。环境和调度器的两类失败保留在debug011。执行设备和正式预算仍按完整阶段准备和当时空闲资源确定，不能把一次快照当永久占卡。

学习新增[无输入模态时的零梯度与AdamW](../../docs/learning/03_omni_data_and_loss.md)：真实监督loss有限、参数变化，也可能只来自dummy梯度、weight decay和历史moment。CPU机制检查在 `learning/inactive_projector_v1/`，正式模型未受修改。
