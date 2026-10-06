# 198M Full Attention＋MoE

官方8层、hidden768、4专家Top-1；这是自己从零训练的语言基座和其他分支的来源。

## 当前状态

**2026-10-06 17:30：用户要求停止该分支，已在第13,209步保存退出，自动SFT接续已终止。** 累计431,460,605监督token，预训练预算未完成；不再依据旧计划自行恢复。训练器状态`paused`只表示完整恢复点可用。[停止记录](../../runs/stop_moe_by_user_20261006/request.json) · [恢复状态核验](../../runs/stop_moe_by_user_20261006/verification.json) · [快照](deliveries/20261006_user_stop/README.md)。

历史：14:18曾从第9,768步恢复。[恢复命令/进程](../../runs/pretrain_resume_after_hybrid_20261006.json)、[原始恢复日志](../../logs/pretrain_resume_after_hybrid_20261006.log)、[恢复前核验](../../reports/ar_resume_precheck_20261006.json)。

完整官方预训练语料已校验并分词，198,416,640参数模型的从零训练已经停止。原全量预算58,752次optimizer更新、约19.196亿有效监督token没有完成；保存状态见[实时状态](runs/pretrain_full_v1/status.json)。完整SFT下载、分词、审计和随机吞吐测试已完成，数据保留；自动进入SFT的进程已取消，SFT未开始。

- [正式配置](configs/pretrain_full_v1.json)、[配置与源码快照](runs/pretrain_full_v1/source/)
- [训练曲线](runs/pretrain_full_v1/plots/training.png)、[路由曲线](runs/pretrain_full_v1/plots/routing.png)、[逐步指标](runs/pretrain_full_v1/metrics.jsonl)、[GPU记录](runs/pretrain_full_v1/gpu.csv)
- [验证与生成](runs/pretrain_full_v1/evaluation/)、[checkpoint位置](runs/pretrain_full_v1/checkpoints/)、[保存进度](runs/pretrain_full_v1/checkpoint_progress.jsonl)
- [预训练样本逐token检查](learning/pretrain_sample/)、[SFT样本逐token检查](learning/sft_fixture_sample/)

固定512条验证样本的CE：初始化8.8943 → 第500步3.5487 → 第1000步2.8574 → 第3000步2.3603。生成仍有重复和知识错误，不能把loss下降当作已获得可靠回答能力。官方发布权重用于`runs/official_sft_reference_*`等独立参考评估，并作为另一条Hybrid分支的初始化；没有初始化这里的从零正式训练。

## 从哪里开始读

- 源码：`upstream/model/model_minimind.py`。来源与commit见根目录 `sources/provenance.json`。
- 配置：本目录 `configs/`。每次运行在 `runs/<run_id>/` 独立保存记录。
- 公共数据：根目录 `data/raw/` 和 `data/processed/`，不在模型目录重复复制。
- 先读根目录 `docs/learning/01_training_step.md`，再对照该模型 forward、loss、优化器与生成路径。
- 全部实验的阶段、控制变量和完成口径见根目录 `docs/experiment_plan.md`。
