# 205M Hybrid：完整两轮训练产物

2026-10-06：113,082步、735,833,934监督token；完整模型结构与数据预算。**训练预算完成，评估结果与生成样例已记录。**

- [结果、失败分析与基座对照](../../../../reports/hybrid_final_review_20261006.md)
- [完整FP32恢复状态](training_resume.pt)：最终保存的硬链接快照，含AdamW、RNG、epoch=2/cursor=0，约2.47GB。
- [选定推理权重](../../runs/hybrid_sft_official_mini_2ep_v1/checkpoints/best_validation.pth)：FP16，选择第113,082步，约421MB。
- [最后一步推理权重](../../runs/hybrid_sft_official_mini_2ep_v1/checkpoints/model_step_0113082.pth)：FP16，约421MB。
- [CPU完整性审计](report.json)、[文件大小与SHA256](files.json)、[审计源码](source_verifier.py)。
- [本轮锁定config](../../configs/hybrid_sft_official_mini_2ep_v1.json)、[训练曲线](../../runs/hybrid_sft_official_mini_2ep_v1/plots/training.png)、[全部评估](../../runs/hybrid_sft_official_mini_2ep_v1/evaluation/)。

这些是有完整血缘和失败证据的实验权重，不是通过可靠问答验收的成品。`best_validation`按固定512条选择；最终全1,066条CE为1.638091，不能与固定子集CE混为一条相同评估序列。完整性审计不是额外的下一步逐位resume等价实验。
