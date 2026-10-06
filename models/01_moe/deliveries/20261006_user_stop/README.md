# 用户停止普通MoE后的完整保存点

2026-10-06 17:30用户要求停止当前普通MoE，不再继续该分支。训练器接收SIGTERM后在optimizer边界第13,209步保存，累计431,460,605监督token，epoch0、cursor1,902,096。原58,752步预训练预算未完成，SFT未开始。

`training_resume.pt`是最后完整恢复文件的硬链接快照，不额外复制约2.38GB数据。保留FP32参数、170组AdamW状态、四类RNG、数据位置、step/config/optimizer LR；FP16导出在`../../runs/pretrain_full_v1/checkpoints/model_step_0013209.pth`。保存可恢复状态不表示授权恢复，当前决定是不再训练此分支。

[停止动作](../../../../runs/stop_moe_by_user_20261006/request.json) · [CPU完整性与SHA核验](../../../../runs/stop_moe_by_user_20261006/verification.json) · [核验脚本](../../../../runs/stop_moe_by_user_20261006/verify_stop.py) · [原始日志](../../../../logs/pretrain_resume_after_hybrid_20261006.log)。核验没有进行新的GPU训练或恢复等价性实验。自动SFT接续进程同步终止。
