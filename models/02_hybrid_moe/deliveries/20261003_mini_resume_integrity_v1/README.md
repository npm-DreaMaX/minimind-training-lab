# 正式Hybrid第100步恢复文件完整性

这是一份正式训练中间恢复点，不是完整两轮的最终模型。

CPU核验通过：完整205,623,072参数，所有tensor形状/FP32/finite、共享embedding/head、AdamW全部参数的两份moment与step、RNG和cursor、固定配置及模型源码hash一致。新线性模块与继承模块都已发生真实更新。详细证据见 `report.json`，原样恢复文件为 `training_resume.pt`。

本次检查只验证状态完整性，不模拟下一步恢复的逐位等价，也不证明生成质量；先前数值恢复合同在模型runs/resume_tf32x3_v5。后续训练继续，最新状态见根README。
