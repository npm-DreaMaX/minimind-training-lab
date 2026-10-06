# 指定时间的 Hybrid 训练快照

**这是中间训练结果，不是完整模型训练完成证明。后台正式训练继续。**

捕获时间：2026-10-03T13:48:45.544918+08:00。阶段：`hybrid_adapt_official_full_v1`，该阶段第 500 / 5,000 步，2,712,906 个有效训练 token（仅此阶段，不包含基座历史）。

`hybrid_205m_fp16.pth` 是完整模型参数导出；`training_resume.pt` 保留FP32模型、AdamW、RNG和cursor。导出保留完整205M结构；它不是官方权重改名。官方预训练初始化来源和新增模块迁移见 `transfer.json`。

如果存在 `checkpoint_evaluation.json`，其中验证/生成对应相同步数的训练模型；FP16导出未单独重评。不能把其他步数的最好分数归给这一份快照。`receipt.json` 保存权重SHA与实际阶段状态。

完整配置和源码快照在本目录；持续更新的日志、曲线和后续权重见模型的 `runs/` 及项目根README。
