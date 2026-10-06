# 198M 第9,768步完整恢复起点

2026-10-06恢复原全量预训练前已用CPU mmap核对step=9,768、监督token=319,137,649、epoch=0/cursor=1,406,592、171个FP32 state键、170组AdamW和四类RNG，config与model/train/data源码匹配。原恢复文件在下一个自动保存前建立硬链接，避免latest替换后丢失本次恢复起点；没有再次复制2.38GB。

- [原始完整恢复状态](training_resume.pt)
- [恢复前检查](../../../../reports/ar_resume_precheck_20261006.json)
- [实际恢复后的连续更新检查](../../../../reports/ar_resume_verification_20261006.json)
- [启动命令与PID](../../../../runs/pretrain_resume_after_hybrid_20261006.json)

这不是一个完成训练的模型，也没有宣称本次额外做了逐位确定性恢复实验。
