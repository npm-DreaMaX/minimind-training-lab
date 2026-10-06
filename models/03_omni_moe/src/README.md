# Omni 数据集兼容性修复

原始来源：`sources/minimind-o/dataset/omni_dataset.py`，官方固定 commit `f900448c608318c53314ebf8a947ab05cd8c038e`。原始 checkout 未修改。

`omni_dataset_role_aware.py` 是该文件的副本；完整差异保存在 `role_aware_image_placeholders.patch`。它只把 user 角色的 `<image>` 当作输入占位符，assistant 回答里的字面 `<image>` 保持普通文本，不再展开成64个图像控制 token。保留原随机调用次数，正常样本的增强不受影响。

这是本项目根据真实数据发现问题后做的修复，不是宣称官方已修复。通过 `image_placeholder_policy: user_only` 显式选择；默认 `official` 保留旧行为，历史运行不改写。

证据：[数据质量诊断](../../../docs/debug/008_i2t_split_and_placeholders.md)、`../runs/i2t_role_policy_contract_v1/result.json`、`../runs/i2t_role_policy_contract_v3_noise_workers/result.json`。修复改变问题样本的输入与监督目标，必须作为数据 recipe 变更记录，不能声称纯性能优化。
