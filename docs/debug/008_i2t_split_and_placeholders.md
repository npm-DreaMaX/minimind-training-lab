# 008：视觉数据的验证污染、文本混合与占位符误展开

## 现象与检查路径

完整 I2T Parquet 已通过固定 revision、字节数和 SHA256 校验，包含2,904,511条记录。按 conversation hash 划分虽能避免完全相同的对话跨集合，却不能自动保证图像独立。对全量编码图像做 SHA256：原验证2929条中只有15条使用训练未见的资产，训练/验证共享2702个图像。原报告保留在 `reports/omni_i2t_image_asset_audit_v1.json`，不能把该划分的loss当作新图像泛化能力。

接着经过真实 Dataset 读取256个均匀抽样记录、两个epoch，观察 `[9,L-1]` 输入、文本 labels、pixels 与图像 marker。L768时235个样本有64个marker，17个为0，4个为128。最初可能是坏图像、图像丢失或错误占位符；不能只把非64样本一律删除。

## 证据与根因

全量按角色审计原始对话，结果在 `reports/omni_i2t_placeholder_audit_v1.json`：

- 230,266条没有 user 图像占位符，附带8×8缩略图，实际是纯文本问答混合样本。它们不通过64个图像位置读入视觉特征。这个事实不等于已经证明作者混合它们的动机，也不能仅因缩略图小就判为损坏。
- 2,674,245条有 user 图像占位符。没有多重 user placeholder；原始数据没有直接写入特殊 token `<|image_pad|>`。
- 19,422条的 assistant 正文也含字面 `<image>`。官方 `create_chat_prompt` 对所有角色重排它，随后 `__getitem__` 全局替换为64个特殊图像 token。这会把回答文本误当成第二幅输入图像，并给这些控制 token 生成监督 labels。
- 以原始第48行为例，修复前有128个输入图像 token、64个受监督图像 token；修复后为64和0。回答中的字面文本仍保留，不是删除整条训练记录。

## 修复与实验含义

第一项修改是分组划分，`tools/group_i2t_split.py` 将实际使用图像的记录按图像 SHA256 连接，再把所有完全相同的对话连接，按连通分量整体分配。纯文本记录不因共同的未使用缩略图而全部捆在一起。新索引 `data/processed/sft_i2t_shards_v1/split_image_v2/`：训练2,901,800条、验证2711条；其中视觉训练2,671,732条、视觉验证2513条，验证包含606个独立图像资产。独立检查确认共享视觉资产和完全相同对话均为0。旧 `audit_v1` 不变，4020条记录的集合归属改变，新旧loss不能直接视为同一评测。仍未做感知近重复或语义去重。

第二项修改在独立源码副本 `models/03_omni_moe/src/omni_dataset_role_aware.py`，附来源与patch。只有 user 的占位符重排并展开；assistant 里的字面文本保持原位。保留原随机抽样调用，避免给无关样本重新分配增强。显式配置 `image_placeholder_policy=user_only` 才启用；官方 checkout 和历史 run 保留原样。

这是数据与评估 recipe 的正确性修复，不改315M模型结构，不是严格数学等价的工程加速。正式视觉投影对齐使用新 split 的 visual_train/visual_val，避免只训练投影时纯文本样本没有有效视觉梯度；后续全参数视觉阶段使用完整混合样本，并分别报告视觉和文本验证、检查语音遗忘。全参数视觉阶段在官方train.sh的full建议中也存在，先前归类为“仅自有消融”已在recipe_decisions纠正。

## 验证与保留的失败

`i2t_role_policy_contract_v1`：两个epoch共504次正常/纯文本读取逐项完全一致，32次问题样本的错误监督图像 token 全部清除，pixels未变。`v3_noise_workers` 在官方0.05随机 token 替换概率下重复通过，并验证268条样本在两个 DataLoader worker 和主进程中完全一致。

`v2_noise_workers` 未通过：验证脚本的 argparse 变量被样本元组覆盖，抛出 AttributeError；未执行完比较，不是模型/数据修复失败。原日志 `logs/i2t_role_policy_contract_v2_noise_workers.log` 保留，变量改名后在新目录重跑。后续脚本也会归档自身源码。

官方名为 scheduled_sampling 的逻辑实际在输入中随机替换 token，并非从模型分布采样；随机替换仍可能抽到特殊 token。这与原始回答里的字面 placeholder 是两个问题，本补丁没有顺便修改增强分布。带噪样本不能机械要求总图像 ID 计数恒为64，要区分真实输入占位符与噪声。

## 以后如何快速识别

先按角色数原始 placeholder，再检查 tokenize 后输入 marker 与受监督 marker，最后对照视觉特征 `[B,64,768]` 和输入注入位置。验证独立性应检查模型真正看见的资产，而不只检查记录ID或问答文本。缺 marker 先确认是否纯文本混合，再判断是否数据错误。完整图像头部可读性已检查，但这不等于所有图像内容和回答语义都正确。
