# 64M Dense AR 对照

与 dLM 形成可解释对照；基线不是低级版本。保留同一 tokenizer 与数据切分。

## 当前状态

后续独立实验，未开始训练，不挤占前三项核心训练预算。

## 从哪里开始读

- 源码：`upstream/model/model_minimind.py`。来源与commit见根目录 `sources/provenance.json`。
- 配置：本目录 `configs/`。每次运行在 `runs/<run_id>/` 独立保存记录。
- 公共数据：根目录 `data/raw/` 和 `data/processed/`，不在模型目录重复复制。
- 先读根目录 `docs/learning/01_training_step.md`，再对照该模型 forward、loss、优化器与生成路径。
- 全部实验的阶段、控制变量和完成口径见根目录 `docs/experiment_plan.md`。
