# Dense dLM

从 Dense AR 权重转换为双向掩码去噪训练，研究训练目标和生成过程；不把 AR perplexity 直接当成扩散模型可比指标。

## 当前状态

作者代码已归档，并完成完整63,912,192参数Dense的CPU检查：AR权重严格兼容、双向可见性、真实SFT mask与同位置loss、forward/backward均验证。零监督NaN和生成残留mask的受控反例也保留。尚未开始正式dLM训练，不挤占前三项GPU预算。详见[AR与掩码扩散学习记录](../../docs/learning/07_ar_and_masked_diffusion.md)及 `runs/author_contract_cpu_v1/`。

## 从哪里开始读

- 源码：`sources/experimental/dlm/model_dlm_author.py`。来源与commit见根目录 `sources/provenance.json`。
- 配置：本目录 `configs/`。每次运行在 `runs/<run_id>/` 独立保存记录。
- 公共数据：根目录 `data/raw/` 和 `data/processed/`，不在模型目录重复复制。
- 先读根目录 `docs/learning/01_training_step.md`，再对照该模型 forward、loss、优化器与生成路径。
- 全部实验的阶段、控制变量和完成口径见根目录 `docs/experiment_plan.md`。
