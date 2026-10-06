# MiniMind：从真实实验学习模型训练

从[十章学习手册](学习手册/README.md)开始；首次使用请读**[其他电脑怎样运行](SHARING_GUIDE.md)**。

主案例：205M Hybrid MoE，6层Gated DeltaNet＋2层Full Attention＋每层4专家Top-1 MoE。官方198M SFT权重转换后执行完整两轮SFT：113,082次更新、735,833,934有效监督token。**没有Hybrid预训练，评估结果与生成样例已记录。** 这里保留学习过程、真实失败和排查证据。

## 不用作者的电脑，也能开始

Python3.10或以上；部分系统请将python替换为python3：

```bash
git clone https://github.com/npm-DreaMaX/minimind-training-lab.git
cd minimind-training-lab
python tools/study_minimind.py summary
python tools/study_minimind.py replay --step 12000
```

这两项只用Python标准库，无需GPU、作者的D盘、完整数据集或虚拟环境。网页阅读教程和PNG甚至不需要安装Python。

想执行真实样本解码、完整模型CPU forward/backward或推理，按[分享指南](SHARING_GUIDE.md)在自己的电脑安装环境。仓库带一条真实T768样本；最终权重约421MB，下载时验证SHA：

```bash
python tools/fetch_learning_assets.py weight
python tools/study_minimind.py sample --index 51023
python tools/study_minimind.py infer --prompt '你好，请介绍一下自己。' --max-new-tokens 64 --out 学习手册/实操输出/我的推理_01
```

学习工具不创建optimizer，不更新模型。最后两项需要先安装分享指南中的依赖。

## 导航

| 要学习什么 | 入口 |
|---|---|
| 数据→tokenizer→mask→forward→loss→梯度→AdamW | [课程与练习](学习手册/README.md) |
| 全部训练更新、验证和显存曲线 | [图表](学习手册/图表/) · [逐步数据表](学习手册/数据表/) |
| 模型结构与实际源码 | [Hybrid原理](学习手册/03_Hybrid和MoE原理.md) · [模型](models/02_hybrid_moe/src/model_hybrid.py) · [训练器](lab/train.py) |
| 真实质量结果与坏答案 | [最终评估](reports/hybrid_final_review_20261006.md) |
| OOM、异常梯度、恢复、数据问题 | [debug实战](学习手册/08_评估和Debug.md) · [原始诊断](docs/debug/) |
| 数据来源、SHA、完整数据准备 | [数据导航](data/README.md) · [下载/重建步骤](SHARING_GUIDE.md) |
| 实验来源、官方/作者/本项目修改 | [源码来源](sources/provenance.json) · [修复diff](models/02_hybrid_moe/src/) |
| 最终推理模型 | [Release](https://github.com/npm-DreaMaX/minimind-training-lab/releases/tag/v1.0-learning) |

每个模型有独立目录：`models/01_moe`、`02_hybrid_moe`、`03_omni_moe`、`04_dense_ar`、`05_dlm`。只有Hybrid上述两轮SFT完成；普通MoE从零预训练于13209步停止，其余方向只有各自标明的准备/预检，未完成正式训练。

## 分享范围

这是学习档案，不是本机整个211GiB目录的备份。Git中有教程、源码、图表、原始训练指标、config/diff、评估、清单和真实样本；最终推理权重另放Release，官方数据按固定来源下载。完整optimizer恢复二进制、所有历史权重、大型profiler trace、缓存及运行控制器未上传。

历史报告描述原实验，不等于在你的机器重做过验证。部分历史链接指向未上传的大文件，使用前查[分享范围与补齐方式](SHARING_GUIDE.md)。机器地址已用占位符处理；原始证据在作者本机保留。源码快照保留原许可证与来源，公开版本清单见`SHARE_MANIFEST.json`。
