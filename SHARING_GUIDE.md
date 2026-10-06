# 给其他学习者：下载以后怎样使用

这里是一个真实的模型训练学习档案。主案例是205M Hybrid MoE转换后两轮SFT；没有Hybrid预训练，评估结果与生成样例已记录。坏答案、数值问题和恢复记录也是课程内容。

仓库地址：`https://github.com/npm-DreaMaX/minimind-training-lab`。网页阅读不需要你的电脑有GPU，也不需要下载数据集。

## 第一级：只读教程、图表和历史状态

从[学习手册](学习手册/README.md)按01→10阅读，图表可以直接打开。下载代码后，在**你自己的项目目录**执行：

```bash
git clone https://github.com/npm-DreaMaX/minimind-training-lab.git
cd minimind-training-lab
python tools/study_minimind.py summary
python tools/study_minimind.py replay --step 12000
```

使用Python3.10或以上；部分系统命令名是`python3`。这两条学习命令只用Python标准库和已提供的真实日志，不依赖作者的D盘、Python虚拟环境、GPU或完整数据集。

## 第二级：解码真实样本

在仓库根目录建立自己的环境，Python3.10与本次实验一致：

```bash
python -m venv .venv
```

Linux、WSL或macOS：`source .venv/bin/activate`。Windows PowerShell：`.venv\Scripts\Activate.ps1`。如果不激活，也可以直接使用对应`.venv`内的Python解释器。

```bash
python -m pip install -r env/learning_cpu_requirements.txt
python tools/study_minimind.py sample --index 51023
```

这里只需要Torch、Transformers和NumPy，不需要FLA/CUDA。Linux/Windows如果只做CPU练习，可以先从PyTorch官方CPU索引安装匹配的Torch，再装上述依赖：`python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`。不同系统的wheel与Python支持范围以实际安装结果为准；本项目验证环境为Linux/WSL，未声称逐一验证所有操作系统。

安装平台选择可对照[PyTorch官方说明](https://pytorch.org/get-started/locally/)。

`data/examples/hybrid_sample_51023.json`是从真实正式缓存提取的**一条完整T768样本**，保留原dataset index、cache row、token和labels。缺少全量缓存时，学习工具明确报告使用这条随仓库提供的样本；它不是新的训练集，也没有拿它替代正式训练预算。其他index需要完整缓存。tokenizer文件随`upstream/model/`提供。

## 第三级：运行完整205M权重和梯度检查

最终FP16推理权重约421MB，放在GitHub Release中，不进入普通Git历史。下载工具会验证完整文件大小和SHA256，拒绝覆盖不一致的已有文件：

```bash
python tools/fetch_learning_assets.py weight
python tools/study_minimind.py infer --prompt '你好，请介绍一下自己。' \
  --max-new-tokens 64 --out 学习手册/实操输出/我的推理_01
python tools/study_minimind.py trace --index 51023 --length 128 \
  --out 学习手册/实操输出/我的梯度检查_01
```

以上两条长命令用的是Bash续行写法；PowerShell中请写成各自一行。输出目录必须尚不存在，避免覆盖你的答案。

这是完整模型的CPU执行，不创建optimizer、不更新权重。CPU入口关闭了FLA依赖硬检查，实际仍使用原模型已有的CPU递推路径；没有减少层数、专家数或参数量。正式CUDA训练配置与源码保持原定义，不因教学入口而改写。

模型加临时张量会使用数GiB RAM，计算速度取决于CPU；没有承诺所有设备实时生成。即使模型回答错误，也要原样保存。原始质量评测在[最终报告](reports/hybrid_final_review_20261006.md)。

## 第四级：下载本次完整数据并重建缓存

只读学习不需要这一步。若要检查任意数据行或准备复现实验：

```bash
python -m pip install datasets==3.6.0
python tools/fetch_learning_assets.py raw-data
python tools/prepare_tokens.py --raw data/raw/minimind/sft_t2t_mini.jsonl \
  --out data/processed/sft_t2t_mini_full_v1 --stage sft
```

下载约1.739GB原始JSONL，缓存额外约1.818GB。原始数据使用固定官方revision和完整SHA；预处理沿用固定seed20261003。生成缓存后，对照`data/manifests/sft_mini_token_cache_v1.json`中的二进制数组SHA；metadata含机器路径/时间等，不能要求跨机器整个JSON字节相同。

未下载其他阶段的数据不影响本次Hybrid学习；Omni、完整预训练等来源见[data导航](data/README.md)。完整长训练还需要匹配CUDA环境、起点权重、profile与恢复验证；这里的教学命令不会自动开始训练。

## 公开仓库包含什么、不包含什么

包含课程、图表、逐步CSV/JSONL、源码、官方源码快照和许可证、config/diff、评估、数据清单、真实样本与只读工具。第三方来源和固定commit见[sources/provenance.json](sources/provenance.json)，原许可证保留在各来源目录。

不包含全部历史checkpoint、完整恢复二进制、大型训练数据、原始大型profiler trace、运行中的控制器、作者Git历史和本机缓存。公开目录中相关来源/审计记录保留，但其中的历史本机路径不是已上传文件的承诺。共享清单`SHARE_MANIFEST.json`记录范围；最终模型Release可按需下载。

历史训练、恢复与数值报告描述作者当时的实验，不等于在你的机器重新完成了那些检查。服务器地址和机器用户名在公开副本中改为占位符，原实验档案仍在作者本机。先看源码/config，再设计你自己的实验。

GitHub是分享入口，不是已经保存全部211GiB实验资料的备份；上传也不会自动删除作者电脑上的文件。阅读和计算可以在不同设备上进行。
