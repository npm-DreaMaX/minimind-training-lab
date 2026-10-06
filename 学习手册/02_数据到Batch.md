# 02 一条对话怎样进入模型

## 先区分文件、样本和token

Hybrid实际使用`data/raw/minimind/sft_t2t_mini.jsonl`：1,739,201,170字节，905,718条JSON记录。来源为Hugging Face `jingyaogong/minimind_dataset`，固定revision和SHA在[mini清单](../data/manifests/minimind_official_mini.json)。其他已下载数据在[数据总表](../data/README.md)；不能将14GB SFT或普通MoE的8GB预训练语料算进这次Hybrid的训练量。

一条JSON是一组conversation；一个token是tokenizer的离散编号；一次batch是多条token序列补齐后形成的张量。同一句中文的字数、UTF-8字节数、token数通常不同。

## 实际管线

原始conversations → 官方对话预处理/模板/后处理 → tokenizer → 无截断token缓存 → 固定train/val划分 → 取前768位置 → padding → DataLoader。

预处理源码[tools/prepare_tokens.py](../tools/prepare_tokens.py)调用[官方SFTDataset](../upstream/dataset/lm_dataset.py)。离线增强固定Python seed=20261003，每条只抽样一次；这与作者每次读取可能重新增强不同，已经记为recipe差异。

缓存`data/processed/sft_t2t_mini_full_v1/`里，`tokens.bin`为uint16，`labels.bin`为int16，`offsets.npy`为记录边界；`train.npy/val.npy`是行索引，不是数据副本。送入模型前转换为int64张量。词表6400能装进缓存dtype，但其他tokenizer不能直接套用这个限制。

同一conversation的规范JSON做SHA256，前8字节按little-endian整数取模1000，小于1进验证。于是1,066条留出，其余904,652条。T768截断后8条没有assistant目标，被过滤；最终904,644条训练。46,862条额外完全重复记录没有删除；同样内容放在同一侧，但不代表语义去重或排除了官方基座见过这些数据。

## 三种mask不要混淆

| 名称 | 控制什么 | 本实验实现 |
|---|---|---|
| causal mask | 位置t能否读取未来token | Full Attention下三角；Linear因果递推 |
| padding/attention mask | padding位置是否参与attention | 本训练器右侧padding，没有另外传attention_mask；因果前缀不读取右侧pad |
| loss mask | 哪些位置贡献CE | labels=-100，忽略用户/模板中相应位置和pad |

右padding的因果性不等于整个训练完全忽略pad：当前MoE aux统计在模型里包含pad位置；日志里的`router_first_microbatch_nonpadding`则另行排除了pad。两者口径不同，不能拿一张路由图断言aux严格衡量真实文本的负载。

## 跟着真实第一批走

第1步的随机顺序由seed=20261019产生，第一条dataset index为51023，对应cache row51081、原文件第51082行。这条有507个未截断token；原始对话和前64位置在[真实首批说明](../models/02_hybrid_moe/learning/official_mini_first_batch_v1/README.md)，完整两条在[batch.json](../models/02_hybrid_moe/learning/official_mini_first_batch_v1/batch.json)。

`input_ids`与`labels`都是`[16,768]`，但内容不同。用户文字仍在input里，只是不作为该位置的目标。首条位置23是`<think>`，label=25；位置22的logits负责预测它。位置23的logits预测位置24，而不是预测当前位置自己。

模型代码执行：`logits[:, :-1, :]` 对 `labels[:, 1:]`。有效位置必须按`labels[:,1:] != -100`计数。第1步12,288个名义输入位置，9,228个非pad输入，**7,818个有效监督标签**，与原始metrics精确匹配。

“用户位置没有CE”不表示用户文字不能影响参数：回答位置通过attention依赖用户上下文，梯度可传回用户位置的embedding输出。第04章的完整模型CPU实测可以直接观察这一点。

## 截断改变了什么

T768下约5.26%记录尾部被截断，保留约98.80%监督token；T512只保留约86.47%，且1,118条失去全部监督。这是数据/训练语义改变，不能仅说“省显存”。[全量长度审计](../reports/sft_official_mini_T768_audit_v1.md)

一轮367,916,967监督token，两轮735,833,934；重复两遍不是新增两份知识。数据中存在长思考、重复和错误示范，读取成功不等于内容优质。检查原始记录时分别看问题、推理过程、最终答案与模板，不能因为JSON合法就判定它适合训练。

检查顺序：确认raw SHA → tokenizer版本 → 模板special token → 真实解码 → labels跨度与shift → 有效标签数 → 截断/重复/划分。全为-100时不要先调学习率；先修复没有训练目标的问题。

练习：运行`python -B tools/study_minimind.py sample --index 51023`，解释位置22/23/24；然后说明为什么不能直接使用B×T统计“训练了多少token”。
