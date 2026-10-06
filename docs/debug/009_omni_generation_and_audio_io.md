# 009：语音评估不能只看loss，也不能把生成截断当成发音错误

## 固定参考与真实输出

为了在我们的Omni正式训练前验证完整评估链，另行下载固定版本官方315M权重。来源、revision和SHA256在 `data/manifests/omni_reference_model.json`，用途只限参考校准，不作为我们Omni的初始化。我们的主线仍从自训语言模型分支。

`evaluation/omni_probes_v1.json` 固定文本、干净音频、图像探针；图像参考事实由实际像素核对。例如文件名写hello的熊猫图，牌子实际写的是“你好”，不能从文件名猜答案。

`tools/evaluate_omni_probes.py` 调用官方stream generator，记录prompt IDs、模型/资产hash、原始文字、Mimi codes和FLOAT WAV。文字采用官方随机采样temperature0.7/top_p0.85，不叫greedy；音频内置temperature0.2/top-k50/最近3code重复惩罚1.05。每题固定seed。CPU用FP32，GPU模式会另记BF16 autocast，不能混为同一数值协议。每帧必须有8个0..2047 code，异常不会偷偷替换成0。

## 音频加载的真实失败

第一次ASR检查报错发生在音频文件加载，还没进入识别模型：Torchaudio2.11的文件接口需要TorchCodec，本环境没有它；FunASR退回外部ffmpeg，但PATH里也没有ffmpeg。它不是音频codes错误、模型NaN或训练未收敛。原始完整链在 `logs/official_omni_t2a_asr_v1.log`，失败目录 `reference_evaluation/official_t2a_cpu_v1/asr_consistency_v1/` 保留源码。

先用已有soundfile读取FLOAT WAV、平均为mono，用torchaudio.functional明确重采样到16kHz，再把PCM数组送入ASR，绕开文件后端。这条路径通过，后续作为固定ASR预处理保留。系统sudo需要密码，因此没有改系统包；独立Omni环境额外安装官方PyPI的imageio-ffmpeg0.6.0固定wheel，hash约束在 `env/omni_extra_requirements.txt`，将其ffmpeg7.0.2静态程序链接到venv/bin。程序hash与版本见 `reports/omni_local_ffmpeg.json`。FunASR原文件加载路径也已复测通过，中文3.68秒音频读得58,880个16kHz采样点。

## 生成步数同时限制文字和语音

初版参考T2A在 `models/03_omni_moe/reference_evaluation/official_t2a_cpu_v1/`：中文“春天”一句话生成46帧，即3.68秒语音；ASR转写与16个归一化文字字符完全一致。单例CER=0不表示整个模型语音完美，也不是我们的训练成果。

英文咖啡题要求一句话，参考模型却生成长段落，还包含不可靠的说明。256生成步时文字已有EOS，语音只有248帧、19.84秒，达到步数上限。把Thinker完整文字当目标，ASR字符错误率35.7%，其中包含尾部缺失，不能都解释为发音错误。

控制实验只把上限改为512，同一题保留原modality内索引对应的随机seed：文字IDs完全一致，旧248帧codes与新输出前248帧逐项相同。新输出291帧、23.28秒，未达到512上限。证据在 `reports/omni_generation_budget_and_audio_backend_v2.json` 与 `official_t2a_cpu_budget512_v2/`。

延长后ASR字符错误率降至25.6%，仍缺少后面的部分文字。因此原问题有预算截断因素，但预算不是全部解释；剩余可能涉及模型音频提早结束/漏说或ASR遗漏，需要结合保留的WAV听辨。当前不把ASR转写直接等同于声学真值。`four`与`4`的ITN写法也会产生字符级差异，虽可能说的是同一个数。

## 以后如何分开判断

先看文字是否满足问题，再看文字EOS、生成预算、8路audio停止与实际输出帧数，最后比较语音转写和文字。Mimi帧率12.5Hz，理论音频时长是帧数/12.5。独立记录ASR一致性、声音自然度、回答正确性；三者不能合并成一个“loss很低所以会说话”。

本工具保存FLOAT WAV，没有通过归一化、裁剪或删除失败句来美化输出。采样上限变化也独立版本化，不能用更长的参考输出和更短的自训输出做不公平比较。正式Omni完成后复用同一协议，并另加按图像分组的held-out数据评估；这些仓库示例不是标准未见测试集。

官方发布权重可能已训练过我们后来分出的视觉验证图像。因此即使将来计算它在新split上的loss，也只能作为带此限制的参考，不能称作与自训模型公平的未见图像泛化对照。
