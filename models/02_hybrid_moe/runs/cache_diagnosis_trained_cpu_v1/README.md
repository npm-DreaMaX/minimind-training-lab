# 第12,000步训练权重的CPU缓存检查

目的：在不占用训练GPU的情况下，检查实际训练权重的缓存推理和完整causal forward是否一致。训练继续运行，配置未变。

执行命令（在项目根目录；输出目录必须不存在）：

```sh
source env/activate.sh
python -u -B tools/check_hybrid_trained_cache_cpu.py \
  --run models/02_hybrid_moe/runs/hybrid_sft_official_mini_2ep_v1 \
  --out models/02_hybrid_moe/runs/cache_diagnosis_trained_cpu_v1 \
  --tokens 32
```

`training_resume.pt`是第12,000步不可变恢复快照，205,623,072参数；大文件保存在磁盘，不加入Git。`source_check.py`保存实际执行源码；`report.json`保存全部数值和原始生成，日志在根`logs/hybrid_cache_diagnosis_trained_cpu_v1.log`。

结果：CPU FP32、无autocast，三个问题与跨128位置边界检查全部通过；相同序列逐位置argmax一致，最大相对RMS 6.94e-7，最大绝对差1.77e-5。检查使用CPU回退，日志中无Triton平台的警告是预期行为，不表示正式GPU训练回退。

限制：没有验证GPU FLA/BF16内核；不是生成质量通过证明，CPU输出仍有错误和重复。3.05秒仅为对照计算时间，不含加载/import。诊断没有对模型作任何修复或训练。
