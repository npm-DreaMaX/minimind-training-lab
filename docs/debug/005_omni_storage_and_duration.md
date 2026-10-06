# Omni：压缩文件、内存与音频时长的三种计量

本次没有在主训练旁边故意加载几十GB数据触发系统OOM。先读取固定版本Parquet的footer，再取一个真实row group，验证可行的存储修复。

## 现象与初始假设

官方OmniDataset构造函数使用`pa.Table.from_batches(ParquetFile(...).iter_batches())`，最终仍把整个数据集留在内存。T2A文件约5.35GB，容易误以为16GB物理RAM足够；本机WSL限额实际约7.6GiB。另一个初始估算来自README的T2A 1636.01小时、A2A 423.40小时。

## 检查、证据与排除

固定数据revision、SHA见`data/manifests/minimind_omni_full.json`。`tools/inspect_omni_parquet.py`读取footer并记录所有row group的非空code计数和范围；结果在`reports/omni_parquet_footers.json`。

| 数据 | 行数 | row groups | answer code数量 | 仅int64数值数组 |
|---|---:|---:|---:|---:|
| T2A | 1,248,923 | 251 | 3,533,787,080 | 28,270,296,640 bytes |
| A2A | 414,024 | 1 | 914,546,656 | 7,316,373,248 bytes |
| I2T | 2,904,511 | 581 | 不适用 | 另有图像bytes |

这些还不包含offset、文本、输入音频、ref codes、Python对象等。A2A只有一个巨大row group，因此“每次读一个row group”也不是足够小的边界。Parquet压缩尺寸、编码后的uncompressed尺寸、Arrow内存尺寸三者不能混用。

`tools/audit_omni_sample.py`只下载T2A第一个row group（约10.31MB网络数据、3702行）；所有code在0..2047范围。`tools/verify_mimi_decode.py`用已校验的官方Mimi权重解码其中第7行：文本为“The capital of France is Paris.”，208个交错codes还原为`[1,8,26]`，输出49920个24kHz采样点，严格对应2.08秒。Mimi配置也明确为12.5Hz，没有缺失或形状不匹配权重。

这排除了把这批codes按75Hz估算的合理性。`codes / 8 / 75 / 3600`恰好重现README两个小时数；按实际12.5Hz应为T2A约9816.08小时、A2A约2540.41小时。**可以确认时长口径相差6倍；没有看到作者生成统计的脚本，不能声称已找到其具体bug行。** 这些是目标音频时长，不是GPU训练wall-clock，也未包含输入语音时长。

## 修复与验证

本项目`tools/repack_omni.py`使用`memory_map=True, pre_buffer=False, iter_batches(128)`，转换为每片最多4096行或约64MiB的Arrow IPC。answer/ref codes逐批核验0..2047后无损转为uint16，其他列保持原值。不是重新量化音频，也没有改变codec词表或模型。

`lab/omni_table.py`以有限LRU缓存映射这些片段；`lab/omni_data.py`继承官方Dataset，保留其模板、随机增强、delay pattern和labels逻辑。需要一个真实schema样本初始化官方构造器，不能用无法推断schema的空表。

`tests/test_omni_storage.py`两项测试已通过：跨片段和缓存淘汰后，各列值逐项一致；重置Python/NumPy/Torch随机种子后，官方与新存储得到完全相同的`[9,1535]`输入、文本labels、`[8,1535]`音频labels及其余返回项。原始结果在`logs/omni_storage_contract.log`。fixture重打包完成耗时0.342秒；这不能外推为完整巨大A2A row group已通过内存验证。

## 代价、限制与快速识别

需要额外一份无损训练索引/分片与顺序转换时间；原始Parquet集中保留，不为各模型复制。uint16把code数值存储降至int64的四分之一，实际总文件比例受其他列影响。随机读取映射缓存大小也会影响吞吐，须在真实训练再量测。

完整三个数据集转换时记录peak RSS与精确行数，失败保留中间目录；仅在完整index原子生成且计数一致后交给训练。后续A2A/I2T须验证真实encoder输入，不能由T2A等价性测试直接推断所有模态已经正确。

以后遇到“数据文件没多大、训练还没forward就内存耗尽”：先看构造函数是否积累所有batch，再检查列的物理dtype、nested offsets、row group数量与最大块大小。估算音频时长时同时确认codebooks数量、交错布局、codec帧率与一次真实解码。

## 完整 T2A 转换实测

现已完成全部1,248,923行转换：210.81秒，`/usr/bin/time -v`记录peak RSS 2,691,476KiB（约2.57GiB），退出码0。全量逐批code范围校验与最终计数通过。完整内容hash划分train1,247,657、val1,266，完全相同conversation的额外重复10,398行，最高重复539次；未做语义去重。原始报告在`reports/omni_t2a_full_data_audit.json`，转换日志在`logs/repack_sft_t2a_full.log`。这仍不证明单巨大row group的A2A已通过。
