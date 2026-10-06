# 002：DataLoader 卡住，根因是 Windows 盘上的 Unix socket

## 现象

`recipe_batch8_probe` 把 TMPDIR 设为 `/mnt/d/minimind/data/tmp` 后，数据 worker 报 `OSError: [Errno 95] Operation not supported`，主进程等待数据。显存仍被模型占用，但没有产生训练 step。

## 假设与检查

最初需要区分 OOM、数据损坏、worker 崩溃和 IPC 错误。traceback 指向 Python multiprocessing 的 `socket.bind`，不是 CUDA allocator 或 tokenizer。使用同一段 `socket(AF_UNIX).bind` 在两个目录执行：D 盘路径失败，Linux `/tmp` 成功；同一数据此前的 pipeline smoke 正常。

## 根因与修复

Windows DrvFS 挂载不支持此次 multiprocessing 所需的 Unix socket。把所有临时文件都放项目 D 盘的假设不成立。只将 TMPDIR 移至 Linux `/tmp/minimind-lab`，持久化学习材料仍在项目根目录。停止的是已确认 PID/命令属于本项目的卡住测试进程和子进程，没有结束用户其他程序。

另外为多 worker DataLoader 加入 60 秒等待超时，避免将来以无限等待隐藏错误。此项改变失败处理行为，不改变模型结构或数学训练目标。

## 证据与验证

- 原始日志：`logs/recipe_batch8_probe.log`。
- 失败记录：`models/01_moe/runs/recipe_batch8_probe/failure.json`。
- 修复复测：`models/01_moe/runs/recipe_batch8_probe_v2/`。
- 统一环境入口：`env/activate.sh`。

以后遇到“GPU 留着显存、DataLoader 没有 batch、socket.bind 报错”，先检查 TMPDIR 的文件系统与 AF_UNIX 支持，不能只减小 batch 或设 num_workers=0 来掩盖原因。
