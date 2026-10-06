# 012：训练作业、环境部署与“完成”的含义

2026-10-03。此次工作期间本机198M训练没有暂停。以下是配套系统问题，不是模型收敛失败。

## 返回PID不等于后台进程已存活

现象：两次WSL shell `nohup ... &` 返回PID，却没有产生控制器状态文件，日志长度为0；随后`ps`中进程不存在。不能仅凭“命令返回成功”报告已排队。

检查：核对`ps`、日志、controller目录，与已持续运行的主训练PID分开检查。没有主训练退出、OOM或配置损坏证据。确切退出信号未捕获，因此不能断言是某一种shell信号造成的。

修复：用Python `subprocess.Popen(..., start_new_session=True, stdin=DEVNULL)`显式创建会话，并重定向日志；启动后再次检查实际PID、`events.jsonl`和`waiting_dependency`状态。Hybrid控制器PID554336、交付桥PID557321已确认存活。这个修复改变进程生命周期管理，不改变模型、样本、梯度或训练算法。

## 独立环境会暴露先前继承的隐含依赖

服务器短测原环境是读取另一环境的Torch/base packages。正式环境现在位于`/new_data/REMOTE_USER/minimind/envs/omni-formal`，自己的Python、Torch2.6.0+cu124与其他包都在项目目录，`sys.prefix == sys.base_prefix`，不再继承第三方site-packages。

源码阅读发现`model_omni.py`无条件import ONNX Runtime；原overlay继承了它，独立环境最初依赖清单没声明。`pip check`只能检查包声明的依赖，不能发现项目脚本遗漏的import。因此补充明确版本的`onnxruntime==1.23.2`，并安装`imageio-ffmpeg==0.6.0`提供项目目录内的ffmpeg。没有改系统库或其他人的环境。

验证分为独立前缀检查、pip依赖检查、真实全315M的B16累积8 T2A更新，以及语音/视觉各模式真实更新。T2A三步的allocated峰值从首步9.10GiB升到后续11.45GiB，说明Adam状态建立之后的峰值必须实测；本次保留13.5GiB限额。短测速度约8.9～9.1样本/s，不能直接证明正式质量。

环境原始日志与freeze保存在`env/server_omni_formal_freeze.txt`和`logs/server_omni_formal_*.log`。正式代码的数据/目标不变；不同依赖集合仍必须真实验证，不能凭相同Torch版本认为全部路径已测过。

## 全量部署不能用“开始传输”冒充“数据已就绪”

本机为1307个最初部署文件计算了约37.76GB内容的SHA256，包含完整Arrow语料、划分与源码。最初tar流约237秒只传437MiB（约1.84MiB/s），且网络中断后不便恢复。仅停止自己的部署进程组，保留已传前缀和v1清单/日志；主训练不受影响。

v2使用项目自己安装的rsync和zstd压缩，复用前缀、支持部分传输恢复，再对照完整期待SHA256检查远端每个文件。原v1证据在`runs/omni_formal_deployment_v1/replacement.json`；v2在`runs/omni_formal_deployment_v2/`。实际是否完成要看`result.json`，不能从配置文件存在推断全量数据就绪。

部署清单检查还补上了`train_sft_omni.py`间接导入的`trainer_utils.py`。仅复制入口脚本不构成完整源码部署；远端字节校验后还实际import官方模型、trainer与collator。v2保持数据期待hash不变，补充源码清单，没有重新抽样或重编号训练记录。

## 三种完成状态

1. `status=complete`：训练器结束它收到的预算，可能只是3步preflight。
2. 正式接续器验证完成：必须匹配预先记录的完整数据量、steps、必要token数、源码/config、最终全验证与训练过的best，不能拿测试目录代替。
3. 研究目标完成：还要分析生成、音频/视觉结果、对照、故障与局限。自动队列只输出`pipeline_budget_complete`且保留`quality_review_pending=true`，不会自动声称训练成功或整个项目完成。

11项CPU合同测试覆盖缩小预算/数据、配置变化、NaN、未训练best、丢失恢复点、失败依赖和重启不重复训练。原始结果：`logs/formal_pipeline_contract_v2.log`。这些是作业管理验证，不是模型训练结果。

Omni交付桥每分钟保活并检查状态、回传记录，按小时触发模型导出回传，完整Adam/RNG恢复包按6小时或阶段结束触发；到达时间仍受带宽影响。选择该频率是因为约3.8GB完整状态在当前链路可能占用半小时，没必要每小时重复占满链路。服务器仍持续按正式配置保存最新恢复点，本机拷贝可能落后；阶段结束及时同步全部并核对best的SHA。不会传播删除操作。SSH认证会话丢失时保留错误并重试，远程作业不会被随意终止。WSL关闭/断电仍可能中断本机作业，后台进程不等于保证无人干预的永久服务。

### 部署v2的120秒I/O超时

压缩rsync的v2又失败了：原日志明确为Receiver `io timeout after120seconds`、code30，随后sender Broken pipe。启动器v1正确停止放行，未把前缀当完整数据。SSH仍正常、远端空间仍约11TB、本机训练持续正常，因此不是模型OOM或目标盘已满。

初始哈希清单的37.76GB扫描约334秒，而v2额外使用`--checksum`再次预扫描；长时间本地计算期间接收方可能没有数据，120秒I/O阈值过短，这是当前主要假设，尚未用系统调用trace单独证明。v3保留同一清单和部分文件，取消传输前的全量checksum预扫描，允许900秒I/O间隔并记录progress2，最终仍强制逐文件SHA256和实际import检查。数据、模型和训练recipe没有变；传输候选文件可依据size/mtime跳过，但**只有最终全量hash才是放行证据**。瞬时传输失败最多重试三次，持久失败继续保留并阻止正式训练。

v2失败在`runs/omni_formal_deployment_v2/`；v3当前状态在`runs/omni_formal_deployment_v3/status.json`。替换启动器在`runs/omni_formal_arming_v2/`，原v1失败不覆盖。
