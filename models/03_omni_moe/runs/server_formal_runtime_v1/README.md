# 独立正式运行环境的短检查

全部六项检查通过：T2A B16×累积8三步，以及T2A、A2A全参数/投影、I2T全参数/投影五条真实路径。使用完整315M随机初始化模型；**这不是正式模型训练结果**。正式315M仍要等待完整198M SFT基座和全量部署校验。

服务器Python/Torch都位于`/new_data/REMOTE_USER/minimind/envs/omni-formal`，自己的conda运行时和pip包；freeze与conda显式包清单在根目录`env/server_omni_formal_*`。旧readiness项目仅提供相同已校验源码/真实fixture/编码器，没有继承其Python环境。新正式部署在`/new_data/REMOTE_USER/minimind/omni_formal_v1/project`，启动前还要比较实际预检源码hash。

T2A三步约8.9～9.1样本/s，第一步峰值9.10055GiB，后续11.44665GiB。差额2.34611GiB，与315M参数的两个FP32 Adam moment理论2.34610GiB几乎相同：第一步尚无moment时forward能通过，不能据此推断第二步。完整资源数值在根`reports/omni_formal_runtime_resource_v1.json`，本例PNG在`t2a_B16_acc8/plots/training.png`。

这里保留config、源码、原始日志、loss、gradient、内存、逐记录验证、输入shape、非零projector梯度和控制器结果。测试checkpoint仍在服务器，列表见`remote_checkpoint_inventory.json`；它是大小/时间清单，不是SHA校验结果。正式权重的回传由另一条交付桥负责。
