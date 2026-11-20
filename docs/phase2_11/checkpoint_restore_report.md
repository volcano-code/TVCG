# N002 的 checkpoint 恢复验证

RTX 5090 D 上更新一次后保存 checkpoint，重新实例化 Full V1、AdamW 与 CosineAnnealingLR，恢复模型、训练状态和随机数状态。恢复得到第 1 步，调度器 last_epoch 为 1。模型状态字典的 245 个条目、优化器的 230 个参数状态和调度器状态通过零容差检查。

更新后的原模型与恢复模型分别使用同一原始几何快照重新编码，比较全部 64 个查询。目标混合精度路径采用逐元素 `atol=1e-08, rtol=1e-05`。

| 恢复输出 | 形状 | 最大绝对误差 | 最大相对误差 |
|---|---|---:|---:|
| Moment | 64 × 10 | 5.46493892e-10 | 1.23505005e-07 |
| 原始系数 Jacobian | 64 × 10 × 10 | 3.90721943e-09 | 0.000483056554 |

相对误差的分母为原模型输出绝对值与 dtype epsilon 的较大者。接近零处的相对误差应结合绝对容差解读。

- Checkpoint SHA-256 为 `5bf17463c2bb0788552a6c2779e6da5fa480bbbc004f515d7e5c78cb3af5aa56`。
- Work manifest SHA-256 为 `a0f914df030228b0f7fc930141b97968238250e43dd671e6f64e88ed882e42ee`。
- 本机 checkpoint 位于 `/data/xvxinyao/TVCG/runs/phase2_11/acceptance/checkpoint.pt`，报告记录的保存大小为 62,795,693 字节。

[实测记录](n002_smoke_report.json) · [复现命令](REPRODUCE.md)
