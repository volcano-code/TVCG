# N002 原生几何的 5090 D 显存与单步结果

Full V1 在 NVIDIA GeForce RTX 5090 D 上完成一次联合 AdamW 更新。峰值 allocated **10.125 GiB**，峰值 reserved **10.195 GiB**，编码开始至 optimizer 完成 **88.66 秒**。该数据来自同步打点的单步 smoke。

| 输入与目标 | 实测 |
|---|---:|
| 原生区间 / Gauss 节点 | 1,177,764 / 9,422,112 |
| 全批次 Moment / 完整 J / 受限 J | 64 / 38 / 26 |
| 受限 J 维度 | 13 × 3 维，13 × 8 维 |
| 总损失 / Moment / Jacobian | 0.2295477755 / 0.1520476703 / 3.030608167e-05 |
| 完整 J / 受限 J 损失 | 3.318655567e-05 / 2.609615813e-05 |
| 裁剪前梯度范数 / 参数变化 L2 | 406.650431 / 0.233424999 |
| 有变化的参数张量 | 226 |
| CPU RSS 边界采样最大值（含恢复验证） | 4.049 GiB |

报告的设备容量为 31.348 GiB。扣除 PyTorch 峰值 reserved 后的容量差为 21.153 GiB；驱动、桌面和其他进程的占用也使用这部分容量。原失败日志给出的 allocated 为 27.15 GiB，最后一笔 2 MiB 的 Key Projection 分配失败。旧运行没有逐阶段峰值记录。

## 训练时各阶段的实测高水位

| 阶段 | 耗时 s | 进入 allocated GiB | 离开 allocated GiB | 峰值 allocated GiB | 峰值 reserved GiB |
|---|---:|---:|---:|---:|---:|
| rle_decode (other) | 1.849 | 0.023 | 0.023 | 0.023 | 0.041 |
| training_batch_transfer (other) | 0.001 | 0.023 | 0.023 | 0.023 | 0.041 |
| geometry_metadata_cache (cache) | 1.174 | 0.023 | 0.023 | 0.023 | 0.041 |
| encode_geometry (forward) | 29.364 | 0.023 | 5.050 | 5.172 | 6.465 |
| pcraft_native_encoding (forward) | 19.629 | 0.023 | 3.252 | 4.546 | 5.432 |
| transolver (forward) | 0.165 | 2.857 | 3.704 | 4.546 | 5.432 |
| sqno_gnn (forward) | 6.993 | 3.269 | 4.799 | 5.150 | 6.465 |
| sqno_fusion (forward) | 0.458 | 4.808 | 4.817 | 4.820 | 6.465 |
| sqno_quadrature (forward) | 2.030 | 4.826 | 5.041 | 5.050 | 6.465 |
| m80_coefficient_jacobian (forward) | 4.022 | 5.050 | 5.075 | 5.075 | 6.465 |
| Moment Operator (forward) | 17.008 | 5.356 | 5.356 | 5.384 | 6.465 |
| geometry_query_jacobian (forward) | 0.835 | 5.357 | 5.368 | 5.371 | 6.465 |
| geometry_physics_attention (forward) | 0.834 | 5.357 | 5.368 | 5.371 | 6.465 |
| attention_moments (forward) | 0.831 | 5.364 | 5.365 | 5.371 | 6.465 |
| cross_scale_attention (forward) | 0.119 | 5.368 | 5.392 | 5.392 | 6.465 |
| Loss (forward) | 0.006 | 5.392 | 5.392 | 5.392 | 6.465 |
| Backward (backward) | 36.829 | 5.392 | 1.493 | 10.125 | 10.195 |
| attention_moments (backward) | 2.478 | 6.237 | 6.239 | 6.249 | 6.758 |
| Optimizer Update (optimizer) | 0.034 | 1.493 | 1.531 | 1.551 | 10.195 |
| cache_invalidation_check (cache) | 0.001 | 1.531 | 1.531 | 1.531 | 10.195 |

最高水位出现在反向阶段。Loss 阶段结束时 allocated 为 5.392 GiB，反向峰值为 10.125 GiB。阶段峰值包括进入该阶段之前仍然存活的张量。嵌套阶段的峰值与反向跨度有重叠。

编码返回时两个子缓存的 28 个张量字段共享 25 块底层存储，共 1.435 GiB。其中 [1,177,764,192] 的 FP32 局部特征在 P-CRAFT 与 SQNO 之间共用同一块 904,522,752 字节存储。缓存字段计数和 CUDA allocator 全进程峰值采用不同的统计范围。

Full V1 工厂保留全部配置模块。按原训练契约，UQ 与事件预测头属于辅助输出头，此次联合损失监督 M80、SQNO、融合和注意力路径。更新后，调用旧学习特征缓存触发 stale 错误。

逐阶段张量尺寸、计数、RSS 和重算耗时见 [原始计量](memory_profile.json)。各训练边界的反向跨度、重算次数和计时口径见 [计时报告](memory_profile.md)。

## 独立复跑

同一源码、同一初始化种子 830251，分别启动两个进程。每个进程从头训练一个完整 N002 批次，然后独立保存、重建和比较输出。

| 运行 | 单步 s | allocated GiB | reserved GiB | 总损失 | 参数变化 L2 | 恢复 Moment 最大绝对差 | 恢复 Jacobian 最大绝对差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| acceptance | 88.66 | 10.125 | 10.195 | 0.2295477755 | 0.233425 | 5.465e-10 | 3.907e-09 |
| repeat | 80.85 | 10.125 | 10.195 | 0.2295477755 | 0.233425 | 3.514e-10 | 2.519e-09 |

数值通过相同的逐元素误差阈值。CUDA 规约下的末位变化保留在原始记录中。[两次独立运行](n002_repeatability.json)
