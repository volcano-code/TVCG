# Chat 端复核清单

| 审核问题 | 本轮实际记录 | 包内复核入口 |
|---|---|---|
| 协议、数据和历史基线可否定位？ | 预登记提交 9115f23；父提交 54e8025；正式四划分发布的训练/开发银行通过原发布哈希核验 | `protocol.json`、`native/host_inputs.json`、压缩包根目录 `SOURCE_COMMIT` |
| 原生成功轨迹能否重放？ | 25 次调用，24 次返回相同几何哈希和十矩，另一次复现同一拒绝；数值导数也经独立四面体积分复核 | `native/evidence/native_replay.json` 的 `trajectory`、`native_targets.npz`、`native/verify.py` |
| 预测是否偷用了布尔切后的几何或教师返回量？ | 预测输入只有几何静态测度、预布尔的有限面、操作侧、运动；逐行物理真值进入独立评分步骤。两个单比例对照使用细粒度预测给出的质量 | `replay.py`、`native/replay.py`、`compact_pino_v21.py` |
| 是否比较了相同查询和相同误差定义？ | 面条各方法的 M/J 共同集合均为 24 条，Jv 16 条；其他几何 M/J 共同集合均为 62 条。含 3 个冻结面互补操作的额外十矩计在独立列 | `evidence/comparison.json` 中的 `*_matched_ids` 与 `*_all_available`；从 `verify_results.py` 独立复算 |
| 是否支持不同来源与保留比例？ | 14 件静态几何包括 6 件历史网格的原生 256³ 重编码；有数值完整 J 的其他几何面型限于椭球、双曲面。低保留比例最差 J 达 41.80% | `native_replay.json` 的 `separately_reencoded_256`、`comparison.json` 的 `strata_other_geometry_probes` 和 `query_strata` |
| 子体素表示是否真的保留材料位置？ | 正权重测度通过独立网格十矩核验；长柱、薄壁、空腔和断连均有解析对照；同一冻结面允许互补，重建正负系数有独立测量 | `evidence/static_material.json`、`evidence/manufactured.json`、`native_replay.json` 的 `controls` |
| 部署 Full、R3、R8 是否各有真实资格？ | 数值 Full J 为 86 条，Full 事件证书为 0；生产配置的开发集 R3 为 0/64、R8 为 0/25。替代配置单列。分子均为有数值目标的条数 | `evidence/gate_status.json`、`native_replay.json` 的 `strict_development_R3`、`strict_development_R8`、`alternative_chart_diagnostic` |
| 限制阈值能否独立保证 Newton 成功？ | S001 在起点十矩相同、单位方向 J 扰动 2.0994% 时约束矩阵秩从 13 降到 12，原生求解器一次调用后停止 | `calibration/evidence/threshold_sensitivity.json` 的 `rank_counterexample`，以及 `native_training_receipts.tar.gz` |
| 是否出现合格新权重、盲测或可发表的加速？ | 真实 Pilot 参数更新为 0；无入选权重、盲测成绩和合格解耗时。双核计时回放 25 次固定查询，包含拒绝、切面构造和读出 | `evidence/gate_status.json`、`profile_inference/profile.json` 与 `summary.md` |
| 后续跑训练的条件是什么？ | 同一生产有限面规则公开可复核的 R3/R8 受限运动与事件资格，训练/开发发布查询具备可审计的同源覆盖；随后执行预登记三臂、小规模联合更新和恢复 | `protocol.json` 的 `pilot` 与 `solver` |

**本轮建议签署的范围为确定性材料及轨迹复核。训练晋级、候选陀螺成功率以及取得合格解的加速倍数均待后续审核。** 本目录的 `README.md` 提供三条可直接执行的审核命令及归档机上的完整复现方法。
