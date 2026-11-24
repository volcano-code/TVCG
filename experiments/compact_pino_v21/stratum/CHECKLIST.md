# Chat 端审核清单

先读本目录的 `README.md`。从解压包根目录运行 `python experiments/compact_pino_v21/stratum/review_check.py` 可以复算主要数字。下列问题对应本轮研究接口能支持的结论。

| 审核问题 | 复核位置与判据 |
|---|---|
| 用的是哪一版正式数据，有没有换掉困难查询？ | 查 `protocol.json`、`qualification/evidence/published_inventory.json`。核对 Work A 四划分清单摘要，再用父几何、原查询 ID、十个原始系数及有序基的摘要匹配 64 条 R3 和 25 条 R8。4 件几何中 combB 与 maskB 同属一个来源组。 |
| 教师和求解回调是否真的切了同一个物体？ | 查 `shared_operator.py` 与 `physical_replay/evidence/registered_physical_replay.json.gz`。逐行核对有限面的顶点、面索引、封盖、操作侧、坐标变换、材料与物理代码身份。再用斜平面切四面体的闭式解复核固定材料、非单位系数及空切拒绝。 |
| C++ 求解器能否正确使用曲面的受限导数？ | 查 `solver_bridge/receipt.json`。从十个原始系数恢复归一化和交叉项的系数 2，复算静平衡 Newton 方程中 R3 的三列与 R8 的八列导数。另核对同一平面的 FIDGET_SPINNER 目标超出给定空间时，求解器是否拒绝组装。回执同时记录测试的迭代次数与目标。 |
| 原始圆整误差和导数单位是否保留？ | 查 `native_backend/source_smoke.json`、`physical_replay/protocol.json`。R8 的真实查询带有小的非零行列式，读中心原始系数以及保持最小特征值的扰动。受限导数应为原始基下的 10×3 或 10×8，顺序与原文件一致。 |
| 89 条物理结果中，哪些数值有独立支持？ | 查 `physical_replay/evidence/summary.json`，再抽查完整压缩回执里的正负两侧。分别复算有向四面体十矩及其导数、两档中心差分、匹配顶点的速度。任何拒绝都要计入各自分母。 |
| 差分一致是否掩盖了保留网格的变化？ | 查每条记录的 `steps`、`evidence/restricted_targets.json` 和 `evidence/event_scale_example.json`。用实际顶点、边和面数复核欧拉特征数，并核对顶点构造和有向面对应。区分给定步长上观测到的变化与中心附近已证明的范围。 |
| 哪些查询可以声称局部切割稳定？半径有多大？ | 查 `qualification/evidence/neighbourhoods.json`，核对每行的分支、半径、全部原始几何三角形、所有封盖及接触判据。对 R8 核对对应的 `qualification/native_admission/evidence/` 穷举回执。分别统计已有充分证明、条件成立和未知。 |
| 体素物理原型的误差在哪里？ | 查 `evidence/prototype.json`，按曲面、几何、来源和保留比例分组。独立复算 M 的无量纲最大分量误差与原始单位切向 J 的最坏方向误差，再取查询 P95。核对每组分母和最差查询。 |
| 原生面条求解是否仍能得到同一件陀螺？ | 查 `evidence/noodles_legacy_direct.json`、`evidence/noodles_full_rank.json`、`evidence/noodles_event_comparison.json`。比较 25 次实际调用、第 3 次拒绝、Newton 矩阵、43 个事件和独立验收网格。原始事件摘要包含计时与 Python 出错位置，包内同时保存原始记录与按字段比对。 |
| 能否启动三臂 Pilot 或长训练？ | 查 `evidence/decision.json` 的 `source_coverage`、局部证明和预检记录。正式训练集 7,162 条、开发集 222 条都要按同一物理接口核对。分开审查原型的 5% 方向 J 筛查值，以及学得模型相对本臂 Epoch 0 的 5% 宏平均 Moment 晋级门槛。 |
| 本轮能否支持替代 Spin-It Faster 或求解提速？ | 查看投入的权重、原生验收的求解臂和计时范围。对照同一查询上的物理误差、原生回归与经过学习的候选在配对种子上获得合格解的总耗时，各自单列。 |
