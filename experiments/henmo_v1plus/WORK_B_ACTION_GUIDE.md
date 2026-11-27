# Work B：HENMO V1+ 验证、训练与求解执行指南

**职责：Work B 运行和训练框架，Work A 交付数据与资格凭据。** 本轮主线是粗边界查询、完整一阶切向和必要细化；优先确认它确实减少工作量并保持物理精度，再扩训练预算。已有长训权重归档为比较对象，不将新头声称为旧头的等价替换。

## B0. 软件环境与分支

- [ ] 使用研究分支 `research/henmo-v1plus`，记录实际 commit、Python、PyTorch/CUDA、GPU 型号及原生后端版本。
- [ ] 不覆盖正在运行的旧实验目录；新模型/新快照/新选择协议有独立输出目录。
- [ ] 运行新的软件验证。CPU 软件结果只能复核代码，目标 GPU 必须再做真实 M+J 更新与断点恢复。

```bash
python -m pytest -q tests/research/test_henmo_operator.py \
  tests/research/test_henmo_hierarchy.py tests/research/test_henmo_execution.py \
  tests/research/test_henmo_response.py tests/research/test_henmo_training.py \
  tests/research/test_henmo_metrics.py tests/research/test_henmo_deployment.py

python -m experiments.henmo_v1plus.cli smoke --output /absolute/new/henmo_software_smoke
```

谱卷积参数包含复数。直接构造时使用 `model.to(device)` 再 `.float()` 或 `.double()`，不要把整个模型以实数 `dtype` 的 `.to(...)` 转换；新 CLI 已按此处理，checkpoint 还校验实际张量 dtype。FP32 网络与 FP64 矩聚合是默认候选，不默认启用 AMP。

## B1. 先做同输入的算子对照

- [ ] Work A 数据审计通过；优先复用当前合格 256 快照，512/1024 分别作为表示对照，避免把换数据和换响应的收益混在一起。
- [ ] 同一查询比较：已归档旧模型、确定性 GL2、确定性 GL3、新模型初始权重/短训权重。
- [ ] 每组报告 Moment 每分量、来源宏平均、最差/P95/P99、低/中/高保留、末段/终点及 R3/R8/Full-J 有效分母。
- [ ] 曲面事件使用资格掩码；J 缺失计为拒绝/不足证据，不算数值零误差。
- [ ] 在训练来源和单独开发诊断上执行同一点 M/J 四组合与原生一步方向/线搜索对照；固定点验收只替换 M 的实验继续保留。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.henmo_v1plus.cli compare \
  --manifest /absolute/frozen/training_manifest.json \
  --config experiments/henmo_v1plus/configs/pilot512.json \
  --mode trial --device cuda:0 --dtype float32 \
  --output /absolute/new/henmo_initial_comparison.json
```

不指定 checkpoint 时是未训练候选，报告明确记录这一点。该入口的 GL2/GL3 是同材料平滑响应的数值积分；仍需以原网格教师检验偏差。论文强对照应增加可控精度的确定性方法，不能只挑误差较大的两点规则证明神经网络有用。

## B2. 性能剖析：先确认节省发生在哪里

逐阶段保存真实计时与工作量：

| 阶段 | 应记录 |
|---|---|
| 冷启动 | 网格读取、材料构建/加载、层级构建、一次几何编码、峰值 CPU/GPU 内存 |
| 每次查询 | 原生构面、有限面准备/最近点、层级分类、活跃点打包/传输、响应和 J、细化与原生接管 |
| 查询工作量 | selected_coarse_blocks、evaluated_atoms、active_atoms、neural_network_calls、padded_response_atoms、refined_nodes、deterministic_leaf_evaluations、compression fallback |
| 完整求解 | 所有候选、失败、校正、回退、最终验收及取得合格解的总时间 |

`material_cache_bytes` 是派生材料规则/特征的缓存预算，不是整个进程 RAM 上限；原材料、树节点、稠密 64/128 上下文和训练图还占内存。默认缓存各 128 MiB 也不代表实际显存总量。

同带宽粗节点已经真实批量运行神经响应及一阶切向；树遍历和有限面查询仍有 CPU/Python 工作。先看剖析，不能因峰值显存低就断言提高 query batch 会按比例变快。改变训练 batch 后，等查询曝光预算与等 optimizer 更新预算通常不能同时成立，应分别比较并完整报告另一项数量；不得把新增曝光带来的改善写成纯吞吐收益，也不能改变原有精度筛选口径。

若几乎所有节点都细化到原生 GL3，先定位粗响应误差和指示阈值，而非宣称已经得到层级加速。容差在看新候选之前冻结，不为通过速度测试临时放宽精度。

## B3. 可开始的短训练

首轮配置见 `configs/training_pilot.json`：Adam、seed 830251、学习率 3e-5、batch 1、M/J 权重 1、辅助物理损失 0，312 次实际更新/624 次最大 attempts。新头可从守恒平滑响应零初始化；验证 J-only 与联合损失都能真实更新参数。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.henmo_v1plus.cli train \
  --manifest /absolute/frozen/training_manifest.json \
  --config experiments/henmo_v1plus/configs/pilot512.json \
  --training-config experiments/henmo_v1plus/configs/training_pilot.json \
  --mode trial --device cuda:0 --dtype float32 \
  --evaluate-every 104 --output /absolute/new/henmo_pilot
```

- [ ] 先在目标 GPU 完成真实 M+J 更新与恢复后下一步连续性检验，再消耗 312 步预算。
- [ ] 保存开发初值与各评估节点；统计 committed update、attempt、实际曝光、训练层级和非有限/零梯度原因。
- [ ] 当前每条既有训练查询在 32/64 粗层级中选择一个，随机状态随断点保存；开发评估使用冻结的自适应配置。没有因此产生新的教师样本。
- [ ] 新运行以初始/恢复权重作 incumbent，记录 selected_step 与是否发生新增更新；这不冒充旧 V2.1 只选训练后检查点的协议。
- [ ] 同预算与无学习基线、旧 E16 兼容臂比较，不以增加更新量掩盖结构变化。

E16 迁移须使用 `--initialize-from`、`--initialize-sha256`、`--transfer-rules` 指定源 SHA256 与显式模块语义规则；checkpoint 的模型字段不是 `model` 时还须使用 `--initialize-state-key`，裸 state dict 用 `none`。新响应头语义不同就重新初始化。形状相同不是语义相同，初始化也不能载入旧 optimizer 冒充精确续训。

真正续训用 `--resume`，保留 checkpoint 及同名 `.pt.json` sidecar，输出新会话目录，并保持完整模型配置、数据 binding、dtype 和训练配置不变，包括 `max_steps` 与 `max_attempts`。以下示例只恢复尚未完成原预算的运行，SHA256 使用冻结回执中的值：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.henmo_v1plus.cli train \
  --manifest /absolute/frozen/training_manifest.json \
  --config experiments/henmo_v1plus/configs/pilot512.json \
  --training-config experiments/henmo_v1plus/configs/training_pilot.json \
  --resume /absolute/henmo_pilot/last.pt \
  --resume-sha256 REPLACE_WITH_FROZEN_CHECKPOINT_SHA256 \
  --mode trial --device cuda:0 --dtype float32 \
  --evaluate-every 104 --output /absolute/new/henmo_pilot_resumed
```

已完成 312 步不会因恢复而自动增加预算。扩大预算或换数据/训练范围应另行冻结新阶段，按显式迁移初始化并重新建立 optimizer；不要修改原配置后继续声称精确恢复。`--mode long` 不修改预算，也不代表物理或训练晋级。恢复报告只在当前 continuation 内选模；`selected_has_new_updates=false` 表示本次续训未产生优于起始 incumbent 的新选中权重，并不抹去此前已完成的更新。

例如，短训审核通过后，将新预算写入独立的 `training_next_stage.json`，核对准备复用的模块语义并冻结 `reviewed_semantic_transfer_rules.json`，即可用已有初始化入口启动下一阶段：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.henmo_v1plus.cli train \
  --manifest /absolute/frozen/training_manifest.json \
  --config experiments/henmo_v1plus/configs/pilot512.json \
  --training-config /absolute/frozen/training_next_stage.json \
  --initialize-from /absolute/henmo_pilot/selected.pt \
  --initialize-sha256 REPLACE_WITH_FROZEN_CHECKPOINT_SHA256 \
  --initialize-state-key model \
  --transfer-rules /absolute/frozen/reviewed_semantic_transfer_rules.json \
  --mode trial --device cuda:0 --dtype float32 \
  --evaluate-every 104 --output /absolute/new/henmo_next_stage
```

规则文件采用 `migration.TransferRule` 的 JSON 列表：每项给出 source/target prefix 及两侧 `ModuleSemantics`。完全相同模型配置且全部语义已核对时，可用空 prefix 映射完整 state dict；有任何结构或语义变化则逐模块审核，不能为了通过形状检查给两侧机械填同一句话。新阶段从 step 0 计数，保留旧阶段回执及源 checkpoint 哈希，重新建立 Adam 和采样随机状态；应报告各阶段与累计预算，不把它写成此前 optimizer 的无缝续跑。

## B4. 继续使用的研究筛选与物理门槛

为保持历史对照，5% 相对 Moment 改善与各受保护项最多 +5% 退化可继续作为工程筛选。它们不是从 Newton 的 1e-12 容差推导出的保证；新 HENMO 选模/初始 incumbent 规则必须单独预登记。

- [ ] 主要 Moment 指标相对本轮冻结初值至少改善 5%，或明确报告未达到；同时与同输入确定性基线比较净收益。
- [ ] 高保留、中保留、R3、R8、Full-J 与逐来源结果完整；缺标签为证据不足。
- [ ] 在终点/末段检查原生驻点与约束残差；单纯平均十矩改善不自动获得最终通过。
- [ ] 直接替代和原生精修的混合路线分别报告。混合路线必须使用相同原生停止条件与最后原网格验收。
- [ ] 归档所有未通过尝试，不能只在成功子集计算速度比。

这些筛选不要求未训练网络在第一步前已达到最终精度。数据/数值正确性过关后可以进行短训练；尚未证实端到端收益时，不直接放大成长期算力投入。

## B5. 原生部署与完整验收

按 `DEPLOYMENT_GUIDE.md` 接入真实 backend、源任务与同一个 solver 回调。候选输出始终是提案；最终通过来自现有独立原网格 `verify`。不得用体素代理自身的 residual 替代它，也不能承诺只需 2–4 次校正。

原生与候选使用相同初值集合、几何、目标、约束、线搜索、停止条件及完整 settings。冷启动和缓存复用分别配对，最终目标值与原生的容许差依原冻结协议（现有建议为 1%）执行。只有成功数/总数、物理残差和完整时间均合格，才讨论加速。

当前提供有界串行多起点调度，不宣称 GPU 并行全部优化轨迹。先取得单起点合格解时间优势，再决定是否实现专门的批量求解器；可选模块不应阻塞当前主线验证。

## B6. 回传清单

1. git commit、环境/设备、模型/训练配置与所有哈希。
2. 软件/GPU 联合更新和恢复报告、材料审计及缺失覆盖。
3. 初始与选中 checkpoint、sidecar、逐步 journal、原始开发逐查询结果。
4. GL2/GL3/旧模型/新模型同输入对照，完整时间分解及活跃工作量。
5. 原生同点 M/J 四组合、方向/线搜索与终点验收结果。
6. 注册部署、逐初值求解日志、所有失败/回退/精修费用、独立原网格最终验收。
7. 明确决策：继续短训/需要修复/进入长训；不得仅以“代码测试全部通过”签发物理 GO。
