# 给 Work 的 Phase 2.10.4 整改指导 — Pilot 数据维护、教师资格与真实初始状态闭环

**日期：** 2026-11-21  
**仓库：** `volcano-code/TVCG`  
**依据：** Chat 端收到的 `CHAT_HANDOFF(1).md`（Phase 2.10.3）

## 0. 当前共同基线（不要回退旧统计）

本轮 Work 交接报告：
- 数据门槛 8/8 通过，`pilot_ready=true`，但该标记只适用于 operator-pilot 数据准入；
- 审计在 CPU 完成，训练步数为 0；
- Train：31 几何 / 21 可追溯生成来源组 / 2,019 查询；
- Strict Development：9 几何 / 8 来源组 / 36 查询；
- Fixed Benchmark：5 几何 / 2 论文重建来源组 / 320 查询；
- 更宽开发档案：13 几何 / 64 查询；严格开发视图隔离命中公开 TRELLIS-500K 的来源家族；
- 31 个训练父网格中，实物采集数为 0；两套原始模板仍保留在训练集；
- Diamond 是一个新封存的真实来源工程用例，512³ 二值材料经原生重放和独立检查；
- Flowerpot v2 仍为 `external_candidate`；
- 教师仍为 `operator_pilot`，不是作者等价的生产 Faster 教师。

Chat 端 Full V1 CUDA Smoke 尚未通过；当前阻塞发生在 Geometry–Physics Cross Attention 对百万区间特征做 Key/Value 全量投影时 OOM。Work 不负责修改 Chat 端网络或显存代码。

## 1. P0 — 正式 Manifest 与增量可复现性

以当前正式 Manifest 为唯一基准，保留历史快照，不覆盖旧文件。

重新交付：
- `manifest.json`
- `manifest_delta.json`
- `SHA256SUMS`
- `pilot_readiness_update.json`
- `development_counts.json`
- `split_leakage_audit.json`
- `README.md`
- `REPRODUCE.md`

必须明确：
1. Manifest 的 SHA-256；
2. 每个 split 的几何数、来源组数、查询数；
3. 每条记录的标签支持数；
4. train/development/fixed_benchmark 的角色；
5. 哪些门槛通过、哪些仍是 operator-pilot 限制。

请提供一条从原始资产到正式 Manifest 的完整复现命令，确保 Chat 端可以独立重建和核验数据快照。

## 2. P0 — Development 监督质量扩充

当前严格开发集为 9 几何 / 8 来源组 / 36 查询，其中：
- 完整 Jacobian：11；
- 3D restricted Jacobian：4；
- 8D restricted Jacobian：3；
- Moment-only：18。

这足以支持接口检查和有限 Pilot 诊断，但对多类型几何的泛化结论仍然偏弱。

本轮优先：
- 保留现有严格来源隔离规则；
- 优先为现有 9 个开发几何增加有效查询，而不是盲目增加相似几何；
- 扩充不同曲率、长宽比、凹凸结构及一般二次曲面查询；
- 优先补齐完整 Jacobian 或明确维数的 restricted Jacobian；
- 保留 Moment-only 样本，但不得把它们计为 Jacobian 监督；
- 每个查询记录教师版本、积分域、坐标系、事件状态、导数有效原因。

目标是把严格开发集逐步扩展到至少 64 条有效查询，并让不同来源组都覆盖不止一种查询类型。达不到的部分如实报告，不得通过复制、扰动查询或伪造导数补数。

## 3. P0 — 21 个训练来源家族与模板偏置审计

当前 31 个 Train 几何归入 21 个可追溯生成家族，这是比旧版 2 个批次组更可信的来源口径。但报告仍显示：
- 31 个父网格中实物采集为 0；
- 两套原始模板仍留在训练集。

请追加：
- 每个来源家族的父几何、生成方式、模板依赖；
- 两套模板对训练查询和标签的贡献比例；
- 删除任一模板后的训练覆盖变化；
- 训练来源组级别的几何类型分布；
- 与 Development / TRELLIS-500K 公开来源的近重复审计。

不要将 21 个生成家族描述成 21 个真实采集对象。请将“生成家族独立性”和“真实物体分布代表性”作为两个不同指标报告。

## 4. P1 — Diamond 原生封存用例完善

Diamond 是当前唯一新封存的真实来源工程用例。请把它做成可复现的完整证据包：

- 原始 OBJ 和 SHA-256；
- 原生 Spin-It 输入 TaskSpec；
- 原生求解器版本、提交、参数和日志；
- 初始材料状态导出 JSON；
- 初始状态和几何指纹；
- 固定材料掩膜、设计域掩膜；
- 512³ 独立重载；
- 十矩重积分及误差；
- CoM、惯量张量、主轴、连通性；
- 接触/支撑条件；
- 失败分类和完整状态码。

特别说明：Diamond 的成功只证明这一对象的工程闭环，不得推广为全部 31 个训练对象均已通过原生 Spin-It 封存。

交付一个独立可重放命令和 `diamond_native_seal_report.json`。

## 5. P1 — Flowerpot 与其他失败对象的失败归因

Flowerpot v2 保持 `external_candidate`，不能升级为 native Spin-It。

继续分析其 `RELAXATION_UNKNOWN`：
- 是初值不满足约束；
- 还是优化器停滞；
- 还是接触/拓扑切换；
- 还是数值条件差；
- 还是确实存在不可行证据。

没有证书时不得宣称不可行。

Chess Pawn 的不可行证书单独保存，禁止与 Flowerpot 失败混为一类。

交付 `native_solver_failure_taxonomy.json` 和每个案例的日志引用。

## 6. P1 — 教师资格边界与 Faster 对照

教师当前维持 `operator_pilot`。请保持这一资格，不要因为数据门槛 8/8 就自动升级。

补充报告：
- 当前本地 libigl/CGAL 教师所实现的具体数学公式；
- 与 Spin-It Faster 2024 论文公式逐项对照；
- 输入、输出、坐标、二次曲面系数约定；
- 十矩与 Jacobian 的独立重放；
- 有限差分步长和误差；
- 事件边界、不可微切换和失败状态；
- 已知偏差和未实现部分。

如果无法访问作者发布实现，应明确标记“公式复现/本地教师”，不得称为作者代码复现或作者等价。

## 7. P2 — Event 标签与事件边界数据

Work Manifest 中的 `event_valid` 和 `restricted_event_valid` 必须继续与 Moment/Jacobian Mask 分离。

请提供：
- 事件类别定义；
- 事件标签来源；
- 事件边界的几何/查询条件；
- 事件标签是否来自教师真实返回，而非后处理推断；
- 事件不确定或未知的编码；
- 每个 split 的事件类别计数。

当前 Chat 端 V13 Pilot 先训练 Moment/Jacobian，不代表 Event Head 已有监督训练。Work 应先交付真实事件标签合同，不要求为配合当前 Pilot 伪造 Event 标签。

## 8. P2 — 交付包和验收

交付目录建议：

```text
phase2_10_4_work_delivery/
  manifest.json
  manifest_delta.json
  SHA256SUMS
  pilot_readiness_update.json
  development_counts.json
  split_leakage_audit.json
  train_source_lineage.jsonl
  template_bias_audit.json
  diamond_native_seal_report.json
  native_solver_failure_taxonomy.json
  teacher_formula_equivalence_report.json
  event_label_contract.json
  README.md
  REPRODUCE.md
```

每次增量必须包含：
- 新旧 Manifest 指纹；
- 变化记录；
- 新增/删除/升级状态的几何；
- 查询数量变化；
- 来源组变化；
- 教师版本变化；
- 复现命令。

## 9. 明确禁止事项

- 不修改 Chat 端神经网络架构；
- 不为了增加查询数复制标签；
- 不将 Moment-only 样本统计为 Jacobian 有效；
- 不把 generated source family 称为真实采集物；
- 不将 Diamond 单例泛化为所有对象均已封存；
- 不将 Flowerpot external_candidate 改成 native_spin_it；
- 不把 operator_pilot 教师升级成作者等价教师；
- 不将 `pilot_ready=true` 解释为模型训练成功或研究结论成立。

## 10. 本轮完成定义

本轮结束时 Work 应交付可复现的正式数据快照、严格开发集监督扩充、模板偏置审计、Diamond 封存证据、教师公式资格报告和 Event 标签合同。

Chat 端负责 Full V1 CUDA Smoke 与训练工程。两端共同评审后，才决定是否扩大 Pilot 或进入正式长训练。
