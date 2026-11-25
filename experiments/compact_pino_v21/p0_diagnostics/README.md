# Compact PINO V2.2 P0：已实现的优化与诊断入口

更新日期：2026-11-25（北京时间）。计划标识：`CPINO-20261125-1756-R1`。

本次从 V2.1 提交 `54a6e717a9bf49f423e8882dd476d1256197629a` 开发。**这是可运行的 P0 软件实现，尚未构成 V2.2 物理验收或新训练结果。** 17:56 Pilot 的第 312 步仍维持 NO-GO；原有 5% 主指标、保护项和原生验收标准继续有效。

## 1. 本次实际做了什么

| 文件 | 实际行为 |
|---|---|
| `spin_adapter/unified_factory.py` | 修复 V2.1 读取配置时提前导入旧 SQNO／图扩展的依赖问题；仅在构建旧 V1 模型时加载其扩展。配置指纹、网络结构、参数和默认带宽不变。 |
| `endpoint_attribution.py` | 先核对存档的全原生数值回放，再做同点 Moment 替换、四臂 M/J 一步比较和重复性检查。支持同点无学习／初始响应；不会自创原生验收或 Newton 公式。 |
| `baseline_cases.py` | 从现有材料缓存、有限操作数及独立标签加载配对 case，验证文件和内容身份、坐标、单位、操作侧、导数资格；提供可导出的合成有限盒体示例。 |
| `deterministic_quadrature.py` | 真实调用 `evaluate_packed(..., model=None)`，对同一材料和有限切割面比较 1／2／4 体素带宽及两侧互补。全量材料十矩核对不通过时停止查询比较。 |
| `exposure_audit.py` | 在实际训练事件处记录曝光，区分预取、前向、正权重损失、反向、成功更新和跳过；断点恢复时区分全部执行记录与所选 checkpoint 的训练血缘。 |
| `test_*.py` | 对回放隔离、无效导数、真实保守材料与有限面响应、生产 Token16 的 J-only 梯度及缓存失效进行针对性验证。 |

已证明的工程缺陷是依赖耦合。**该缺陷不能解释面条的驻点误差。** 生产 Token16 的软件测试没有发现通用的 J 梯度断链；这也不能证明本地 Pilot 的训练驱动正确使用了标签或损失掩码。

## 2. 当前可运行范围

源码和测试可在普通 CPU 上运行。本次验证使用 Python 3.12、NumPy／SciPy、PyTorch `2.8.0+cpu`、pytest，以及材料构建所需的 C++17／OpenMP 编译器。生产环境的 GPU 数值结果仍须单独验证。旧 V1 模型继续需要它自己的扩展依赖。

从仓库根目录运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  experiments/compact_pino_v21/p0_diagnostics \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_v21_response.py \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_masked_derivative_supervision.py
```

测试内有一次仅用于验证软件梯度路径的临时 SGD 更新，不生成训练 checkpoint。它不计入正式 Pilot 更新，不作为精度收益证据。

本次最终整合验证覆盖新增 58 项、原有响应 15 项和导数掩码 44 项。最终运行结果、测试命令、源码哈希、环境警告和明确未执行项记录在本目录 `validation.json`。配置工厂另有 6 项选定回归通过。

### 2.1 立即复算真实算子代码路径的合成有限盒体

```bash
task_p0_root="$(mktemp -d)"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m \
  experiments.compact_pino_v21.p0_diagnostics.deterministic_quadrature \
  --write-synthetic-case "$task_p0_root/finite_fixture" \
  --output "$task_p0_root/finite_report.json"
```

此命令导出完整 case 示例，再走原网格 → 保守材料 → BlockMeasure → 有端盖有限操作数 → 无学习响应。示例具有非对齐边界体素和非零固定材料，使用独立盒体／平面解析矩作参考。它在 256³ 地址网格中使用小型材料，便于 CPU 复算；不代表真实几何的分辨率精度或面条性能。

也可以对导出的 `case.json` 再运行 `--case`；输出文件应使用新路径。省略输入选项时保留旧的解析图形合成实验，其竖直间隙并非有限面的欧氏距离，报告已明确区分。

## 3. Work B：接入真实终点和同轨迹归因

### 3.1 所需的真实资产

本次开发环境未挂载本轮实际训练机的 checkpoint、17:56 冻结标签、`diagnostic_training/train.py` 或冻结 `solve.py`。以下接入应在拥有这些资产的训练机上执行。不要把仓库旧证据或合成测试的数值替换进去。

每个查询导出一个 NPZ 和一个 JSON。NPZ 数值键：

| 键 | 约定 |
|---|---|
| `coefficients` | 同一查询的原始 10 系数。 |
| `predicted_moments`、`native_moments` | 各为 `[10]`，均含同一个固定材料；次序为 `1,x,y,z,xx,yy,zz,yz,xz,xy`。 |
| `predicted_jacobian`、`native_jacobian` | 有资格时为 `[10,D]`，D=3／8／10，已映射到同一个固定参数 chart。 |
| `chart_basis` | 有资格时为 `dc/dq`，形状 `[10,D]`；满列秩。Full J 使用原始系数 identity。物理 Jacobian 本身允许秩亏或为零。 |
| `direction` | 可选，真实轨迹的 chart 方向。零方向保留 Moment 分析，只标记单位方向检查不可用。 |
| `no_learning_moments`、`initial_moments` | 可选，来自同一查询的无学习响应／本轮初始响应。可另附对应 Jacobian，须显式声明其有效资格。 |

JSON metadata 应包含：

- 查询身份：`geometry_id`、`query_id`、`source_group_id`、`split`、`trajectory_stage`、`teacher_version`、`solver_version`、`coordinate_frame`、`units`、`chart_id`。
- 约定：`moment_order`，`coefficient_basis="raw_coefficients"`，`coefficient_multipliers=[1,1,1,1,1,1,1,2,2,2]`；正的 `editable_mass`、`length_scale`；有非零实际方向时提供冻结的 `coefficient_scale`。
- 资格：布尔值 `derivative_valid`；有效时 `derivative_dimension`；无效时明确 `derivative_reason`。不能以零 Jacobian 替代无效标签。
- 六份实际文件：`artifacts.geometry/operand/manifest/checkpoint/solver/solver_state.path`。前五个分别对应顶层同名 `*_sha256`；`solver_state` 对应 `adapter_contract.external_state_sha256`。CLI 会读取字节验证，并将绝对路径放入 `case['verified_artifacts']`。
- 原生契约：`thresholds.stationarity/balance` 来自冻结验收；`adapter_contract.acceptance_reads_jacobian` 为布尔值；`multiplier_policy` 为 `native_recompute` 或 `native_frozen`，必须如实描述实际求解器。
- 回放容差：`replay_tolerances` 明确列出 residual／direction／coefficient 各自的 `atol`、`rtol`，在查看候选结果前冻结，不允许用调松容差来掩盖接入错误。
- 回放锚点：`native_replay.accept` 保存存档验收的 `stationarity_residual`、`balance_residual`、`accepted`；一步实验另提供 `native_replay.step`，字段与下文 `step` 返回值一致。如果已有预测十矩对应的存档验收，提供 `predicted_replay` 作为第二个锚点，检查 M 改变时的乘子行为。

### 3.2 实际 solver adapter

实现 `factory(case)`，返回有 `accept(c,M,J)` 和按需提供 `step(c,M,J)` 的对象。`--adapter` 使用实际可导入的 `module:factory`。本包没有冒充部署求解器的默认 adapter。

**必须直接调用冻结原生实现。** 工具不会验证重写公式是否等价。每个新对象恢复同一份外部状态：目标、约束、chart、线搜索设置、固定材料、输入坐标、初值及其他部署设置。若原生函数会从输入 M 重算乘子，按其规则重算，不能把原生 M 对应的乘子固定给其他臂。

adapter 必须使用 `case['verified_artifacts'][role]['path']`，并将加载的查询系数、有限操作数、固定材料、标签与 checkpoint 预测关系逐项核对。文件哈希与残差数值回放都不能单独证明这些关系。CLI 另记录 adapter 源码路径和哈希；可用 metadata 的 `adapter_sha256` 强制匹配。

`accept` 返回：

```python
{
    "stationarity_residual": finite_nonnegative_float,
    "balance_residual": finite_nonnegative_float,
    "accepted": actual_native_boolean,
    # 可以另保留真实乘子、目标、约束向量等诊断量。
}
```

`step` 返回：

```python
{
    "direction": actual_chart_direction_or_none,
    "line_search_accepted": actual_native_boolean,
    "reason": actual_native_reason,
    "trial_coefficients": actual_raw_trial_or_none,
    "trial_native_stationarity_residual": actual_native_trial_value_or_none,
    "trial_native_balance_residual": actual_native_trial_value_or_none,
}
```

接受步必须有方向、试点和独立原生试点残差；原生线性求解拒绝可保留 `direction=None` 和原因。不得用 `inverse(dM/dc)` 替代真实 Newton 方程。四个混合 M/J 臂只用于局部诊断，不构成可部署响应。

下面使用占位的资产路径和真实模块名，应替换为训练机上的对应文件；CLI 及参数已经实现：

```bash
python -m experiments.compact_pino_v21.p0_diagnostics.endpoint_attribution \
  --case /absolute/path/to/matched_endpoint.npz \
  --meta /absolute/path/to/matched_endpoint.json \
  --adapter actual_native_adapter:factory \
  --output /absolute/path/to/new_endpoint_report.json
```

在有资格的轨迹位置增加 `--one-step`。先核对全原生回放，再做候选比较；结束时重复原生与预测更新，以检测底层共享状态污染。固定点验收不读取 J 时，换 J 后的残差必须在冻结重复容差内一致。

## 4. 同材料、同有限切割面的确定性基线

真实数据入口：

```bash
python -m experiments.compact_pino_v21.p0_diagnostics.deterministic_quadrature \
  --case /absolute/path/to/baseline_case.json \
  --cache-directory /absolute/path/to/static_block_cache \
  --output /absolute/path/to/new_baseline_report.json
```

case 最小外层结构为：

```json
{
  "schema": "tvcg.v21.baseline-case.v1",
  "meta": {},
  "assets": {
    "geometry": {"path": "geometry.npz", "sha256": "actual_hash"},
    "material": {"path": "material.npz", "sha256": "actual_hash"},
    "operand": {"path": "operand.npz", "sha256": "actual_hash"},
    "targets": {"path": "targets.npz", "sha256": "actual_hash"}
  },
  "block_voxels": 8
}
```

上面的空 `meta` 和哈希文字是格式示意，不能直接运行。`--write-synthetic-case` 导出的 `case.json` 提供完整字段示例。真实材料与操作数使用现有 `PreparedMaterial.save`、`FiniteOperand.save` 格式；不能传入切割后的真实保留网格充当模型输入。

`targets.npz` 必须包含 JSON 标量 `metadata`、`coefficients`、`fixed_moments`、`native_moments`、`native_full_editable_moments`。有资格的导数另附 `native_jacobian`、`chart_basis`、正的 `parameter_scales`，以及可选 `direction`。`metadata` 精确重复 `baseline_cases.LABEL_IDENTITY_FIELDS`。该 loader 保留现有字段 `source_group`，与曝光／归因中的 `source_group_id` 应映射到同一个来源 ID。

冻结身份同时包含 `length_scale`、非空的 `length_scale_definition`、`editable_mass`、`density`。合成示例采用 `editable_bbox_max_extent`；真实记录使用其既定尺度定义。改变外层 L 而不匹配标签身份会被拒绝，不能通过改归一化尺度放宽材料门槛。

完整材料十矩按 `m L^k` 缩放，任何分量超出 `1e-8` 时输出 `STOP_FULL_MATERIAL_REFERENCE`，不继续切割比较。通过后使用同一组积分点、同一 surface 与原生运动，分别计算 1／2／4 体素带宽，并对同面换侧核对：

\[
M_\mathrm{inside}+M_\mathrm{outside}=2M_\mathrm{fixed}+M_\mathrm{editable}.
\]

两次独立构面的 `c/-c` 不适用此互补检查。全材料守恒和同面互补都不能证明部分切割的精度，报告另列与独立标签的误差。

### 导数与计时口径

曝光数量、J 资格、数值 J 非零性分别处理。若独立导数无资格或活动节点的距离导数不唯一，报告明确不可用；饱和区域经资格确认的零导数可以保留。

两个工具都报告真实方向的 `Jv`。归因工具的单位方向使用原始系数空间范数及 `coefficient_scale`；基线工具使用冻结 chart 的 `norm(v / parameter_scales)`，报告明确记录该口径。未经转换，不合并两种单位方向指标；原始矩阵、方向和尺度都保留供复核。

基线的逐查询时间不含加载、全材料审计和 surface 建立；这些时间分别报告。**本次没有端到端合格解加速结论。** 部署计时仍需从原网格读入开始，包含原生构面、编码、传输、推理、失败、校正、回退和最终验收。

## 5. 实际训练曝光记录

`exposure_audit.py` 已实现，但尚未接入未随仓库提供的本轮训练驱动。应在真实事件边界调用，不能事后根据 epoch、batch 数或 `1164=388×3` 重建历史曝光。

先构造包含冻结快照身份的 `inventory`：

- `snapshot_id`、`manifest_sha256`、`registered_count`。
- `records` 为逐条已准入记录。每行包括 `record_id`、`query_id`、`geometry_id`、`source_group_id`、`moment_valid`、`jacobian_valid`、`loader_eligible`、`derivative_type`；有效 J 另有 `chart_id`、`derivative_dimension`。
- 建议保留 `surface_family`、`retention_bin`、`trajectory_stage`，用于真实曝光的分层汇总。

接入顺序：

```python
from experiments.compact_pino_v21.p0_diagnostics.exposure_audit import ExposureRecorder

audit = ExposureRecorder("new_segment.jsonl", inventory=inventory,
                         run_id=run_id, initial_checkpoint_sha256=initial_sha)

# 前向执行成功后报告；预取应另调 prefetch，不能替代 forward。
audit.forward(batch_id, accumulation_window_id, record_ids)
audit.loss(batch_id,
           moment_mask=final_moment_mask, moment_weight=final_moment_weights,
           jacobian_mask=final_j_mask, jacobian_weight=final_j_weights)

# 对应的实际 backward 已成功返回后：
audit.backward(batch_id, success=True)

# 仅在确认实际 optimizer 更新执行后；AMP 跳过应 committed=False 并写原因。
audit.optimizer_step(accumulation_window_id, committed=True)
```

mask 和权重应为实际传入损失的最终值；正权重包含全局 Jacobian 权重等因素。记录“进入损失”不等同于证明梯度非零。不要根据 `GradScaler.step` 返回 `None` 判断是否更新，因为很多成功的 optimizer step 同样返回 `None`。

checkpoint 只在无待提交累积梯度的边界、实际保存成功后登记。恢复使用新 segment 文件和原 checkpoint 回执；提供父日志才能核实血缘。日志汇总：

```bash
python -m experiments.compact_pino_v21.p0_diagnostics.exposure_audit \
  --events original_segment.jsonl resumed_segment.jsonl \
  --checkpoint actual_checkpoint_id \
  --output new_exposure_summary.json
```

输出同时列出所有执行记录和所选权重的有效血缘，包含成功更新的 Moment/J 曝光次数、独立查询、R3/R8/Full-J 分项、未曝光记录和跳过原因。缺日志报告 `INSUFFICIENT_EVIDENCE`；有中断或未提交窗口报告不完整。无日志不等于训练更新为零。

## 6. 训练机下一步按此顺序执行

1. 冻结 17:56 快照、初始和第 312 步权重、原生设置；保留历史结果。
2. 在真实训练驱动接入曝光记录，先核实实际 loader 可选记录、参与损失的唯一查询与各类有效 J。历史记录不足就明确不足。
3. 导出面条原生成功轨迹末段、失败终点，以及训练来源的同类位置；接入冻结原生 adapter。先通过原生数值和资产关联核对，再执行同点 M 和合格 J 替换。
4. 对同一批 query 运行无学习基线，并把同点无学习 M 放入归因 case；不能比较不同方法各自返回点就宣称定位了共同偏差。
5. Work A 继续按独立来源、曲面、保留比例、轨迹阶段补标。面条用于诊断，开发 222 条保留配对，封存盲测不进入本工具。
6. 依据真实结果选择下一轮单因素短实验。全原生回放失败先修接口；无学习也偏差先查共同积分近似；曝光／覆盖不足先修数据路径；学习独有退化再检验训练目标或局部修正。通过既定门槛前，不自动延长训练。

本轮软件工作已经为这些实验提供执行入口。真实 checkpoint 的终点归因、更新数据后的配对 Pilot，以及取得合格解的完整耗时仍待训练机执行，报告不得填成已通过。
