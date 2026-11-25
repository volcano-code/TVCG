# Compact PINO：P0 到实际训练的接入

本轮继续实现计划 `CPINO-20261125-1756-R1`。V2.2 是研发计划名称，模型仍为 V2.1 的 256³ 守恒材料、Token16、P-CRAFT/U-FNO 和同一响应的 Moment/Jacobian。新训练器不是不可见的 17:56 本地 `diagnostic_training/train.py` 的复刻。

**旧 Pilot 仍为 NO-GO。软件回归、合成数据上的参数更新、仓库数值核编译成功，都不代表 Noodles 已通过原网格验收。**

接入文档索引：[实际训练器](../training/README.md)、[研发筛选](PROMOTION_GATE.md)、[原生数值核](NATIVE_SOLVER_ADAPTER.md)。机器可读的本轮验证结论另见 `integration_validation.json`。

## 1. Work A 的可执行交付

入口：

```bash
python -m experiments.compact_pino_v21.p0_diagnostics.export_cases \
  --request /absolute/path/frozen_export_request.json \
  --output /absolute/path/new_frozen_snapshot
```

输出目录必须是新目录。仅当全部条目通过检查，才写出 `training_manifest.json`。失败时保留 `EXPORT_FAILED.json` 和已完成的中间资产供定位，不生成可直接训练的清单。

### 1.1 冻结请求

顶层字段：

| 字段 | 要求 |
|---|---|
| `schema` | `tvcg.v22.case-export.v1` |
| `snapshot_id` | 新冻结快照身份；后补数据使用下一快照 |
| `registered_train_count` | 登记训练条目数，不能用已准入数覆盖未完成差额 |
| `records` | 本次已准入、可导出的条目列表 |
| `authority` | 真实数据必须提供冻结 Work A 准入声明 `{path, sha256}`，不能只在本次请求中自行声明 split |
| `teacher_contract_sha256` | 准入声明中完整 `teacher_contract` 的 canonical JSON SHA256 |

每个条目包含：

| 字段 | 要求 |
|---|---|
| `record_id` | 稳定唯一标识；禁止重复 geometry/query/chart |
| `role` | `train` 或 `development`；面条诊断与封存盲测不进入此清单 |
| `meta` | 原始教师标签身份，见下文 |
| `assets` | `geometry`、`operand`、`targets`、`qualification`，每项均为 `{path, sha256}`；相对路径以请求文件所在目录为根 |
| `assets.material` | 可选已有静态材料缓存；只有网格、坐标、格点、密度和实现版本身份完全相同时复用 |
| `grid` | `{origin: [x,y,z], pitch: 正数, side: 256}`，与网格处于同一物理坐标 |
| `block_voxels` | 默认 8，必须整除 256 |
| `moment_weight` / `jacobian_weight` | 默认 1，非负有限值；权重不赋予导数资格 |
| `surface_family` | 有证据的曲面类别；未知显式写 `unknown` |
| `trajectory_stage` | 初值、一般迭代、末段、终点邻域、拒绝/事件等已核实阶段；未知写 `unknown` |
| `retention_bin` | 可省略，由可编辑材料真值自动算出；若提供必须一致 |

训练和开发的来源组、几何 ID、原网格文件哈希必须互不相交。冻结开发 222 条继续保持原配对身份；新增诊断面板另建清单，不能在查看候选表现后混入原固定开发口径。

### 1.2 原冻结来源与教师契约

`authority.json` 使用 `tvcg.v22.handoff-authority.v1`，包含：

- 与请求完全相同的 `snapshot_id`。
- `teacher_contract`：`teacher_version`、`backend=cherchi`、`map_profile=finite_directed_capped_intersection_difference`、`level=2`、`growth_policy=GROW_TO_CERTIFIED_V1`、`runtime_sha256`、`config_sha256`。当前 v1 不允许把另一后台或构面设置包装成同一部署教师。
- `assets`：原始 `source_manifest`、`teacher_runtime`、`teacher_config` 的实际路径和 SHA；附加证据放在 `evidence` 列表。
- `geometries`：原冻结表中的 `geometry_id`、`source_group`、`split`、`geometry_sha256`。不仅核对当前子集，整个表中的来源与几何内容都不能跨 split。
- `admitted_records`：每行 `record_id`、完整原始 `identity`、原 `geometry/operand/targets` 的 `assets_sha256` 和同一 `teacher_contract_sha256`。

这落实既有的“使用正式来源拆分和部署教师、由 Work A 签发准入回执”要求，不是重新申请用户许可。输入只导出 R02 一件几何时，也不能把它从冻结的 development 改为 train。未知或被锁定的查询在打开目标标签前被拒绝。

输出会复制并绑定这些原始证据，训练 manifest 中的 `source_authority` 使用相对输出根目录的路径。加载时整份快照只核验一次支持资产，再逐行查冻结表，避免每条查询重复读取大的教师二进制。它核验外部声明的交接一致性，不独立重做源数据划分或教师算法的科学审计。

### 1.3 原始文件格式与资格

`geometry.npz` 包含原网格在**已登记求解坐标**下的 `vertices[N,3]` 与 `faces[F,3]`。导出器不会猜测单位、居中或缩放。外壳和空腔的定向必须正确。

`operand.npz` 使用现有 `FiniteOperand.save()` 格式：有限切割面顶点、面片、端盖、保留侧和可用时的原始参数运动。不能放入真实布尔后的保留网格作为网络输入。

`targets.npz` 包含：

- `metadata`：JSON 标量，逐字段重复 `export_cases.RAW_IDENTITY_FIELDS`；不包含导出器尚未生成的 `material_identity`。
- `coefficients[10]`、`fixed_moments[10]`、`native_moments[10]`、`native_full_editable_moments[10]`。
- 只有合格导数行才读取 `native_jacobian[10,D]`、`chart_basis[10,D]`、`parameter_scales[D]`。D 为 3、8 或 10；Full-J 的 chart 是原始单位矩阵，**不要求物理 Jacobian 自身满秩**。
- 可选 `direction[D]` 必须绑定实际原生 Newton 方向来源。缺失方向不补造为零。

Moment 顺序固定 `[1,x,y,z,xx,yy,zz,yz,xz,xy]`，交叉矩不加倍；切割系数交叉项乘子为 `[1,1,1,1,1,1,1,2,2,2]`。`native_moments` 包含固定材料一次；`native_full_editable_moments` 仅包含完整可编辑材料。

`meta` 的完整键由代码中的 `RAW_IDENTITY_FIELDS` 定义；核心身份包括几何/查询/来源/划分/教师版本、坐标/单位/密度、上述矩与系数约定、原网格/切割面/查询指纹、保留侧、导数资格与 chart、完整可编辑质量和长度。当前导出长度定义为 `editable_bbox_max_extent`。

`qualification.json` 的结构：

```text
schema: tvcg.v22.teacher-qualification.v1
record_id: 与请求一致
identity: 完整重复 RAW_IDENTITY_FIELDS
assets_sha256: {geometry: ..., operand: ..., targets: ...}
moment_valid: true
ambient_event_valid: 1 / 0 / -1
restricted_event_valid: 1 / 0 / -1
neighbourhood: {scope: ..., radius: ..., radius_definition: ...}
evidence: [{path: 原始验收/导数证据文件, sha256: ...}, ...]
```

合格 Full-J 要求 `ambient_event_valid=1`、`scope=ambient_raw_coefficients`；合格 R3/R8 要求 `restricted_event_valid=1`、`scope=fixed_rank_chart`。有效邻域半径必须为正，记录其实际参数空间和范数定义。不能把“一点有有限矩阵”写成“有效邻域已确认”。证据文件的存在与哈希只能验证交接一致性，真实资格仍由 Work A 的独立教师/事件审计负责。

未合格行在 `meta` 写 `derivative_valid=false` 和明确 `derivative_reason`，保留 Moment；导出器不会读取或复制该行无效 J，不会把它改成零标签。未获 Moment 准入的行保留在上游拒绝台账，不混入可训练条目。

### 1.4 导出时实际执行的物理检查

每个新材料身份只执行一次静态保守预处理；不同查询复用结果。所有十矩分别与独立网格积分和完整可编辑教师量比较，按 `m L^k` 缩放，最大误差必须 ≤ `1e-8`。保留材料质量须位于完整可编辑质量范围内。固定材料的零至二阶矩必须符合非负质量测度。

加载整份快照时，还共享静态材料打包结果和文件校验缓存；大材料 NPZ 不为每条查询重新解包。文件大小、inode、修改时间或状态时间变化会触发 SHA 重查。缓存只存静态 memmap、十矩与密度，不保存跨参数更新的学习编码。

输出包含可直接被已有确定性基线读取的 case 文件、原始教师与资格证据副本、材料缓存、训练清单和 `export_receipt.json`。覆盖回执按来源 × 曲面 × 保留比例 × 轨迹阶段 × 导数类别列数。它报告“导出/准入”，**不报告虚构曝光次数**。

合成数据只允许额外显式参数 `--synthetic-fixture`，且输入本身必须标为 `analytic_fixture/synthetic`。这种输出不能成为真实长训练或物理验收的证据。

## 2. Work B 的执行顺序

1. 保存 17:56 的实际初始权重、选中权重、逐查询结果、教师版本和原生状态，原样归档 NO-GO。
2. 使用与该状态相符的真实求解适配器做全原生重放；不能重放就停止该案例的混合归因。
3. 固定同一查询比较原生、无学习、初始与候选的 Moment；在已资格确认的参数空间做四种 M/J 一步对照。
4. 用同一 case 运行确定性基线，先检查完整材料，再比较有限切割面两侧和既定带宽。合成误差不能替代面条与跨训练几何结果。
5. 根据归因只选择一个短 Pilot 因素。新的数据采样/训练驱动需要新的冻结配置及 E0；不得声称与不可见的旧优化器、batch 或损失完全相同。
6. 真正执行 Moment＋合格 Jacobian 更新、保存与恢复；按日志区分池大小、曝光次数、唯一查询和参加 J 损失的样本。
7. 使用冻结协议从逐查询结果复算主指标和保护项。短测过关才进入观察和有条件扩展预算。
8. 候选终点由原网格重新验收；直接替代与原生热启动/校正路线分别计时。失败候选不能产生“合格解加速比”。

## 3. 保持条件性的研发分支

本次没有在证据不足时同时更改编码器、分辨率、初始化、带宽、响应结构和损失。物理残差匹配损失、从无学习响应精确起步、局部积分细化都仍须由真实归因触发。

若选择物理损失，目标应为同一冻结原生规则下预测矩与真值矩产生的残差之差；不能把一般查询强制成零残差驻点。若选择边界细化，应先证明无学习与学习响应在同点、跨几何出现共同偏差。保持同一响应求 Moment/Jacobian，以及每次参数更新后失效的学习编码缓存。

真实数据、原生教师环境、RTX 5090 D 和训练回执未到位时，本仓库只能完成软件接入与受限软件验证；不得把这些条件改写成“所有规划已验收”。

## 4. 已有命令如何连起来

下面的全链演示是真实模型执行，但输入仍全部是合成解析几何，输出目录必须不存在：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m \
  experiments.compact_pino_v21.p0_diagnostics.integration_smoke \
  --output /absolute/new/export-train-screen-smoke
```

它生成两条训练查询和一件实际缩放后的开发几何，经过 Work A 格式导出、材料检查、真实两步 M+J 更新、E0/候选/无学习预测、冻结节点选模和研究筛选。它会明确报告缺失的 R8、Full-J 与保留比例覆盖，绝不因为软件通路完成而授予长训资格。

精确恢复的软件验证单独运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m \
  experiments.compact_pino_v21.training.smoke \
  --output /absolute/new/exact-resume-smoke
```

真实短 Pilot 使用 [训练器文档](../training/README.md) 的 `--mode trial`、冻结数据、配置、评估协议与预算记录。用户已批准的短试验授权仍然有效；预算文件记录该授权适用的快照和条件，不要求重复询问。长训继续受既定 NO-GO/GO 条件约束。

新驱动显式采用 Adam、batch size 1 等默认实现选择，未证明它们与旧 17:56 驱动一致。Work B 应核对旧实际配置；若采用新驱动，配对 A/B 都使用同一新驱动和共同初始权重，仅改变被选中的实验因素。不能把新驱动的默认配置称为旧 Pilot 的精确续训。

原生数值核目前只在自己的明确 profile 下执行原仓库公式和 Full-J 一步求解，详见 [原生适配文档](NATIVE_SOLVER_ADAPTER.md)。正式 Cherchi 的私有验收接口与原网格轨迹仍需要真实 Work 资产适配；不要把另一个 CGAL 工厂接到 Cherchi 标签上。

## 5. 真正关闭研发条目的剩余证据

| 条目 | 所需的真实证据 |
|---|---|
| 17:56 旧 Pilot 复核 | 实际协议、E0/选中 checkpoint、逐查询输出、曝光日志和同版本 native 状态 |
| 终点与一步归因 | 正式部署求解器的全原生重放、同点十矩替换；合格参数空间内的四种 M/J 一步记录 |
| 确定性物理基线 | 正式原网格/有限面/运动/教师标签上的同点结果，及跨训练来源边界误差 |
| 定向补数 | Work A 的末段/终点、中高保留、跨来源曲面、合格 R8/Full-J 逐行回执 |
| 新候选短测和长训 | 目标 GPU 的联合更新/恢复验证、新冻结配对短测、过门槛后才执行的训练权重 |
| 最终实用价值 | 原生热启动/校正实测、原网格验收、协议 0–49 配对初值及含失败/回退的冷/热完整计时 |

真实资产或执行入口缺失时，以上条目保持未完成。分支已实现的工程接口可以继续使用，不能用制造数据补齐这些物理证据。
