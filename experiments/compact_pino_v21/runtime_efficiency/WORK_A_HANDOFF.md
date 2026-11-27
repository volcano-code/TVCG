# Compact PINO V2.3：Work A 数据与物理凭据交付清单

版本日期：2026-11-27
交付对象：Work A（数据、原生教师与逐查询资格）
关联工作：Work B 复用 E16 权重，验证 V2.3 执行等价、查询性能和完整求解路径。

## 1. 本轮决策与立即执行顺序

**先让 Work B 用冻结的 E16 权重开展执行等价和速度对照；Work A 同时审计当前实际数据，并定向补充已有失败证据所指向的覆盖。无需等扩数完成，才能开始本轮运行时优化。** E16 的具体文件、配置、训练快照和 SHA256 以 Work B 提交的冻结回执为准；本清单不预设该权重已通过原生物理验收。

本轮执行优化复用既有物理响应和权重。新查询的表面准备、缓存、距离计算、M-only 请求分流等运行时改进，先在相同输入和相同权重下核对输出与拒绝行为。它们是否等价、是否变快，不以新增若干条训练样本作为前置条件。

- [ ] 向 Work B 交付当前 E16 对应的准确数据快照入口、manifest SHA256、教师契约及已有合格查询身份；不可只发送旧聊天中的汇总数量。
- [ ] 先运行第 2 节的元数据审计，确认当前数据的来源、曲面、保留比例、轨迹阶段和标签资格。
- [ ] 并行运行现有 loader 的完整资格检查，形成可核对的覆盖回执；对于暂时不可访问或未通过的资产，提交明确原因。
- [ ] 从当前训练来源优先补第 3 节中有用途的原生轨迹点。每批合格数据完成后即可交付，不等待凑齐固定总量。
- [ ] 固定开发 222 条继续作为原协议的配对集合；新增独立开发证据另建版本与报告，不改变正在运行实验的身份。
- [ ] 在 Work B 请求改变训练数据或损失时，交付新的冻结快照与差异清单。当前 E16 的执行对照继续使用原输入，不混入新数据后宣称“仅执行优化”。

**不设新的任意数量门槛。** 不要求为了启动执行优化先凑够某个几何数、某个总样本数或固定的终点配额。证据必须足以支持实际声称的范围：只有一个来源时只能作该来源诊断；没有合格 Full-J 时，该指标仍是证据不足；逐行资产、教师身份和导数有效性要求保持严格。

## 2. 当前数据审计：以本地 manifest 为准

### 2.1 冻结审计对象

- [ ] 确认输入是当前训练实际使用的 `compact_pino.training-manifest.v1`，记录绝对路径、`snapshot_id` 与文件哈希。
- [ ] 对齐 E16 的 checkpoint/run receipt，区分“注册的记录”“loader 允许的记录”“实际读取的标签”“实际参与优化器更新的曝光”。
- [ ] 不把曾经出现的 388、2,158、1,164 或其他旧快照数量写成当前覆盖。**当前最新覆盖数量待本地 manifest 与步骤日志审计后填写。**
- [ ] 如果现在只有原始 R1 case 包，先按现有 [数据集成契约](../p0_diagnostics/INTEGRATION.md) 导出带教师资格和资产绑定的 training manifest。覆盖工具不接受任意目录作为正式训练集，不自动补写来源或资格。
- [ ] 输出写入新的审计目录，保留原 manifest、case、标签、资格凭据和旧审计文件。

### 2.2 第一步：只读元数据审计

在仓库根目录，使用已安装本项目依赖的 Python。把以下绝对路径占位项替换为当前真实路径：

```bash
python -m experiments.compact_pino_v21.runtime_efficiency.coverage \
  --manifest /ABS/CURRENT_SNAPSHOT/training_manifest.json \
  --output /ABS/NEW_AUDIT/coverage_metadata.json
```

这一步读取 manifest 和 case JSON，并核对 case SHA256；不读取材料或标签资产，不运行原生教师。

- [ ] 保存 `coverage_metadata.json` 和执行日志。
- [ ] 检查 `source_or_identity_overlap` 与 `duplicate_query_identities`，处理训练/开发来源、几何、轨迹交叉和重复查询身份。
- [ ] 检查 `cells`：它按 `role × source_group_id × surface_family × retention_bin × trajectory_stage` 分组。
- [ ] 对 `UNKNOWN` 补充真实来源记录；没有证据时继续保留未知，不依据文件名、模型预测或默认比例猜测。
- [ ] 明确元数据中的 `derivative_declared_valid` 是教师声明；此时 `moment_loader_verified`、`jacobian_loader_qualified` 等为 `null`，不能计作已加载的合格标签。

正常且未检测到身份交叉时退出码为 `0`；检测到交叉或重复查询时写出报告并返回 `2`。资产哈希、记录格式或身份不一致会直接报错。退出码 `0` 仅说明该审计完成并未检测到这些冲突，不是训练或物理精度 GO。

### 2.3 第二步：复用现有 loader 检查标签与材料

```bash
python -m experiments.compact_pino_v21.runtime_efficiency.coverage \
  --manifest /ABS/CURRENT_SNAPSHOT/training_manifest.json \
  --verify-load \
  --mode trial \
  --cache-directory /ABS/MATERIAL_CACHE \
  --output /ABS/NEW_AUDIT/coverage_verified.json
```

`--verify-load` 调用现有 `training.manifest.load_training_manifest()`：检查外部准入包、教师契约、资产哈希、标签绑定、逐行导数资格及完整材料十矩。它可能加载较大的材料资产；静态缓存按现有身份复用。

- [ ] 核对 `verification_mode="existing_training_loader"`，保存完整日志。
- [ ] 核对 `splits` 中训练/开发各自的合格 M、R3、R8、Full-J 与残差标签数量，区分声明与 loader 通过。
- [ ] 核对保留比例来自原生标签：`(native_moments[0] - fixed_moments[0]) / native_full_editable_moments[0]`；不可用总质量比例或网络预测比例代替。
- [ ] 使用既有分档：近零 `<0.01`，低保留 `[0.01, 0.1)`，中保留 `[0.1, 0.7)`，高保留 `[0.7, 1]`。声明与原生复算分档冲突时先修复数据，不修改边界迁就结果。
- [ ] 检查完整材料十矩按各分量物理尺度归一化的最大误差不超过现有 `1e-8` 门槛。这个材料门槛不等于切割后预测精度或求解成功。

**范围限制：** 该命令验证现有凭据与载入契约，不重新计算原生布尔、重新积分所有切后网格，也不独立重做导数邻域证明。教师资格的真实性仍由 Work A 的原始证据支撑。`--mode smoke` 只供显式解析测试夹具使用，不能用来绕过真实数据的准入检查。

### 2.4 第三步：核对实际训练曝光与损失分母

如果对应训练运行保留了现有 trainer 的 `steps.jsonl`：

```bash
python -m experiments.compact_pino_v21.runtime_efficiency.coverage \
  --manifest /ABS/CURRENT_SNAPSHOT/training_manifest.json \
  --verify-load \
  --mode trial \
  --cache-directory /ABS/MATERIAL_CACHE \
  --steps /ABS/E16_RUN/steps.jsonl \
  --output /ABS/NEW_AUDIT/coverage_with_exposure.json
```

- [ ] 步骤日志必须来自同一实验快照。若原日志没有逐行 `training_manifest_sha256`，同时附运行级 manifest 绑定凭据；工具只会报告明确存在的哈希绑定行数，不替旧日志补造绑定。
- [ ] 核对 forward 尝试曝光、`committed=true` 的优化器曝光、唯一训练记录数，各自单独报告。重复曝光不能算新增独立标签或新增几何。
- [ ] 检查每项损失实际记录的 mask、有效成员数及 denominator；缺失字段保留未知，不能按标签资格推算“肯定进入了损失”。
- [ ] 不把开发评估次数算入训练曝光。工具遇到开发记录进入优化器日志会拒绝；开发评估曝光目前不由此工具推断，须用独立评估日志核对。
- [ ] 记录没有步骤日志、没有显式哈希绑定或没有损失分母等限制，不用模型 epoch 数替代实际曝光证据。

## 3. 定向补标优先级

补标依据是原生求解的具体失败模式和当前覆盖审计，不追求先把总量做大。开始前先从既有合格资产检索可复用数据，避免重复布尔和积分。

| 优先级 | 应补内容 | 使用目的与所需证据 |
|---|---|---|
| P0 | 训练来源的原生成功轨迹末段、终点及其有效邻域，优先中保留点 | 检验模型是否因十矩偏差误判最优；记录原生成功验收、位置、目标/约束配置、乘子和实际终点残差。末段或邻域点不自动标成已收敛。 |
| P0 | 不同训练来源、不同曲面上的高保留对照 | 定位高保留误差是否跨来源重复；包括完整保留附近的有效查询，不能仅从一个几何大量复制相似查询。 |
| P0 | 跨来源抛物类 R8 及真实原生 Newton 方向上的 Jv | 在合法固定秩参数空间验证曲面导数与优化方向；保存 chart、方向、参数尺度和邻域资格。 |
| P1 | 有有效十维邻域证据的 Full-J | 仅当实际构面与几何事件允许时生成；没有合法邻域时记录拒绝原因，不把 R8 补零或外推成 Full-J。 |
| 配套对照 | 低保留、近零、一般迭代、初值及切割/拓扑拒绝事件 | 保留风险覆盖，帮助区分局部终点误差与一般切割误差；不以它们替代本次已定位的中保留终点和高保留问题。 |

- [ ] 对每条新增记录给出 `source_group`、`geometry_id`、`query_id`、`trajectory_id`、`task_id`、曲面家族、保留比例和轨迹阶段。阶段可采用 `initial/general_iteration/near_convergence/endpoint/endpoint_neighborhood/rejected_event`，但须附判定依据。
- [ ] `endpoint` 必须由部署原生求解器按冻结条件验收通过，并保存原始回执；不能根据网络 residual 或迭代停止状态命名为成功终点。
- [ ] 同一轨迹近邻属于相关样本。除查询数外，单列独立来源、几何、任务和轨迹数；来源宏平均不按查询数重复加权。
- [ ] 检查 `m、z、zz` 的终点敏感性是否跨几何复现；不要据单一面条位置永久提高某个 Moment 的标签权重。
- [ ] 将面条作为诊断轨迹单独保存，不进入训练来源；封存的独立盲测几何保持封存。
- [ ] 将已合格的小批次立即交给 Work B，用于物理复验或后续训练候选；尚未合格的数据保留拒绝记录与待补证项，不混入 loss。

**Jv 的接口边界：** 当前训练 case 接口接收已合格的 `10×d` Jacobian（`d=3/8/10`），并可附该 chart 内的真实 `direction`。由它形成的 Jv 可用于方向诊断。若只有单个方向的差分证据而没有该矩阵的资格，单列为方向诊断资产；当前 loader 不会将独立 Jv 资产自动转换为合格 Jacobian 监督。

## 4. 每条新增数据必须携带的准入凭据

### 4.1 几何、切割和十矩身份

- [ ] 原网格、有限切割面、原始教师 targets、资格 JSON 和所有支持证据分别有路径及 SHA256；打包后能按哈希回溯。
- [ ] 固定部署教师版本、构面配置、端盖、操作侧（`intersection`/`difference`）、坐标变换、单位、密度与固定材料；同查询不得改用另一套反号重新构面语义。
- [ ] 原始十系数、切割面内容身份和原网格身份一致。十矩顺序保持 `1,x,y,z,xx,yy,zz,yz,xz,xy`；原始系数交叉项乘数保持 `1,1,1,1,1,1,1,2,2,2`。
- [ ] `native_moments` 明确包含固定材料，同时保存 `fixed_moments` 与 `native_full_editable_moments`；固定材料只计一次。
- [ ] 完整可编辑质量和 `editable_bbox_max_extent` 长度尺度与原网格一致，全部十矩采用现有物理尺度校验。
- [ ] 静态材料缓存只在网格、坐标/网格定义、密度和实现身份一致时复用；不把参数更新后的学习编码当作静态材料资产。

### 4.2 导数资格

- [ ] 对每条数据明确 `derivative_valid`、`derivative_dimension`、`chart_id`；不合格时记录具体原因。有限数值矩阵本身不是有效性证明。
- [ ] R3/R8 提供固定秩 chart、满秩 `chart_basis`、正的 `parameter_scales`，以及与该 chart 一致的切割面顶点运动。
- [ ] Full-J 使用原始十系数的单位 chart，提供 `ambient_event_valid=1`；R3/R8 提供 `restricted_event_valid=1`。资格凭据须有正的有效邻域半径及其定义，范围分别为 `ambient_raw_coefficients` 和 `fixed_rank_chart`。
- [ ] 在有效邻域内保存原生导数/允许方向及多步长有限差分对照；记录端盖、拓扑和事件变化、不可定义方向及拒绝原因。不同 chart 的导数不得直接混作相同矩阵。
- [ ] 实际 Newton 方向保存来源、方向坐标和对应任务状态；真实任务方向不得用任意随机向量替代后仍标为 `native_newton`。
- [ ] M 合格而 J 不合格的行仍可进入 Moment 监督；导数 loss 必须保持无效掩码，禁止补零后当真值。

### 4.3 终点与物理残差标签（需要时）

- [ ] 绑定同一查询的原生验收公式、源文件哈希、目标/约束配置、固定乘子和尺度、原生十矩参考值以及 case SHA256。
- [ ] 使用既有 `cpino_fixed_residual_target_v1` 契约；当前实现的精确 repository profile 与原生冻结线性化 profile 不可互换或冒充私有部署公式。
- [ ] 若采用原生冻结线性化，提供同一参考状态下的 Lagrangian Hessian、约束 Jacobian 和原始残差；它只代表参考态的一阶敏感性，不是完整部署验收。
- [ ] 非终点、末段和邻域点使用其真实非零参考残差，不把所有样本的残差目标统一设为零。
- [ ] 无法取得绑定目标时设 `valid=false` 并说明原因。保留已有合格 Moment/Jacobian 用途；物理 residual loss 的资格单独控制。

## 5. 固定开发、新开发与防泄漏

- [ ] 固定 222 条开发查询的身份、教师语义、来源角色和原协议保持冻结；补充数据不得静默替换其难例或删除失败查询。
- [ ] 新增开发诊断另建 manifest/版本与报告，并列说明来源、几何、轨迹和各指标的有效分母。它不能被合并进旧 222 条后继续称为同一配对指标。
- [ ] 训练与开发按来源组隔离，同时核对几何身份/哈希与轨迹身份；同一成功轨迹的终点和邻域必须留在同一侧。
- [ ] 若需要训练使用当前开发来源的数据，先重新设计独立拆分与正式协议，不能直接改角色继续沿用旧评估结论。
- [ ] 对面条、封存盲测和其他诊断来源给出角色清单；禁止进入训练优化器或混入固定开发选模。
- [ ] 新开发只补齐新证据。原固定开发缺某类合格标签时，旧指标继续标明缺失，不能把新增标签悄悄填进旧分母。
- [ ] 本轮不自动重设 5% 等历史训练晋级线；执行等价、物理通过率和端到端速度分别由 Work B 按对应冻结协议裁决。

## 6. 向 Work B 提交的最小可审查交付包

以下是内容要求，不要求用这些文件名覆盖现有文件。

- [ ] **当前覆盖回执**：manifest 路径、snapshot、SHA256、`coverage_metadata.json`、可用时的 `coverage_verified.json` 与 `coverage_with_exposure.json`，附命令、环境和原始日志。
- [ ] **E16 数据绑定**：该 checkpoint 实际训练/评估快照、数据顺序或步骤日志入口、已存在标签和资产位置，明确不能核验的项目。
- [ ] **新增数据版本**：新的 training manifest、case JSON、哈希寻址资产、外部 source authority 准入包与 teacher contract、逐行 qualification 和支持证据。
- [ ] **覆盖差异表**：旧快照与新快照按来源、曲面、保留比例、轨迹阶段及 M/R3/R8/Full-J/残差资格的增量；不要只报总行数。
- [ ] **终点证据**：原生成功轨迹、终点独立验收及同一状态的目标/约束/乘子、必要的真实 Newton 方向；失败或拒绝记录单列。
- [ ] **新开发说明**：与原固定 222 条的分离方式、来源/几何/轨迹隔离结果、有效分母及可支持的结论范围。
- [ ] **未完成项**：列出 `UNKNOWN`、缺失标签、不可定义导数、资产不可访问和未运行检查；不将“工具可运行”写成“物理验证已通过”。

### 本轮交付验收

- [ ] 当前快照已审计；所有写出的计数可以追溯到本轮 manifest，不沿用旧聊天计数。
- [ ] 没有检测到训练/开发来源、几何、轨迹泄漏；重复查询身份已解释或修复，并重新冻结版本。
- [ ] 可训练行通过现有 loader 检查；无法通过的行和目标有明确隔离/掩码与原因。
- [ ] 终点、R8、有效 Jv 和 Full-J 证据各自说明实际覆盖及限制，不互相冒充。
- [ ] 原固定 222 条保持冻结，新增开发证据单列，盲测保持封存。
- [ ] Work B 已拿到足以开始 E16 执行等价对照的旧资产；定向补标进度没有被设置为该对照的等待条件。
- [ ] 对外结论限定为“数据与资格交付通过”，不由覆盖 CLI 自动宣布模型达到物理精度、加速或长训晋级。

## 7. 完成回报模板

```text
当前 snapshot_id：
当前 training_manifest 路径 / SHA256：
与 E16 checkpoint 的绑定凭据：
审计模式：metadata_only / existing_training_loader
训练 / 开发：已知来源数、几何数、轨迹数及 UNKNOWN 行数：
合格 M / R3 / R8 / Full-J / residual 数量（分别列训练、开发）：
实际 forward / committed optimizer 曝光；目标 mask/denominator 是否完整：
中保留末段与终点、高保留跨来源、R8/Jv/Full-J 的新增证据：
原固定 222 条是否保持冻结；新开发包位置：
来源 / 几何 / 轨迹重叠与重复查询检查结果：
未通过、缺失或未运行项目：
已交 Work B 的资产路径与可复验命令：
```

接口依据：现有 [training/manifest.py](../training/manifest.py)、[export_cases.py](../p0_diagnostics/export_cases.py)、[baseline_cases.py](../p0_diagnostics/baseline_cases.py)、[residual_matching.py](../training/residual_matching.py)；本轮审计实现为 [coverage.py](coverage.py)。这些检查核对数据身份与现有资格，不代替部署原生几何和完整求解验收。
