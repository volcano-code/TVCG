# Compact PINO V2.2：局部积分规则与固定参考物理误差

本目录提供同一查询上的积分精度对照，并承接训练侧新增的固定参考物理误差匹配。
共享编码器、Token16 默认、native256 材料网格、有限 operand、十矩顺序及默认
`h = pitch` 保持原有定义。完整单元规则可以独立切换，查询缓存会区分规则，静态几何编码可继续复用。

发布分支：`research/compact-pino-v22-precision`。基准提交：
`6ffaa99a86d084e8efa2be383edacf4bca24aad8`。
运行 `git rev-parse HEAD`，并与 Chat 发布回执中的提交号核对。

本轮完成的是实现、软件一致性验证和可执行实验入口。CPU 上的完整 Token16 合成更新与恢复
已通过 15 项检查，实际执行了 3 次 optimizer 调用：连续两步，再从第一步断点重放第二步。
该检查使用解析合成 fixture 和默认 32³/64³ U-FNO，不是正式 256³ 真实几何训练。
GPU 5090、Noodles 私有原生公式及最终求解验收尚未运行；当前证据不能判定面条终点问题已经解决。
执行顺序见 [WORK_HANDOFF.md](WORK_HANDOFF.md)。

## 1. 两类独立的积分选项

### 完整材料单元：运行时选择

| `full_cell_rule` | 每完整单元节点数 | 适用用途与精度范围 |
|---|---:|---|
| `tetra4` | 4 | 默认兼容基线；保留原节点次序、坐标、权重，二次多项式精确。四点并非中心对称，乘上三次响应后可能产生高阶混叠。 |
| `gl2` | 8 | 正权重、中心对称的张量 Gauss 规则；各坐标最高三次精确。适合作为首先测量成本和收益的 full-cell 单因素候选。 |
| `gl3` | 27 | 正权重张量 Gauss 规则；各坐标最高五次精确。可作较高阶对照，实际代价由所选块和活跃节点决定。 |

完整单元仍以紧凑 full runs 表示。`BlockMeasure.build` 保存默认 tetra4 节点供原路径及编码器使用；
`blocks.chunks(mask, ..., full_cell_rule=...)` 仅为选中的查询块临时展开 gl2/gl3。
`PreparedMaterial.moments()` 和 `block_moments()` 的完整 run 解析归约保持不变。
参数布局没有新增学习参数，切换规则本身不需要重训编码器。

gl3 对无学习、单一平面最近特征、整个单元处于同一个非饱和 cubic 分支的五次被积函数具有
明确精度依据。跨 `d=±h`、最近特征切换、有限 operand 边/顶点/封盖、空间变化的学习系数时，
不能直接沿用该多项式精确结论。更多积分点收敛到当前平滑响应；它不会自动消除有限带宽偏差。
本轮没有缩带宽、硬指标替换、查询自适应细分或全局提高材料分辨率。

### 部分材料单元：材料预处理选择

| `partial_rule` | 构造方式 | 单因素含义 |
|---|---|---|
| `fit_compressed` | 原支持内四点拟合；失败后使用 `GL(3,2,2)` sweep 并压缩。 | 保留原默认行为。 |
| `sweep_compressed` | 所有 partial cells 使用原网格竖直分解的 sweep，然后保十矩压缩。 | 与 `sweep_raw` 共享同一原始几何分解，隔离压缩因素。 |
| `sweep_raw` | 使用同一 `GL(3,2,2)` sweep，保留原始正权重节点。 | 去掉降点，通常增加 partial 存储和查询节点；没有同时提高 sweep 阶数。 |

`prepare_material(..., partial_rule=...)` 与 `material_identity(..., partial_rule=...)` 支持上述选项。
真实冻结导出请求中的每行 `grid.partial_rule` 也可指定它；省略时为 `fit_compressed`。
这是导出/材料选项，`compare_rules` 和 `TrainConfig` 没有 `--partial-rule` 参数。
RLE 对照的 `source='rle'`、`partial_rule='none'` 表示中心占用 full-cell 材料，其几何表示误差应单独解释。

partial 四点拟合检验实际材料支持，sweep 从原始面计算材料区间；压缩只调整正权重并保留原节点位置。
新节点不能通过对旧四点镜像、抖动或仅从十矩反推细域获得。full-rule 对照时固定 partial 规则；
partial 压缩对照时固定 full 规则、原网格、网格坐标、带宽和查询。

## 2. 材料与缓存版本

材料 NPZ 使用 `compact-v21-cell-material-v2`，记录 `source`、`partial_rule`、默认 full 规则、
算法 revision 和十矩 audit。partial 规则进入材料 identity；运行时 full 规则不改变材料 identity。

新 block pack 在原 `nodes.npy` 和块矩旁增加：

- `material_full_runs.npy`
- `material_partial_cells.npy`
- `material_boundary_nodes.npy`

`metadata.json.material_provenance` 绑定这些数组的来源与摘要。打开时做一次校验，此后使用只读 mmap；
查询签名可以使用只读文件 stamp，避免每次扫描全部 partial 节点。规则、带宽、有限 operand 内容和
查询语义共同进入 query preparation key；同一材料的 geometry signature 不随 full 规则切换变化。

旧 block pack 仍可按默认 tetra4 读取。缺少 compact provenance 的旧 pack 会明确拒绝 gl2/gl3，
`build` 不会悄悄改写旧目录。请使用新的 `--cache-directory` 构建新 pack。
若冻结 case 指向旧版本材料 NPZ，单换 block 目录仍不足够：须从原始几何与原冻结教师资产重新导出
新的材料/case 清单，保留旧快照。导出器可复用满足当前 identity/revision 的材料；旧材料不满足时，
新导出请求应从原网格重建，不能用重新填写摘要掩盖版本差异。

## 3. 同一查询的无训练对照 CLI

以下命令在仓库根目录执行，输出 JSON 的父目录应先建立，文件必须不存在。

```bash
mkdir -p runs/v22_precision/frozen
python -m experiments.compact_pino_v21.precision.compare_rules \
  --manifest /abs/frozen/training_manifest.json \
  --mode trial \
  --roles train development \
  --rules tetra4 gl2 gl3 \
  --chunk-size 65536 \
  --repeats 3 --warmup 1 \
  --cache-directory /abs/new/precision_block_cache \
  --output runs/v22_precision/frozen/no_learning.json
```

上例只运行 `model=None` 的 CPU 积分，不更新网络。要在同一报告中增加固定 checkpoint 对照：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.precision.compare_rules \
  --manifest /abs/frozen/training_manifest_with_physics.json \
  --mode trial --roles train development \
  --rules tetra4 gl2 gl3 \
  --checkpoint /abs/frozen/common_checkpoint.pt \
  --checkpoint-sha256 EXACT_64_CHARACTER_LOWERCASE_SHA256 \
  --device cuda:0 \
  --chunk-size 65536 --repeats 3 --warmup 1 \
  --cache-directory /abs/new/precision_block_cache \
  --output runs/v22_precision/frozen/paired_checkpoint.json
```

替换路径和 SHA 后即可运行。支持 `cpu` 或 `cuda[:index]` 的 checkpoint 推理；无学习部分始终在 CPU。
没有 checkpoint 时不要设置非 CPU 的 `--device`。checkpoint 与其 SHA 必须成对提供。
没有单独的 `--modelconfig`：加载器读取当前 runner checkpoint 自带的 `TrainConfig`，核对 config SHA、
`compact_pino.training-state.v1` schema、有限 tensor、dtype 与严格 `state_dict`。私有/旧包装权重
不会被猜测转换；只有私有 checkpoint 时，先补交格式映射与来源证明，再生成明确绑定的兼容权重。

每条 case 的 `records` 保存无学习三规则，`learned_rules` 保存冻结网络三规则；同一 case 的所有学习规则
共享同一次几何编码。报告记录 checkpoint、输入及执行源码摘要、编码耗时、cache 状态、各次查询耗时、
band/active 节点、十矩分量、合格 J/Jv 与可用物理误差。参数更新数始终为 0。
缓存淘汰可能让不同 case 重新编码；这不改变同一 case 三规则共享编码的约定。

也可重复使用 `--case /abs/case.json` 输入单个冻结 case；真实 case 仍须有已导出的教师资格证据。
`--roles` 只适用于 manifest。物理目标由严格 manifest 的 inline/sidecar 记录进入；direct case CLI
不接受额外未绑定 profile。需要保持完整来源拆分和物理目标绑定时优先使用 manifest。

查询耗时、编码耗时、checkpoint 加载、材料准备和原生积分必须分别阅读。现有标签不能提供本轮原生
积分计时；冻结缓存收益不能代表不断变化的 Newton 轨迹，也不能直接形成端到端加速结论。

## 4. 固定参考物理目标与标签缺口

新增匹配项比较同一查询、同一组固定乘子下的残差向量差：

$$
\Delta r = r(\widehat M;p,\lambda,\mu)-r(M_*;p,\lambda,\mu),
\qquad \Delta c=c(\widehat M)-c(M_*).
$$

参考 $M_*$ 包含固定壳。参考残差可以非零；任意训练查询不被强迫成为驻点。
共同的 `mu*p` 在差中解析抵消，不重新拟合乘子。尺度是冻结的正数，不以可能为零的参考残差充当尺度。

当前支持的 profile：

| profile | 计算与证据范围 |
|---|---|
| `TVCG_REPO_DUAL_MONOMIAL_RAW_V1_CABI_V1` | 精确计算明确绑定的仓库 objective/constraints 在预测矩与参考矩间的固定乘子残差差；检查仓库公式文件 SHA。只有实际部署公式确实与该 profile 相同时才可用于对应部署。 |
| `NATIVE_FROZEN_MOMENT_LINEARIZATION_V1` | 使用原生同一参考状态导出的 $H_L,Dc$，计算 $\Delta r\approx H_L(M_*)(\widehat M-M_*)$、$\Delta c\approx Dc(M_*)(\widehat M-M_*)$。输出标记 `first_order_at_native_moments`，属于局部敏感度近似。 |

Noodles 私有 `solve.py` 公式未随本轮代码获得验证，不能用仓库 profile 代替其精确验收。
linearized profile 不重算私有公式，也不代表非线性 exact acceptance；$H_L$ 必须已包含同一组
固定 lambda 的约束二阶项，不能替换为仅 objective Hessian。这里的 $Dc\in\mathbb R^{n_c\times10}$
是约束对十矩的导数，不是十矩对切割参数的 $J\in\mathbb R^{10\times D}$。

有效 target 的完整字段及数值校验在 [residual_matching.py](../training/residual_matching.py)。关键约定为：

- `schema='cpino_fixed_residual_target_v1'`，`valid=true`，明确 `profile` 和 `teacher_role`。
- `formula_provenance.source_profile` 及非空源码 SHA256 映射。
- 十矩顺序为 `1,x,y,z,xx,yy,zz,yz,xz,xy`；raw 交叉矩本身不乘 2。
- `reference_moments` 与该 case 的原生含壳十矩逐值一致。
- `p_convention='unit_dual_monomial_v1'`；`p` 必须等于原始几何系数后三项乘 2 后的单位向量，程序不会静默归一化。
- 冻结的 `multipliers`、`multiplier_mu`、正的 `stationarity_scale[10]`、`constraint_scale[n_c]`。
- exact 仓库 profile 还需要 `objective={mode,weight,mass_scale}`；linearized profile 还需要
  `lagrangian_hessian[10,10]`、`constraint_jacobian[n_c,10]`、`reference_stationarity[10]`、`reference_constraints[n_c]`。
- 最终绑定精确 case SHA，并核对 record/query/geometry 身份。合成目标与真实原生目标不能互换。

标签缺失时明确标为 `valid:false` 并附原因，或不提供 target；不得补造零目标。
无标签仍可进行 M/J 积分对照与 T0/T2/D1，报告的物理指标保持不可用。
启用物理损失却没有正权重合格训练目标时，runner 在分配模型前拒绝运行。

外部已经提供真实 target 后，用下列入口生成新 manifest：

```bash
python -m experiments.compact_pino_v21.training.attach_residual_targets \
  --manifest /abs/frozen/training_manifest.json \
  --targets /abs/frozen/native_physics_targets.json \
  --mode trial \
  --cache-directory /abs/new/precision_block_cache \
  --output /abs/new/training_manifest_with_physics.json
```

请求文件格式是 `{"schema":"cpino_residual_attachment_v1","records":[{"record_id":"...","target":{...}}]}`。
attachment 工具不生成乘子、Hessian 或原生标签；它验证现有数据，补齐一致的 binding，保留原角色及
教师资格回执，并拒绝覆写冻结 manifest。原生 M/J 资格和物理 target 资格各自核验。

## 5. 损失、曝光与配置矩阵

总损失为 `L_M + w_J L_J + w_r L_r + w_c L_c`。四项各用自己的有效样本分母。
`residual_weight` 是样本权重，默认继承该行 `moment_weight`；residual 与 constraint 项根据各自
全局系数分别进入 loss。缺失物理标签的样本不进入物理分母。`steps.jsonl` 和 `exposure.jsonl`
分别记录四项有效权重/掩码、分母及实际 optimizer 结果。

最终 `result.json.physical_supervision` 绑定 last checkpoint；
`selected_candidate_physical_supervision` 绑定选中候选自身的 lineage。
`physical_objectives_exercised` 表示启用的目标确实进入了对应 checkpoint 的已提交更新。
清单登记数、forward 数、后续 checkpoint 的曝光不能替代选中 checkpoint 的实际物理曝光。

| 配置模板 | full 规则 | residual / constraint 系数 | 主要配对 |
|---|---|---|---|
| [T0_old_loss_tetra4.json](configs/T0_old_loss_tetra4.json) | tetra4 | 0 / 0 | 统一参考臂。 |
| [T1_physics_tetra4.json](configs/T1_physics_tetra4.json) | tetra4 | 0.1 / 0.1 示例 | 与 T0 比较预先冻结的物理匹配方案。 |
| [T2_old_loss_gl2.json](configs/T2_old_loss_gl2.json) | gl2 | 0 / 0 | 与 T0 仅比较 full-cell 规则。 |
| [T3_physics_gl2.json](configs/T3_physics_gl2.json) | gl2 | 0.1 / 0.1 示例 | 与 T2 比较同一物理方案；与 T1 比较规则。 |
| [D1_old_loss_gl3.json](configs/D1_old_loss_gl3.json) | gl3 | 0 / 0 | 较高阶诊断臂；是否训练由冻结对照结果决定。 |

**0.1 只是已经可执行的示例值。** 正式值仅使用训练来源测量损失与梯度尺度后校准、冻结；
开发数据用于既定评估与选模，不参与物理系数校准。
T1/T3 使用相同冻结值。这里将 residual+constraint 作为一个预声明物理方案；该矩阵不能单独归因两项
各自的收益。`old_loss` 仅指当前 runner 的原 M/J 目标，不表示完整复刻不可见的旧 Pilot。

原研究筛选的 5% 改善及 5% 保护门槛保持不变。新物理指标和曝光独立报告，不自动替换主指标、
开发集、候选选择协议或原生最终验收。Noodles 两个邻近返回点只作复核，不能用于全项目调参或放宽门槛。

## 6. 可复现的软件证据

```bash
mkdir -p runs/v22_precision/software
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.precision.analytic_cells \
  --rules tetra4 gl2 gl3 \
  --output runs/v22_precision/software/analytic_cells.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.training.precision_smoke \
  --output runs/v22_precision/software/actual_updates_resume
```

`precision_smoke` 的输出目录必须不存在；该 CLI 当前只有 `--output`，执行 CPU 完整编码器检查。
不要向它添加未实现的 `--device`。目标 GPU 的实际更新命令见交接文档。

发布证据入口：

最终受影响回归为 **369 passed / 7 skipped / 0 failures / 0 errors**。7 种仓库任务的独立原始 C++
公式 oracle 已实际运行并通过；另外 7 个旧 native adapter 测试因当前环境未提供
`CPINO_NATIVE_LIBRARY` 而跳过，两组测试不能混淆。冻结 checkpoint 对照实际运行 2 条合成 case ×
3 种规则，编码 1 次、跨 case 编码缓存命中 1 次、权重更新 0 次。

- [validation.json](validation.json)：最终验证汇总。
- [actual_updates_resume.json](evidence/actual_updates_resume.json)：完整 CPU Token16、15 项检查、3 次实际 optimizer 调用。
- [analytic_cells.json](evidence/analytic_cells.json)：有限 operand 的斜向单元解析积分对照。
- [frozen_checkpoint_comparison.json](evidence/frozen_checkpoint_comparison.json)：固定 checkpoint 的同查询对照。
- [software_tests.xml](evidence/software_tests.xml)、[diagnostic_tests.xml](evidence/diagnostic_tests.xml)：对应自动测试记录。

完整代码依据：[材料与规则](../../../research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/v21_material.py)、
[block 来源与惰性展开](../../../research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/v21_blocks.py)、
[冻结对照](compare_rules.py)、[checkpoint 加载](frozen_checkpoint.py)、[目标绑定](../training/attach_residual_targets.py)、
[训练入口](../training/runner.py)、[原筛选门槛](../p0_diagnostics/promotion_gate.py)。
