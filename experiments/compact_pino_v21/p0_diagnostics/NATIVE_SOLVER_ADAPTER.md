# 原生数值核接入与证据边界

本模块直接执行仓库 `src/spin/QuadricHollowingSolver.cpp` 的数值实现。它不是私有
17:56 / Noodles `solve.py` 的替身，也没有接通正式部署的 Cherchi 教师。当前 Pilot
仍为 **NO-GO**；这些接口和测试不改变晋级线，不授予新导数资格，不启动训练。

## 1. 已实现的入口

| 入口 | 实际行为 | 范围限制 |
|---|---|---|
| `NativeKernel.accept(c, M, J=None)` | 原 `evaluateState` 重算目标、约束、乘子；原 `converged` 判定 | 完全不读取 J；M 应为已经包含固定材料的总十矩 |
| `NativeKernel.step(...)` | 原 `solveSpinMomentSystem`，一次 Newton 迭代、原回溯、逐试探重新求值 | 必须有合格 Full-J 和真实试探求值器；数值核本身不证明几何 |
| `native_solver_adapter:factory` | SHA 绑定资产、版本、外部状态，逐 arm 新建适配器 | 先由 `endpoint_attribution` 复现该 profile 自己的全原生参考 |
| `repository_fullrank_geometry_factory` | 现有 canonical fullrank CGAL binding，携带同调用的材料拓扑证据 | 独立的仓库 CGAL profile；禁止承载 Cherchi 标签 |

本次实际执行的正质量制造输入，覆盖 STATIC_MAXIMUM_MASS、SPINNING_TOP、
SPINNING_TOP_LOWER_CENTER。后两项返回有限目标和残差；相同 M 换任意 J 的结果不变，
改变轴向一阶矩会使原生乘子重新计算。制造映射上的一次原生 Newton 回溯接受了一步，
原始方程残差从约 `0.3168343` 降至 `0.01044457`；这不是合格陀螺或实际几何收敛证据。

R3/R8 一步接口明确拒绝。现有 supported-space 示例只证明特定点的线性化接入；
没有当前受支持子空间和移动 chart 的完整凭据时，不将 R3/R8 零填成 Full-J。
Full-J 本身可以秩亏；真零 J 保持“有资格但零敏感”的含义，随后由原优化器决定能否走步。

## 2. 两层 profile 必须同时匹配

数值核 profile 为：

```text
TVCG_REPO_DUAL_MONOMIAL_RAW_V1_CABI_V1
```

非 synthetic 的几何/教师 map profile 必须另行声明为：

```text
TVCG_REPO_CGAL_CANONICAL_FULLRANK_EXACT_BINARY64_V1
```

这一几何 profile 还要求完整、已版本化的
`libigl::copyleft::cgal::mesh_boolean;libigl_source=...` backend identity。
元数据、固定构面配置、实际载入扩展和每一次 native 返回必须一致；`unversioned`、
Cherchi、缺少 map profile 或切换 backend 都拒绝。中心 M/J 相等只能补充检查单点重放，
不能替代整条几何映射的身份约束。

每次真实 CGAL 调用返回的 `scientific_hash` 作为该查询的 `evaluation_map_identity`；
另记录 `source_scientific_hash`、实际 backend、geometry hash 和同调用的
`provenance_topology_hash`。这些字段进入 C ABI 原生事件和 Python 试探回执。
被人为交叉替换的中心 M/J 明确标记 `ATTRIBUTION_FIXED_CENTER_MJ_V1`，不伪称它是一个
重新执行布尔运算后得到的物理响应。

数值核用 `fromNormalized` 保留已保存的单位原系数，适配器拒绝非单位输入，不擅自
正缩放、更换保留侧或重构反号切割面。具体 CGAL 接口自身使用其已冻结的
`fromUnnormalized` 构造约定；其标签和复现参考必须来自这个独立 profile，不能从
另一构面实现改一个版本字符串迁移过来。

## 3. 构建和测试

构建器不下载安装依赖，不修改仓库原生源码。它通过隔离 C ABI 编译原数值实现，
用符号隐藏和 section GC 排除不可达的几何/制造导出入口。仍需要真实 Eigen、CGAL、
Boost、MPFR/GMP 头，因为原头文件公开了相关类型。

本次在仓库根执行的命令如下，路径可按 Work 环境替换：

```bash
/workspace/scratch/fcf5369c0b51/pino-venv/bin/python -m experiments.compact_pino_v21.p0_diagnostics.build_native_solver_bridge \
  --eigen /workspace/scratch/fcf5369c0b51/native-deps/eigen \
  --include /workspace/scratch/fcf5369c0b51/native-deps/debian/usr/include \
  --output /workspace/scratch/fcf5369c0b51/native-build/libcpino_native.so \
  --link-library gmp

CPINO_NATIVE_LIBRARY=/workspace/scratch/fcf5369c0b51/native-build/libcpino_native.so \
  /workspace/scratch/fcf5369c0b51/pino-venv/bin/python -m pytest -q \
  experiments/compact_pino_v21/p0_diagnostics/test_native_solver_adapter.py

/workspace/scratch/fcf5369c0b51/pino-venv/bin/python -m \
  experiments.compact_pino_v21.p0_diagnostics.test_native_solver_adapter \
  --smoke-library /workspace/scratch/fcf5369c0b51/native-build/libcpino_native.so \
  --smoke-output /workspace/scratch/fcf5369c0b51/native-build/native_kernel_smoke_final.json
```

未设置 `CPINO_NATIVE_LIBRARY` 时，依赖真实库的测试显式 SKIP，不能报告原生通过。
一旦显式指定库路径，缺文件、源文件改变或构建收据不符都是失败。测试不自动编译或下载。

构建输出旁的 `.so.build.json` 为 `cpino_native_numeric_build_v1`，记录 binary SHA、
实际编译命令、编译器版本，以及依赖到的仓库源码、头文件和第三方头文件 SHA。
载入时复核 binary 和仓库源码；更新相关原生代码后必须重编，不能继续用旧库。

数值 smoke 回执为 `cpino_native_kernel_execution_receipt_v1`。除制造映射，它只读两条
仓库档案中的单位查询和十矩做公式接口演算；没有读到原网格或重新生成历史标签，
不宣称复现历史验收，不升级档案中未通过的事件/导数资格。

## 4. endpoint case 的新增状态契约

沿用 `endpoint_attribution.py` 的完整 meta / NPZ / artifact / replay 契约。新增：

| 位置 | 必须字段 |
|---|---|
| `meta` | `native_solver_profile`、`teacher_map_profile`、`native_geometry_backend_identity` |
| `adapter_contract` | `acceptance_reads_jacobian=false`、`multiplier_policy=native_recompute`、实际 solver_state SHA |
| `artifacts.solver` | 本次 C ABI `.so` 路径；`solver_sha256` 为其实际 SHA |
| `solver_state` | 下述 JSON；文件 SHA 被 meta 绑定，不能复用可变的全局状态 |

`solver_state` 的字段如下；数值和版本由实际登记任务填写，不从本示例推测：

```text
schema: cpino_repo_native_state_v1
profile: TVCG_REPO_DUAL_MONOMIAL_RAW_V1_CABI_V1
equation_profile: DUAL_MONOMIAL_RAW_V1
coefficient_convention: dM_d_unnormalized_geometry_at_unit_v1
geometry_map_profile: TVCG_REPO_CGAL_CANONICAL_FULLRANK_EXACT_BINARY64_V1
boolean_backend_identity: 实际返回的版本化 libigl CGAL identity
external_state_kind: repository_stateless_spherical_newton_v1
multiplier_policy: native_recompute
thresholds: 与 meta 完全相同的 stationarity / balance
objective: {mode, weight, mass_scale}
solver_settings:
  maximum_iterations: 1
  maximum_restarts: 1
  maximum_function_evaluations_per_restart: 明确预算，2..60，包括中心
  line_search_reduction: 0.5
  maximum_geodesic_step: 0.5
bindings: 与 meta 一致的 geometry_sha256 / operand_sha256 / teacher_version / coordinate_frame / units
center_geometry_hash: 当前真实材料几何 hash
center_topology_hash: 同调用 provenance topology hash
center_evaluation_map_identity: 当前 native scientific_hash
kernel_build_receipt: {path, sha256}
trial_evaluator:
  factory: experiments.compact_pino_v21.p0_diagnostics.native_solver_adapter:repository_fullrank_geometry_factory
  source_sha256: 上述实际入口源文件 SHA
  runtime_artifact: {path, sha256}，必须是实际载入的几何扩展
```

`objective.mode` 支持 `STATIC_MAXIMUM_MASS`、`STATIC_LOWER_CENTER`、`STATIC_PARETO`、
`FIDGET_SPINNER`、`YOYO`、`SPINNING_TOP`、`SPINNING_TOP_LOWER_CENTER`。
pure-height spinning-top 的额外 `weight` 必须为零，并保留原来的四项约束。
目标配置还会与具体几何请求逐项核对。禁止向状态注入冻结的 `multipliers` 或 `multiplier_mu`。

没有合格导数时仍可以执行 M-only accept；加入 `--one-step` 才要求完整合法 Full-J、
原系数 identity chart、当次 native 中心复算以及当前试探求值器。
当前 native replay 必须先复现，任何 mismatch 都在混合 M/J 实验前停止。

## 5. 真实 CGAL 几何接口所需资产

- `geometry` 为 NPZ，含 `outer_vertices`、`outer_faces`、`design_vertices`、`design_faces`。
- `operand` 为固定 JSON recipe，schema 为 `cpino_repository_fullrank_geometry_input_v1`，
  包含 `geometry_map_profile`、`boolean_backend_identity`、`request`、`evaluator_settings`。
- `request` 是现有 `SpinItFasterRequest` 的非网格参数；包括固定材料十矩、目标坐标、
  单位、密度、任务模式、目标权重和显式 `SpinSolverConfig`。不自动重心平移或改单位。
- `evaluator_settings` 是完整冻结的 `SpinEvaluatorConfig`；要求
  `DISCRETE_FULL_RANK_QUADRIC`、精确源边界和精确材料边界策略，随后请求同调用 material proof。
- 固定材料只由原几何求值器加一次；数值核接收已经包含固定材料的 `M_total`，不再次加壳。

此 recipe 格式与学习侧 `FiniteOperand` NPZ 不可互换。前者描述该独立 CGAL map 的
构面配置，后者保存正式查询已经生成的有限面。两者不能靠重新命名文件互相替代。

运行四臂诊断仍使用已有 CLI：

```bash
python -m experiments.compact_pino_v21.p0_diagnostics.endpoint_attribution \
  --case PATH/case.npz --meta PATH/meta.json --asset-root PATH \
  --adapter experiments.compact_pino_v21.p0_diagnostics.native_solver_adapter:factory \
  --output NEW_PATH/attribution.json --one-step
```

省略 `--one-step` 进行终点 M-only 归因。结果中的 `trial_evaluations`、原生事件、
`native_center_revalidation_calls` 和 `total_native_geometry_calls` 分开列出。方向来自原生
事件保存的完整 bordered system，用同一个原生线性求解器恢复，再映射回单位原系数切空间；
它不是 `J^{-1}`。一步被接受不代表最终最优性通过。

## 6. 尚缺的正式物理证据

本环境没有私有 17:56 的 solver、冻结 checkpoint/完整输入资产，也没有已编译的完整
CGAL 几何扩展。因此当前执行证据覆盖数值核和制造映射，真实 fullrank 几何工厂仍待
在其明确独立的 CGAL profile 下实跑。正式 Cherchi 教师需要单独的真实接口和版本绑定；
不得借用本工厂绕过它。当前 R3/R8/Full-J 覆盖缺口、面条求解失败和长训 NO-GO 均未据此解除。
