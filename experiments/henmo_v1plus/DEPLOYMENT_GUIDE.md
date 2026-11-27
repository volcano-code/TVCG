# HENMO V1+ 部署与原生求解桥接指南

本文面向 Work B。代码入口为本目录的 `deployment.py`；它复用已有 `runtime_efficiency.deployment.RuntimePredictor` 和 `runtime_efficiency.acceptance.verify`，不另写一套 Newton 方程或放宽最终验收。

**当前完成的是注册、冻结权重回调、有限切割面求值、拒绝传播、原生校正调度与计时接口。实际原生后端和冻结求解器必须由 Work B 接入；真实几何上的通过率、校正次数及端到端速度仍须实测。CPU 回放测试不能代替这项验收。**

## 1. 三种结果必须分别记录

| 结果 | 含义 | 能否算作合格解 |
| --- | --- | --- |
| HENMO 返回有限的十矩和 Jacobian | 同一神经响应的数值预测；可能还存在细化或导数资格警告 | 否 |
| 原生优化器使用 HENMO 回调返回 `CONVERGED` | 原生求解器在预测物理量下收敛 | 否，必须重新验收 |
| `solve_hybrid(...)["accepted"] is True`，且证据来自真实原网格后端 | 重新在原网格上构造切后材料、积分，并使用冻结的原生公式验收通过 | 是；另查任务要求的网格、制造与动力学检查 |

`evaluate_moment_formula` 可以做终点 Moment 归因，但它只运行给定十矩上的公式，不能提供独立原网格验收。不得把神经预测写入该函数后产生的“通过”当作物理解。

当前验收沿用 `WORK_SPINNING_TOP_DUAL_MONOMIAL_RAW_V1`：任务为 `SPINNING_TOP`，目标为 `INERTIA_AND_LOWERING`，原生系数约定为 `DUAL_MONOMIAL_RAW_V1`，方程容差为 `1e-12`、物理相对容差为 `1e-6`。**当前目标不是将保留质量设为唯一目标。** 如果需要换目标、约束或验收公式，应另行冻结任务与验收协议，不得复用本配置的通过结论。

## 2. Work A / Work B 提供什么

| 输入 | 要求 | 运行时检查 |
| --- | --- | --- |
| 原网格和原生请求 `backend.request` | 可编辑原网格、固定材料、坐标、单位、密度、轴、原点、目标与完整求解设置一致 | 网格哈希、任务哈希、固定十矩、密度 |
| 材料 `source` | 已准入的 `PreparedMaterial`、`BlockMeasure` 或 `SparseMaterialHierarchy`；保留原材料来源 | 材料身份、实际数组内容、原生分辨率与配置一致 |
| HENMO checkpoint | `HENMOTrainer` 生成的 `henmo.training-state.v1`；结构、训练绑定和所有参数完整 | checkpoint 哈希、训练绑定哈希、架构、参数名/形状/类型/有限性 |
| 部署注册回执 | 独立核对后冻结为 `ADMITTED`，并提供冻结注册摘要 | canonical JSON 摘要及所有绑定字段 |
| 原生后端 | Work 当前真实的有限切割面构造、原网格 Boolean、认证积分实现 | 既有源面接口与验收 profile；每次验收必须有新的原网格积分凭据 |
| 求解回调 | 同一个已登记原生求解器，接受外部物理量 evaluator | 回调源文件摘要、任务摘要；每次执行前重查 |

原生材料分辨率与神经上下文网格是不同的参数。例如输入材料可用 512³/1024³，粗节点与上下文网格使用较小尺度。这里不会把体素代理上的自洽误当成对连续原网格的验收。

注册的 `task_sha256` 包含完整 native settings，包括嵌套的线搜索设置、迭代预算等。设置必须可序列化为明确的 dataclass、字典，或具有明确字段的对象；不可序列化值会被拒绝，不会静默漏掉。回调源文件检查不等于所有原生依赖均已被锁定，Work 的环境回执还应记录部署后端、编译库与教师版本。

### 2.1 读取已经准入的材料

下列加载函数确实存在于仓库；`material_path` 和 `expected_identity` 必须来自 Work A 冻结回执。

```python
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_material import load_material

source = load_material(material_path, expected_identity=expected_identity)
```

这只读取静态材料，不能凭读取成功推断边界体素、十矩或薄壁的物理精度已经合格。新材料的一次预处理、原网格十矩核对和来源登记仍归 Work A；冷启动必须计入实际预处理成本。

### 2.2 生成待核对清单与冻结注册

```python
import json
from pathlib import Path
from experiments.henmo_v1plus.deployment import candidate_registration

# backend/source/config/checkpoint 均为本次实际部署对象。
draft = candidate_registration(
    backend, source, checkpoint, config,
    admission_id=source_admission_id,
)
Path("deployment-registration.draft.json").write_text(
    json.dumps(draft, indent=2, ensure_ascii=False, allow_nan=False),
    encoding="utf-8",
)
assert draft["status"] == "DRAFT_NOT_ADMITTED"
```

`candidate_registration` 生成的是当前对象清单，**不产生教师资格或模型晋级资格**。Work A/B 需将该清单与已经冻结的材料、任务、教师与 checkpoint 回执核对，形成正式 `ADMITTED` 文件。仅把草稿的 status 改掉、再对自己刚读入的文件取摘要，不构成独立核对。

`expected_registration_sha256` 应来自冻结注册表。它是 `registration_digest` 定义的 canonical JSON 摘要，不是直接对 JSON 文件字节运行 `sha256sum` 的结果。正式回执冻结后可用下式记录摘要；部署消费者应从冻结注册表读取预期值。

```python
from experiments.henmo_v1plus.deployment import registration_digest

digest_to_record_in_frozen_registry = registration_digest(admitted_registration)
```

更换 response profile、带宽、粗层级、细化设置、材料、checkpoint 或任务设置，都需要更新对应科学配置与回执。`compact_warp` 与 `bounded_residual` 具有不同语义，即使权重形状相同也不能相互静默替换。

factory 在 `load_state_dict` 前逐 tensor 检查 dtype，设备迁移仅执行 `model.to(device)`，不执行全模型的实数 dtype 强制转换。与默认构造器 dtype 不同的 float64 / complex128 checkpoint 当前会明确拒绝，不能静默降精度；需要支持这种精度时，应显式扩展并登记构造精度配置后重新验证。不要使用 `model.to(device, dtype=实数类型)` 处理含复数 FNO 权重的模型，这可能丢弃虚部。

## 3. 原生后端和求解回调如何接入

### 3.1 原生后端保留原实现

HENMO 查询沿用 `v21_native.native_operand(backend, q, ...)` 构造**同一个有限、有向、带端盖的原生切割面**及其运动，再由算子处理材料。普通神经查询不会请求原网格切后材料十矩。

后端应当是 Work 现有实现，而非一个仅填上标识字符串的空壳。当前接口需要原有 `native`、`target_design`、`triangulation`、`request`、`fixed`、`length`、`normalization_origin` 等字段，以及原生求值入口。最终验收还要求原网格 Boolean 身份、网格/拓扑哈希与当次认证积分证据。

当前冻结认证积分 profile 为 `CERTIFIED_MPFR256_GMP_FALLBACK_V1`，有理数后端为 `NATIVE_GMP_SURFACE_V1`。其中 `precision_bits >= 256` 指数值积分证据中的精度位数，**不是材料的 256³ 体素分辨率**。本目录没有新增或编造官方原生后端导入路径；Work B 必须使用其当前部署且经过核对的入口。

如果实际原生后端提供符合既有协议的 Moment-only 能力，最终验收可避免无用的切割 Jacobian。否则 `verify` 会明确记录 `joint_fallback` 并计入实际联合求值成本。HENMO 的 value-only 查询也不自动代表原生构面已省去顶点运动；现有默认构面入口可能仍支付这部分成本，须按实际路径报告。

### 3.2 求解回调契约

仓库真实接口为：

```python
from experiments.henmo_v1plus.deployment import SolverCandidate

# 接口示意：Work B 需在其现有原生 solver 上编写这层薄适配。
def native_solver_callback(evaluator, initial_query):
    # 调用已冻结的原生求解器，将所有 Moment / Jacobian 查询交给 evaluator。
    # 保留目标、约束、停止条件、线搜索、迭代预算、状态含义。
    # evaluator(q) 返回原生求解器已有的物理量 packet。
    # 若其入口要求不同的字段，做字段适配，不改变数学定义。
    raise NotImplementedError("此处接入 Work B 当前冻结的原生求解入口")
    # 返回 SolverCandidate(query=q_end, status=status, objective=objective)
```

上段明确是需要 Work 接入的伪代码，不是一个已经实现的原生求解器。回调必须位于真实 `.py` 文件内，才能进行源文件摘要绑定；交互式匿名函数不是合适的部署入口。

关键约定：

- 输入为十个有限非零原始系数。HENMO 内部输出原始系数 Jacobian，复用的 `RuntimePredictor` 按已有部署约定**只做一次**乘 `||q||` 的单位系数转换。回调不要重复转换。
- 十矩顺序固定为 `m, x, y, z, xx, yy, zz, yz, xz, xy`。固定材料只加一次；packet 同时提供总十矩和可编辑十矩。
- 回调必须使用传入 evaluator。神经提案时传入 HENMO，校正时传入同一实际原生后端；不得在闭包中偷偷固定另一套物理量来源。
- 返回 `SolverCandidate` 或包含 `query/status/objective` 的字典。额外的 `accepted=True` 不会被采信。
- `objective` 使用冻结原生目标的同一约定。当前顺序筛选按其数值升序排列，不将保留质量擅自当成当前任务目标。
- 事件、拒绝和数值异常应向上抛出或保留明确状态，不能用零 Jacobian 或上次结果补齐。

这里复用原生优化器，没有实现新的流形 Gauss–Newton 求解器，也没有用网络二阶导重写现有求解步骤。是否另做一阶拟牛顿算法实验，应作为单独的 solver 对照，不能混入“同原生求解器”的速度比较。

## 4. 可直接使用的调度函数

下面代码仅依赖仓库现有 API。唯一需要 Work 提供的两部分是 `prepare_native_case`（真实原网格后端和材料准备）与 `solver_callback`（上一节的原生入口适配）。

```python
from time import perf_counter

from experiments.henmo_v1plus.deployment import (
    RegisteredSolver, make_predictor, solve_hybrid,
)


def run_registered_case(
    *, prepare_native_case, checkpoint, registration,
    expected_registration_sha256, solver_callback, solver_source_path,
    expected_solver_sha256, initial_queries, regime="cold", device="cuda:0",
):
    # 必须早于原网格读入、材料构建或缓存打开、后端初始化。
    read_start = perf_counter()
    backend, source, preparation_receipt = prepare_native_case(regime)
    # preparation_receipt 由实际准备流程填写，不能给预先构建对象补填 True。
    # registration 为冻结回执字典；预期摘要来自独立冻结注册表。
    metadata = {}
    with make_predictor(
        backend, source, checkpoint, registration,
        expected_registration_sha256=expected_registration_sha256,
        device=device, metadata=metadata,
    ) as predictor:
        solver = RegisteredSolver(
            solver_callback,
            source_path=solver_source_path,
            expected_source_sha256=expected_solver_sha256,
            expected_task_sha256=registration["task_sha256"],
        )
        timing_context = {
            "regime": regime,
            "read_start_perf_counter": read_start,
            "includes_original_mesh_read": preparation_receipt.get(
                "includes_original_mesh_read", False),
            "includes_material_preparation_or_cache_open": preparation_receipt.get(
                "includes_material_preparation_or_cache_open", False),
            "includes_backend_initialization": preparation_receipt.get(
                "includes_backend_initialization", False),
            "includes_checkpoint_and_geometry_encoding": True,
        }
        result = solve_hybrid(
            predictor, solver, initial_queries,
            max_proposals=1,
            max_native_attempts=1,
            verify_direct=True,
            timing_context=timing_context,
        )
    # 不在这里伪造合格结论；直接保存 result 和 metadata。
    return result, metadata
```

`max_native_attempts=1` 表示最多一次完整的原生 solver 尝试，**不表示一次 Newton 更新或一次 Boolean**。原生回调内部仍保留登记的迭代预算。直接验收和校正后验收的成本分别记录。当前实现不保证原生校正只需 2–4 步；应从运行记录统计真实次数。

可选多起点通过增加 `initial_queries` 与 `max_proposals` 启用，候选完整返回后按同一目标排列，并使用预先固定的 `max_native_attempts`。**当前这些求解轨迹按顺序运行，并非 GPU batch multi-start。** 算子内部已有边界块批量求值，不等于完整的多起点迭代已经批量化。结果明确记录 `proposal_execution="sequential_registered_solver"` 与 `gpu_parallel_multistart=False`。

## 5. 拒绝、回退与缓存失效

| 条件 | 当前行为 | Work B 应如何记录 |
| --- | --- | --- |
| 注册、材料、固定壳、密度、架构或任务不一致 | factory 拒绝创建 predictor | 配置失败；先纠正身份绑定，不能进入吞吐统计 |
| checkpoint 或配置在 predictor 生命周期中改变 | 拒绝；要求重建 predictor / geometry cache | 禁止把旧编码用于新权重 |
| 原网格请求或材料存储被替换 | 拒绝 | 不复用跨任务静态缓存 |
| `NATIVE_REQUIRED_DERIVATIVE`、`NATIVE_REQUIRED_REFINEMENT` 等状态 | 即使 M/J 有限也拒绝作为神经回调结果 | 记录原因；由显式原生路线处理 |
| 普通 Moment-only 查询 | 可返回 `MOMENT_ONLY_REQUIRES_NATIVE_ACCEPTANCE` 的十矩 | 仅用于确实不需要 J 的请求，不能冒充 Newton 回调 |
| 神经提案异常且没有完整候选 | 从原初值执行预算内原生回退 | 神经失败时间照计，不能当作零成本 |
| 预测收敛、原生验收拒绝 | 对候选执行预算内原生校正，再重新验收 | 保留直接拒绝与校正两段记录 |
| 原生尝试或最终验收失败 | 返回 `accepted=False`，或保存明确异常 | 计为失败，不能用候选用时计算“合格解加速比” |

`make_predictor` 会固定模型参数、注册材料与任务，几何编码在此生命周期内只做一次。不要在同一个 predictor 中训练或原地改权重；新 checkpoint 必须重建。完整材料缓存可以跨查询复用，学习特征不能跨参数更新复用。

当前细化差异是经验误差诊断，不是严谨的误差上界。compact warp 在光滑支撑外保持常量，显式 Jacobian 包含 attention、cut descriptor 与响应链；它不承诺真实硬切割在拓扑事件处存在连续一阶导数。资格拒绝不可被“数值仍有限”覆盖。

## 6. 冷启动、缓存复用与端到端统计

`make_predictor` 的 `setup_elapsed_s` 包括读取/核对 checkpoint、材料内容核对、模型创建及几何编码；**它不包括已经在上游完成的原网格读入和材料预处理**。不能把这项时间和单次预测时间相加后称为全部端到端成本。

完整计时要求同一进程的 `perf_counter` 在读原网格前开始，并按前述示例提交覆盖回执。支持两个 regime：

- `cold`：读原网格、实际新材料预处理、后端准备、checkpoint/编码、所有查询、失败、校正与最终验收都在区间内。
- `material_cache_reuse`：读原网格、打开并核对已有静态材料缓存、后端准备、checkpoint/编码、求解与验收都在区间内；另报该缓存的首次构建成本及复用次数。

第二种模式不等于“提前完成全部编码，然后从第一次神经查询才计时”。若要研究跨多次求解复用编码，应另外报告共享预处理成本及摊销方法，不挪用本模式的计时标签。

缺失完整回执时，结果保留 `workflow_seconds` 和 `factory_setup_seconds`，同时明确：

```json
{
  "timing_scope": "PARTIAL_WORKFLOW_ONLY",
  "end_to_end_seconds": null,
  "eligible_for_end_to_end_comparison": false,
  "acceleration_proven": false
}
```

即使覆盖回执完整，单次运行也不会把 `acceleration_proven` 自动改为 true。仍需在冻结几何和配对初值上比较纯原生、强确定性基线和混合 HENMO 的合格率、目标质量与取得合格解总时间，并按预注册统计区间。CPU fixture 的认证标记只为测试验证调度，结果不能晋级为端到端加速证据。

若分项诊断启用 `profile_sync=True`，CUDA 会增加同步，适合定位瓶颈。正式吞吐另跑默认模式；回调返回 host 数组会完成其所需计算，但无同步的分项 host wall time 不是单独 GPU kernel 时间。

确定性强基线须使用同样的材料、同一有限切割面、细化预算及最终原网格验收。**冷启动对比不要给无学习积分器强行加载无用神经编码器的成本**；可直接使用 `henmo_hierarchy.evaluate_deterministic`，并另计它真正需要的材料/层级准备。

## 7. Work B 最少验证清单

1. **部署桥接**：先使用全原生 evaluator 和已登记回调，复现成功轨迹、终点与冻结验收；确认未改目标、固定壳、系数尺度与线搜索。
2. **真实 factory**：加载实际 HENMO checkpoint 与独立注册；在同一实际 operand 上核对 operator 原始 J 和 callback 单位系数 J 只相差规定的一次转换。
3. **值与导数资格**：检查 Moment-only、Full J、有效 R3/R8 数据诊断；部署全十系数入口不要把固定秩导数伪装成 Full J。事件或细化拒绝必须触发明确回退。
4. **原网格验收**：模型终点、原生校正终点分别重新计算原网格十矩；保留完整方程残差、物理约束、目标、`mu`、网格身份与当次积分凭据。
5. **成本归属**：保存每条起点的所有提案、失败、原生校正和验收时间；保留 `neural_query_calls`、`native_solver_evaluator_calls`、`native_verification_calls`。这些计数分别对应接口调用，不能直接等同内部 Boolean 调用数。
6. **GPU 实测**：在正式设备上检查峰值显存、有效吞吐、缓存增长、不同粗层级和支撑宽度的误差与成本。当前 CPU 梯度/重放测试不证明 GPU 吞吐或跨设备完全一致。
7. **推广证据**：固定开发集继续配对；新增诊断集单列；面条用于复验；独立几何封存到约定评估。按正式协议报告所有初值，不能只挑校正成功的最快案例。

仓库回归入口：

```bash
python -m pytest -q tests/research/test_henmo_response.py tests/research/test_henmo_deployment.py
```

响应测试覆盖正权重十矩、同面换侧互补、两个 response profile、D=3/8/10 显式切向导数与独立 AD/多步长有限差分、非零 attention 链、Jacobian loss 参数梯度、饱和边界和真实 checkpoint factory。部署测试覆盖注册错配、过期缓存、完整 settings、原生必要状态拒绝、同求解器校正、独立 verifier、计时范围与 Moment-only。它们验证工程合同，不能替代上面第 1–7 项的真实训练/物理证据。

## 8. 本阶段完成度与剩余责任

| 项目 | 状态 |
| --- | --- |
| 正权重固定支撑局部响应、十矩同源积分、显式完整 J 链 | 已实现并有 CPU float64 独立求导测试 |
| coarse-support compact warp、饱和外分支、同 operand 互补 | 已实现；真实切割事件与粗积分误差仍保留诊断/回退 |
| 注册材料/任务/结构/权重、缓存冻结与原生回调桥接 | 已实现；Work 填入真实登记与后端 |
| 同一原生求解器的神经提案与原生校正、最后重新验收 | 调度已实现；实际 solver 适配与真实原网格闭环待 Work B |
| 冷启动与材料缓存复用计时、失败与回退计费 | 接口已实现；需 Work 完整准备回执及原始测时 |
| GPU batch multi-start 或新流形 Gauss–Newton solver | 未实现；当前分别为串行可选多起点与复用既有原生求解器 |
| 512³/1024³ 真实复杂材料的精度、网络收益、校正次数、论文级加速 | 未由本地 CPU 工程测试证明；需训练与封存评估 |

本交付不承诺固定 2–4 步精修、毫秒级完整求解、10–50 倍加速或必然达到论文发表标准。可以开始真实注册和桥接验证；是否进入长训练、是否替代原生生产路径，以 Work 实验回执为准。
