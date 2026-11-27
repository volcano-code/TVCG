# Work B：Compact PINO V2.3 执行优化交付与实机验证清单

日期：2026-11-27。范围：复用已训练 E16，验证执行等价、定位动态查询成本、测量完整合格解耗时。本轮不要求重新训练。

**当前交付是保持原物理响应的执行实现，尚未取得 RTX 5090 D 上的新查询或完整原生轨迹加速证据。** 先在同权重、同查询、同教师语义下验证，再比较完整求解。不能把新实现能运行、CPU 回归通过或某段内核变快，写成独立几何已通过物理验收。

## 1. 已交付什么，哪些结论仍需实测

| 项目 | 本轮实现 | Work B 应核验的边界 |
|---|---|---|
| `CompactPINOv23` / `RuntimeConfig` | 保持 Token16、共享多尺度编码器、注意力、局部响应及 E16 `state_dict` 布局；新配置只控制执行 | 严格加载同一 E16，不将默认配置当作 checkpoint 的响应语义证明 |
| 联合 FP64 收缩 | 同一个积分 basis 对保留率及其运动导数作联合收缩 | M 与 J 仍来自同一响应；允许浮点归约顺序导致舍入差异，不能要求不同 GEMM 路径逐位相等 |
| 推理投影缓存 | 有界缓存静态 token table 在局部响应第一层的投影；按 table 与权重版本失效 | 不缓存预测 M/J，不跨参数更新沿用旧投影，不保留训练图；不是新查询完整准备的缓存 |
| 切面执行上下文 | CPU 亲和感知、可复用线程池，算子持有并显式关闭；非零 query-cache 准备也借用该上下文 | 默认上限不等于实际线程数；用实际亲和和统计回执说明，不声称旧实测必然发生“8 线程抢 2 核” |
| `QueryTrace` / `metrics.py` | 记录 callback、互不嵌套的阶段、请求模式、重复查询、异常及完整配对求解成本 | 只是计时和差异工具，不判断物理合格；样本 P95/P99 不是置信区间 |
| 按需有限构面 | `with_jacobian=False` 可不向算子附带运动；已注册 R3/R8 value source 可省源运动 | 默认 full-rank 原生入口仍可能调用 source-motion；必须报告 `motion_fallback`，不能把丢弃运动说成源端没有计算运动 |
| 原生 Moment-only | Cherchi `evaluate_moments` 在真实原网格布尔/积分后跳过材料 Jacobian 路径 | 不跳过真实切割与十矩积分；实际能力以 backend capability 回执为准 |
| 严格验收复用 | `acceptance.verify` 缓存原 Work 公式的符号构造，逐次重算原网格十矩；支持真实 native Moment-only，否则明确 joint fallback | 不缓存某个 q 的物理验收结果；不会用预测十矩冒充独立验收 |
| 部署与回放入口 | `deployment.make_predictor` 向已注册原生求解器提供 callback；`replay_cli` 是查询回放入口 | 不更换优化器、约束、停止条件或接受准则；命令参数以本提交入口的 `--help` 和源文件为准 |

共享编码器、边界带稀疏积分、静态材料/编码缓存和解析局部斜率原先已经存在；本轮不能将这些既有功能重复列为新增收益。

本轮 CPU 证据包括串并行见证、精确歧义处理、M/J、训练参数梯度、缓存失效、资源关闭及 native/验收接口回归。部分 native 接口测试使用可控测试后端；它们不代替实际 Cherchi 二进制、E16 权重和面条轨迹实测。合并后统一回归为 179 passed，见本目录 [README](README.md) 与 [原始日志](evidence/software_validation.log)。

## 2. P0：冻结 E16 与对照身份，先开始无需重训的验证

- [ ] 保存本次 Git commit、实际源文件 SHA256、dirty diff、依赖锁/版本、原生扩展构建身份和 CUDA/驱动版本。不能仅依据一个 Python 文件相同就判定全部依赖与旧受测版本一致。
- [ ] 冻结 E16 checkpoint 路径与 SHA256、响应类型、原训练快照绑定和选模记录；新旧臂均 `strict=True` 加载同一份权重，记录参数数量与 state_dict keys。不得在验证过程中继续更新权重。
- [ ] 冻结材料包 manifest、原网格、256³ 定义、密度、固定壳、坐标/单位、切割构面配置、端盖、保留侧与原始十系数。manifest 哈希须来自已冻结登记，不是读取任意目录后临时算一个哈希就当作准入证明。
- [ ] 保留固定 222 条开发查询及原有效标签分母。新补开发诊断另建版本，不替换旧难例、不删失败、不把缺失的 R8/Full-J/低保留分母写成通过。
- [ ] 冻结面条成功轨迹、E16 提案轨迹与失败点、原生校正点、原生拒绝事件；明确初值、一般迭代、末段和终点角色。面条用于诊断，不能据其调参后再称为独立盲测。
- [ ] 记录 CPU 亲和集合、进程/BLAS/OMP/PyTorch 线程配置、GPU 型号与其他负载。性能测量使用空闲窗口；不同臂按配对顺序交替，记录顺序及预热策略。
- [ ] 新 Newton 查询的生产对照统一 `query_cache_bytes=0`。几何可只编码一次，静态材料可按身份复用；不同 q 的构面、距离、积分准备和运动仍应计时。重复固定 q 的缓存基准单列。
- [ ] 与 Work A 并行推进覆盖审计和定向补标；这些新数据不是本轮 E16 执行等价比较的等待条件。数据工作见 [WORK_A_HANDOFF.md](WORK_A_HANDOFF.md)。

### 公平对照矩阵

当前 `v22` 与 `v23` 同在本提交中，二者会共享部分底层调度、原生接口与通用验收改动。因此需要区分以下比较。

| 对照臂 | 应使用的实现 | 回答的问题 |
|---|---|---|
| 历史纯原生 | 能复刻历史受测身份的旧 checkout/环境 | 历史完整成本是否可复现；只作历史实现基线 |
| 通用优化后的纯原生 | 同一严格求解器，复用已验证的通用准备/验收优化 | 当前公平的纯原生比较对象；不能故意保留其重复符号构造或无用验收 J |
| 历史 E16 混合 | 原受测 E16 的完整旧实现、原生校正和验收 | 相对此前整条流程是否缩短；必须用隔离旧 checkout，不能把当前 `v22` 类名当成旧实现 |
| 当前 `v22` 执行臂 | 本提交 `CompactPINOv21`，同 E16，query-cache 关闭 | 包含共同底层收益的消融基线 |
| 当前 `v23` 执行臂 | 本提交 `CompactPINOv23`，同 E16，query-cache 关闭 | fused/投影缓存/私有执行管理等额外贡献，以及完整新流程效果 |

本提交 `make_predictor(arm="v22")` 是“当前 v22 执行臂”，不是历史旧二进制/旧线程池基线。分别报告共同优化收益与 V2.3 专属增益，不能相加重复归因。若旧环境暂不可复刻，明确标为缺失；可以继续当前两臂对照，但不得据此宣布相对整个历史版本的已测提升。

## 3. P0：同一新查询的执行等价与拒绝回归

### 3.1 首轮检查

- [ ] 在实际 E16 和正式材料上，按固定查询顺序比较旧/新十矩、J 的可用性、每个分量误差、合法 chart 内的 Jv、保留侧和拒绝原因。
- [ ] 低层算子 J 对 raw q 求导，部署 callback 含既定的单位坐标映射；比较同一接口的同一种坐标。不得重复乘归一化，也不得直接将 R3/R8 列与 Full-J 列混比。
- [ ] 对 R3/R8 用既有资格证明对应的固定秩 chart；没有 Full-J 有效邻域时继续记录“不可定义/证据不足”，不补零凑十列。
- [ ] 复算同一有限切面换保留侧的互补、非零固定壳只加一次、薄/断连/端盖邻域及已有精确距离歧义反例。禁止用独立反号构面代替“同面换侧”。
- [ ] 同时比较 `joint` 与 `M` 请求的十矩一致性；Moment-only 不得假称获得了有效 J，也不能被传给需要 J 的 Newton callback。
- [ ] 核对 inference/no-grad、训练态缓存绕过和原联合 M/J loss 的参数梯度。CPU 已有回归；GPU 使用同一数学案例作必要复核，确认没有由 fused 路径、dtype 或投影缓存引入额外偏差。
- [ ] 保存 raw 分量差、物理尺度归一化差和近零分量的绝对差；在查看候选性能结果前冻结与 FP64 归约误差相容的等价容差。发现差异超过数值误差先定位，不通过放松原生物理阈值解决。
- [ ] 复验加载权重、普通 optimizer 更新、table 修改、设备/dtype 迁移和 train/eval 切换后的缓存失效。测试更新使用单独副本，不覆盖 E16。
- [ ] 使用 `with model:` / `with predictor:` 或 `finally: close()` 释放执行上下文。非零 query-cache 的研究对照也应使用模型私有上下文，关闭时先清缓存再关池。

**本轮无需新增“Moment 至少改善 5%”门槛。** 该门槛属于训练候选的预注册筛选；相同权重的执行改写目标是数值/物理行为等价和效率改善。固定 222 条仍需复算以证明旧质量与失败没有被隐藏，不要求执行改写凭空提高预测精度。训练候选、执行实现和独立物理推广分别裁决。

### 3.2 已有 Python API：向冻结求解器提供 callback

下面使用当前实际接口；`backend` 必须是绑定该原网格和任务的已注册原生 evaluator。`registered_geometry`、`registered_manifest_sha256` 从冻结登记取得。此函数仅示范固定查询回放，不自动执行 Newton、原生校正或最终验收。

```python
import numpy as np
from experiments.compact_pino_v21.runtime_efficiency.deployment import make_predictor
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.compact_pino_v23 import RuntimeConfig

def replay_frozen_queries(backend, pack, checkpoint, queries, *,
                          registered_geometry, registered_manifest_sha256,
                          arm="v23", device="cuda:0", surface_workers=None):
    rows, packets, metadata = [], [], {}
    runtime = RuntimeConfig(surface_workers=surface_workers) if arm == "v23" else None
    with make_predictor(
        backend, pack, checkpoint, metadata,
        expected_geometry=registered_geometry,
        expected_pack_sha256=registered_manifest_sha256,
        device=device, arm=arm, runtime=runtime,
        trace_sink=rows.append, profile_sync=True,
    ) as predictor:
        for query in queries:
            q = np.asarray(query, dtype=np.float64)
            try:
                packet = predictor(q)  # joint；部署求解器的系数映射已由 callback 处理
            except Exception as error:
                packets.append({"defined": False, "error": str(error), "query": q.tolist()})
                continue
            packets.append({"defined": True, "query": q.tolist(), "packet": packet})
    return rows, packets, metadata
```

比较臂对同一 `queries` 分别调用。异常行继续回放只为收集固定序列证据，不代表求解器可以忽略该拒绝。`packets` 含 NumPy 数组，写 JSON 前按字段转换，不能直接当作 `QueryTrace` JSONL。实际求解时将 `predictor` 传入原有冻结求解器，不在此处重新设计线搜索/停止条件。

完整低层路径可直接使用 `CompactPINOv23` 的 `encode_geometry(..., inference=True)` 与 `evaluate_operand(..., with_jacobian=True/False, trace=trace)`。它接收有限切面的顶点、面片、端盖、保留侧及可用运动；不得给它输入布尔后的材料十矩来降低提案成本。

## 4. P0：5090 D 上拆分新查询成本

- [ ] 固定查询回放分别报告 `joint/new_query`、`joint/repeated_query`、`M/new_query`、`M/repeated_query`。重复 q 标签仅表示输入重复，不等于 query-cache 命中；记录真实 cache hits/misses/builds/bytes。
- [ ] 保存每条 `QueryTrace`，包括失败/拒绝。至少解释 `native_source`、`surface_prepare`、`block_classification`、`material_rule`、`positive_measure`、`distance_stencil`、`context_projection`、`response_moment_jacobian` 和 `host_export`；prepared-cache 路径单列其准备阶段。
- [ ] 当前阶段名有分辨率边界：`native_source` 包含其内部构面/运动/源检查；`distance_stencil` 含距离及见证准备；`response_moment_jacobian` 可能含张量转移和收缩。没有更细证据时不能将某一整段全称为“网络前向”。
- [ ] GPU 分段归因时用 `profile_sync=True` 或等效显式同步，保证计时覆盖已完成工作。`QueryTrace.stage()` 不支持嵌套：不要在已有内部 trace 外再包一个同 trace 的整体 stage。
- [ ] 另做无逐阶段插桩的完整求解墙钟测量；边界仍需覆盖 GPU 完成、返回 CPU 数组及所有失败。分段同步会影响调度，其总时长不能直接当作最优生产时长。
- [ ] 材料/模型/编码准备与逐查询成本分开；冷启动从读取原网格开始，不能把预先生成的 pack 当作零成本。`make_predictor` 打开既有 pack 的时间不是材料生成时间。
- [ ] 保存算子 `efficiency_statistics()`，包括 encoder 调用数、投影 hit/miss/bytes、surface pool 创建/关闭次数和实际 affinity。参数更新、换材料、换权重的行为单列。
- [ ] 以原始序列报告总和、median/P95/P99 和计数。几条查询的 P95 仅作描述；不从同一轨迹内相关查询构造“独立初值”置信区间。

已有汇总命令，以下绝对路径替换为本轮真实输出位置：

```bash
python -m experiments.compact_pino_v21.runtime_efficiency.metrics \
  --queries /ABS/RUN/queries.jsonl \
  --output /ABS/RUN/query_summary.json

python -m experiments.compact_pino_v21.runtime_efficiency.metrics \
  --queries /ABS/RUN/queries.jsonl \
  --solve-pairs /ABS/RUN/solve_pairs.json \
  --output /ABS/RUN/runtime_summary.json
```

`solve_pairs.json` 是 JSON 数组，每项至少形如：

```json
{
  "pair_id": "geometry-and-initial-condition-id",
  "reference": {"accepted": true, "total_s": 100.0},
  "candidate": {"accepted": false, "total_s": 120.0}
}
```

这些数值只是结构示例，不是新实测。`accepted` 必须来自独立原网格验收。工具会保留新增失败和所有尝试成本，提供共同成功对的描述性时间比；**不会自动给出 95% 置信区间或模型物理 GO**。

### CPU 微基准的真实结果

本提交已保存 [surface_cpu_benchmark.json](evidence/surface_cpu_benchmark.json)：合成有向盒体 12 个三角面、16,384 个点、实际两核亲和，每路径预热 3 次、测量 30 次，交替执行。此次结果为：

| 纯 CPU 内核调度路径 | Median | P95 |
|---|---:|---:|
| 每调用新建池 | 6.173 ms | 7.153 ms |
| 复用池 | 6.381 ms | 7.390 ms |

**此次没有测到提速。** 复用路径 33 次调用只创建 1 池，逐次输出与串行完全一致；共享 CPU 负载未隔离。这里既未运行原生构面，也未运行完整神经响应或 GPU。不能选择性引用此前未完整留存的更快一轮，不能从本测试外推面条合格解速度。

可复算命令；输出到新位置以保留交付证据：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.runtime_efficiency.surface_benchmark \
  --output /ABS/RUN/surface_cpu_benchmark.json
```

脚本保存版本、亲和、线程环境、源文件哈希、每轮数据和统计口径。它仅用于检查调度实现和合成成本，不是 5090 D 性能验收替代物。

## 5. P1：原生校正与严格验收同样优化、公平计费

- [ ] 将实际 Cherchi Moment-only 与 joint 在同一组合法 q 上比较原网格十矩、几何/拓扑身份、认证积分凭据与拒绝语义。确认 native capability 来自实际部署扩展，而非仅 Python 方法存在。
- [ ] `acceptance.verify` 缓存的是公式构造，不是切后网格、十矩或验收决定。每个最终 q 都必须执行独立原网格切割与认证积分。
- [ ] 校验原 Work 验收 profile、目标权重、混合系数 dual 约定、`CONVERGED` 状态、正 μ、方程残差 `1e-12` 和物理相对残差 `1e-6` 全部保持。该验收不读取网络切割 J，不能因切割 J 不再需要就省掉真实十矩。
- [ ] 记录 `verification_backend_mode`。支持时应为 `native_moment_only`；不支持时可明确 `joint_fallback` 完成同等质量验收，但不能声称已经省去材料 J。对已支持路径的真实错误不得默默切换映射重试。
- [ ] 符号 warmup、静态网格/context 准备必须计入冷启动一次；热启动说明究竟复用了哪些资源。纯原生和混合臂采用同样通用准备/验收优化。
- [ ] 保留独立验收与求解器提案的职责：可复用不可变原网格和已核对的静态上下文，但不能拿原生校正缓存的切后十矩或网络十矩替代验收调用的新积分。

已有 API：

```python
from experiments.compact_pino_v21.runtime_efficiency.acceptance import verify

# exact_verifier 与 request 绑定同一原网格、坐标、密度、固定壳及冻结任务。
# returned_q 与 solver_status 来自本次实际求解结果。
result = verify(exact_verifier, returned_q, request, solver_status,
                prefer_moment_only=True, require_moment_only=False)
```

部署能力验证明确要求不得 fallback 时，可另用 `require_moment_only=True`，无能力将报错。这个选项不应变成阻塞所有物理复验的新审批流程。`evaluate_moment_formula` 只供归因诊断；即使它对预测 M 返回 `accepted=True`，也不是独立原网格验收。

## 6. P1：完整混合求解与实用收益

### 6.1 重测历史 E16 基线，再测 V2.3

历史 Work 回执中，已知 Noodles v8 / seed 0 的 E16 混合流程冷启动约 115.04 s：提案 34.26 s、原生校正 38.42 s、独立验收 21.00 s、其余约 21.36 s；同轮纯原生冷启动约 266.86 s。纯提案残差约 `2.7166e-3` 未合格，校正后约 `1.28e-13` 合格。这些是旧回执的定位信息，不能当作本提交在新环境下的对照测量。原始速度审计包的 `cold.json`、提案/校正 JSONL 和 native 回执应随本次证据保留。

- [ ] 使用相同初值、原网格、目标/约束、线搜索、停止条件、预算和最终验收，复跑当前纯原生与候选混合路线。
- [ ] 同时保留提案是否合格、校正是否成功、是否回退三个结果；“候选很快返回”不等于取得合格解。
- [ ] 生产总时长从读入原网格开始，覆盖材料建立/索引、权重载入、编码、逐查询构面与运动、CPU/GPU 转移、M/J、拒绝尝试、校正、回退和最终验收。
- [ ] 对失败初值计入全部耗时；报告成功数/总数、纯原生成功而候选失败的配对、共同成功对耗时比，以及所有尝试总耗时/合格解数。不得删除失败后只报快速成功平均值。
- [ ] 每段必须互不重复计费。用 `metrics.reconcile_timing` 核对总额、阶段和及未归因部分；负余量表示重叠或边界错误，不能裁成零。
- [ ] 如果某个提案的额外原生验收用于决定是否校正、继续或回退，它属于生产成本；若只是全流程结束后独立追加的诊断，单列全 job 成本与生产成本。必须有实际调用顺序支持，不能为了更快的数字将生产决策排除。
- [ ] 冷启动与复用缓存分开实测。旧回执材料复用约 115.97 s 并未快于冷启动；本轮也不能假设复用必然提速。
- [ ] 最终原网格物理质量不降：沿用已注册目标差标准和约束，报告每个初值目标值与原生对照。取得合格解的完整成本是速度结论主体。

### 6.2 早交接给原生校正的研究分支

- [ ] 先在训练来源的冻结轨迹上校准提前交接策略，使用运行时可得信号和真实校正成本，记录策略版本及触发点；之后在独立开发来源验证。
- [ ] 先单独比较“当前提案停止点交接”和“预注册提前交接”，不要同时换数据、权重、积分精度或停止阈值。
- [ ] 交接可以改变提案预算，但原生校正和最终物理验收标准保持。策略若需要额外原生查询，必须计费；不能每步读取不计成本的 oracle 标签。
- [ ] 无法交接、交接后失败与回退均计入成本。未校准前不直接把当前单个面条种子的最佳交接点推广到所有几何。
- [ ] 最终独立原网格验收不可省略；一个近似驻点或优化器的内部成功标志不能替代它。

## 7. 根据本轮证据决定下一步，不自动启动长训

| 已观测到的结果 | 下一项工作 |
|---|---|
| 新旧 M/J 或拒绝行为出现超出舍入的差异 | 先修执行实现、坐标映射或缓存生命周期；暂停该候选的速度宣传 |
| 动态有限构面/距离/见证准备占主导 | 在同切面语义下做向量化、并行/设备实现或严格认证的活动区筛选；保留端盖、精确歧义和 J 连续性证据 |
| 局部神经响应/传输/收缩占主导 | 以 profile 选择下一项算子融合、布局或传输优化；不能只凭模型参数多就重做编码器 |
| 编码占比确实显著且跨新几何重复 | 再评估编码器精简/蒸馏，另立训练对照；现有单几何编码已一次复用，不能先假定它是主瓶颈 |
| 提案误差导致多次昂贵校正或失败 | 先用训练来源末段/终点 M/J 归因和 Work A 定向数据评估改进空间；再决定损失、边界响应或学生模型实验 |
| 纯实现优化结果等价且完整成本下降 | 保留 E16 权重继续配对物理试验，无需为执行版本重新套用训练改善 5% 的准入线 |
| 合成内核快但完整合格解不快 | 保留各段证据，按总成本定位其他瓶颈；不能宣布神经算子部署已加速 |

更快系数条件学生尚属后续研究，不在本次无重训交付里。若启动，应保持有限有向带端盖切割的真实条件和一个 Moment 响应导出的 J；不能直接改成无限二次曲面而称语义等价，也不能只用 E16 预测当作物理真值。

## 8. 推广、交付包与最终回报

- [ ] 已知 seed 0 只支持该输入上的局部结论。论文 Noodles 表中的多初值统计、几何规模和计时边界不同，不从一次本地配对推断论文级加速或跨几何泛化。
- [ ] 在按既有协议封存的独立几何与初值上运行配对验证；若沿用注册的 seed 0–49，则如实报告实际完成数量和未完成项。不能看结果后删种子、换停止条件或将诊断几何改称盲测。
- [ ] 按原协议报告成功率、无新增失败、最终合格解目标差和完整配对时间比。若使用 95% 区间，加工方法、统计单位和配对范围必须预先固定；不能把一次轨迹内的查询当作独立求解样本制造区间。
- [ ] 先完成已获授权且可运行的本地/设备验证，缺失独立证据明确列出；不要把所有研究扩展做完当作提交这一轮可审查执行版本的前置条件。

最小交付包：

1. **身份与配置**：代码/构建/环境哈希、E16 权重哈希、材料/原网格/任务/查询身份、设备线程与缓存策略、各对照臂完整定义。
2. **数值等价**：固定 222 条和轨迹逐查询 M/J 差异、chart/资格/拒绝、同面两侧与固定材料检查、GPU 必要梯度复核。
3. **执行证据**：原始 `queries.jsonl`、汇总、分阶段同步 profile、无插桩墙钟、缓存/执行器统计及峰值内存范围；性能变化和未变化都保留。
4. **物理求解**：每步 q、M/J 或拒绝、方向/线搜索、提案交接、校正与回退记录；逐初值原网格独立验收与完整计时。
5. **决策报告**：历史完整基线、共同优化贡献、V2.3 专属贡献分别回答；明确已知种子范围、独立几何证据和剩余不确定项。

回报模板：

```text
实现版本 / dirty diff / 原生构建身份：
E16 checkpoint SHA256 / 数据和材料登记：
对照臂及历史旧实现能否复刻：
设备 / CPU affinity / 线程 / GPU 负载 / 预热方式：
固定222与轨迹：数值差异 / J可用性 / 拒绝差异 / 失败：
query-cache预算 / 投影与执行器实际统计：
new-query joint与M：样本数 / 总耗时 / median / P95 / 失败成本：
主要阶段及未归因时间；生产计时是否另测：
纯原生与候选：成功数/总数、完整成本、合格解目标和残差：
独立验收后端：native_moment_only 或 joint_fallback；原始凭据：
新增失败与回退；所有尝试总耗时/合格解数：
独立几何与预注册初值覆盖；区间方法或证据不足：
结论：执行等价是否通过 / 实测速度是否改善 / 物理推广证据是否齐备：
下一项工作与直接支撑它的profile或误差证据：
```

代码依据：[deployment.py](deployment.py)、[metrics.py](metrics.py)、[acceptance.py](acceptance.py)、[surface_benchmark.py](surface_benchmark.py)。具体接口以本提交实际源码为准；不通过更名、放宽验收或忽略失败取得速度结论。
