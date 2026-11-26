# V2.2 精度改进：Work 执行交接

## 当前交接状态

分支为 `research/compact-pino-v22-precision`，基于
`6ffaa99a86d084e8efa2be383edacf4bca24aad8`。
运行 `git rev-parse HEAD`，并与 Chat 发布回执中的提交号核对。

可执行能力已经落地：tetra4/gl2/gl3 按需 full-cell 积分、三种 partial 预处理规则、显式材料来源、
同查询无学习与冻结 checkpoint 成对对照、严格 case 绑定的固定参考物理损失、独立物理曝光和恢复审计。
规则与 target 详情见 [README.md](README.md)。

CPU 完整 Token16 smoke 已 PASS：15 项检查、2 次连续更新加独立重放第 2 步，总共 3 次真实 optimizer
调用。该检查使用解析合成 fixture 和默认 32³/64³ U-FNO，不是正式 256³ 真实几何训练。
物理损失单独反传可到达共享编码器、attention 和局部响应。目标 GPU 5090、Noodles 私有原生验收
仍未运行。当前结论是代码与软件路径已具备后续实验条件，终点残差是否下降由 Work 的真实同点结果判定。

本交接承接已确认的实施安排。Work 可以立即执行环境核验、软件复现、新材料缓存准备及现有冻结查询
的只读对照；不需要等待 Chat 再修改这些入口。真实训练沿用 runner 已有预算与冻结协议文件校验，
已具备的授权直接使用，不新增聊天确认环节。物理标签缺失时先推进可完成的 M/J 工作，并补交第 4 节数据。

## 1. 获取独立工作树，记录精确版本

在现有 TVCG 仓库中执行；以下新工作树目录不得已存在：

```bash
git fetch origin refs/heads/research/compact-pino-v22-precision:refs/remotes/origin/research/compact-pino-v22-precision
git worktree add --detach /data/xvxinyao/TVCG_v22_precision refs/remotes/origin/research/compact-pino-v22-precision
cd /data/xvxinyao/TVCG_v22_precision
git rev-parse HEAD
```

核对 HEAD 与发布回执的最终 SHA，保留旧 Pilot 的代码、权重、快照和结果。
记录 Python、NumPy、PyTorch、CPU 线程数、CUDA、驱动、GPU 型号/compute capability 和可用内存。
继续 `amp=off`；当前 Fourier 复数参数不支持把本轮结果解释为 FP16 AMP 验证。

以下命令均从该工作树根目录运行，使用已经装有本项目依赖的 Python 环境。
所有 `/abs/...` 均须替换成 Work 的真实路径，所有输出使用新文件或新目录。

## 2. 先复现 CPU 软件检查，再在 5090 上检查实际更新

### 2.1 CPU 复现

```bash
mkdir -p runs/v22_precision/software
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.precision.analytic_cells \
  --output runs/v22_precision/software/analytic_cells.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.training.precision_smoke \
  --output runs/v22_precision/software/actual_updates_resume
```

第二个命令会生成合成训练清单、物理 target、checkpoint 和 `precision_smoke_report.json`。
应核对 `status=PASS`、全部 15 项 `checks=true`、`real_optimizer_calls_executed=3`、
`native_solver_acceptance=NOT_EVALUATED`，并保留完整报告，而非只记下通过文字。

要运行本轮限定软件回归，可执行：

```bash
PYTHONPATH="$PWD/research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction:$PWD" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_v21_cubature_rules.py \
  research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction/spin_adapter/tests/test_v22_precision_cache.py \
  experiments/compact_pino_v21/training/test_residual_matching.py \
  experiments/compact_pino_v21/training/test_precision_training.py \
  experiments/compact_pino_v21/precision/test_precision.py \
  experiments/compact_pino_v21/precision/test_frozen_checkpoint.py
```

发布记录共 **369 passed / 7 skipped**，无失败或错误。独立原始 C++ 公式 oracle 的 7 种仓库任务已实际
通过；复现时给 `CPINO_EIGEN_ROOT` 指定包含 `Eigen/Core` 的目录，并确保 `g++` 可用：

```bash
CPINO_EIGEN_ROOT=/abs/eigen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q experiments/compact_pino_v21/training/test_residual_matching.py \
  -k compiled_unmodified_native_values_and_loss_hessian_chain
```

当前跳过的另外 7 项属于旧 native solver adapter，原因是未提供已编译 bridge。Work 本机有对应
bridge 时可立即补跑，无需新增审批：

```bash
CPINO_NATIVE_LIBRARY=/abs/native/libcpino_bridge.so \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q experiments/compact_pino_v21/p0_diagnostics/test_native_solver_adapter.py
```

路径应替换为本机已按 `p0_diagnostics.build_native_solver_bridge` 构建的实际库。
adapter 使用制造的数值映射 fixture；补跑通过也不能替代 Noodles 历史查询或原生网格验收。
不设置 `CPINO_EIGEN_ROOT` 会使独立公式 oracle 在该复现环境跳过；这种结果必须明确记录，不能照搬发布记录。

### 2.2 5090 实际更新

`precision_smoke` 本身没有 `--device`。GPU 使用它生成的明确合成清单交给实际 runner：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.training.runner \
  --manifest runs/v22_precision/software/actual_updates_resume/assets/precision_manifest.json \
  --mode smoke --device cuda:0 \
  --config experiments/compact_pino_v21/precision/configs/T3_physics_gl2.json \
  --cache-directory runs/v22_precision/gpu_block_cache \
  --output runs/v22_precision/gpu_updates \
  --max-steps 2 --max-attempts 4 --checkpoint-every 1
```

这里使用模板中的 0.1 物理系数仅检查软件路径。应看到真实 `global_step=2`，M/J/residual/constraint
四项有效分母为正，编码器、attention、response 梯度有限且非零。
从该 GPU 运行 step 1 的 runner-issued `.resume.json` 启动新的 replay 目录，恢复到绝对 step 2：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.training.runner \
  --manifest runs/v22_precision/software/actual_updates_resume/assets/precision_manifest.json \
  --mode smoke --device cuda:0 \
  --config experiments/compact_pino_v21/precision/configs/T3_physics_gl2.json \
  --cache-directory runs/v22_precision/gpu_block_cache \
  --resume /abs/gpu_updates/STEP_1.resume.json \
  --output runs/v22_precision/gpu_replay \
  --max-steps 2 --max-attempts 4 --checkpoint-every 1
```

`STEP_1.resume.json` 应替换为实际第一步 receipt，不是直接给 `.pt`。核对第二批 record IDs、loss、
模型/Adam/scheduler/scaler/RNG/sampler/step 状态和实际曝光。CPU 的逐值恢复结果不能替代 5090 上的核验。

## 3. 准备当前版本材料与完整冻结查询

先确定每个真实 case 的 `source`、材料 identity、origin/pitch/density、fixed moments、有限 operand、
原生教师/源码版本、query SHA、derivative 资格及 train/development 角色。不能把 R1 的两个邻近终点
自动视为本次已绑定数据，也不能把它们用于选择全项目物理系数。

### 3.1 新缓存与重新导出

若已有当前版本材料 NPZ，只需给 `--cache-directory /abs/new/precision_block_cache`，建立带来源的新 pack。
旧 pack 可继续 tetra4，但 gl2/gl3 要求 full runs 和 partial 支持来源。
若 case 仍引用旧 schema/revision 的材料 NPZ，需要用原始冻结导出请求生成新快照：

```bash
python -m experiments.compact_pino_v21.p0_diagnostics.export_cases \
  --request /abs/frozen/precision_export_request.json \
  --output /abs/new/precision_export
```

真实导出不要加 `--synthetic-fixture`。新请求保留原 source authority、角色、原始网格、有限 operand、
教师目标与资格资产；每行 `grid` 可显式加入 `"partial_rule":"fit_compressed"`。
旧 material 资产不符合当前 identity 时，从原网格重建；不得在旧 NPZ 或冻结清单上覆盖内容。
导出器保持教师数值，并为新材料/case 生成新身份与回执。新 case SHA 出现后重新绑定物理 target，
不能照抄旧 case SHA；先核对原生 M、query、geometry 仍为同一参考状态。

### 3.2 本轮主对照固定的因素

第一轮固定 `partial_rule=fit_compressed`、native256、原 pitch、`h/pitch=1`、相同原生标签、相同
query/operand 和乘子。只改变 full-cell 规则。不要同时换材料源、细化全局网格、缩带宽或重选查询。

partial 的附加实验另开冻结快照：相同原网格分别选择 `sweep_compressed` 与 `sweep_raw`，固定 full 规则。
这两臂检验相同原始 sweep 的压缩影响；`fit_compressed` 与 `sweep_raw` 的差异同时包含四点拟合的影响，
不能全部称为“压缩误差”。增加 partial 节点需要原始几何，不能从旧缓存十矩猜测材料内部形状。

## 4. 需要补交的真实物理标签

原生 M/J 标签本身不足以生成新增物理目标。真实目标至少需要以下同点资料；已有可信归档可直接使用，
不需要为本轮重新发明一套教师。

| 数据 | 要求 |
|---|---|
| case/query/geometry | 原始材料与有限 operand 资产、原生教师版本、case SHA、query ID、geometry SHA、十矩与系数约定。 |
| 参考矩 | 同一查询的十个原生 raw 矩，包含 fixed shell，与导出 case 逐值相等。 |
| 乘子 | 该参考状态固定的全部 lambda 和 mu；保持原生非零残差，不按每个预测结果重新拟合。 |
| 对偶方向 | 原始系数后三项乘 2 后的单位 `p`；显式 `p_convention`。 |
| 公式来源 | 明确 profile、公式/求解器来源标识与源码 SHA；不能将不同私有公式重标为仓库 profile。 |
| 物理归一化 | 冻结正的 stationarity/constraint 尺度；训练侧校准并冻结，不从零参考残差生成。 |
| 私有原生线性化 | 原生同一状态的 `H_L[10,10]`、`Dc[n_c,10]`、参考 stationarity 与 constraints。 |

若部署确实使用精确仓库 profile，补齐 `objective.mode/weight/mass_scale` 及对应源码摘要后可用该精确
残差差。若 Noodles 私有公式尚未提供，先使用有原生来源的
`NATIVE_FROZEN_MOMENT_LINEARIZATION_V1`：它度量局部敏感度误差，不计算私有非线性 exact acceptance。
完整原生求解验收仍由 Work 原机器执行。来源/公式不全时返回 UNAVAILABLE，不补零、不替换教师。

将真实目标整理为 attachment 请求：

```json
{
  "schema": "cpino_residual_attachment_v1",
  "records": [
    {
      "record_id": "实际导出清单中的记录ID",
      "target": {
        "valid": false,
        "reason": "该记录尚未提供同点原生乘子与公式/线性化标签"
      }
    }
  ]
}
```

上例是诚实的缺标签格式。真实 `valid:true` target 的字段见 README 与 `residual_matching.py`，不能把
示例 reason 改成 true 后以猜测数值填充。整理好供应方的实际 target 后执行：

```bash
python -m experiments.compact_pino_v21.training.attach_residual_targets \
  --manifest /abs/new/precision_export/training_manifest.json \
  --targets /abs/frozen/native_physics_targets.json \
  --mode trial \
  --cache-directory /abs/new/precision_block_cache \
  --output /abs/new/training_manifest_with_physics.json
```

核对返回的 `manifest_sha256`、`attached_count`、`valid_train`、`valid_development`。
该回执表示绑定和数值约定已经验证，不能作为独立原生物理认证。

## 5. 第一阶段：冻结 checkpoint，同 query，无训练对照

先使用已登记训练几何的固定切割面，并保留完整开发集逐条身份。面条 A/B 作为后续固定复核点。
不要让每种规则自行运行求解器寻找不同的终点再比较；这一步的输入 query 与权重必须完全相同。

```bash
mkdir -p runs/v22_precision/frozen
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.precision.compare_rules \
  --manifest /abs/new/training_manifest_with_physics.json \
  --mode trial --roles train development \
  --rules tetra4 gl2 gl3 \
  --checkpoint /abs/frozen/common_checkpoint.pt \
  --checkpoint-sha256 EXACT_64_CHARACTER_LOWERCASE_SHA256 \
  --device cuda:0 \
  --chunk-size 65536 --warmup 1 --repeats 3 \
  --cache-directory /abs/new/precision_block_cache \
  --output runs/v22_precision/frozen/paired_rules.json
```

如果暂时没有兼容 checkpoint，先省略 checkpoint、SHA 和 `--device` 三项，立即完成无学习 CPU 对照。
如果暂时没有物理目标，使用尚未附 target 的有效 manifest；M/J 对照仍可执行。记录实际缺口，不把它
改写成零物理误差。私有 checkpoint 需要明确格式映射；现有 loader 仅接收 runner schema 和严格参数布局。

同一报告逐条核对：

1. `parameter_updates=0`，相同 query/operand、原生 M/J、fixed moments、checkpoint SHA 和 (h/pitch=1)。
2. `records` 的无学习结果与 `learned_rules` 的冻结网络结果成对齐全；学习三规则共享编码。
3. raw 十矩完整差向量，重点联合查看 m/z/zz；不得只按未加权十矩范数决定物理优劣。
4. J 只在原生资格和响应导数均有效时统计；缺失 J/Jv 明确保留 unavailable。
5. 若有物理 target，检查 profile 和 approximation。记录整个 residual/constraint 差向量及其范数，
   不把可相互抵消的 Hessian 列贡献范数当作误差占比。
6. 比较 band/active 节点、逐次查询耗时、编码耗时和缓存命中；独立保存材料准备和原生积分耗时。

`COMPARISON_COMPLETE_NOT_NATIVE_ACCEPTANCE` 只表示对照完成。报告中的
`original_five_percent_gate=UNCHANGED_NOT_EVALUATED`、`native_solver_acceptance=NOT_EVALUATED`
是预期状态。此报告不产生“模型已晋级”或“原生终点已通过”的结论。

## 6. 第二阶段：固定实验矩阵，单因素短训

先用第 5 节判断积分改动是否值得训练，再冻结正式物理系数和更新预算。模板位置为
`experiments/compact_pino_v21/precision/configs/`：

| 臂 | 规则 | 物理系数 | 因素解释 |
|---|---|---|---|
| T0_old_loss_tetra4 | tetra4 | 0 / 0 | 同一新 runner 的 M/J 基线。 |
| T1_physics_tetra4 | tetra4 | 校准后冻结；模板 0.1 / 0.1 | 与 T0 比较固定物理匹配方案。 |
| T2_old_loss_gl2 | gl2 | 0 / 0 | 与 T0 仅比较积分规则。 |
| T3_physics_gl2 | gl2 | 与 T1 相同 | 与 T2 比较物理方案，与 T1 比较规则。 |
| D1_old_loss_gl3 | gl3 | 0 / 0 | 高阶对照，是否进入短训由无训练结果决定。 |

T1/T3 的 0.1 是可执行示例，正式值仅使用训练来源校准冻结；开发数据用于既定评估与选模。
残差与约束分别记录损失和梯度尺度，不能用新
数值偷偷改写原主指标。若要分别归因 residual 和 constraint，须另有预声明的分项对照；本矩阵把二者
作为一组固定的物理匹配方案。partial 规则、输入 manifest、共同初始权重、采样顺序、seed、学习率、
batch、带宽、更新次数、开发评估节点及保护规则在对应配对中保持一致。

一次真实短训的实际入口如下。数字预算来自 Work 已冻结协议，不使用文档自行发明的更新上限：

```bash
CPINO_ARM=T2_old_loss_gl2
CPINO_AUTH="/abs/frozen/authorization/$CPINO_ARM.json"
CPINO_MAX_STEPS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_optimizer_steps"])' "$CPINO_AUTH")
CPINO_MAX_ATTEMPTS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_attempts"])' "$CPINO_AUTH")
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.training.runner \
  --manifest /abs/new/training_manifest_with_physics.json \
  --output "/abs/new/precision_trials/$CPINO_ARM" \
  --mode trial --device cuda:0 \
  --config "experiments/compact_pino_v21/precision/configs/$CPINO_ARM.json" \
  --initial-weights /abs/frozen/common_checkpoint.pt \
  --initial-weights-sha256 EXACT_64_CHARACTER_LOWERCASE_SHA256 \
  --authorization "$CPINO_AUTH" \
  --evaluation-protocol /abs/frozen/screen_protocol.json \
  --registered-protocol /abs/frozen/original_protocol.json \
  --cache-directory /abs/new/precision_block_cache \
  --max-steps "$CPINO_MAX_STEPS" --max-attempts "$CPINO_MAX_ATTEMPTS" \
  --checkpoint-every 1
```

两项上限直接读取现有预算文件；`evaluation_steps` 必须在预算内有合法评估节点。
每臂的 authorization 文件须绑定该臂实际 manifest/config/protocol SHA。
这是 [现有 runner 的数据和预算校验](../training/README.md)，不新增对已授权工作的审批请求。
正式物理系数改变后保存新配置与其 canonical SHA，按已有流程更新绑定，不能沿用不匹配的文件。

使用 `--initial-weights` 创建新实验时只加载共同 model，optimizer/RNG/sampler/run 重新开始；
它与 `--resume` 互斥。只有同版本、同配置、同数据和同环境的 runner-issued receipt 可做严格续训。
不要跨本轮源码更新把旧 run 宣称为逐值恢复，也不要让各臂从各自不同的已训练终点初始化。

## 7. 读结果、曝光和原门槛

先核对 `steps.jsonl` 中的四项 denominator/mask/weight，再核对实际 optimizer 调用结果。
启用物理项的臂应在最终 checkpoint lineage 和选中候选自身 lineage 中分别拥有正的 committed
residual/constraint 曝光；不能用后续更新证明较早 best 已接受物理监督。
合法无参数依赖 batch 会消耗 attempt 而不增加 optimizer step，不能把它计成有效更新。

汇总曝光可直接运行：

```bash
python -m experiments.compact_pino_v21.p0_diagnostics.exposure_audit \
  --events /abs/run/exposure.jsonl /abs/resumed_run/exposure.jsonl \
  --checkpoint ACTUAL_CHECKPOINT_ID_FROM_RECEIPT \
  --output /abs/new/selected_checkpoint_exposure.json
```

没有恢复分支时只提供一个 journal；checkpoint 参数使用 receipt 中的实际 ID。
该工具核算已观察曝光，不替代物理标签和原生验收资格。

保留原研究门槛：主指标要求 5% 改善，既有保护量允许的最差倍率仍为 1.05；冻结开发集、R3/R8/Full-J
资格、保留比例保护与候选选择程序继续使用原协议。物理误差另列，不以它自动代替主指标或绕过门槛。
训练 E0、实际更新后的候选、no-learning 基线分别记录；首次候选即便劣于 E0 也不能伪装为 E0 的成绩。

最后在 Work 原生环境对冻结查询以及实际求解返回点分别执行原始验收。固定 query 比较和完整求解轨迹
是两个实验：前者归因局部响应，后者回答终点精度、成功率及取得合格解的总耗时。保持原教师与验收条件。
若原生标签只有线性化 profile，仍需补跑完整私有非线性验收后才能评价面条是否解决。

## 8. 应回传的完整证据

1. 最终 Git SHA、环境信息、实际命令、正式 config SHA 和各臂初始 checkpoint SHA。
2. 新导出 manifest、材料 identity/source/partial_rule、原生 M/J 资格回执及物理 target 绑定回执。
3. 冻结三规则无学习/学习完整 JSON，包含 unavailable 与失败记录、各查询向量、节点及分项耗时。
4. 5090 软件更新与恢复报告；保留 actual optimizer 次数与梯度证据。
5. 每臂 `run.json`、`config.json`、`steps.jsonl`、`exposure.jsonl`、`result.json`、checkpoint/receipt、
   `last.json`、实际存在的 `best.json`、开发评估与 screen 文件。
6. 选中 checkpoint 的独立物理曝光、原 5% 筛选结果，以及原机器最终验收记录。
7. 原生私有公式/线性化标签仍缺失的明确列表；变慢、退化和非规则导数样本同样保留。

本轮发布证据汇总见 [validation.json](validation.json) 和 [README 的证据入口](README.md#6-可复现的软件证据)。
不需要等待新编码器或全局提分辨率，Work 已可按上述顺序推进。是否进入后续更大训练预算，依据既有
冻结协议和真实结果判断；本交接不预先宣告 Noodles 原生终点合格。
