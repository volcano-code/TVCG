# Compact PINO V2.2：响应修复、短训与受控长训交接

日期：2026-11-26（Asia/Shanghai）。目标分支：`research/compact-pino-v22-response-training`。

本文件对应本次已经实现的命令行。按顺序执行，所有结果写入新的实验目录。旧 T1 仍维持历史 NO-GO；当前交付证明了软件实现与合成夹具上的训练／恢复路径，尚未证明 Work GPU、真实新数据或原生求解已经通过。

**执行顺序：准备合法 24 点 → 选定共同积分规则 → 同版本 CUDA smoke → 校准 → 冻结两臂配置和曝光表 → 绑定真实 G0 凭据 → 两臂各至多 104 步 → 复核原筛选 → 只对完整 GO 的选中 checkpoint 原样续训。** 数据生成、CUDA 检查和现有 T1 的原生校正可并行；单卡训练与计时不要彼此争抢资源。

## 1. 固定本次比较

| 项目 | A_old_response | B_monotone_response |
|---|---|---|
| 局部响应 | 原有有界修正 | 新的单调、带内可饱和变换 |
| 编码器／静态 attention | 相同已冻结参数和 buffers | 完全相同 |
| 响应隐藏层 | 相同初始 tensor | 完全相同 |
| 响应输出层 | 两个输出均清零 | 两个输出均清零 |
| Epoch 0 | 同一无学习三次响应 | 同一无学习三次响应 |
| 学习率／种子 | 3e-5／830251 | 相同 |
| 损失 | M＋合格 J＋驻点向量匹配＋约束向量匹配 | 同一批训练侧校准系数 |
| batch／短训上限 | 1／104 更新、104 尝试、6 小时 | 相同 |
| 开发与选模 | 原固定开发查询；26 监控，52/78/104 可选 | 相同 |

整个日程预先冻结到 **10,000 个查询条次**，以便合格候选原样恢复。每次尝试消费一个条次；无活动区、AMP 跳过或其他未更新不能计成有效训练，也不会自动补造曝光。长训的 10,000 步、10,000 次尝试和 48 小时是**包含已恢复历史的累计上限**，并非额外再给一份预算。

## 2. 填写真实绝对路径，隔离检出新分支

下面 `REPLACE_...` 均由 Work 替换成实际存在或计划新建的绝对路径。没有在此交接中提供任何 Work 私有数据或权重路径。

```bash
export TVCG_SOURCE_REPO='/REPLACE_EXISTING_TVCG_REPOSITORY'
export TVCG_RAPID_REPO='/REPLACE_NEW_ISOLATED_WORKTREE'
export TVCG_PYTHON='/REPLACE_VERIFIED_CUDA_ENV/bin/python'
export TVCG_RUN_ROOT='/REPLACE_NEW_EXPERIMENT_DIRECTORY'
export TVCG_MANIFEST='/REPLACE_ADMITTED_MANIFEST_WITH_PHYSICS.json'
export TVCG_COHORT='/REPLACE_WORK_A_LEGAL_COHORT.json'
export TVCG_OLD_EVAL_PROTOCOL='/REPLACE_FROZEN_222_EVALUATION_PROTOCOL.json'
export TVCG_OLD_REGISTERED_PROTOCOL='/REPLACE_ORIGINAL_REGISTERED_PROTOCOL_FILE'
export TVCG_INITIAL_CHECKPOINT='/REPLACE_CANONICAL_RUNNER_INITIAL_CHECKPOINT.pt'
export TVCG_G0_RECEIPT='/REPLACE_WORK_VERIFIED_CORRECTNESS_RECEIPT.json'
export CPINO_EIGEN_ROOT='/REPLACE_EXISTING_EIGEN_INCLUDE_ROOT'
export TVCG_DEVICE='cuda:0'
export TVCG_RULE='gl2'

git -C "$TVCG_SOURCE_REPO" fetch origin research/compact-pino-v22-response-training
git -C "$TVCG_SOURCE_REPO" worktree add --detach "$TVCG_RAPID_REPO" FETCH_HEAD
cd "$TVCG_RAPID_REPO"
git rev-parse HEAD
git status --short
mkdir -p "$TVCG_RUN_ROOT"
```

新 worktree 路径必须尚不存在。不要在正在训练的旧工作目录中 reset、切分支或覆盖数据。随后记录实际提交与代码状态，G0、短训和精确恢复期间保持相关执行源码不变。

建议使用已核验的共同初始 checkpoint 继承编码器、attention 和响应隐藏层。加载器只接受显式 runner checkpoint 格式并校验 SHA；私有格式需要先做有证据的格式对接，不能靠猜 key 或 `strict=False` 装入。若决定使用完全新的随机初值，须在校准前明确预登记，并认识到冻结随机编码器并不等同继承现有模型能力。

## 3. Work A：首批合法 24 点和任务证据

先交付 4 件训练几何，至少 3 个来源组，优先 4 个独立来源。每件严格采用下面的独立查询构成：

| `trajectory_stage` | 每件数量 | 要求 |
|---|---:|---|
| `terminal` | 1 | 原生成功轨迹终点，含固定材料的最终几何合法，原生终点验收通过 |
| `near_terminal` | 2 | 同一或其他成功轨迹末段的不同位置；保留真实非零参考残差 |
| `high` | 2 | 原生可编辑保留率 ≥0.7 且 <1−1e-8，具有实际切割响应 |
| `ordinary` | 1 | 一般合法轨迹位置 |

共 24 个不同查询，不能重复命名同一点凑数。支点检查针对**包含固定材料**的最终实体；保留支点不等于终点验收通过。原先四个普通切割仍可在普通训练池内保留，但不能改名为成功终点。

每个 case 必须已经通过现有严格数据 loader：原网格、坐标／单位／密度、固定材料、有限 operand 的方向／端盖／保留侧、原始系数与顶点运动、教师版本及各项哈希均可追溯。驻点和约束标签必须绑定同一任务定义、目标附加权重、旋转轴、支点、初值、同点原生十矩和参考乘子。普通位置匹配原生非零向量，不能将教师都改成零。

M、物理向量、R3、R8、Full J 分别保留资格。没有合格 J 的行仍可贡献 M／物理损失，J 掩码关闭。本首批 cohort 的每条记录必须具有正权重、合格的驻点及非空约束目标。

`cohort.json` 的结构如下，示例省略的 23 条必须由真实数据补齐：

```json
{
  "schema": "compact_pino.rapid-cohort.v1",
  "training_manifest_sha256": "REPLACE_EXACT_FINAL_MANIFEST_SHA256",
  "records": [
    {
      "record_id": "REPLACE_ACTUAL_TRAIN_RECORD_ID",
      "trajectory_stage": "terminal",
      "legality_evidence": {
        "path": "/REPLACE_ACTUAL_TASK_LEGALITY_DECLARATION.json",
        "sha256": "REPLACE_ACTUAL_FILE_SHA256"
      }
    }
  ]
}
```

每个 `legality_evidence` 指向实际生成的 `compact_pino.task-legality.v1` 声明，包含 `record_id`、`case_sha256`、`geometry_legal`、`pivot_retained`、`native_terminal_accepted`，以及：

- `task_asset: {path, sha256}`：原登记任务文件。
- `original_checks: [{path, sha256}, ...]`：原生合法性、轨迹／末段身份及终点验收的原始证据文件。

`prepare` 校验外部声明、身份与原始文件哈希，**不会自己重做网格布尔或独立证明几何合法**。真假字段由真实检查结果生成，不能为通过脚本而填入。

如物理标签尚未绑定到新 manifest，可复用现有 `training.attach_residual_targets`，其具体输入约定见 `../precision/WORK_HANDOFF.md`。完成绑定后再生成 cohort 的 manifest SHA；之后不要继续往同一冻结 manifest 里追加记录。

原固定开发 **222 条的查询身份、来源角色和教师语义保持不变**。面条继续作诊断，盲测来源封存。新增独立开发证据必须单列；本次没有实现一个可以用补充开发集合替换原缺失 Full J／低保留分母的独立晋级模块。

## 4. 一次训练侧数值选择与同版本 CUDA smoke

先固定积分规则，再做后续所有步骤。默认候选 `gl2` 不是已证明的最优规则；可用以下现有命令在训练角色做一次 tetra4／gl2／gl3 比较：

```bash
"$TVCG_PYTHON" -m experiments.compact_pino_v21.precision.compare_rules \
  --manifest "$TVCG_MANIFEST" --mode trial --roles train \
  --rules tetra4 gl2 gl3 --warmup 0 --repeats 1 \
  --cache-directory "$TVCG_RUN_ROOT/numerical_cache" \
  --output "$TVCG_RUN_ROOT/training_rule_comparison.json"
```

此命令按 manifest 的 train 角色比较，未实现 `--cohort` 子集参数；不要伪造 flag 或删改冻结开发清单。无学习路径在 CPU 上执行。优先从训练侧 Moment、有效 J/Jv 与成本决定共同规则，再设定 `TVCG_RULE`；不靠开发高保留结果搜索新超参数。若 gl2/gl3 的旧材料包缺 provenance，须从原材料生成新包，不能只改名称。

在启动任何 CUDA Python 进程前设置确定性环境：

```bash
export CUBLAS_WORKSPACE_CONFIG=':4096:8'
export OMP_NUM_THREADS='1'
export MKL_NUM_THREADS='1'
export OPENBLAS_NUM_THREADS='1'
export PYTHONUNBUFFERED='1'

"$TVCG_PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()'

"$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.smoke \
  --device "$TVCG_DEVICE" --full-cell-rule "$TVCG_RULE" \
  --output "$TVCG_RUN_ROOT/cuda_smoke"
```

检查 `cuda_smoke/report.json` 的 `status=PASS`、真实 `actual_device=cuda:...`、6 次真实 optimizer 调用、两臂零输出 E0 对齐、M/J 联合更新及中断恢复逐值重演、冻结参数／buffers 不变和跨更新缓存行为。此 smoke 使用合成材料和已绑定的仓库物理公式，不能代替 Work 私有教师语义及原生终点验收。

本次 G0 绑定会自动核对 smoke 状态、两臂 profile、实际 CUDA 设备、当前执行源码摘要及 smoke 工具自身摘要。改动执行源码、换积分规则或环境后，不复用旧 CUDA 通过记录。

## 5. 校准物理损失，然后冻结完整两臂包

从真实 checkpoint 计算 SHA，使用同一个 checkpoint 校准两臂。`geometry_and_hidden` 迁移保留共同几何／隐藏层，仅清零两个响应输出；旧头输出权重不能直接赋给新语义。

```bash
export TVCG_INITIAL_SHA="$("$TVCG_PYTHON" -c 'import hashlib, os; from pathlib import Path; p=Path(os.environ["TVCG_INITIAL_CHECKPOINT"]); print(hashlib.sha256(p.read_bytes()).hexdigest())')"

"$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.calibrate \
  --manifest "$TVCG_MANIFEST" --cohort "$TVCG_COHORT" --mode trial \
  --full-cell-rule "$TVCG_RULE" --seed 830251 --device "$TVCG_DEVICE" \
  --max-records 24 --initial-weights "$TVCG_INITIAL_CHECKPOINT" \
  --initial-weights-sha256 "$TVCG_INITIAL_SHA" \
  --cache-directory "$TVCG_RUN_ROOT/material_cache" \
  --output "$TVCG_RUN_ROOT/calibration.json"

"$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.prepare \
  --manifest "$TVCG_MANIFEST" --cohort "$TVCG_COHORT" \
  --evaluation-protocol "$TVCG_OLD_EVAL_PROTOCOL" \
  --registered-protocol "$TVCG_OLD_REGISTERED_PROTOCOL" \
  --calibration "$TVCG_RUN_ROOT/calibration.json" --mode trial \
  --full-cell-rule "$TVCG_RULE" --seed 830251 --total-steps 10000 \
  --cache-directory "$TVCG_RUN_ROOT/material_cache" \
  --output "$TVCG_RUN_ROOT/prepared"
```

校准只使用训练 cohort，不做 optimizer 更新。它输出 M/J/R/C 梯度范数、夹角及两臂共用的损失系数；“约 10% 梯度预算”是初始校准政策，并不保证以后始终保持同一比例。

`prepare` 生成 `prepared/bundle.json`、两臂配置、共同 `schedule.json`、新预注册与 `protocol.json`。其中：

- 前 26 次计划曝光覆盖 24 个不同新增查询，前 52 次使每个至少两次；选中 checkpoint 还须用自己的有效曝光回执证明。
- 前 104 条次至少 26 次 high 曝光；不同角色可重叠。来源均衡采样，低／近零／有效 J 缺池时明确记录 fallback。
- 26 只监控，52/78/104 可选；以后 500 的倍数及 10,000 预先登记。
- 从第 52 步开始，连续两个已登记检查点“主项劣于 E0 且 high 超过 1.05”触发早停。没有 high 分母则报告不足，不伪造早停结论。
- 状态仍为 `PREPARED_NOT_AUTHORIZED_NOT_TRAINED`，不产生教师、数据准入或模型 GO。

## 6. 用真实 G0 凭据绑定已获授权的试训预算

Work 提交 `compact_pino.rapid-correctness.v1` 外部回执，绑定本次最终 manifest SHA。以下五项均须引用实际检查文件的 `{path, sha256}`：

| `checks` 键 | 原始证据 |
|---|---|
| `teacher_and_operand_identity` | 部署教师、有限 operand、原任务公式、原生十矩与参数归一化同点复核 |
| `source_split_isolation` | 原有来源准入及 train/development/诊断/盲测隔离核验 |
| `material_ten_moments` | 完整可编辑材料十矩按原物理尺度的双精度审计，原 1e-8 条件 |
| `derivative_qualification_masks` | R3/R8/Full J 与事件／chart 的独立资格和无效掩码；缺标签如实保留 |
| `finite_joint_update_and_resume` | 本机同代码、同响应 profile、同积分规则的 `cuda_smoke/report.json` |

结构示意如下。`NOT_RUN` 和空证据是不可准入的占位，必须由真实检查生成 PASS 与文件绑定：

```json
{
  "schema": "compact_pino.rapid-correctness.v1",
  "training_manifest_sha256": "REPLACE_EXACT_FINAL_MANIFEST_SHA256",
  "checks": {
    "teacher_and_operand_identity": {"status": "NOT_RUN", "evidence": []},
    "source_split_isolation": {"status": "NOT_RUN", "evidence": []},
    "material_ten_moments": {"status": "NOT_RUN", "evidence": []},
    "derivative_qualification_masks": {"status": "NOT_RUN", "evidence": []},
    "finite_joint_update_and_resume": {"status": "NOT_RUN", "evidence": []}
  }
}
```

不得把本次云端 CPU smoke 充当 Work CUDA 证据。其他 G0 项的工具检查仍是外部声明与原文件身份绑定，并未自动执行私有网格物理复算。

用户已授权本轮研发与试训；下面是绑定既有授权与证据，不是再次请求用户批准：

```bash
for TVCG_ARM in A_old_response B_monotone_response; do
  "$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.authorize \
    --bundle "$TVCG_RUN_ROOT/prepared/bundle.json" --arm "$TVCG_ARM" \
    --correctness-receipt "$TVCG_G0_RECEIPT" \
    --authorized-by '本会话用户已授权；Work根据实际G0证据执行' \
    --mode trial --max-steps 104 --max-attempts 104 --wall-hours 6 \
    --output "$TVCG_RUN_ROOT/authorization_${TVCG_ARM}_trial.json"
done
```

G0 不要求未训练模型先通过原生 1e-12 终点；无 Full J 标签也不伪装为有标签。它只允许这一段受限研究试训，完整长训晋级继续由原协议决定。

## 7. 两臂各至多 104 步

```bash
for TVCG_ARM in A_old_response B_monotone_response; do
  "$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.run \
    --bundle "$TVCG_RUN_ROOT/prepared/bundle.json" --arm "$TVCG_ARM" \
    --authorization "$TVCG_RUN_ROOT/authorization_${TVCG_ARM}_trial.json" \
    --mode trial --device "$TVCG_DEVICE" --max-steps 104 --max-attempts 104 \
    --cache-directory "$TVCG_RUN_ROOT/material_cache" \
    --output "$TVCG_RUN_ROOT/$TVCG_ARM"
done
```

脚本顺序运行两臂，避免单卡资源竞争。不要把输出目录设成已有 run；工具不会覆盖它。观察实际 `committed`、loss 有效分母、冻结摘要、缓存统计、每步耗时及显存，不能从查询池大小推断训练已发生。

各臂的关键输出：`run.json`、`config.json`、`steps.jsonl`、`exposure.jsonl`、逐检查点 development JSON、checkpoint 和 `.resume.json`、`last.json`、若有合格候选则 `best.json`，以及可运行时的 `screen_report.json`／`screen/bundle.json`。

`best.json` 只会从预登记选模节点产生，并要求 cohort 每条驻点／约束各至少两次有效 committed participation。即使第 26 步分数更好，也不能成为本轮候选。后续曝光不补记到早期权重。

## 8. 训练梯度探针、运行预算与原筛选复算

训练结束后对存在的真实 `best.json` 做无更新探针。`probe --checkpoint-receipt` 接受 `best.json`／`last.json` 这类 `{path, sha256}` 记录；它与长训 `--resume` 的输入不同。

```bash
for TVCG_ARM in A_old_response B_monotone_response; do
  if [ -f "$TVCG_RUN_ROOT/$TVCG_ARM/best.json" ]; then
    "$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.probe \
      --manifest "$TVCG_MANIFEST" --cohort "$TVCG_COHORT" --mode trial \
      --checkpoint-receipt "$TVCG_RUN_ROOT/$TVCG_ARM/best.json" \
      --device "$TVCG_DEVICE" --max-records 24 \
      --cache-directory "$TVCG_RUN_ROOT/material_cache" \
      --output "$TVCG_RUN_ROOT/${TVCG_ARM}_selected_gradient_probe.json"
  fi
  "$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.runtime \
    --run "$TVCG_RUN_ROOT/$TVCG_ARM" --total-steps 10000 --warmup 3 \
    --protocol "$TVCG_RUN_ROOT/prepared/protocol.json" \
    --output "$TVCG_RUN_ROOT/${TVCG_ARM}_runtime_projection.json"
done
```

探针保留已经训练的输出层，不重新清零，不创建 optimizer，不从开发集取梯度。若需检查 26／52／78 等位置，先由真实 checkpoint 的 `{path, sha256}` 构造对应记录，不能将未选中的点混成 `best.json`。

`runtime` 至少需要超过 3 次 committed update；样本不足时拒绝估计。它报告实测 P50/P95/P99、峰值 allocated 显存、剩余更新与开发评估耗时预测；这不是置信区间或端到端加速证据。恢复自早期 best 时，其步数可能早于 trial 末步；排期须以选中位置计算剩余预算，不把被丢弃的后续更新算成续训进度。

对生成了 screen bundle 的臂可以独立复算原指标：

```bash
export TVCG_REVIEW_ARM='REPLACE_WITH_A_old_response_OR_B_monotone_response'
"$TVCG_PYTHON" -m experiments.compact_pino_v21.p0_diagnostics.promotion_gate \
  --bundle "$TVCG_RUN_ROOT/$TVCG_REVIEW_ARM/screen/bundle.json" \
  --output "$TVCG_RUN_ROOT/${TVCG_REVIEW_ARM}_screen_recomputed.json"
```

`promotion_gate` 的非 GO 退出码为 2，这表示筛选结论，不是应当绕过的脚本故障。保留 Moment 相对本轮 E0 至少改善 5%、原保护项最多退化 5%、原 no-learning 及来源要求。缺 Full J 或低保留分母仍是 `INSUFFICIENT_EVIDENCE`。没有 `best.json` 或没有可导出的 screen 时，保留 `result.json` 中原因，不能手工创建 GO。

本次没有实现把新增独立开发诊断直接并入旧 222 条晋级的模块。因此不能仅旁挂一份新的 Full J 文件就让旧完整门槛变成 GO；标签及其查询身份必须通过正式数据准入和既有评估绑定。需要改变 manifest／protocol 的试验须重新冻结并重新开始 trial，不能修改正在续训的科学身份。

## 9. 只有完整 GO 才精确恢复长训

从完整 GO 的臂中选择真实胜出者，填写下面变量。若两臂均失败或证据不足，停止在研究结果；不执行本节，不自动继续 208／312。

```bash
export TVCG_SELECTED_ARM='REPLACE_WITH_ACTUAL_FULL_GO_ARM'
export TVCG_SELECTED_RUN="$TVCG_RUN_ROOT/$TVCG_SELECTED_ARM"
export TVCG_SELECTED_RESUME="$("$TVCG_PYTHON" -c 'import json, os; from pathlib import Path; r=Path(os.environ["TVCG_SELECTED_RUN"]); best=json.loads((r/"best.json").read_text()); screen=json.loads((r/"screen_report.json").read_text()); assert screen["decision"] == "GO" and screen["scope"] == "development"; assert screen["identity"]["candidate_checkpoint_sha256"] == best["sha256"]; print(best["receipt_path"])')"

"$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.authorize \
  --bundle "$TVCG_RUN_ROOT/prepared/bundle.json" --arm "$TVCG_SELECTED_ARM" \
  --correctness-receipt "$TVCG_G0_RECEIPT" \
  --authorized-by '本会话用户已授权；Work依据完整GO绑定同一候选续训' \
  --mode long --max-steps 10000 --max-attempts 10000 --wall-hours 48 \
  --screen-report "$TVCG_SELECTED_RUN/screen_report.json" \
  --resume "$TVCG_SELECTED_RESUME" \
  --output "$TVCG_RUN_ROOT/authorization_selected_long.json"

"$TVCG_PYTHON" -m experiments.compact_pino_v21.rapid_training.run \
  --bundle "$TVCG_RUN_ROOT/prepared/bundle.json" --arm "$TVCG_SELECTED_ARM" \
  --authorization "$TVCG_RUN_ROOT/authorization_selected_long.json" \
  --mode long --resume "$TVCG_SELECTED_RESUME" \
  --device "$TVCG_DEVICE" --max-steps 10000 --max-attempts 10000 \
  --cache-directory "$TVCG_RUN_ROOT/material_cache" \
  --output "$TVCG_RUN_ROOT/selected_long"
```

必须恢复 `best.json` 所引用的**选中权重 `.resume.json`**，不能直接给 `best.json`、任意 `.pt` 或最后一个 checkpoint。长训 wrapper 不重新初始化输出层。

续训保留同一 config、manifest、完整日程 SHA、protocol、代码与 runtime 身份、optimizer／scheduler／scaler／RNG／曝光历史。计数与墙钟预算均累计；达到任一上限或预登记早停即停止。48 小时是硬预算，不是保证能完成 10,000 次更新。

解冻编码器、改变学习率／损失、增加训练数据、修改积分规则或重新划分开发集合，都必须成为新的 trial；不能借精确恢复接口隐式改变这些条件。执行源码更新后也不能冒称原样恢复。

## 10. 原生校正与最终验收仍由 Work 接入部署求解器

当前代码提供算子、训练、独立公式匹配与软件证据；**本次没有实现 Work 私有 Noodles 求解器的完整自动热启动适配**。

Work B 可立即并行使用已冻结 T1@104 的真实候选参数：检查含固定材料的实体任务合法性，原生重算同点 M/J，依部署求解器重算乘子和归一化，用原线搜索续算。首个诊断预算维持最多 5 次接受更新、10 次新增原生查询，以先到者为限；初始化、线搜索和验收查询全部计入。

纯神经、神经候选＋原生校正、纯原生分别报告。原方程 1e-12、约束 1e-6 及其他登记验收条件不降低；未通过候选不能计算取得合格解的加速比。同期同机原生对照、原网格冷启动、缓存复用、失败／回退和最终验收成本均保留。原预登记独立几何及配对初值要求继续有效，少量功能对照不能替代它们。

## 11. 回传材料

- 实际提交、运行环境、CUDA smoke 原报告及执行源码摘要。
- 新 manifest／cohort／任务合法性声明及原生原始证据；教师、材料、初始权重的哈希。
- `calibration.json`、完整 `prepared/`、G0 回执与预算授权。
- 两臂 `run.json`、`config.json`、完整步骤／曝光日志、所有预登记开发评估、E0／best／last 权重与恢复记录。
- 原 gate 报告；逐来源、high、各 Moment 分量及有效 J 的原始结果；所选权重自身的最低物理曝光证明。
- 梯度探针、真实耗时与显存、原生校正和独立网格验收。缺失部分明确标记，不以测试数量替代物理证据。

这份交接的完成标志是“可复现的实际训练结果及原判据下的明确结论”。软件完成、短训完成、完整筛选 GO 和原生部署／加速通过分别报告。
