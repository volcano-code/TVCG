# Work B：旧配置失败后切换 HENMO R3 的执行清单

日期：2026-11-28（北京时间）。执行框架仍为 HENMO R3；本文件更新实验次序和验收操作，不修改网络或宣称已有合格权重。

研究分支：`research/henmo-v1plus`。代码基线：`74e47e7ee5c7266b1d0492ea81e633e08df07b21`。总决策见 [联合清单](WORK_EXECUTION_R3_2026-11-28.md)。本文件取代 11 月 27 日的 Work B 执行顺序及旧配置隔夜续跑安排。

## B0. 停止旧配置，冻结证据

- [ ] 通过现有进程管理在完整更新边界停止旧任务，保存已提交的状态，确认 supervisor/循环脚本不会继续启动下一轮。当前 CLI 没有 `stop` 子命令；不要对整个 GPU 或不相关进程执行批量终止。
- [ ] 保留 E0、所有完整 epoch，至少包括 E4、E5、实际最后权重及 `.pt.json`，不要覆盖或删除失败权重。
- [ ] 按 checkpoint 内部 `step_count` 登记身份。在每轮 273 次已提交更新的原运行中，E4 为 1092 步、E5 为 1365 步；1366 已经是 E6 的一次更新。隔夜最终步数以实际 checkpoint 为准。
- [ ] 保留模型/Adam/随机状态、源代码提交和本地 diff、原始配置、manifest/资产哈希、查询顺序、逐步日志及逐查询评估。
- [ ] 回传 `old_run_stop_receipt.json`，写清 PID/任务身份、停止时间、实际最后已提交步数、可恢复 checkpoint 及哈希、是否仍有自动续跑。该文件由 Work 据实生成，不是已有 CLI 自动产物。

当前可见 E5 汇总为 raw Moment `4.0817e-7`、高保留 `4.0954e-7`、R3 `0.104842`。用户已确认隔夜没有改善，但本清单没有收到隔夜最终原始报告，不能补写后续数值。R8 两条的改善不代替其余指标验收。

## B1. 隔离更新，核对软件环境

先激活 Work 已有 PyTorch/CUDA 环境。下面所有 `/absolute/...` 都必须替换为实际路径；不要求更换现有 Python/CUDA 版本。

```bash
set -euo pipefail
export HENMO_PYTHON="$(command -v python)"
export HENMO_REPO=/absolute/existing/TVCG
export HENMO_WORKTREE=/absolute/new/TVCG_henmo_r3_1128
export HENMO_CODE_BASE=74e47e7ee5c7266b1d0492ea81e633e08df07b21

git -C "$HENMO_REPO" status --short --branch
git -C "$HENMO_REPO" fetch origin
git -C "$HENMO_REPO" worktree add --detach "$HENMO_WORKTREE" origin/research/henmo-v1plus
cd "$HENMO_WORKTREE"
git merge-base --is-ancestor "$HENMO_CODE_BASE" HEAD
git rev-parse HEAD

export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$HENMO_PYTHON" - <<'PY'
import sys
import torch
from pathlib import Path
import experiments.henmo_v1plus.cli as cli
root = Path.cwd().resolve()
source = Path(cli.__file__).resolve()
if not source.is_relative_to(root):
    raise RuntimeError('import resolved outside the new worktree: '+str(source))
print({'python': sys.executable, 'torch': torch.__version__,
       'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
       'cli': str(source)})
PY
"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli --help
"$HENMO_PYTHON" -m pytest -q tests/research/test_henmo_*.py
```

- [ ] 使用新 worktree 的目的是保留旧运行与 Work 自有原生接口。新目录必须不存在；已有目录不能直接覆盖。若已有独立、干净的同版本 checkout，可复用并登记身份。
- [ ] 复用原生扩展与依赖时记录其真实路径/版本，防止 `PYTHONPATH` 使神经算子导入旧仓库。不要 `reset --hard` 覆盖 Work 修改。
- [ ] 新 worktree 默认为 detached；若后续需要修改代码，先为这些修改建立明确的研究分支。
- [ ] 预检失败先修复环境或数值问题，不能改成忽略失败。已发布 CPU 证据为 170 项通过；Work 本机结果另记，测试数量不是物理精度证明。

## B2. 严格绑定源 checkpoint，不换近似配置

优先使用 E0 和一个失败代表状态（E5 或隔夜最终状态）；E4 用于复放 E4→E5 跳变点。每个状态使用单独的新目录。先设置一组真实路径：

```bash
export HENMO_MANIFEST=/absolute/frozen/training_manifest.json
export HENMO_CHECKPOINT=/absolute/old_run/initial.pt
export HENMO_RESULTS=/absolute/new/results/henmo_r3_1128_e0
export HENMO_DTYPE=float32

"$HENMO_PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
import torch

checkpoint = Path(os.environ['HENMO_CHECKPOINT']).resolve()
manifest = Path(os.environ['HENMO_MANIFEST']).resolve()
output = Path(os.environ['HENMO_RESULTS']).resolve()
if output.exists():
    raise ValueError('choose a new results directory')
def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
digest = sha256(checkpoint)
receipt = json.loads(checkpoint.with_suffix(checkpoint.suffix+'.json').read_text())
if receipt.get('schema') != 'henmo.checkpoint-receipt.v1' or receipt.get('sha256') != digest:
    raise ValueError('checkpoint receipt/schema/hash mismatch')
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
if state.get('schema') != 'henmo.training-state.v1':
    raise ValueError('expected original HENMO training-state checkpoint')
config = state['architecture']['config']
output.mkdir(parents=True)
(output/'source_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
(output/'source_checkpoint.sha256').write_text(digest+'\n')
identity = {
    'checkpoint': str(checkpoint), 'sha256': digest,
    'step_count': state['step_count'],
    'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
    'binding_sha256': state.get('binding_sha256'),
    'tensor_dtypes': sorted({str(value.dtype) for value in state['model'].values()}),
    'scope': 'metadata extraction; CLI must still verify exact architecture/tensors/dataset binding'
}
(output/'source_identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(identity, ensure_ascii=False, indent=2))
PY

export HENMO_SOURCE_CONFIG="$HENMO_RESULTS/source_config.json"
export HENMO_CPSHA="$(cat "$HENMO_RESULTS/source_checkpoint.sha256")"
git rev-parse HEAD > "$HENMO_RESULTS/code_commit.txt"
git diff --binary > "$HENMO_RESULTS/code_local_changes.patch"
```

`HENMO_DTYPE` 必须匹配实际源模型张量精度；原阶段为 FP32 网络及 FP64 矩聚合，谱权重可为 complex64。不要把全部 checkpoint 转成一种 dtype 后声称是同权重重放。上面仅导出配置与身份，后续 CLI 还会核对 tensor digest、参数清单、形状、dtype、有限性以及数据 binding。

### 数据预检

```bash
"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli audit-data \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
  --mode trial --output "$HENMO_RESULTS/data_audit.json"
```

- [ ] 原快照应核对为实际训练 273 条/4 件几何、原开发 222 条/4 件几何/3 个来源；数量相同还需核对全部 `record_id` 和角色。
- [ ] 原快照 ID 为 `CPINO-20261125-1756-R1`，原 manifest SHA256 为 `3f4efce1fc6e5db863a7faac179e20da33bfd0f2dfae19f5ca9eca4c34d49701`。这些是现有选择文件绑定的快照，不能通过改哈希或角色强行兼容另一快照。
- [ ] 后续若换数据，建立新阶段及合法初始化/迁移记录；不要绕过旧 checkpoint 的严格 binding。
- [ ] 原开发 64 条合格 R3、2 条 R8、Full-J/终点等覆盖缺口据实列出；缺少新标签不阻塞现有合法查询的只读诊断。

## B3. 同权重复放与八路径归因

### 先少量查询

下列选择文件已经冻结 12 条旧开发诊断及 13 条训练探针。先复用它们即可，不必等 Work A 新数据；其中训练探针 9 条有旧联合导数记录。

```bash
"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli diagnose-paths \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
  --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
  --mode trial --device cuda:0 --dtype "$HENMO_DTYPE" --threads 2 \
  --split development \
  --selection experiments/henmo_v1plus/configs/diagnostic_development_r2.json \
  --coarse-grids 32 64 --diagnostic-nodes 128 \
  --output "$HENMO_RESULTS/development_paths.json"

"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli diagnose-paths \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
  --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
  --mode trial --device cuda:0 --dtype "$HENMO_DTYPE" --threads 2 \
  --split train \
  --selection experiments/henmo_v1plus/configs/diagnostic_train_r2.json \
  --coarse-grids 32 64 --diagnostic-nodes 128 \
  --output "$HENMO_RESULTS/train_paths.json"
```

- [ ] 首次若成本高可加 `--limit 6`，但要标记实际子集；不能称全开发结果。已存在的报告路径会被拒绝，重跑使用新名字。
- [ ] 依据 E4/E5 实际逐查询误差追加 3–5 个新增异常点及正常来源对照，另存带原 snapshot/manifest/split/唯一 `record_ids` 的 selection；不能假定旧 12 条已覆盖 E5 的最差点，也不能把开发诊断加入训练。
- [ ] E0 与失败代表权重做主要诊断，E4 优先只复放同一组跳变点；不要求每个历史 checkpoint 都运行全 222×八路径。
- [ ] 同一个 checkpoint 在旧/新代码中比较相同路径的原始 M/J；先冻结浮点容差、dtype、设备和误差尺度，再查看候选结果。若旧评估没有原始向量，用旧入口只读补导出代表查询。RMSE 相同不等于两个向量相同。
- [ ] `routes.<route>.rows` 保留原始预测与教师 M/J、尺度、资格和执行诊断。诊断不完整会先保存含失败行的报告，再非零退出；不要删除失败行。

八条路径为 coarse32、coarse64、对应的两个 zero_correction、adaptive32、adaptive64、deterministic_gl2、deterministic_gl3。

**解释约束：** adaptive64/leaf64 的边界输出直接由细积分承担，不能当成神经分支收益；同粗零修正不等于 GL2；GL2/GL3 是同材料平滑参考，不是原网格硬布尔真值。节点 trace 包含同步开销，不是吞吐基准。多层节点计数不能当材料比例重复相加。

### 再做一次完整开发比较

选定一个代表 checkpoint 后执行下列命令。不传 `--limit`，才能覆盖该 manifest 的全部 admitted development：

```bash
"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli compare \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
  --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
  --mode trial --device cuda:0 --dtype "$HENMO_DTYPE" --threads 2 \
  --output "$HENMO_RESULTS/full_development_compare.json"
```

- [ ] 只有审计确认身份集合为原 222 条，才称“固定 222 全开发”。HENMO 的实际执行路径来自源 config，必须记录它确实是要复查的原部署路径。
- [ ] 分别检查各臂 `complete`、`derivative_complete`、`failed_predictions`、实际行数与合格导数分母。`compare` 即使写入失败行也可能退出 0，退出码不能代替完整性检查。
- [ ] `compare` 的 `arms.<arm>.rows` 是逐查询误差/资格结果，不包含完整原始 M/J 矩阵；原始向量看 `diagnose-paths`。全开发聚合复核与少量原向量复放分别报告，不将其混称为全量逐元素等价。
- [ ] 此命令也会计算 GL2/GL3；报告中单次查询计时未控制重复预热，不能直接当正式 P95/P99 或完整求解加速证据。

**本阶段通过：** 同权重同输入数值保持检查通过，原退化可复现，评估口径/身份无错误。若旧/新同权重不同，先解释并修正差异，不将差异直接写成模型学习进步。

## B4. 训练侧分项梯度：区分 M 与 J 的作用

```bash
for HENMO_GRID in 32 64; do
  "$HENMO_PYTHON" -m experiments.henmo_v1plus.cli diagnose-gradients \
    --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
    --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
    --mode trial --device cuda:0 --dtype "$HENMO_DTYPE" --threads 2 \
    --selection experiments/henmo_v1plus/configs/diagnostic_train_r2.json \
    --coarse-grid "$HENMO_GRID" --jacobian-weight 1 \
    --output "$HENMO_RESULTS/gradients_${HENMO_GRID}.json"
done
```

- [ ] 只用 train-role 查询。核对 `qualified_derivative_objectives`、`observed_derivative_objectives`、`derivative_objective_complete` 以及 `actual_parameter_updates=0`。
- [ ] 分别记录 shared_encoder、local_geometry、geometry_attention、response 的 M/J 梯度范数、同查询夹角与合成梯度范数 `combined_gradient_norm`。当前报告不导出完整梯度向量或 Adam 更新方向；不同查询的总梯度不可当同点冲突证据。
- [ ] E0 零输出头可能暂时使部分编码器梯度为零；在已非零的中间权重再次检查。标量 J loss 占比不能直接证明梯度冲突。
- [ ] 归因可以使用 E4/E5；校准训练系数 λ 的报告必须与下一阶段实际起点使用同一个 checkpoint、同一 snapshot 和 grid。不能拿 E5 梯度去给 E0 校准而不重新测量。
- [ ] 困难导数先检查教师、chart、单位/尺度及事件有效邻域，不能靠删除困难行取得改善。

**下一步按证据选择：** coarse 好而 adaptive 差，先查最终响应与层级连接；M-only 好而 joint 差且梯度支持失衡，才校准 λ/更新日程；训练探针也拟合不了，查表示/教师；训练好而开发差，优先来源覆盖和采样。当前尚不能预先宣布其中任何一项为唯一根因。

## B5. 只运行有界的新阶段：各 128 步

### 开始前冻结

- [ ] 已完成 B1–B4，或已发现并修复明确的口径/代码问题；把唯一要检验的因素写入 `experiment_decision.json`（Work 记录，不是自动产物）。
- [ ] 默认首先检查原 E0 的可比起点；如选用其他起点，登记规则并同时列原 E0、同路径零修正和 GL2/GL3。不得为容易超过基线而默认选择较差 E5。
- [ ] 将本节变量重新绑定到实际短训练起点；失败代表 checkpoint 只用于诊断时，不得遗留在 `HENMO_CHECKPOINT` 中误作初始化。
- [ ] 固定层级来自路径诊断。下例 64 为需要登记的选择值，不是已证实最优值。

```bash
export HENMO_GRID=64
for HENMO_ARM in moment joint; do
  "$HENMO_PYTHON" -m experiments.henmo_v1plus.cli screen \
    --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
    --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
    --mode trial --device cuda:0 --dtype "$HENMO_DTYPE" --threads 2 \
    --coarse-grid "$HENMO_GRID" --arm "$HENMO_ARM" \
    --steps 128 --seed 830251 --learning-rate 3e-5 \
    --evaluate-every 128 --save-every 16 \
    --output "$HENMO_RESULTS/screen_$HENMO_ARM"
done
```

固定配置：batch1、同源精确权重、同训练池/种子/查询顺序、FP32 网络及 FP64 矩聚合、辅助损失 0；Moment 臂 λJ=0，joint 臂 λJ=1；**新建 Adam，不加载旧 optimizer**。这是登记的新阶段，不是精确恢复旧运行。若源 dtype 不同，先登记精度实验，不能把 dtype 变化混入两臂对照。

`screen` 强制训练和内部评估为同一 coarse 路径；默认只在起点与第 128 步完整评估，每 16 步保存 last。内部选模只看 source-macro raw Moment，**不会自动执行 J/高保留保护线**。若 `selected_has_new_updates=false`，说明仍选中起点。

- [ ] 到 128 次已提交更新后保存并停止，不因最后一次小回升自动追加。CLI 最多支持 256 步、最多 `2×steps` attempts；本清单授权的首轮预算仍是每臂 128。
- [ ] 回传 `run.json`、`stage.json`、配置、initial/last/selected/step 权重与 sidecar、steps.jsonl、所有评估及失败。失败/跳步导致实际曝光不同必须说明。
- [ ] 不同时改采样、学习率、材料、头结构或多个损失。数值非有限或异常更新先保存已提交 checkpoint 并停止排查，不让监督脚本无限重试。

### 条件满足后才增加校准臂

lambda 需要根据同源训练侧审计显式冻结，不能只凭 scalar loss 比例设定。下面的占位值必须替换成已登记的正数：

```bash
export HENMO_J_WEIGHT=REPLACE_WITH_REGISTERED_POSITIVE_VALUE
"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli screen \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
  --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
  --mode trial --device cuda:0 --dtype "$HENMO_DTYPE" --threads 2 \
  --coarse-grid "$HENMO_GRID" --arm calibrated --jacobian-weight "$HENMO_J_WEIGHT" \
  --calibration-report "$HENMO_RESULTS/gradients_${HENMO_GRID}.json" \
  --steps 128 --seed 830251 --learning-rate 3e-5 \
  --evaluate-every 128 --save-every 16 \
  --output "$HENMO_RESULTS/screen_calibrated"
```

CLI 会核对同 checkpoint、manifest、snapshot、grid、train split、完整合格导数、非零联合梯度与零参数更新。缺失这些证据不能使用 calibrated 臂；不接受开发梯度调 λ。

## B6. 短训练后复评部署路径

先使用该 screen 的 **model_config.json + 新权重 + 新 SHA** 严格加载，再做显式路径诊断。不能把旧源 config 与新 screen checkpoint 搭配，也不能直接编辑源 config 规避绑定。

```bash
export HENMO_SCREEN_DIR="$HENMO_RESULTS/screen_joint"
export HENMO_SCREEN_CHECKPOINT="$HENMO_SCREEN_DIR/selected.pt"
export HENMO_SCREEN_CPSHA="$("$HENMO_PYTHON" -c 'import hashlib,sys; f=open(sys.argv[1],"rb"); print(hashlib.file_digest(f,"sha256").hexdigest()); f.close()' "$HENMO_SCREEN_CHECKPOINT")"

"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli diagnose-paths \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SCREEN_DIR/model_config.json" \
  --checkpoint "$HENMO_SCREEN_CHECKPOINT" --checkpoint-sha256 "$HENMO_SCREEN_CPSHA" \
  --mode trial --device cuda:0 --dtype "$HENMO_DTYPE" --threads 2 \
  --split development \
  --selection experiments/henmo_v1plus/configs/diagnostic_development_r2.json \
  --coarse-grids 32 64 --diagnostic-nodes 128 \
  --output "$HENMO_SCREEN_DIR/selected_development_paths.json"
```

先检查小样本上 joint 是否恢复两个目标，以及 Moment-only 的变化是否能解释原失败；不只报告 selected，last 的失败也保留。

在小样本支持继续、且数据审计确认仍是原 222 条后，可对选中的一个候选执行完整路径复评：

```bash
"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli diagnose-paths \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SCREEN_DIR/model_config.json" \
  --checkpoint "$HENMO_SCREEN_CHECKPOINT" --checkpoint-sha256 "$HENMO_SCREEN_CPSHA" \
  --mode trial --device cuda:0 --dtype "$HENMO_DTYPE" --threads 2 \
  --split development --limit 222 \
  --coarse-grids 32 --diagnostic-nodes 0 \
  --output "$HENMO_SCREEN_DIR/selected_full_adaptive32_paths.json"
```

**限制必须写入报告：** `--coarse-grids 32` 仍运行 5 条路径（coarse32、同粗零修正、adaptive32、GL2、GL3），没有仅执行 adaptive32 的 CLI 开关。省略 selection/limit 默认只选前 12 条，不能称全开发。原部署起始网格如果实际不是 32，先核对原 config，再显式登记相应覆盖。

这一步是 screen 权重在声明路径上的复评；coarse 内部评估不能冒充原 adaptive 晋级。仅当数据、教师、尺度、聚合口径和部署路径可比时才应用原正式筛选线。它仍不等于原网格求解验收。

## B7. 效率与原生终点验收

- [ ] 在实际运行内用 Python API `model.efficiency_statistics()['static_input_cache']` 采集命中/容量；当前 CLI 报告没有自动写出这一字段。如测 cache0，对照构造使用 `HENMOOperator(config, static_input_cache_bytes=0)`，不是不存在的 CLI 参数。学习编码不能跨参数更新复用。
- [ ] 旧训练停止后独占 GPU，固定查询与配置、统一预热/同步，分别测静态准备、树构建、编码、查询、反向、optimizer、记录/保存/评估。`stages_host_s` 是 Host 计时。
- [ ] 报告冷/热、P50/P95/P99、样本量、重复次数及总任务耗时。详细 trace、一次 compare 的时间和 CPU 局部缓存实验不能冒充正式吞吐。
- [ ] Work A 的数据整理可并行；正式性能计时期间限制 CPU/磁盘/原生补标争用。单 GPU 的新验收与 screen 排队执行。
- [ ] 在原生成功轨迹末段和候选终点，同点分别替换 M 或 J；固定终点验收的 M 敏感性与迭代路径的 J 影响分别报告。
- [ ] 使用已注册的原生 solver callback、原网格 `acceptance.verify` 与完整 timing context。当前 `deployment.py` 提供接口，CLI 没有 `solve`/`evaluate-native`/`benchmark`；不能构造一条假的命令假装已经接通。
- [ ] 完整求解计入读入网格、预处理、构面、传输、编码、推理、失败、校正、回退、最终验收。失败候选不能计算合格解加速比。

## B8. 通过线、预算与回传

| 决策 | 所需证据 |
|---|---|
| 允许只读诊断 | 真实源权重/配置/数据身份一致，软件环境可运行 |
| 允许首轮各 128 步 screen | 口径与代码错误已排除或修复，有明确单因素假设；不要求失败旧模型先达到精度线 |
| 考虑扩大训练预算 | 声明部署路径 Moment 至少较声明 E0 改善 5%，预登记保护项最多退化 5%；另报同粗零修正、GL2/GL3、分来源、最差查询和终点证据 |
| 原生可用性与最终推广 | 同原生任务/停止条件/初值、独立原网格验收；成功率、目标质量和全成本耗时；独立来源最终评测 |

若沿用旧 E0，raw Moment 门槛为 `2.405867433300587e-7`。固定 coarse screen 有自己的同路径 E0，两类绝对值不能混用。新阶段不能只选择较差 E5 作容易超过的起点。缺少 Full-J、低保留或终点有效分母时单列“证据不足”，不能把两条 R8 外推为曲面普遍通过。

50 初值与独立来源最终验收继续保留，不阻塞本轮 P0/128 步归因；也不能省略后就宣称论文级加速。所有自动产物中的 `physics_promotion=false` 和 `long_training_promotion=false` 必须如实解释。

### 回传目录最低内容

```text
old_run_stop_receipt.json          # Work据实生成，旧任务实际停止记录
checkpoint_inventory.json        # Work据实生成，E0/E4/E5/最终身份与哈希
code_commit.txt / code_local_changes.patch
source_identity.json / source_config.json / source_checkpoint.sha256
data_audit.json
development_paths.json / train_paths.json
gradients_32.json / gradients_64.json
full_development_compare.json
experiment_decision.json          # Work据实生成，只改一个因素的决定
screen_moment/ / screen_joint/    # 仅在条件满足并实际运行后存在
```

首次回传路径和梯度报告即可，不必等所有数据补齐或整个最终验收矩阵跑完。Chat 本次只更新执行清单与核对 CLI，没有替 Work 停进程、执行生产查询或完成真实 GPU 训练。
