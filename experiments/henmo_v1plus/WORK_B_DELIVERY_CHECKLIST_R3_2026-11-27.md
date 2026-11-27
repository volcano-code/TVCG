# Work B：三轮退化后的执行清单 R3

日期：2026-11-27。框架/训练/求解由Work B负责，数据来源与教师资格由Work A负责。研究分支 `research/henmo-v1plus`。

## B0. 更新与冻结

- [ ] 原819步阶段归档，不原样接长训。保留E0/273/546/819及其`.pt.json`、原始配置、快照、逐行结果。
- [ ] 保存Work本地改动，再获取本研究分支；不要reset覆盖本地原生接口。

```bash
git status --short --branch
git fetch origin
git switch research/henmo-v1plus
git pull --ff-only origin research/henmo-v1plus
python -m experiments.henmo_v1plus.cli --help
python -m pytest -q tests/research/test_henmo_*.py
```

若分支存在本地提交且不能fast-forward，保留改动并按正常合并流程集成。新诊断代码不要求覆盖Work自有原生solver。

## B1. 绑定真实源权重与配置

下面路径必须替换为Work实际位置。先选E0；第819步用另一新结果目录复算。第一阶段不需要新训练。

```bash
export HENMO_PYTHON=python
export HENMO_MANIFEST=/absolute/frozen/training_manifest.json
export HENMO_CHECKPOINT=/absolute/old_run/initial.pt
export HENMO_RESULTS=/absolute/new/henmo_r3_e0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HENMO_RESULTS"

"$HENMO_PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
import torch
path = Path(os.environ['HENMO_CHECKPOINT'])
out = Path(os.environ['HENMO_RESULTS'])
with path.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
receipt = json.loads(path.with_suffix(path.suffix+'.json').read_text())
if receipt.get('sha256') != digest:
    raise ValueError('checkpoint sidecar/hash mismatch')
state = torch.load(path, map_location='cpu', weights_only=True)
if state.get('schema') != 'henmo.training-state.v1':
    raise ValueError('expected original HENMO training checkpoint')
target = out/'source_config.json'
if target.exists():
    raise ValueError('use a new result directory')
target.write_text(json.dumps(state['architecture']['config'], indent=2)+'\n')
(out/'source_checkpoint.sha256').write_text(digest+'\n')
PY
export HENMO_SOURCE_CONFIG="$HENMO_RESULTS/source_config.json"
export HENMO_CPSHA="$(cat "$HENMO_RESULTS/source_checkpoint.sha256")"
```

这只是安全导出源配置；后续CLI仍重新校验完整checkpoint tensor、配置、manifest和snapshot binding。不能用当前仓库某个近似preset冒充源权重的实际配置。

## B2. 八路径归因：先小样本，后扩大

```bash
"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli diagnose-paths \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
  --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
  --mode trial --device cuda:0 --dtype float32 --split development \
  --selection experiments/henmo_v1plus/configs/diagnostic_development_r2.json \
  --coarse-grids 32 64 --diagnostic-nodes 128 \
  --output "$HENMO_RESULTS/development_paths.json"

"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli diagnose-paths \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
  --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
  --mode trial --device cuda:0 --dtype float32 --split train \
  --selection experiments/henmo_v1plus/configs/diagnostic_train_r2.json \
  --coarse-grids 32 64 --diagnostic-nodes 128 \
  --output "$HENMO_RESULTS/train_paths.json"
```

- 选择文件冻结本轮12条开发查询和13条训练探针；训练探针中9条有本轮联合导数记录。只对原快照有效，发生变更应另建经审查的选择文件。
- 第一次可加 `--limit 6` 控制成本，报告会记录实际身份。不要把小样本当全222开发结果。
- E0与819先跑；273/546优先只复放三条跳变查询。避免四个checkpoint各跑全部222×八路径。
- 原始预测在每个 `routes.<route>.rows`，包含教师、缩放、资格、求解状态和执行诊断。
- 默认八条路径：coarse32/64、coarse32/64_zero_correction、adaptive32/64、deterministic_gl2/gl3。
- **adaptive64/leaf64直接由细积分承担边界输出**，不据此宣称网络精度；原部署32→64的连接看adaptive32。
- 新实现跳过adaptive64/leaf64中无用的粗神经响应和切面token构造，`skipped_unused_coarse_frontier_nodes`记录该项。分类、fine积分和验收不变；一次编码尚未由这项优化省掉，不能夸大节省范围。
- 节点trace详情有上限，但汇总计数与初始frontier质量完整。父子多层计数不能当材料比例重复相加。
- GL2/GL3不运行学习编码器，属于同材料平滑参考。关闭粗学习修正并不等于GL2。
- 详细trace包含同步开销；阶段时间用于定位，不用它计算正式GPU吞吐或原生求解加速。

## B3. 同查询分项梯度

```bash
for HENMO_GRID in 32 64; do
  "$HENMO_PYTHON" -m experiments.henmo_v1plus.cli diagnose-gradients \
    --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
    --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
    --mode trial --device cuda:0 --dtype float32 \
    --selection experiments/henmo_v1plus/configs/diagnostic_train_r2.json \
    --coarse-grid "$HENMO_GRID" --jacobian-weight 1 \
    --output "$HENMO_RESULTS/gradients_${HENMO_GRID}.json"
done
```

- [ ] 核对 `qualified_derivative_objectives`、`observed_derivative_objectives`、`derivative_objective_complete`。
- [ ] 看shared_encoder/local_geometry/geometry_attention/response各组和整体M/J范数及夹角。
- [ ] 诊断只允许训练role；不会更新参数，不把描述性范数比自动转换为λ。
- [ ] E0零输出头可能使编码器梯度暂为零，不能据此判定链路失效；在头已非零的权重另测。用于校准的报告和screen源权重必须相同，不能拿819步的梯度报告去调E0起点而不登记新的实验。
- [ ] 复谱参数使用实Hermitian内积。不能用不同查询M-only与M+J的总梯度作同点冲突证据。
- [ ] 困难J先核对教师、chart、尺度、有效邻域，不能直接删除。

## B4. 根据结果选择一个修正，再运行128步对照

若路径/标签发现明确错误，先修该处再开screen；不要用screen掩盖已知错误。若已支持固定层级诊断，选择经配对比较的层级，下例64只是示例值。

```bash
export HENMO_GRID=64
for HENMO_ARM in moment joint; do
  "$HENMO_PYTHON" -m experiments.henmo_v1plus.cli screen \
    --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
    --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
    --mode trial --device cuda:0 --dtype float32 \
    --coarse-grid "$HENMO_GRID" --arm "$HENMO_ARM" \
    --steps 128 --seed 830251 --learning-rate 3e-5 \
    --evaluate-every 128 --save-every 16 \
    --output "$HENMO_RESULTS/screen_$HENMO_ARM"
done
```

两臂均为batch1、同源精确权重、同训练池、同种子、同更新预算；moment为λJ=0，joint为λJ=1。源optimizer不载入，明确作为新阶段；如出现失败/跳步，另外比较曝光与attempts，不能假设等更新就等曝光。

128步默认仅E0和末步做完整开发评估，过程每16步保存last；避免小训练预算反而被反复全222条评估主导。若使用更频繁固定探针，另记探针身份，不冒充全开发指标，不与正式性能测量并发争用GPU。

`screen` 强制训练与主要评估都使用所选coarse路径，单次最多256步。它用于定位学习目标与表示，**不代表旧adaptive正式协议的复验或长训批准**。若要继续同一screen精确恢复，使用其导出的model_config.json/training_config.json与last.pt，全部配置与原预算保持不变。

经训练侧审计后才运行calibrated臂，必须显式登记选择的正λ；命令不代替判断：

```bash
export HENMO_J_WEIGHT=REPLACE_WITH_TRAIN_SIDE_FROZEN_VALUE
"$HENMO_PYTHON" -m experiments.henmo_v1plus.cli screen \
  --manifest "$HENMO_MANIFEST" --config "$HENMO_SOURCE_CONFIG" \
  --checkpoint "$HENMO_CHECKPOINT" --checkpoint-sha256 "$HENMO_CPSHA" \
  --mode trial --device cuda:0 --dtype float32 \
  --coarse-grid "$HENMO_GRID" --arm calibrated --jacobian-weight "$HENMO_J_WEIGHT" \
  --calibration-report "$HENMO_RESULTS/gradients_${HENMO_GRID}.json" \
  --steps 128 --seed 830251 --learning-rate 3e-5 \
  --evaluate-every 128 --save-every 16 \
  --output "$HENMO_RESULTS/screen_calibrated"
```

校准报告必须来自相同源checkpoint、snapshot、manifest和grid，且合格导数完整、存在联合梯度、无参数更新。不接受开发侧梯度选择λ。先不同时改学习率、采样分布、响应头或材料。

### 每个screen结束后

- [ ] 保留stage.json、model_config.json、training_config.json、initial/last/selected/step权重及sidecar、steps.jsonl、全部评估。
- [ ] 如果selected_has_new_updates=false，说明仍选中阶段起始权重，不能写“新模型晋级”。
- [ ] 使用该screen的 **model_config.json** 与 **last或selected.pt及其新SHA** 再运行B2。旧源config不能用来加载新阶段checkpoint。
- [ ] 同时报固定训练探针、旧222完整开发与新增轨迹诊断；固定coarse和原部署adaptive32结果分别标记。
- [ ] 如M-only也不能改善固定训练探针，再做局部表示能力/教师检查；如训练改善开发退化，优先数据泛化。

## B5. 训练效率与实际求解

- [ ] 新静态输入缓存默认256 MiB/模型，统计位于efficiency_statistics.static_input_cache；可构造cache0作为对照。不得跨参数更新复用learned编码。
- [ ] 分清静态准备、层级构建、编码、查询、反向、optimizer、审计/日志/保存/评估。stages_host_s是Host计时，不是GPU kernel计时。
- [ ] 不与另一训练/评估进程争用GPU后宣称吞吐变化；同步profiling与稳定吞吐分别记录。
- [ ] 验证缓存数值等价后，用同查询/配置测真实5090D收益，不将CPU局部准备倍率当总体倍率。
- [ ] 新CLI默认每16步保存last，评估前保存last及step快照；被中断评估不应再迫使重做整段训练。
- [ ] 对原生成功轨迹末段及候选终点，分别替换M或J检查验收/下一步方向。终点驻点公式只读M的部分与迭代路径用J的部分分开归因。
- [ ] 只有在原网格合格解与完整成本下比较旧最佳V2、确定性路线和HENMO。失败/校正/回退全部计入；保留原任务目标，不擅自改成另一个优化问题。

## B6. 回传与通过口径

| 回传 | 必需内容 |
|---|---|
| 第一次 | 八路径与梯度JSON、源权重/配置/快照、异常及正常对照、实际CPU/GPU环境 |
| 第二次 | 128步配对原始产物、所有失败、原部署路径复评、来源/保留/曲面/阶段保护项 |
| 第三次 | 新轨迹同点M/J归因、原网格验收、真实求解成功及全成本计时 |

原正式路径仍检查来源宏平均raw Moment改善至少5%及预登记保护项最多+5%；缺分母写证据不足。另比较同粗零修正及确定性基线。screen和诊断报告均不自动签发长训资格。

最终独立来源/50初值/同质量全成本评测继续保留，但不阻塞本轮P0和128步归因。当前真实材料、原生切割和GPU未在Chat环境执行，Work必须回传实际结果，不能把软件测试数量当精度/速度证明。
