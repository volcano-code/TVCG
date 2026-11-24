# 复算与重新运行

## Chat 审核包

在审核 ZIP 的解压目录执行。离线复核需要 Python 3.11 或更新版本及 NumPy，读取的是包内逐条证据。

```bash
sha256sum --quiet -c SHA256SUMS
python experiments/compact_pino_v21/stratum/review_check.py
```

原始系数、切向基、预测、身份、资格及分母逐项精确核对。原型的派生误差在不同 NumPy 线性代数实现之间允许 64 倍 float64 机器精度的相对舍入差。物理汇总中两项接近零的特征值诊断，按原始二次型矩阵的 Frobenius 范数核对浮点舍入上界。5% 门槛及分组判断逐项复核。

`restricted_targets.npz` 使用 `R3_`、`R8_` 两组字段。`raw_query` 和 `raw_basis` 保留原始有序系数及切向列。通过 `moment_row_index` 和 `restricted_j_row_index` 读取实测矩阵。`restricted_event_valid` 的 1 表示相应半径内有几何证明，−1 表示未知。`native_output_triangulation_event_valid` 给出整邻域的原生输出三角剖分资格。`sampled_output_incidence_stable` 的 0 表示至少一对规定扰动出现面片对应变化，1 表示规定采样全部对应，−1 表示证据不足。文件中的多种数值和事件掩码可分别使用。

通过 `certificate_row_index`、`certificate_raw_linf_radius` 和 `certificate_domain` 可查每份证书的对象和半径。R8 的两个正值限于实际可表示的 binary64 输入，在 2⁻⁵² 半径内覆盖源分支准入及物理边界关联。更大步长的雅可比差分在独立字段中记录。

## 在原生物理档案机重算

从完整 Git checkout 执行。数据、冻结的工具链、三个原生物理库以及已发布几何档案都需要可读。本轮在 `/data/xvxinyao/TVCG_compact_v21` 验证。以下命令把新运行的结果写入独立目录。逐几何的顺序入口也可在 CPU 少于 32 个逻辑核心的机器上使用。

```bash
cd /data/xvxinyao/TVCG_compact_v21
stage=$(mktemp -d /tmp/tvcg-typed-replay.XXXXXXXX)
study="$PWD/experiments/compact_pino_v21/stratum"
align=/data/xvxinyao/TVCG-phase2-12-operator/runs/native_teacher_alignment_v1
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export TVCG_STRATUM_BUILD="$stage/build"
python "$study/native_backend/build.py" --build-dir "$TVCG_STRATUM_BUILD"
mkdir -p "$stage/replay" "$stage/certificates" "$stage/qualification"
for geom in maskB combB R01 R02; do
    "$align/run.sh" "$cpu" "$study/physical_replay/replay_registered.py" \
        --asset "$geom" --workers 1 --mode centers-first --integrator fast --out "$stage/replay"
    "$align/run.sh" "$cpu" "$study/qualification/run.py" \
        --geometry "$geom" --build "$TVCG_STRATUM_BUILD" --out "$stage/certificates"
done
python "$study/physical_replay/archive.py" --evidence "$stage/physical" package --out "$stage/replay"
bash "$study/qualification/native_admission/reproduce.sh" \
    "$stage/replay/centers" "$stage/certificates" "$stage/source_admission"
python "$study/qualification/report.py" --rows "$stage/certificates" --admissions "$stage/source_admission" \
    --centers "$stage/replay/centers" --output "$stage/qualification/neighbourhoods.json"
python -m experiments.compact_pino_v21.stratum.restricted_targets \
    --replay "$stage/physical/registered_physical_replay.json.gz" --admissions "$stage/source_admission" \
    --certificates "$stage/qualification/neighbourhoods.json" --output "$stage/targets"
```

本机 32 核心调度入口为 `bash "$study/physical_replay/run_all.sh" "$stage/replay"`。每个中心和方向均有独立指纹，中断后可以在相同目录继续。运行时间、临时文件路径会随运行改变。复核时请核对数值、原始系数、面片、物理映射、拒绝条件与资格掩码。两条 R8 的独立复算也可以仅用审核包里的输入和归档工具链执行，见 `qualification/native_admission/README.md`。

如需重算原型预测，使用父研究生成的守恒几何索引。档案机已核验的索引在 `/tmp/tvcg-v21-stage/final`。若临时目录已清理，使用父研究的 `reproduce.sh` 或 `pack_geometry.py` 从原始几何重建。

```bash
indexes=/tmp/tvcg-v21-stage/final
python -m experiments.compact_pino_v21.stratum.prototype predict \
    "$stage/replay/operands" "$indexes" "$stage/predictions"
python -m experiments.compact_pino_v21.stratum.prototype score \
    "$stage/physical/registered_physical_replay.json.gz" "$stage/predictions" "$stage/prototype.json"
"$align/run.sh" "$cpu" "$study/full_rank_control.py" "$stage/direct.json" --entry legacy_direct
"$align/run.sh" "$cpu" "$study/full_rank_control.py" "$stage/versioned.json" --entry versioned
"$align/run.sh" "$cpu" "$study/full_rank_control.py" "$stage/compare.json" --compare "$stage/direct.json" "$stage/versioned.json"
"$align/run.sh" "$cpu" "$study/event_scale_example.py" "$stage/replay" \
    "$stage/certificates/combB_R3_021.json" "$stage/scale_example.json"
```

## 使用同一接口

```python
from research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.spin_adapter.v21_stratum import (
    RawChart, SharedStratumOperator,
)

service = SharedStratumOperator.from_request(request, normalization, physical_vertices)
chart = RawChart(surface="PARABOLOID", basis=published_raw_basis)
operand = service.condition(raw_query, chart)
target = service.teacher(raw_query, chart)
solver_packet = service.solver(raw_query, chart)
```

`condition` 只返回实际切割面、封盖、操作侧及输入单位下的顶点速度。`teacher` 与有类型的 `solver` 使用同一布尔材料。无类型的 `service.solver` 可作为原生十参数 Newton 回调。`target.moment_valid`、`target.restricted_valid` 及独立证书分别表示值可用、受限导数可用以及已证明的局部范围。

若要调用带固定支持空间的原生 Newton，使用同一模块提供的 `supported_stratum_callback(service, current_chart, support_basis=dual_support)`。将同一个 `dual_support` 传给 C++ 求解器，`current_chart` 负责为每个实际请求提供当前切向基。适配器将 10×3 或 10×8 的原始切向导数转换到正交单位切向基。一般 R8 的切空间随查询变化，固定支持只覆盖通过校验的查询。真实原生入口、方程线性化与拒绝对照见[求解接口复算](solver_bridge/README.md)。
