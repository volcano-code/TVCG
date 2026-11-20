# 在 5090 D 上复现 Phase 2.11

从本次提交的仓库根目录运行。实测环境为 Linux、Python 3.13.5、PyTorch 2.8.0+cu128、CUDA runtime 12.8、NVIDIA 驱动 580.178.04、torch-scatter 2.1.2+pt28cu128、torch-geometric 2.8.1、NumPy 2.1.3 和 SciPy 1.15.3。Work 的几何与标签存放在本机数据目录。

## A · 确定性等价测试

```bash
cd /data/xvxinyao/TVCG
D=research/NeuralMoments/models/pcraft_ufno/candidate_projects/quadrann_reproduction
export PYTHONPATH="$PWD:$PWD/$D"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q "$D/spin_adapter/tests/test_geometry_physics_attention_analytic.py" \
  "$D/spin_adapter/tests/test_phase211_encoder.py" \
  "$D/spin_adapter/tests/test_batched_moment_operator.py" \
  "$D/spin_adapter/tests/test_geometry_physics_attention_chunked.py" \
  "$D/spin_adapter/tests/test_quadrann_chunked_gnn.py" \
  "$D/spin_adapter/tests/test_unified_m80_sqno.py" \
  "$D/training_v13/tests" \
  research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/test_transolver_chunked.py
```

下面的跨提交比较从原提交导出独立的实现，用相同随机种子、非零 attention 输出头和同一张小几何产生对照结果。物理量和查询为 FP64，神经网络沿用 Full V1 的训练精度。测试同时检查直接对融合 Moment 求出的 raw-coefficient Jacobian、各分支输出、所有活跃参数梯度，以及从一个注意力权重方向传播到全网络的混合二阶导数。算子级 FP64 测试另外覆盖所有注意力参数的 Hessian 向量积。

```bash
REFERENCE_DIR=$(mktemp -d)
git archive a21a200075a6703f5c7adef43058af43457a6aa5 research/NeuralMoments/models/pcraft_ufno \
  | tar -xf - -C "$REFERENCE_DIR"
PYTHONPATH="$REFERENCE_DIR" python docs/phase2_11/compare_revisions.py --record "$REFERENCE_DIR/reference.pt"
PYTHONPATH="$PWD" python docs/phase2_11/compare_revisions.py --compare "$REFERENCE_DIR/reference.pt"
```

跨提交的逐元素容差为 `atol=5e-7, rtol=2e-4`。旧版融合 LayerNorm 的直接高阶自动微分在该测试中与旧模型一阶梯度的中心差分存在差异。对于选取的最后一层权重方向及线性测试目标，其他参数的一阶梯度沿该方向为仿射函数，因此以两端梯度差作为独立参考。稠密基础算子 LayerNorm 与中心差分在注意力的独立测试中另行交叉验证。

## B、C · 一次更新、保存和重建

```bash
cd /data/xvxinyao/TVCG
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MANIFEST=datasets/phase2_10_4_work_delivery/manifest.json
sha256sum "$MANIFEST"
python -m research.NeuralMoments.models.pcraft_ufno.candidate_projects.quadrann_reproduction.training_v13.preflight_unified \
  --manifest "$MANIFEST" --allow-absolute-assets --geometry-id N002 \
  --device cuda --config full_v1_candidate --grid-size 64 --seed 830251 \
  --smoke-update --output runs/phase2_11/reproduction/checkpoint.pt \
  --memory-profile runs/phase2_11/reproduction/memory_profile.json
```

Manifest SHA-256 为 `a0f914df030228b0f7fc930141b97968238250e43dd671e6f64e88ed882e42ee`。`--grid-size 64` 选择 P-CRAFT 原有的辅助上下文网格。原生 RLE 的分辨率和每条区间由几何资产决定，N002 的 1,177,764 条全部参与训练，每条保留二阶 Gauss 规则的 8 个节点。先对 31 个训练记录的 2,019 条有效 Moment 标签计算 RMS，再哈希校验所选几何及其查询标签。

进程成功退出后，同一目录有 checkpoint、checkpoint.report.json、memory_profile.json 与 memory_profile.md。报告逐项列出训练使用的标签、参数变化、旧缓存拒用、全批次恢复输出差异。报告中 optimizer step 为 1。AdamW、调度器、模型状态、数据指纹、训练契约和随机数状态随 checkpoint 保存。恢复时使用相同提交的源码。

## 解析导数的范围

单头令 `x` 表示原始十维系数，`u=x/||x||`，`p=Wq Encoder(u)`。区间的 key、value 和 log-mass prior 对 x 为常数，对几何编码器参数可微。定义每个区间的十维分数导数 `t_i=(∂p/∂x)ᵀk_i/√d`。同一个全局 softmax 同时给出 `E[v]`、`E[t]`、`E[vtᵀ]`，因此上下文导数为 `E[vtᵀ]−E[v]E[t]ᵀ`。每个 tile 合并时按新的运行最大分数重缩放旧累计量。所有区间共享同一个归一化常数。

完整导数还经过系数归一化、线性层、GELU、多个 head 的拼接，以及直接送到输出头的 query token。对于 LayerNorm 的标准化向量 z 和切向量 T，标准化切向量为 `(T−mean(T)−z·mean(zT))/sqrt(var+eps)`，再乘通道权重。输出 MLP 和最终缩放参与相同链式计算。流式算子的反向逐 tile 重建 key/value 和全局 softmax 权重，并返回输入和投影权重的伴随梯度。

物理 Moment 算子利用每个积分节点 Hessian 的两个秩一项及单位阵项，在节点维先收缩，只在查询维保留最终 10×10 矩阵。Cross-scale 分支在每个 query 的所有可用尺度上归一化，保留逐输出通道的自动微分；最多 64 个独立 query 共用一次通道反传。M80 和融合 gate 同样保留其原有 raw-coefficient 导数。

## 回归范围

上面的聚焦测试在目标机有 CUDA 时通过 92 项。将 CUDA 隐藏后通过 91 项，跳过 1 项 CPU/CUDA 对照。扩展 CPU 回归覆盖整个 spin_adapter/tests、training_v13/tests、test_native_pcraft.py 与 test_transolver_chunked.py，371 项通过、6 项跳过、1 项失败。失败项为 Neural Newton 的 `test_exact_gate_accepts_only_feasible_decreasing_trial`，相同测试在基线提交 a21a200 也失败。此阶段保留其原有接受门行为。[数值对照](equivalence_report.json)
