# Compact PINO V2.2 算子效率优化：Work 端交接清单

## 交接范围

本次代码完成五项算子执行优化：静态编码输入缓存、同权重评估编码复用、token
特征预投影、同一局部响应的解析斜率、冻结查询的几何准备缓存。

继续使用 256³ 守恒材料、共享 32³／64³ P-CRAFT/U-FNO、Token16 默认及单一物理响应。
参数名称与形状保持兼容。本轮没有更换教师、扩大网络、调整边界带宽或加入新的物理损失。
原有数据准入、导数资格和研究筛选门槛继续适用。

本清单直接承接已确认的研发安排。配置哈希与运行记录用于绑定实验，不要求再次批准同一项工作。

## 1. 接入已发布代码

- [ ] 使用发布分支 `research/compact-pino-v22-efficiency`，核对 Chat 回传的提交号。
- [ ] 保留旧 Pilot 的代码、权重、快照和输出，采用独立工作树执行新对照。
- [ ] 记录 Python、PyTorch、CUDA、GPU、驱动、CPU 线程数与可用内存。
- [ ] 默认保持 `amp=off`。当前 Fourier 复数参数的 FP16 GradScaler 路径尚未获准。

可在现有 TVCG 仓库中获取该分支，随后按本机路径建立独立工作树：

```bash
git fetch origin refs/heads/research/compact-pino-v22-efficiency:refs/remotes/origin/research/compact-pino-v22-efficiency
git worktree add --detach /data/xvxinyao/TVCG_v22_efficiency refs/remotes/origin/research/compact-pino-v22-efficiency
cd /data/xvxinyao/TVCG_v22_efficiency
git rev-parse HEAD
```

以上新目录必须尚不存在；使用既有 Python 环境执行，不在原 Pilot 中途替换源码。

## 2. 先运行 GPU 合成性能检查

该命令使用真实默认 Token16 编码器和两条解析几何查询。两臂载入同一初始权重，先核对
Moment、Jacobian、全部参数梯度，再计时。它不调用优化器、不代表真实几何的预测质量。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.efficiency.benchmark \
  --mode smoke \
  --device cuda:0 \
  --repeats 3 \
  --warmup 1 \
  --output runs/v22_efficiency/gpu_smoke
```

- [ ] `benchmark_report.json` 的数值／参数梯度一致性为 PASS。
- [ ] 两臂使用同一权重摘要、同一查询和同一运行环境。
- [ ] 分别记录缓存冷单查询、重复冻结查询评估、Moment＋Jacobian 前后向耗时。
- [ ] 核对 CUDA 同步计时、显存 allocated 峰值及缓存命中／保留字节。
- [ ] 对没有改善或变慢的场景如实报告，不只选择热缓存结果。

缓存冷单查询已经预热运行时和数值内核，不能解释为从读入原网格开始的冷启动。
重复冻结查询不等同于不断改变切割面的 Newton 轨迹。三次重复所得 P95 仅作样本描述。

## 3. 确认目标 GPU 上能实际更新

上一步生成的解析样本可直接用于两次真实更新检查：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.training.runner \
  --manifest runs/v22_efficiency/gpu_smoke/assets/training_manifest.json \
  --output runs/v22_efficiency/gpu_updates \
  --mode smoke \
  --device cuda:0 \
  --max-steps 2 \
  --max-attempts 4 \
  --checkpoint-every 1
```

- [ ] `global_step=2`，记录实际提交的 optimizer step，不用 forward 数替代。
- [ ] Moment、Jacobian 的有效损失分母均为正。
- [ ] 编码器、attention、局部响应参数均有有限且非零的梯度。
- [ ] 在同版本环境内完成断点恢复核对，再扩大更新预算。

Chat 已检查本版本 CPU 的逐位一致恢复；GPU 的一致性需要在该 GPU 环境中确认。
旧执行版本的 checkpoint 不应跨源码变更做严格恢复。相同参数布局的权重可以作为新运行的初始化；
`--initial-weights` 接口仅接受现有 runner 自己的 checkpoint 格式，私有包装格式须先明确映射。

## 4. 冻结真实训练查询的性能对照

使用 Work A 已通过教师、来源拆分和导数资格准入的 `training_manifest.json`。
下例中的绝对路径由 Work 替换为真实冻结快照：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m experiments.compact_pino_v21.efficiency.benchmark \
  --mode trial \
  --manifest /abs/frozen/training_manifest.json \
  --device cuda:0 \
  --repeats 3 \
  --warmup 1 \
  --output runs/v22_efficiency/frozen_training_queries
```

默认选择同一几何／固定材料上下文中的前两条合格、正权重导数训练查询；也可用两次
`--record-id` 预先指定。报告会记下其身份与导数维数。要跨来源比较，应预先指定各来源的
查询对并分别运行，不能把一对样本推广为所有几何。

- [ ] 先用相同初始权重比对计算一致性，已有兼容 checkpoint 时可以指定
  `--checkpoint` 和 `--checkpoint-sha256`。
- [ ] 分别覆盖边界活跃点少／多、常见曲面以及大切割面。
- [ ] 每一对比均记录来源、材料身份、切割配置、导数类型、活跃点与缓存统计。
- [ ] 如缓存频繁超额或未命中，在相同查询上单独比较缓存预算；不同时改变模型或损失。

默认静态输入缓存 128 MiB、查询缓存 128 MiB、单次评估学习特征缓存 64 MiB。
这些数值限制缓存持有的载荷，不是进程／GPU 峰值内存硬上限；候选构建、编码激活和流式分块另计。

对于每步都产生新切割面的 Newton 路径，可以设 `query_cache_bytes=0`，沿用流式准备，
同时保留静态输入缓存、token 预投影和解析斜率。固定权重时继续复用同一几何编码。
不要把冻结查询的缓存收益计作新切割面的实际收益。

## 5. 接入下一轮正式实验

- [ ] 新配置中记录 `efficiency`、`static_cache_bytes`、`query_cache_bytes`、
  `evaluation_cache_bytes`，重新冻结配置摘要及已授权预算的绑定记录。
- [ ] 保持数据顺序、更新预算、采样、损失和选模规则一致，先比较本轮执行优化本身。
- [ ] 不把缓存命中率、CPU 软件测试或 GPU 合成测试作为长训晋级证据。
- [ ] 继续真实同点 Moment／Jacobian 归因、确定性积分基线和原生最终验收。
- [ ] 若有具体精度失效或 GPU 瓶颈，附最小可复现查询和记录，再决定是否触发下一项模型修改。

## 应回传的文件

1. Git 提交号、环境记录、实际完整命令。
2. 合成及真实查询的 `benchmark_report.json`，包括所有逐次耗时。
3. GPU 更新的 `run.json`、`config.json`、`steps.jsonl`、`exposure.jsonl`、`result.json`。
4. 恢复核对记录及 checkpoint 摘要；必要时提供对应权重。
5. 真实查询身份、缓存统计、显存峰值、失败或变慢案例。

本次交接后无需等待 Chat 再补齐这五项代码。Work 可以直接进行目标硬件验证和既定的
真实数据实验；真实终点精度、原生求解成功率及取得合格解的总耗时仍按原协议单独裁决。
