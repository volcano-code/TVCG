# Compact V2.1 实际训练驱动

本目录实现一个新的、可审计的实际训练驱动。它调用仓库现有的
`CompactPINOv21`、256³ 守恒材料、有限有向操作数和统一 Moment/Jacobian 响应。
它没有复刻不可见的本地 `17:56 diagnostic_training/train.py`，也没有改变主干、
默认边界带宽或十矩顺序。软件 smoke 的几何和监督均为明确的合成解析 fixture。
软件检查通过不代表真实面条通过，也不代表取得物理晋级或训练授权。

## 冻结默认配置

`TrainConfig` 默认值：Token16、seed `830251`、学习率 `3e-5`、batch size 1、
Jacobian 损失系数 1、辅助损失 0、chunk size 65536、AMP off、确定性算法开启。
优化器为 Adam (`betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`)，
当前调度器是显式保存状态的恒定 StepLR。它们是本驱动的明确实现选择，不能作为
旧 Pilot 配置完全相同的证据。

采样只实现完整遍历后的确定性 shuffle，不自动重采样、平衡来源、增加物理损失，
也不修改边界带宽。一个 batch 可以跨 epoch，原顺序和游标均保存在断点中。
学习得到的几何编码只在当前 forward 内共享；optimizer 更新后不复用。

当前真实 Fourier 参数含复数；本驱动拒绝 fp16 AMP，保留 AMP off。
独立 GradScaler 溢出测试证明了“实际 optimizer 是否调用”的审计机制，
不意味着 V2.1 模型的 GPU AMP 已得到验证。

## 数据、身份和损失

输入是 `compact_pino.training-manifest.v1`。真实清单应由同级
`p0_diagnostics.export_cases` 从冻结 Work A 原始资产导出。
每行至少包含 `record_id`、`case_path`、`case_sha256`、
`role` (`train`/`development`) 和显式 `loader_eligible`。
清单还保存 `snapshot_id`、`registered_train_count`。
可选 `moment_weight` 和 `jacobian_weight` 默认都是 1。

加载分两遍：先检查全部小 manifest 的角色、来源组、几何 ID 和几何 SHA，
通过 train/development 隔离后才打开标签。test/benchmark 明确拒绝，
包括 loader-ineligible 行；已登记查询数不会被当作实际曝光量。
真实清单还必须携带 `source_authority` 及其 portable 支持资产，绑定原冻结来源拆分、
准入 query、统一部署教师 runtime/config。每份清单只核验一次支持文件，每行在
打开标签前匹配冻结身份；legacy real 清单缺少该准入回执会拒绝。这是核验已有
Work A 冻结职责的外部声明，不是新增的独立物理认证。
所有真实行通过共享的 `verify_export_qualification`，核对打包数组与原教师完全一致；
qualified J 另要求其冻结身份已声明导数合格，并绑定几何、有限操作数、事件邻域
和外部证据。这个检查验证凭据一致性，不重新声称独立物理认证。未取得导数资格时
由 Work A 明确导出 Moment-only 行；其 Moment 仍须绑定原教师。

每件材料在读取后核对独立完整十矩，按 `m L^k` 缩放的最差分量误差必须不超过
`1e-8`。材料点通过共享 memmap 使用，不为每条 query 重复加载到内存。

Moment 与 J 的损失分母分别计算：

\[
 L_M=\frac{\sum_i a_i\;\mathrm{mean}_{10}
   \left[((\widehat M_i-M_i)/(m_iL_i^k))^2\right]}{\sum_i a_i},
\qquad
 L_J=\frac{\sum_{i\in Q}b_i\;\mathrm{mean}_{10\times D_i}
   \left[((\widehat J_i-J_i)s_i/(m_iL_i^k))^2\right]}{\sum_{i\in Q}b_i}.
\]

这里 `Q` 同时要求标签合格、响应导数存在和正损失权重；`s` 是冻结的 chart
参数尺度。总损失是 `L_M + jacobian_loss_weight * L_J`。没有合格 J 的行不进入
J 分母，缺失 J 不补零。合法完整保留/完整剔除的零 J 仍然是有效监督。
当整个 batch 没有参数依赖时，记录 `NO_PARAMETER_DEPENDENCE`，消耗一次 attempt，
保留实际 forward/loss 曝光，跳过 backward/optimizer，`global_step` 不增加。

## 软件 smoke

在仓库根目录运行；输出目录必须不存在：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.compact_pino_v21.training.smoke \
  --output /absolute/new/software-smoke
```

该命令使用真实默认模型进行两次联合 M+J 更新，并从第一步断点独立重放第二步。
因此实际执行三次 optimizer 调用，两条分支的逻辑终点都是 step 2。
`smoke_report.json` 保存 checkpoint 路径和 SHA、三组参数的有限非零梯度、实际
曝光统计，以及 model/Adam/scheduler/scaler/RNG/sampler/step 的逐值连续性比较。
当前验证范围是同一 CPU 运行环境；GPU 位级连续性必须在目标设备另行核验。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  experiments/compact_pino_v21/training/test_training.py
```

## 有预算的真实诊断与长训练

统一入口：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.compact_pino_v21.training.runner \
  --manifest /absolute/frozen/training_manifest.json \
  --output /absolute/new/trial \
  --mode trial --device cuda:0 \
  --config /absolute/frozen/train_config.json \
  --authorization /absolute/frozen/trial_authorization.json \
  --evaluation-protocol /absolute/frozen/screen_protocol.json \
  --registered-protocol /absolute/frozen/original_protocol.json \
  --max-steps 312 --max-attempts 400 \
  --checkpoint-every 104
```

以上数字只是命令格式示例，不能替代实际冻结的更新预算和准入回执。程序不会生成
训练授权，也不会自动启动任何真实训练。

确定性 CUDA 运行要求在进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`
（或 `:16:8`）；缺失时本驱动提前拒绝，并在断点记录该环境变量、设备型号和
compute capability。这里没有运行 GPU 验证，CPU smoke 不能替代目标显卡检查。

`trial`/`long` 均须有 `compact_pino.training-authorization.v1` 文件，字段包括：

- `mode`、`training_authorized: true`、非空 `authorized_by`；
- `training_manifest_sha256`、`config_sha256`、`evaluation_protocol_sha256`；
- 正整数 `max_optimizer_steps`、`max_attempts` 与正有限 `max_wall_seconds`；
- 显式 `protocol.correctness_gates` 和 `protocol.screening_thresholds`。

必须 PASS 的正确性项为教师/operand 身份、来源隔离、完整材料十矩、导数资格掩码、
有限联合更新和断点连续性。短 trial 无须预先满足“尚未训练模型的最终原生验收”，
但 wall 预算不得超过六小时。长训练额外要求单独的预算授权，以及真实 development
`research_screen` GO，绑定相同训练清单、科学配置、评估协议和将继续使用的精确
候选 checkpoint。screen 自身始终 `training_authorized=false`。

step 与 attempt 上限是包括已恢复历史的绝对上限。attempt 包括 AMP skip 和合法
无参数依赖步骤，只有实际 optimizer 调用增加 step。wall 预算累计各 runner
会话截至 checkpoint receipt 的时长，恢复时继承，不因重启清零；它在更新边界
检查，正在执行的一步可以越过边界后结束。清单加载/材料预处理发生在 runner
计时之前，不属于此训练预算；它也不是最终“从原网格到合格解”的端到端计时。
回滚到较早 checkpoint 时被舍弃的后续分支、以及崩溃中尚未写入 receipt 的时间，
不属于这条 checkpoint 时间链。工作端仍需用外部调度器执行绝对截止和总 GPU
时段上限；本驱动没有声称实现跨所有失败分支的全局计费账本。

## 评估和断点

真实训练必须在第一次更新前绑定开发协议。Epoch 0 是基线；候选只来自实际参数
更新后的开发评估。第一个候选即使比 E0 差，也会进入候选集；之后按冻结协议的
原始十矩来源组宏平均选择严格最小值，同分保留较早节点。E0 不冒充 best。
若没有一次真实更新，则没有训练候选，也不生成候选 screen bundle。

真实协议必须预声明严格递增的正整数 `evaluation_steps`，例如 `[104,208,312]`。
只有这些更新节点能进入候选集；wall 中断和不在列表中的预算终点不增加候选。
允许事前列出后续长训练节点，短 trial 只走预算内节点，扩预算时协议 SHA 保持不变。
若恢复段没有新增评估，结果明确标记为继续使用旧 best。`--evaluate-every` 如非零，
必须与已冻结节点一致；默认零表示遵从协议列表。synthetic 软件协议缺列表时才
允许间隔/末尾 fallback，这种运行不声称复刻真实选模协议。

`evaluate_development()` 另报 `mL^k` 归一化诊断误差，但不拿它替代原始主指标。
R3/R8/Full-J 或保留比例保护不足会由共享 screen 返回证据不足。
`last.json`、`best.json` 是带 SHA 的真实 checkpoint 指针；screen 导出自动调用
共享 `screen_exports.write_screen_bundle`，不手工抄写结果。

精确续训使用 `--resume /absolute/checkpoint.resume.json`。边界 checkpoint 包含：

- model、Adam、scheduler、GradScaler、Python/NumPy/CPU/CUDA RNG；
- 完整 sampler 顺序、随机发生器、epoch/cursor、global step/attempt；
- 清单、inventory、配置、评估协议、执行源码 SHA 和运行环境；
- E0/best 选择状态、累计 runner 时间与可追溯曝光 journal。

checkpoint 文件 SHA 必须与曝光 receipt 的 checkpoint SHA 一致；源码、数据、
配置、运行环境改变会拒绝“精确续训”。每步 journal 不重复写完整 sampler 排列，
完整恢复状态只进 checkpoint，避免长跑内存/日志无谓增长。

数据单因素 A/B 需要共同初始权重时，使用
`--initial-weights /absolute/common-E0.pt --initial-weights-sha256 ...`。
它严格加载本驱动 checkpoint 的 model，创建新的 optimizer/RNG/sampler/run，
与 `--resume` 互斥，明确不是旧 run 的精确续训。

配置 SHA 指完整 `TrainConfig` 对象的 canonical JSON 摘要。mode、输出目录和
执行预算不进入该科学配置，因此通过明确授权后可以从 trial 候选恢复并扩展
长训练预算；改损失、架构或采样规则仍会改变配置或源码身份。
