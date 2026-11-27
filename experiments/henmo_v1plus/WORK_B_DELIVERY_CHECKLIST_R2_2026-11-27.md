# Work B 交付清单 R2：HENMO V1+ 表示诊断、训练与原生求解

版本日期：2026-11-27。研究分支：`research/henmo-v1plus`。本轮已发布的实现基准为 `cd00343b71e63987d38640781aeb6b7876564d05`；运行时另记实际代码 commit 与文档修订 commit。

**Work B 负责框架、诊断工具、训练、性能和求解闭环；Work A 负责材料、原生教师、来源隔离与标签资格。** 本轮冻结主干，先回答三个问题：当前粗原子表示能否容纳真实十矩；网络是否比同输入无学习方法更好；这种改善能否降低取得一个原网格合格解的总耗时。

本清单不要求初始网络先达到训练后精度才准短训。数据与软件正确性通过即可开始有范围声明的短 Pilot；表示诊断和强基线建设并行进行。发现已证实的实现错误或表示瓶颈时，停止受影响配置的扩大投入。

## 一、先分清现有入口与本轮待完成工作

| 项目 | 目前状态 | Work B 责任 |
|---|---|---|
| `config / smoke / audit-data / train / compare` | 已有 CLI | 使用真实准入快照及目标 GPU 验证 |
| CPU synthetic smoke、同源 J、训练恢复测试 | 已有实现和软件回归 | 不把 CPU/合成结果写成真实物理或 GPU 结论 |
| `compare` 的 GL2/GL3、初始/训练后模型对照 | 已有 | 解释它们的不同执行尺度，补分层指标 |
| bounded-retention oracle、同粗规则零修正专用消融 | **待补实验工具** | 实现、冻结输入、提供可复算解与最优性凭据 |
| 强确定性局部积分的精度—耗时曲线 | **待补对照** | 采用同有限切割语义，验证数值收敛，不只比较固定两点平滑规则 |
| 同点 M/J 四组合、真实 GPU 两步恢复、独立来源求解 | 接口可复用，实际实验待运行 | 使用 Work 原生后端与冻结求解器，保存原始结果 |
| 原生校正、独立验收、完整计时 | 已有部署桥接 | 按 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) 接入真实 backend/solver |
| GPU 并行完整多起点、新流形 Gauss–Newton | **未实现、非本轮前置条件** | 先验证单起点收益，再由剖析决定是否开发 |

不要编造 `cli oracle`、`smoke --device` 或“已完成强基线”的命令和结论。以下新报告文件名是要求 Work B 产出的交付约定，除另行注明外，不是现有 CLI 自动生成的产物。

## 二、B0：冻结配置与短 Pilot 准入

- [ ] 新运行目录；冻结原始与选中权重、manifest、材料/教师/查询哈希、模型与训练配置、软件环境、设备、原生版本。旧模型和旧报告保留为对照。
- [ ] 先用合格 native256，保持默认 `context_grids=[64,128]`、`coarse_grid=32`、`leaf_grid=64`、目标压缩至每粗块最多 32 个正权原子、`compact_warp`、有限切割 attention。压缩不达标时保留确定性回退，报告实际原子数，不能将目标当作全路径硬上限。native512/1024 各用单独材料和配置；材料分辨率、编码网格、查询层级不能混为一项。
- [ ] Work A 完成真实模式材料/教师审计，完整材料及固定壳十矩通过既定 `1e-8` 门槛。每条查询明确 M 与 R3/R8/Full-J 资格；未获资格 J 不填零。
- [ ] 检查十矩顺序、固定材料只加一次、有限操作数端盖与保留侧、原始系数到 solver chart 的转换。先用全原生回调复现一个已知成功任务。
- [ ] 目标 GPU 使用真实数据完成两次 M+J 更新；保存第一步后恢复，再与不中断的第二步比较权重、Adam 状态、采样/随机状态、M/J 和损失。使用独立两步 preflight 配置及同一完整预算，不能拿 CPU `smoke` 冒充此项。
- [ ] 检查 M-only、合格 J-only 与联合损失的梯度路径。零输出头初始化可能使编码器第一步梯度为零；在头已非零后再验证编码器和局部响应收到导数监督，不能仅检查 loss 数值有限。

缺 R8 或 Full-J 可以限制在 M/R3 等已验证范围做短 Pilot，但不能签发曲面导数、Full-J 或完整原生 Newton 的推广资格。身份错配、错误标签、NaN、错误求导链或恢复不一致属于需要先修复的阻断项。

## 三、B1：优先补“表示能力”诊断，不先盲目扩主干

### 3.1 固定问题，再定义 oracle

对 Work A 冻结的训练来源末段/终点、薄壁/空腔、曲面及已登记诊断点，先分别固定纯粗 `32/64` 前沿、材料、原子、支撑宽度、有限操作数和保留侧。不要一边拟合一边改变材料或前沿。

定义：

\[
M(r)=M_{\mathrm{const}}+\Phi r,\qquad
\Phi_{i\ell}=w_\ell\phi_i(x_\ell),\qquad
b=M_{\mathrm{native}}-M_{\mathrm{const}}.
\]

`M_const` 包含固定壳、已证明完整保留的节点、当前 support 外 `r=1` 的饱和原子及其他固定执行贡献。support 外 `r=0/1` 的原子不能悄悄释放。以完整可编辑质量和几何长度构成 `S=diag(m L^k)`，其中十个分量阶数为 `0,1,1,1,2,2,2,2,2,2`。

优先用与最差分量门槛一致的有界 minimax 问题：

\[
\min_{r,t}t,\qquad
\ell\le r\le u,\quad t\ge0,\quad
-t\mathbf1\le S^{-1}(\Phi r-b)\le t\mathbf1.
\]

- [ ] 第一层用活动原子的 `0≤r≤1` **乐观放松**；如需放开饱和原子，另建明确标记的更松对照，不能冒充当前执行路径。
- [ ] 第二层按需增加 `compact_warp` 的 profile 可达区间。其幅度受 `2*tanh` 限制，自由 `[0,1]` 拟合好不等于该响应头一定可达；profile 区间拟合好也不等于共享网络可学。
- [ ] 归档矩阵/目标或可重建输入、可行解、solver/status/tolerance、边界违反量、原始与缩放分量残差、原/对偶目标和 gap；独立重算最差分量误差。
- [ ] **可行解残差是最优误差的上界。** 只有可信对偶下界，或由经校验的最优性 gap 得到的下界，超过预先冻结的阈值，才支持该放松表示不足；判定还需计入浮点与教师不确定性。小 KKT 残差与偏大 primal 值本身不当作认证下限。迭代耗尽、数值失败、证书不足记为 `INCONCLUSIVE`，不能据此否决架构。
- [ ] 若使用有界 least-squares，报告其自己的范数和最优性凭据；不能把 RMSE/二范数直接当成最差分量门槛。教师自身数值不确定性和证书浮点容差一并报告。

### 3.2 怎样根据结果行动

| 观察 | 支持的解释 | 下一步 |
|---|---|---|
| free-r 已有可信误差下界，超过冻结预算 | 该材料、原子及冻结粗执行状态不足 | 与未压缩同材料规则、更多原子/更细前沿比较，定位材料误差或压缩误差；暂停该粗配置扩大训练 |
| free-r 可拟合，profile-bounded 不够 | 可能受当前局部响应可达范围限制 | 只改响应参数化/支撑，固定其余因素做短对照 |
| profile 放松可拟合，网络明显较差 | 仍可能是可学性、优化或数据覆盖问题 | 先检验小训练子集可拟合性、梯度/损失尺度，再定向补数据 |
| 网络和无学习细规则在同点都有偏差 | 共用材料、带宽或有限切割处理值得检查 | 做同网格材料与确定性积分收敛对照，不先扩大编码器 |
| Oracle 未收敛或无有效证书 | 尚不能定位表达下限 | 修正数值诊断，保留 `INCONCLUSIVE`；不虚构通过/失败 |

这些结论只针对冻结的粗执行状态。若冻结 adaptive 前沿/混合系数求 oracle，必须一并保存它们，不能以该状态的下界否定允许继续细化的完整架构。全局十矩拟合还可能出现节点间抵消，不能推出每个节点局部积分正确。

逐查询自由选择保留率不证明存在光滑共享函数 `r(c)`，不证明 J/Jv 正确，也不证明该保留率分布对应真实切割。Oracle 不能作为部署输入或推理加速结果。

对应阈值 `tau_M` 应由训练侧原生末段误差敏感性在看候选前冻结；若尚无依据，先报告误差曲线和参考线，不事后发明绝对门槛。配置里的 `moment_tolerance=0.001`、`jacobian_tolerance=0.05` 是当前细化指示参数，**不是经过证明的真实误差上界，也不能代替该标定。**

## 四、B2：同点归因与真正同输入的基线

### 4.1 保留 Moment / Jacobian 四组合

- [ ] 固定同一点、任务状态、乘子、操作数、坐标、合法 chart 和完整 solver 设置，运行：原生 M/原生 J；预测 M/原生 J；原生 M/预测 J；预测 M/预测 J。
- [ ] 先全原生复现。分别记录十矩分量误差、有效 J/Jv、下一步方向/步长、原生线搜索、驻点与约束残差。
- [ ] 若固定点验收公式只读 M，同 M 的两组终点残差应一致；J 的影响通过下一步和后续轨迹检验。不能再把这种终点拒绝一概归为 J 错误。
- [ ] R3/R8 在各自固定秩 chart 检查；Full-J 在合格十维邻域检查。端盖、顶点运动、坐标/系数缩放、attention 和混合门控的链式项均须纳入同源 J。
- [ ] 分开记录人为粗细切换处的连续性和真实切割事件；后者可能没有有效导数，不能强迫全部点 C1 或把有限差分失败统一算成网络错误。

### 4.2 最小对照矩阵

| 对照臂 | 用途与要求 | 入口状态 |
|---|---|---|
| 原生原网格 | 同一有限切割的教师及最终求解基准 | Work 原有 backend/solver |
| fine GL2 / GL3 平滑响应 | 判断同材料细规则与平滑误差 | 现有 `compare` |
| 同粗 atoms/support 的零修正响应 | 分离粗表示和学习贡献，省去无用网络开销 | 待补专用对照；不能直接用改变执行尺度的 deterministic 模式冒充 |
| HENMO E0 | 检查零初始化数值和网络壳成本 | 现有 `compare` 不指定 checkpoint |
| HENMO 训练后 | 学习净收益、有效导数与真实耗时 | 现有 `compare` 指定绑定同快照的 checkpoint |
| 已归档 E16/V2.4 | 保留历史性能/精度参照 | 用其真实冻结配置，标明输入和语义是否可比 |
| 可控精度的确定性局部几何积分 | 论文强基线，比较同质量的速度，而非击败弱规则 | 本轮另建；与短 Pilot 并行 |

`response_zero_init` 只将最后输出层置零，编码器与 attention 仍有计算成本。同 atoms/support/execution 下，E0 与零修正粗响应应可数值复核一致；不能据其耗时差宣称学习改善精度。现有 `response_path=deterministic` 使用 fine/native-width，不是纯粹关掉同粗网络。

强确定性方法必须正确处理有限面、端盖、孔洞与材料分区。改变积分阶数、细化深度、支撑宽度时分别归档，画误差—耗时曲线；只验证平滑积分收敛不等于已对原网格硬切割收敛。无学习臂不承担不需要的神经编码成本。

为核对方法贡献，后续至少做“同粗规则无学习/有学习”“仅局部几何/加入全局编码和切割 attention”“固定粗级/自适应细化”三组单因素消融。完整论文消融无需先于首个短 Pilot 完成。

## 五、B3：首轮训练配置与现有命令

| 项目 | 首轮冻结值 |
|---|---|
| 材料与编码 | 合格 native256；64³+128³ 上下文；512/1024 分臂 |
| 优化器 | Adam，学习率 `3e-5`，seed `830251` |
| Batch / 预算 | query batch 1；312 次实际更新，最大 624 次 attempts |
| 损失 | M/J 权重各 1；残差/约束辅助权重 0；资格掩码与物理尺度一致 |
| 精度 | FP32 网络、FP64 十矩聚合；首轮 AMP 关闭 |
| 执行 | 训练每条查询选 32/64 粗层级；开发评估用冻结自适应配置 |
| 评估/选模 | 每 104 次更新；原始十矩来源宏平均 RMSE，含起始 incumbent |

这些是有界诊断训练配置，不是对长训练 epoch 数的承诺。训练/原子预算、教师曝光量、真实更新与 attempts 都应计数。若 624 次 attempts 用尽仍不足 312 次更新，报告实际数及拒绝原因，不能写成完整 Pilot 已通过或静默增加预算。不能以未变化的 epoch 标签掩盖数据集大小或曝光改变。

在仓库根目录执行，替换示例绝对路径；每次使用新输出目录：

```bash
python -m experiments.henmo_v1plus.cli smoke \
  --output /absolute/new/henmo_cpu_smoke

python -m experiments.henmo_v1plus.cli audit-data \
  --manifest /absolute/frozen/training_manifest.json \
  --config experiments/henmo_v1plus/configs/pilot256.json \
  --mode trial --output /absolute/new/henmo_data_audit.json

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.henmo_v1plus.cli compare \
  --manifest /absolute/frozen/training_manifest.json \
  --config experiments/henmo_v1plus/configs/pilot256.json \
  --mode trial --device cuda:0 --dtype float32 \
  --output /absolute/new/henmo_e0_comparison.json

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.henmo_v1plus.cli train \
  --manifest /absolute/frozen/training_manifest.json \
  --config experiments/henmo_v1plus/configs/pilot256.json \
  --training-config experiments/henmo_v1plus/configs/training_pilot.json \
  --mode trial --device cuda:0 --dtype float32 \
  --evaluate-every 104 --output /absolute/new/henmo_pilot

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m experiments.henmo_v1plus.cli compare \
  --manifest /absolute/frozen/training_manifest.json \
  --config experiments/henmo_v1plus/configs/pilot256.json \
  --checkpoint /absolute/new/henmo_pilot/selected.pt \
  --checkpoint-sha256 REPLACE_WITH_FROZEN_CHECKPOINT_SHA256 \
  --mode trial --device cuda:0 --dtype float32 \
  --output /absolute/new/henmo_selected_comparison.json
```

`smoke` 是 CPU synthetic；上述真实 GPU 两步 preflight 应基于 trainer 另组织，不能加不存在的 `--device`。含复数谱权重的模型不可直接整体 `.to(dtype=实数类型)`；用 CLI 的设备/精度路径。

### 恢复与迁移

- [ ] 精确恢复使用 `--resume`、`--resume-sha256`，保留 checkpoint 同名 `.pt.json`；模型、数据 binding、dtype 和**全部**训练配置不变，包括 `max_steps/max_attempts`。保存第一步和不中断对照用于真实恢复检查。
- [ ] 已完成 312 步不会因 resume 自动追加预算。扩大预算或更换快照，另冻结阶段配置，通过 `--initialize-from / --initialize-sha256 / --initialize-state-key / --transfer-rules` 显式语义迁移，新建 optimizer；按 [WORK_B_ACTION_GUIDE.md](WORK_B_ACTION_GUIDE.md) 的真实入口操作。
- [ ] E16 只迁移语义、维度和规范均匹配的模块；新边界头重新初始化。不能仅因形状相同就加载，也不能加载旧 optimizer 后称无缝续训。
- [ ] 记录 `selected_step`、`selected_has_new_updates`、`selection_scope`。选中起始 incumbent 不等于本轮学出新改进；这套 HENMO 规则不同于旧 V2.1 只选训练后节点。
- [ ] `--mode long` 既不自动扩大预算，也不是晋级许可。

## 六、B4：指标、保护线与训练决策

保留历史工程筛选：相对本轮冻结 E0，**来源宏平均原始十矩 RMSE 至少下降 5%**；预登记的中/高保留、R3/R8/Full-J 等保护项误差至多为其 E0 的 `1.05` 倍。小分母、近零基准的处理必须事先约定，不能看到结果后调整口径。5% 不等于物理准确率或统计显著性。

- [ ] 同时报告 `source_macro_raw_moment_rmse` 和 `source_macro_scaled_moment_rmse`，主指标保持前者；不能互换后称延用协议。
- [ ] 当前汇总的 `qualified_jacobian` 是合格行的 pooled 描述统计，**不是来源宏平均 J**。来源宏平均 J、保留档与轨迹阶段保护项需从逐行结果另做派生分析，冻结公式并保存分母。
- [ ] 失败查询不从主指标删掉；标签合格但预测 J 缺失计为不完整。缺分母写 `INSUFFICIENT_EVIDENCE`，不能写“通过保护”。
- [ ] 逐来源、逐矩分量、最差/P95/P99、中高保留末段/终点、R3/R8/Full-J 和真实 Newton `Jv` 单列，避免平均值掩盖某来源或终点退化。
- [ ] 与同粗零修正、fine GL2/GL3 和强确定性方法分别比较；对准同样的精度与原生验收，评价学习的净收益。
- [ ] 第一次诊断仅改变一个因素。先定位表示/数据/梯度/边界偏差，再选择采样、响应或损失；不同时换材料、主干、数据和优化器。

| 关口 | 可做的事 | 需要的证据 |
|---|---|---|
| G0：短 Pilot | 运行限定范围的 312 步诊断 | 真实数据身份与资格、软件/求导正确、GPU 更新及恢复通过；无需完整论文消融或最终加速 |
| G1：扩大训练预算 | 冻结下一阶段有界训练 | Pilot 主指标及有分母的保护线通过；梯度有效；无已确证且未修复的表示瓶颈；终点/导数诊断与剖析支持明确收益路径 |
| G2：正式大预算长训 | 按冻结阶段预算继续 | G1 之外，在已声明求解范围取得原网格合格解，网络相对同输入基线有可信收益；预算、停止/回退规则与未覆盖范围明确 |
| G3：部署/论文加速声明 | 推广到独立来源、报告加速 | 封存来源、同条件完整求解、合格率/目标/完整时间及强对照全部完成 |

G1/G2 不是 `train` 自动生成的结论。若某项只有小样本趋势，就写出不确定性与本次预算上限，不宣称已证明泛化。最终 50 种子、完整消融和独立来源统计是 G3 证据，不应反过来阻塞 G0。若网络最终不优于确定性路线，应保留最快的合格方案并如实调整论文主张。

## 七、B5：效率剖析和原生闭环

- [ ] 分别计材料构建/读取、树构建、一次编码、逐次原生构面、有限面准备、树分类、打包/传输、响应/J、细化、校正和验收；计所有失败尝试。
- [ ] 保存粗块数、evaluated/active atoms、网络调用与 padding、refined nodes、deterministic leaf 数、压缩回退和峰值 CPU/GPU 内存。若总是细化至 fine，就先定位粗响应与指示阈值。
- [ ] 区分 CUDA 同步剖析与正式吞吐；低显存本身不能证明提高 batch 会变快。新 batch 对照分别控制等曝光与等更新预算，完整报告另一项变化。
- [ ] `compare` 读取预导出 operand，包含有限面准备及 M/J 回传，**不等于从原始系数重新构面的完整部署计时**；数据准入审计时间也不是纯推理时间。
- [ ] 按部署指南接真实 backend 和 `RegisteredSolver`；注册清单要独立核对，不能只把草稿 status 改为 ADMITTED。单次候选返回 CONVERGED 后仍做原网格验收。
- [ ] 直接替代与“神经提案＋原生校正”分别报告。保持原任务 `INERTIA_AND_LOWERING`、约束、初值、完整设置和验收，不擅自改成质量最小化；不承诺固定 2–4 步校正。
- [ ] 原生基准、确定性混合、HENMO 混合共用初值与任务。固定开发、额外诊断和封存独立来源分开；独立资料不改名塞进训练 loader，也不绕过 checkpoint 的快照绑定。
- [ ] 冷启动从读原网格前计时；缓存复用报告首次构建及实际摊销。失败、候选筛选、校正、回退、最终验收全部计入每取得一个合格解的总时间。
- [ ] 最终按已冻结 seeds `0–49` 和独立来源协议报告成功数/总数、原生残差、目标偏差、完整成本及配对区间。既有目标容差为 1% 时按该合同计算，若改变定义须事先修订协议，不能看结果后改。
- [ ] 配对耗时比 95% 区间上界小于 1 只在完整质量与计时证据下解释为加速；保留未成功任务及其费用，并按来源处理相关性。独立来源太少时不能将查询数当作独立样本量。

当前只有边界块响应的批量化与串行可选多起点。新增 GPU 多轨迹求解或更换 solver 要单列算法对照，不混入“同原生 solver”比较。

## 八、Work B 最终回传包

| 交付文件/资产 | 必须包含 |
|---|---|
| `WORK_B_DELIVERY_RECEIPT_R2.md` | 当前关口 G0–G3、已完成/阻断/证据不足、明确下一笔预算及理由 |
| `environment_and_bindings.json` | commit、设备、配置、教师/快照/材料/查询/权重哈希与 source/split |
| `gpu_update_resume_report.json` | 真实更新、联合/J-only 梯度、恢复后下一步对照；原始日志 |
| `oracle_input_index.jsonl` 与 `oracle_cases/` | 与 Work A 共同绑定输入；每例问题、解、证书、固定贡献/原子/support/前沿/门控身份、INCONCLUSIVE 原因 |
| `same_point_mj_attribution.jsonl` | 四组合、chart、原生方向/线搜索、终点残差和有效性 |
| `baseline_comparison.json`、`stratified_metrics.csv` | 同粗零修正/E0/训练后/fine/强基线、来源与阶段分母、5%及保护线、失败行 |
| `performance_breakdown.json` | 冷/复用、实际工作量、同步口径、显存/内存、所有拒绝/回退成本 |
| CLI 训练原始产物 | 初始/选中/last 权重、sidecar、逐步 journal、开发逐查询结果、配置、真实曝光/更新/attempts |
| `native_solve_pairs.jsonl` 与原生验收资产 | 全初值、原网格凭据、成功/失败、目标、残差、校正/回退、完整时间；独立来源单列 |
| `stage_decision.json`、`SHA256SUMS` | 每关口状态及依据，缺失证据、预算和冻结规则；所有可移交文件及引用资产校验 |

先回传 B0 的真实准入和最小运行证据，随后回传 B1/B2 与 312 步 Pilot；强基线、封存评测按关口补齐，不等待所有报告完成才交流结果。本清单是待执行合同，不代表新增实验已跑完、当前模型已通过长训或论文验收。
