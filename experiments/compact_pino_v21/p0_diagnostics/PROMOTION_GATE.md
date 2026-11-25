# 研发筛选与训练器连接

`promotion_gate.py` 从配对的逐查询数据重算研发筛选，不读取训练损失来替代开发误差，也不执行训练。其结果与最终原生物理验收分开。

## 冻结规则

- 主指标始终是原始、未归一化、交叉项不乘 2 的十矩误差，按来源等权宏平均；相对本轮同一候选的 E0 至少改善 5%。
- 原协议中高保留与 Full-J／R3／R8 保护项最多为各自 E0 的 1.05 倍。
- 中高保留保护可以明确选 `scaled10_moment_rmse`，使用每条标签中冻结的 `moment_scale[10]`，例如 `m L^k`。也支持有明确原协议依据的 `raw10_moment_rmse`、`raw_mass_rmse` 或 `scaled_mass_rmse`；不自动替换原指标。
- 导数保护明确选 `raw_jacobian_rmse` 或 `scaled_jacobian_rmse`。后者必须提供正的 `jacobian_scale[10,D]`，按照误差除以该尺度计算。共享导出器使用 `S_i / parameter_scales_j`。
- 必须声明聚合顺序。`source_macro_group_rmse` 表示来源内对查询和分量先求均方、再开方，最后来源等权平均。`source_macro_mean_query_rmse` 表示先逐查询开方，再来源内平均、最后来源等权平均。工具不从旧摘要推断哪一个是冻结的旧公式。
- Full-J、R3、R8 或 low 缺少有效分母时，明确记录 `INSUFFICIENT_EVIDENCE`。已测项目失败时整体仍为 `NO_GO`，同时保留缺失证据列表。
- 缺失预测不能从分母中删除。有效查询推理失败会否决候选；有效教师 J 对应的模型 J 缺失会使该保护项证据不足。未取得导数资格的行不进入 J 误差。
- 无学习基线的差值单列。工具没有把“必须优于无学习基线”冒充旧协议已存在的硬门槛。

逐来源、逐分量、四档保留比例、各类合格导数数量、推理失败和不可用导数均写入结果。近零／low／middle／high 的边界分别为 `<0.01`、`[0.01,0.1)`、`[0.1,0.7)`、`[0.7,1]`。

## 可执行入口

```bash
python -m experiments.compact_pino_v21.p0_diagnostics.promotion_gate \
  --bundle /absolute/path/to/screen/bundle.json \
  --output /absolute/path/to/new-screen-report.json
```

成功完成计算后，`GO` 返回退出码 0，`NO_GO` 或 `INSUFFICIENT_EVIDENCE` 返回 2。输入损坏、哈希不符或语义不完整会抛出错误，不生成伪造的正常回执。已有输出不会覆盖。

输入 `bundle.json` 的 schema 为 `tvcg.v22.research-screen-bundle.v1`，其 `artifacts` 将每个资产映射到 `{path,sha256}`。必需角色为：

- `protocol`：下述明确聚合顺序的冻结评估规格；
- `registered_protocol`：原已登记协议，SHA 必须与规格中的 `registered_protocol_sha256` 相同；
- `training_manifest`、`config`、`evaluation_manifest`；
- `labels`、`epoch0`、`candidate`；
- `epoch0_checkpoint`、`candidate_checkpoint`。

`no_learning` 是可选预测资产；若冻结规格的 `require_no_learning_baseline=true`，缺失它会得到证据不足。实际文件 SHA、预测绑定的检查点、训练清单、评估清单和协议身份均核对。`config_sha256` 按 canonical JSON 计算，与新训练器采用相同规则，不能与任意格式化 JSON 的文件 SHA 混用。

冻结评估规格的字段为：

```python
{
    "schema": "tvcg.v22.research-screen-protocol.v1",
    "protocol_id": "the-registered-comparison",
    "registered_protocol_sha256": actual_registered_protocol_file_sha,
    "primary_aggregation": "source_macro_group_rmse",
    "jacobian_aggregation": actual_registered_jacobian_aggregation,
    "primary_relative_improvement": 0.05,
    "protection_max_ratio": 1.05,
    "protected_retention_metric": "scaled10_moment_rmse",
    "protected_jacobian_metric": actual_registered_jacobian_metric,
    "require_no_learning_baseline": True,
    "expected_record_ids": actual_frozen_development_record_ids,
}
```

这些 `actual_*` 必须来自冻结协议与数据，不是工具替用户选择的值。新增规格或不同聚合口径须作为新版本登记，不能看过候选结果后更换口径。

## 与真实训练器直接连接

新训练器已调用 `screen_exports.write_screen_bundle`。独立调用时也使用同一接口：

```python
bundle = write_screen_bundle(
    data,  # load_training_manifest 返回的 TrainingData
    protocol_path=existing_frozen_protocol,
    registered_protocol_path=original_registered_protocol,
    epoch0_evaluation=actual_epoch0_evaluation,
    candidate_evaluation=actual_candidate_evaluation,
    no_learning_evaluation=actual_unweighted_evaluation,
    run_id=actual_run_id,
    config=actual_train_config_dict,
    epoch0_checkpoint_path=actual_epoch0_checkpoint,
    candidate_checkpoint_path=actual_candidate_checkpoint,
    output_directory=new_directory,
)
```

评估可传训练器的 `rows`／`records`／`predictions` 列表包装或列表本身，无需再造一套字段。每行保留 `record_id`、`status`、`moments`、`jacobian_status`、`chart_id`、`jacobian` 和不可用原因。

导出器读取实际开发案例的原生标签、固定材料、完整材料和尺度，计算真实保留比例；全部已登记开发行必须可装配，不允许静默丢失。源 target 文件、case 文件和 qualification 文件在导出时再次核对 SHA。

教师身份明确为排序后的 `{record_id,targets_sha256,teacher_version}` 列表的 canonical SHA，名称为 `canonical_sorted_record_targets_sha256_and_teacher_version_v1`；它不是捏造的某个单独教师程序文件 SHA。实际 qualification 文件存在且已验证时，导数才进入真实 J 分母；仅有旧 metadata 声明而没有凭据时，保留 `declared_but_unreceipted_jacobians` 并明确证据不足。合成解析 fixture 采用单独标识，不能充当真实开发资格。

导出器绑定调用者交来的预测与检查点字节，不会从检查点重新推理，也不会凭文件 SHA 证明教师的物理正确性。新训练器在实际保存、评估和选模位置调用它；外部调用者必须保持同样的检查点—预测关系。

## 输出及权限边界

输出 schema 为 `tvcg.v22.research-screen.v1`，包括：

- `decision`: `GO`、`NO_GO` 或 `INSUFFICIENT_EVIDENCE`；
- `scope`: `development` 或 `synthetic`；
- `identity`: 实际候选／E0 检查点、训练清单、配置、评估清单、教师和协议身份；
- `primary`、`protections`、`by_retention`、`by_component`、`coverage` 和 `availability`；
- `training_authorized=false`；
- `physical_acceptance="NOT_EVALUATED"` 与 `speedup_claim=false`。

只有真实开发范围的 `GO` 才有 `eligible_for_long_training_review=true`。它仅允许进入单独的长训练审核；训练器还需要绑定该回执、实际候选、预算和协议的明确授权。它不要求事先完成训练后才可能执行的独立几何终验，也不把研发筛选直接当作部署通过。

软件测试使用制造数据检验边界，包括主指标改善 4.54%、高保留退化 29.77%、相对无学习误差高 7.73% 时必须 `NO_GO`。这些制造数据不声称是用户第 312 步检查点的原始逐查询复算。本次没有实现或宣称新的配对物理加速结论。
