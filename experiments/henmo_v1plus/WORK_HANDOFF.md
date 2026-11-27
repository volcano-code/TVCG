# HENMO V1+ Work 交付入口

研究分支：`research/henmo-v1plus`。

**核心代码已经实现；169 项集成回归通过，实际 M+J 两步更新及下一步恢复通过。真实生产精度和端到端 GPU 加速仍待 Work 验证。**

按职责阅读：

1. **Work A（数据集侧）**：[WORK_A_ACTION_GUIDE.md](WORK_A_ACTION_GUIDE.md)。先冻结当前可用快照，补终点/高保留、跨来源 R8/Full-J 和高分材料身份。
2. **Work B（模型与训练侧）**：[WORK_B_ACTION_GUIDE.md](WORK_B_ACTION_GUIDE.md)。先运行实际更新/恢复、同输入确定性对照和 312 步有界 Pilot，再做完整原生求解验收。
3. **部署桥接**：[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)。使用 Work 的真实原生 backend/solver，按注册合同接入，最终由原网格 verifier 验收。
4. **新架构**：[README.md](README.md)。材料分辨率、编码分辨率和查询层级分开；同源一阶 J，粗细光滑混合，批量边界响应。
5. **规划是否全部落实**：[IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md)。逐项列明已完成代码、已验证范围、未实现可选模块与待真实实验项目。

没有承诺固定毫秒耗时、固定校正步数或必然的原论文加速比。当前边界响应批量执行已经实现，多条完整优化轨迹的 GPU 并行尚未实现。不得把软件 PASS 改写成物理 GO。
