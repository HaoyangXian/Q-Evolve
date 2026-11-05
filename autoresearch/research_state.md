# Research State

**Project:** Q-Evolve / Self-Evolving Hidden Capability Drift
**Version:** 1
**Last Updated:** 2026-11-06
**Maintainer:** Codex（执行记录）；重大状态由 Human PI 确认

## Current question
当前要验证：在一次或少数 Self-Evolution transition 中，数学内部候选行为维度是否出现可测量、可区分、且被动观测可能漏掉的变化。

## Confirmed findings
- 远端项目：`/data3/xhy/codes/work/paper_repos/Q-Evolve`
- 可追踪节点：θ0 初始 Qwen3-8B、θ1 `sft_success`、θ2 `policy_chunked8192`。
- Policy 阶段使用初始 Qwen3-8B 生成的 rollout；不是 Policy 自己闭环生成的数据。
- 当前训练链是 Base rollout → SFT → Critic → Advantage → Policy，尚未形成多轮闭环 Self-Evolution。
- 现有 Critic/Drift、Termination、Branching 和 capability_probe 结果只能作为探索性证据。
- 当前队列曾因 `GPU_MIN_FREE=40000` 长期无法推进，已有 rollout 产物可能不完整。

## Active hypotheses
- Probe 的人工设计矩阵 X 不等于经验敏感性矩阵 A_emp。
- X 满秩不保证 A_emp 可识别；需要独立模型状态变化方向和多个模型对。
- 单独使用当前模型成功率会混入题目难度；应使用 paired observation `Z=(Y_reference,Y_current)`。
- 当前 Adaptive simulator 没有完整实现基于后验的选题。

## Current decision
暂不启动完整的 200 题、H0–H6、五策略 G2。先完成 Probe Construction 和 G0 Measurement Validation。

## Next action
1. 建立 evolution manifest，记录 θ0→θ1→θ2。
2. 构造 Type A/B/C 与 Negative Control Probe。
3. 划分 Reference Battery、Discovery、Confirmation。
4. 生成配对 rollout，并估计经验响应。
5. 冻结互斥漂移状态后，再比较 Passive、Random、Stratified、Optimized Static、Adaptive。

## Evidence locations
- 总结文档：`Q-Evolve_G2当前进展总结.md`
- 远端项目：`/data3/xhy/codes/work/paper_repos/Q-Evolve`
- 早期结果：远端 `drift_exp/drift_auc_result_CORRECTED.txt`
