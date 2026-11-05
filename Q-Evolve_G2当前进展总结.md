# Q-Evolve：G2 与 Self-Evolution 研究进展总结

更新时间：2026-11-05

## 1. 研究问题的收敛

研究最初关注 Critic 的 Value 是否测量推理状态质量，还是主要编码位置、长度和终止状态等表面信号。

讨论后，研究问题收敛为：

> 在 Self-Evolution 过程中，多维能力状态如何变化？哪些 Hidden Capability Drift 无法通过常规被动观测及时发现？Active Probing 能否在有限成本下识别和定位这些变化？

三层关系：

```
Self-Evolution Process
        ↓
Hidden Capability Drift
        ↓
Active Capability Probing
```

Self-Evolution 是动态过程，Hidden Capability Drift 是科学现象，Active Probing 是诊断方法。

当前 G2 只能定位为：

> 对一次或少数 Self-Evolution Transition 进行预算受限的能力漂移可辨识性研究。

## 2. 远端项目

目标主机：

```
10.249.45.139
```

实际项目路径：

```
/data3/xhy/codes/work/paper_repos/Q-Evolve
```

主要目录和脚本：

```
drift_exp/
drift_exp/capability_probe/
drift_exp/branch_pilot/
drift_exp/branch_smoke/

drift_exp/capability_probe/probe_bank.py
drift_exp/capability_probe/eval_capability.py
drift_exp/capability_probe/analyze_matrix.py
drift_exp/capability_probe/info_gap.py
drift_exp/capability_probe/adaptive_sim.py
drift_exp/capability_probe/experiments.py

eval_agent/train_sft.py
QEvolve/train_critic.py
QEvolve/train_policy.py
QEvolve/train_policy_chunked.py
QEvolve/inf_critic_chunked_fast.py
```

## 3. 已确认的 Self-Evolution 训练链

当前流程是：

```
初始 Qwen3-8B
    ↓
生成 rollout
    ↓
过滤成功轨迹
    ↓
SFT
    ↓
Critic 训练
    ↓
Advantage 推理
    ↓
Policy / LoRA 更新
```

关键模型关系：

| 环节 | 使用的模型或数据 |
|---|---|
| 训练 rollout 生成 | 初始 `/data3/xhy/codes/models/Qwen3-8B` |
| SFT | 初始 Base rollout 中的成功轨迹 |
| Critic | .139 链中以 `sft_success` 为基础模型 |
| Advantage | `sft_success` + Critic |
| Policy 训练 | 以 `sft_success` 为基座，使用已有轨迹和 Advantage |
| Policy 推理 | `sft_success` + `policy_chunked8192` adapter |

关键结论：

> Policy 阶段使用的 rollout 是初始 Qwen3-8B 生成的旧轨迹，不是 SFT 或 Policy 模型自己生成的新 rollout。

因此现在是一次单向更新链，而不是完整闭环：

```
Base rollout → SFT/Critic/Advantage → Policy
```

尚未形成：

```
Policy → 新 rollout → 新反馈 → 下一轮 Policy
```

## 4. 当前模型和数据资产

主要模型：

```
/data3/xhy/codes/models/Qwen3-8B
runs/olympiad_qwen3_8b_iterative1024_small/sft_success
runs/olympiad_qwen3_8b_iterative1024_small/policy_chunked8192
runs/olympiad_qwen3_8b_iterative1024_small/policy_chunked8192/checkpoint-256
runs/formal200_qwen3_policy_4096/checkpoint-56
```

已检查到的训练状态：

```
SFT：global_step=7，epoch=1
Policy chunked：global_step=256，epoch=2
```

主要数据规模：

```
训练 rollout：约 4314 条
Policy chunked 数据：512 条
positive advantage：215 条
```

可定义的演化节点：

```
θ0 = 初始 Qwen3-8B
θ1 = sft_success
θ2 = policy_chunked8192
```

可先研究：

```
θ0 → θ1：SFT transition
θ1 → θ2：Policy/RL transition
```

目前缺少统一的演化元数据：

```
evolution_round
parent_checkpoint
child_checkpoint
rollout_generator
feedback_source
update_operator
training_seed
```

## 5. 早期 Critic/Drift 结果

`drift_exp/drift_auc_result_CORRECTED.txt` 中已有：

```
AUC(mean_V, drift)       = 0.401
AUC(final_V, drift)      = 0.274
AUC(mean_Qmin, drift)    = 0.363
AUC(mean_A, drift)       = 0.525
AUC(raw_accuracy, drift) = 0.173
```

易题与难题的 `mean_V` 差异约为：

```
0.0013
```

已有 Termination Intervention 条件：

```
E1：正常 assistant turn 结束
E2：EOS 结束
E3：预算截断
E4：继续状态
E5：特殊终止形式
```

部分样本显示 E3 截断下的 Value 与正常/EOS/continue 条件不同，说明 termination signal 可能影响 Value，但这不能单独证明 Critic 具有通用能力判断能力。

Branching Continuation 已有 smoke/pilot 结果，但规模不足以作为正式主实验。

## 6. G2 v0.2 的正式边界

第一轮正式名称：

> G2-Math Pilot: Identifiability of Fine-grained Mathematical Reasoning Behaviors

只研究数学内部的候选行为维度：

```
concept
derivation
logic
planning
```

这些只能称为数学任务内部的候选行为维度，不能直接称为四种已经验证的通用能力。当前没有 Tool Use Probe，因此不能声称已经完成工具使用能力诊断。

## 7. 任务设计矩阵与经验敏感性矩阵

必须区分：

```
X：任务设计矩阵
A_emp：经验敏感性矩阵
```

`X[i,k]` 表示第 `i` 道题被设计为需要第 `k` 个任务需求因素；`A_emp` 则由不同模型状态的响应变化估计。

关键问题：

> X 满秩不代表 A_emp 可识别。

如果所有 Checkpoint 来自同一条训练轨迹，能力变化方向可能高度相关：

```
Δconcept ≈ λ1 Δderivation
Δlogic   ≈ λ2 Δplanning
```

需要构造模型对变化矩阵：

```
C_delta = [Δc_1^T; Δc_2^T; ...; Δc_M^T]
```

并检查秩、条件数和测量误差。G0 不应把“完整识别 A_emp”设为硬性要求，而应先验证预注册任务因素在独立模型状态上是否产生可重复响应差异。

## 8. Probe Construction

第一轮 Probe：

```
Type A：单因素 Probe
Type B：多因素组合 Probe
Type C：Contrastive Probe
Negative Control：只改变表达、长度或 Prompt
```

最重要的是 Contrastive Probe。例如规划对照：

```
A：提供完整操作步骤，模型只执行
B：相同初始条件和目标，但模型自主决定步骤
```

需要控制背景、信息量、计算难度和答案格式。

200 道题只能作为 G0 Pilot：

```
单因素任务：120
双因素任务：60
Anchor：20
```

## 9. Reference、Discovery、Confirmation

必须拆成三个互斥集合：

```
R：Reference Battery
D：Discovery Set
C：Confirmation Set
```

满足：

```
R ∩ D = ∅
R ∩ C = ∅
D ∩ C = ∅
```

用途：

| 数据集 | 用途 |
|---|---|
| Reference Battery | 定义模型对的漂移状态，不参与 Probe 选择 |
| Discovery Set | 拟合观测模型、选择 Probe、优化 Static/Adaptive |
| Confirmation Set | 冻结所有参数后做独立验证 |

Anchor 只能作为测量连接集，不能同时作为独立 Confirmation 样本。每种漂移状态也不能只由一个模型对代表。

## 10. 正式观测应采用 reference/current 配对

研究对象是：

```
Δc = c_current - c_reference
```

因此 Probe 的观测单位应为：

```
Z_p = (Y_reference, Y_current)
```

四种结果：

```
(0,0), (0,1), (1,0), (1,1)
```

不能只拟合 `P(Y_current | H)`，而应拟合 `P(Z_p | H)`。

建议保存：

```json
{
  "model_pair_id": "pair_001",
  "reference_checkpoint": "sft_seed0",
  "target_checkpoint": "policy_seed0",
  "question_id": "q_001",
  "template_family": "factorial_math_01",
  "evolution_round": 1,
  "reference_success": 1,
  "target_success": 0,
  "rollout_seed": 0,
  "n_tokens": 1024,
  "generation_cost": 0.0,
  "evaluator_version": "g2-v0.2"
}
```

## 11. 漂移假设必须互斥

不再使用含义重叠的 H1–H6。定义：

```
Δc = c_t - c_0

H_S:
  Δc_k < -δ_k       for k ∈ S
  |Δc_j| ≤ δ_j      for j ∉ S
```

如果最多允许两个维度退化，四个维度共有：

```
1 + C(4,1) + C(4,2) = 11
```

个状态。

增加：

```
H_ambiguous
```

表示置信区间不足以判断是否跨过实际意义阈值。不能把“证据不足”直接标成 H0。

## 12. 方法比较

正式方法：

```
Passive Natural Sampling
Random Probe
Fixed Stratified
Optimized Static
Adaptive Probe
```

理论上界：

```
Oracle Static
Oracle Adaptive
```

`Optimized Static` 必须在 Discovery 上优化最坏情况风险，而不能只选全局 KL 最大的题目。

Adaptive 至少应比较：

1. Expected Information Gain；
2. Expected Risk Reduction；
3. Worst-case Pairwise Discrimination。

当前 `adaptive_sim.py` 虽然接收 `obs`，但没有正确用后验更新选题，不能直接作为正式 Adaptive 实验。

## 13. 被动分布、成本和信息量

必须区分：

```
Reference Distribution：定义能力状态的参考任务分布
Passive Distribution q0：模拟自然观测来源
Diagnostic Probe Pool：主动选择的任务池
```

第一轮至少比较：

```
Balanced Passive
Synthetic Skewed Passive
```

没有真实生产日志时，偏斜分布只能称为 Synthetic Stress-Test Distribution。

还应在能力需求比例固定时改变任务难度比例，防止 Active 的优势只是难度选择优势。

主指标：

```
R_B(π) = max_H P_H^π(Ĥ ≠ H)
C_α(π) = 达到 R_B(π) ≤ α 所需的最小成本
```

成本包括模型生成、推理 Token、判分和人工验证成本。

经验 KL 接近 0 不能直接证明真实不可辨识，需要 Bootstrap 置信区间、保守信息量上界以及 Le Cam/Fano 分析。

## 14. 当前队列问题

曾检查到 `drift_exp/capability_probe/experiments.py` 长期处于：

```
done=1 running=4 pending=8
```

其 GPU 调度条件为：

```
GPU_WHITELIST = [4, 5, 6, 7]
GPU_MIN_FREE = 40000 MB
```

白名单 GPU 没有达到 40 GB 空闲显存，后续任务无法启动。当前 rollout 文件是部分产物，不能直接作为完整 G2 证据。

## 15. 推荐下一步

```
1. 不直接启动完整 G2
2. 建立 evolution_manifest，记录 θ0 → θ1 → θ2
3. 完成 Type A/B/C 与 Negative Control Probe
4. 先做 G0 测量效度验证
5. 建立 R/D/C 三组数据
6. 生成 reference/current 配对 rollout
7. 从 Reference Battery 定义互斥漂移支持集
8. 在 Discovery 上估计经验观测模型
9. 修正 Adaptive 后验更新
10. 加入真正的 Optimized Static 和 Oracle baselines
11. 在 Confirmation 上冻结参数并比较成本—错误率
12. G2 通过后再进入连续 Sparse Capability Drift
13. 更晚阶段再把 Probe 诊断反馈给 Self-Evolution
```

## 16. 暂时结论

1. Q-Evolve 已有一次可追踪的 Base → SFT → Policy 更新链，可作为 Self-Evolution Transition 素材。
2. Policy 使用的是初始 Base 模型生成的 rollout，而不是 Policy 自己生成的闭环数据。
3. 现有 Critic、Value、Branching 和 capability_probe 结果支持继续做测量研究，但不能直接证明通用 Hidden Capability Drift Detection。
4. G2 v0.2 的核心逻辑应保持：Measurement → Identifiability → Adaptive Diagnosis。
5. 统计模型不能过早冻结，必须先解决 Probe 测量效度、模型状态变化方向、配对观测和数据泄漏问题。
6. 当前阶段最合适的任务是完成 G2-Math Pilot，并将其放在 Self-Evolving Hidden Capability Drift 的更大框架中。
7. 只有在独立 Confirmation 上证明 Passive 存在信息缺口，且 Adaptive 稳定超过强 Optimized Static，才值得进入更强的 Adaptive Diagnosis 或 Sparse Capability Recovery 理论研究。

