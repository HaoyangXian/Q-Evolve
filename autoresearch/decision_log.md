# Decision Log

## 2026-11-06 — Adopt shared research memory

### Decision
当前任务采用 V1 Shared Research Memory：Markdown/Git 文件是唯一权威状态；聊天记录只用于思考和讨论。

### Rationale
Agent 1、Agent 2 和 Codex 的会话上下文不可靠，且当前远端实验状态需要可追溯的事实、配置和原始证据。

### Consequences
- Agent 1 只提交带编号的 Idea 和证据。
- Agent 2 审查证据，提交状态更新建议和决策建议。
- Codex 执行实验、保存原始结果、写实验报告，并只写入已批准的正式状态。
- Human PI 决定是否批准重大研究方向、假设和结论。

## 2026-11-06 — G2 scope

### Decision
第一轮限制为 G2-Math Pilot；不把当前资产表述为四领域能力诊断或连续在线 Drift Monitoring。

### Evidence
现有 Probe 主要是数学题；Policy rollout 来自初始 Base；当前训练链尚未闭环。

## 2026-11-06 — Policy rollout provenance

### Decision
Policy 阶段的训练 rollout 归因于初始 Qwen3-8B，不归因于 SFT 或 Policy 模型。

### Evidence
远端 rollout launcher 使用 `MODEL=/data3/xhy/codes/models/Qwen3-8B`；Policy trainer 使用 `sft_success` 作为更新基座。
