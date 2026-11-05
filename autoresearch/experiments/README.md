# Experiments Directory Protocol

每个实验使用独立目录：

```text
experiments/EXP-YYYYMMDD-name/
├── README.md
├── config.json
├── commands.sh
├── raw/
├── logs/
├── derived/
└── report.md
```

## Required metadata

- experiment_id
- git commit or code version
- remote host and absolute project path
- model/checkpoint paths
- input dataset and split
- random seeds
- evaluator version
- command line
- start/end time
- exit status

## Reporting rule

原始结果不可覆盖；派生统计必须注明脚本和输入；聊天中的结论只有在 `report.md` 和审查后才可进入 `research_state.md`。
