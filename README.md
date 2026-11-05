# Q-Evolve Reproduction and Hidden Capability Drift

This repository contains source code, launch scripts, and lightweight experiment definitions for the Q-Evolve reproduction and the G2-Math Pilot.

Large artifacts are intentionally excluded from Git: model weights, checkpoints, rollout datasets, caches, generated JSONL files, and runtime logs. They remain on the experiment host and should be referenced by absolute paths and checksums in experiment reports.

Main directories:

- `QEvolve/`: Critic and Policy training code.
- `eval_agent/`: rollout, evaluation, SFT, and Q-guided inference.
- `drift_exp/`: drift, branching, termination, and capability-probe experiments.
- `bin/`: launch and data-preparation scripts.
- `qa/`: evaluation utilities.

Current experiment host:

- Host: `10.249.45.139`
- Project path: `/data3/xhy/codes/work/paper_repos/Q-Evolve`

Before running an experiment, record the commit, command line, model paths, input split, random seed, evaluator version, and output paths in an experiment report.
