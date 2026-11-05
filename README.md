# Q-Evolve Reproduction and Hidden Capability Drift

This private repository contains source code, launch scripts, lightweight Probe definitions, and research memory for the Q-Evolve reproduction and G2-Math Pilot.

## Scope

The current research sequence is:

1. Measurement validation for fine-grained mathematical reasoning behaviors.
2. Identifiability under finite passive observation.
3. Comparison of passive, static, and adaptive probing.

The current Self-Evolution assets represent a Base -> SFT -> Critic/Advantage -> Policy transition. They do not yet form a closed multi-round evolution loop.

## Repository layout

- `QEvolve/`: Critic and Policy training code.
- `eval_agent/`: rollout, evaluation, SFT, and Q-guided inference.
- `drift_exp/`: drift, branching, termination, and capability-probe experiments.
- `bin/`: launch and data-preparation scripts.
- `qa/`: evaluation utilities.
- `autoresearch/`: research brief, current state, decisions, and experiment-recording protocol.

## Excluded artifacts

Model weights, checkpoints, rollout datasets, caches, generated JSONL files, runtime logs, and large training outputs are excluded by `.gitignore`. They remain on the experiment host and must be referenced by absolute paths and checksums in experiment reports.

## Experiment host

- Host: `10.249.45.139`
- Project path: `/data3/xhy/codes/work/paper_repos/Q-Evolve`

## Quick validation

From the project root:

```bash
python -m compileall -q QEvolve eval_agent drift_exp bin qa
for f in $(git ls-files '*.sh'); do bash -n "$f"; done
```

These commands validate syntax without starting GPU jobs.

## Running experiments

Read `autoresearch/research_state.md` before starting. Use the launch scripts under `bin/` or `drift_exp/`; do not run large jobs without recording an experiment ID, commit, model paths, data split, random seed, evaluator version, command, and output paths.

Each experiment record belongs under:

```text
autoresearch/experiments/EXP-YYYYMMDD-name/
|-- README.md
|-- config.json
|-- commands.sh
|-- raw/
|-- logs/
|-- derived/
`-- report.md
```

Raw results must not be overwritten. Derived statistics must identify the script and input files. Only reviewed results may update `research_state.md`.

## Git workflow

```bash
git status --short --branch
git add <files>
git commit -m "type: concise description"
git push origin main
```

Use a feature branch for experiment changes and merge only after review.
