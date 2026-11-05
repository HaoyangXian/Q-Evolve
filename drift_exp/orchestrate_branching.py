#!/usr/bin/env python3
"""
Branching 编排：等 branch_pilot 完成后自动补 critic 打分 + 分析。

流程（不硬编码样本数，用进程/文件状态判断阶段）：
  1. 等 stage_traj 完成（cont 文件开始有内容 = 进入 cont 阶段）→ 补 critic 打分
  2. 等 stage_cont 完成（branch_pilot 进程退出）→ 分析

用法：nohup python orchestrate_branching.py > drift_exp/branch_pilot/orchestrate.log 2>&1 &
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")
OUT = ROOT / "drift_exp/branch_pilot"
PY = "/data3/xhy/miniconda3/envs/pettingllms/bin/python"


def count(p):
    try:
        return sum(1 for _ in open(p, encoding="utf-8"))
    except FileNotFoundError:
        return 0


def log(msg):
    print(f"[orchestrate {time.strftime('%F_%T')}] {msg}", flush=True)


def branch_running():
    """branch_pilot 进程是否存活。"""
    r = subprocess.run(["pgrep", "-f", "branch_pilot.py --mode all"],
                       capture_output=True, text=True)
    return r.returncode == 0


def run_score(shard, gpu):
    log(f"补打分 shard{shard}（卡{gpu}）")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [PY, "-u", str(ROOT / "drift_exp/score_prefixes.py"),
           "--traj", str(OUT / f"traj_shard{shard}.jsonl"),
           "--prefixes", "16", "32",
           "--out", str(OUT / f"prefix_scores_shard{shard}.jsonl")]
    subprocess.run(cmd, env=env)


def main():
    log("编排启动")

    # 1) 等 stage_traj 完成：cont 文件开始有内容（branch_pilot 进入 cont 阶段）
    while True:
        n0 = count(OUT / "traj_shard0.jsonl")
        n1 = count(OUT / "traj_shard1.jsonl")
        c0 = count(OUT / "cont_shard0.jsonl")
        c1 = count(OUT / "cont_shard1.jsonl")
        log(f"traj: shard0={n0} shard1={n1} | cont: shard0={c0} shard1={c1}")
        if (c0 > 0 or c1 > 0) and n0 > 0 and n1 > 0:
            log("stage_traj 完成（cont 已开始写入）")
            break
        if not branch_running():
            log("branch_pilot 已退出（可能异常），检查文件后继续")
            if n0 > 0 and n1 > 0:
                break
        time.sleep(120)

    # 补 critic 打分（卡4/卡5 并行）
    log("补 critic 打分（卡4/卡5 并行）")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(run_score, 0, 4)
        ex.submit(run_score, 1, 5)
    log("critic 打分完成")

    # 2) 等 stage_cont 完成：branch_pilot 进程退出
    while True:
        n0 = count(OUT / "cont_shard0.jsonl")
        n1 = count(OUT / "cont_shard1.jsonl")
        if not branch_running():
            log(f"stage_cont 完成（进程退出），cont: shard0={n0} shard1={n1}")
            break
        log(f"stage_cont 进行中: shard0={n0} shard1={n1}")
        time.sleep(300)

    log("开始分析")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    cmd = [PY, "-u", str(ROOT / "drift_exp/analyze_branching.py"),
           "--scores",
           str(OUT / "prefix_scores_shard0.jsonl"),
           str(OUT / "prefix_scores_shard1.jsonl"),
           "--cont",
           str(OUT / "cont_shard0.jsonl"),
           str(OUT / "cont_shard1.jsonl")]
    subprocess.run(cmd, env=env)
    log("分析完成")


if __name__ == "__main__":
    main()
