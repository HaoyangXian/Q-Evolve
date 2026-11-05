#!/usr/bin/env python3
"""
探索性实验队列调度器：定义一批实验，按依赖 + GPU 可用性自动排队执行。

特性：
- 依赖编排：分析实验在数据实验完成后自动触发
- GPU 监控：等有空闲卡（free >= 阈值）才启动 GPU 实验
- 断点续跑：GPU 实验的 rollout jsonl 存在且非空则跳过
- 进程级完成检测：分析实验靠进程退出判定完成

用法：nohup python experiments.py > experiments.log 2>&1 &
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")
PY = "/data3/xhy/miniconda3/envs/pettingllms/bin/python"
ROLLOUT = ROOT / "eval_agent/qa_rollout.py"
ROLLOUT_CHUNKED = ROOT / "eval_agent/qa_rollout_chunked_context.py"
OUT = ROOT / "drift_exp/capability_probe/results"
PROBE = ROOT / "drift_exp/capability_probe/probe_bank/probe_pool.jsonl"
INPUT = OUT / "probe_input.jsonl"

MODELS = {
    "base": {"model": "/data3/xhy/codes/models/Qwen3-8B", "adapter": None},
    "sft": {"model": str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/sft_success"), "adapter": None},
    "policy": {"model": str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/sft_success"),
               "adapter": str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/policy_chunked8192")},
}

GPU_MIN_FREE = 40000  # MB
GPU_WHITELIST = [4, 5, 6, 7]  # 只使用这 4 张卡（等 AgenTracer GRPO 训练空出后自动开跑）


def log(msg):
    print(f"[queue {time.strftime('%F_%T')}] {msg}", flush=True)


def n_gpus_free():
    free = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            idx, mb = line.split(",")
            if int(idx) in GPU_WHITELIST and int(mb) >= GPU_MIN_FREE:
                free.append(int(idx))
    except Exception:
        pass
    return free


class Exp:
    def __init__(self, name, cmd, deps=(), gpu=False, done_file=None):
        self.name = name
        self.cmd = cmd
        self.deps = list(deps)
        self.gpu = gpu
        self.done_file = done_file  # GPU 实验断点续跑标记
        self.status = "pending"
        self.proc = None
        self.gpu_id = None

    def rollout_exists(self):
        if not self.done_file:
            return False
        p = Path(self.done_file)
        return p.exists() and p.stat().st_size > 0


def rollout_cmd(ckpt_key, out_name, max_steps=96, temperature=0, samples=1,
                no_trunc=False, step_tokens=2048, wall_timeout=1800):
    m = MODELS[ckpt_key]
    # no_trunc=True 走 chunked_context 脚本：上下文有界裁剪，可用大 max_steps 而不 OOM，
    # 让模型自然输出 </FINAL_ANSWER> 停止，避免 step 内/步数两层截断污染能力测量。
    # step_tokens / wall_timeout 可按 checkpoint 差异化：
    #   - base 不会解题，死磕满步数，用较小预算避免每题为 30min 兜底浪费时间
    #   - sft/policy 会解题，用大预算保证不截断
    script = ROLLOUT_CHUNKED if no_trunc else ROLLOUT
    cmd = [PY, "-u", str(script),
           "--input", str(INPUT), "--output", str(OUT / out_name),
           "--model", m["model"]]
    if m["adapter"]:
        cmd += ["--adapter", m["adapter"]]
    if no_trunc:
        cmd += ["--samples", str(samples), "--step_max_new_tokens", str(step_tokens),
                "--max_steps", str(max_steps),
                "--temperature", str(temperature), "--termination_mode", "cue",
                "--context_max_tokens", "8192", "--wall_clock_timeout", str(wall_timeout)]
    else:
        cmd += ["--samples", str(samples), "--max_new_tokens", "8192",
                "--step_max_new_tokens", "1024", "--max_steps", str(max_steps),
                "--temperature", str(temperature), "--termination_mode", "strict"]
    return cmd


EXPERIMENTS = [
    # ===== 核心：3 档 checkpoint 能力矩阵（GPU 并行，不截断：chunked 上下文 + 超时兜底） =====
    # base 不会解题（success 恒≈0），用较小预算避免死磕浪费；sft/policy 用大预算保证不截断
    Exp("cap_base",   rollout_cmd("base",   "rollout_base.jsonl",   max_steps=96,  no_trunc=True, step_tokens=1024, wall_timeout=600),  gpu=True, done_file=OUT / "rollout_base.jsonl"),
    Exp("cap_sft",    rollout_cmd("sft",    "rollout_sft.jsonl",    max_steps=256, no_trunc=True), gpu=True, done_file=OUT / "rollout_sft.jsonl"),
    Exp("cap_policy", rollout_cmd("policy", "rollout_policy.jsonl", max_steps=256, no_trunc=True), gpu=True, done_file=OUT / "rollout_policy.jsonl"),

    # ===== 扩展：budget 旋钮维度退化（sft × 4 档，96 复用 cap_sft） =====
    Exp("budget_8",  rollout_cmd("sft", "rollout_sft_b8.jsonl",  max_steps=8),  gpu=True, done_file=OUT / "rollout_sft_b8.jsonl"),
    Exp("budget_16", rollout_cmd("sft", "rollout_sft_b16.jsonl", max_steps=16), gpu=True, done_file=OUT / "rollout_sft_b16.jsonl"),
    Exp("budget_32", rollout_cmd("sft", "rollout_sft_b32.jsonl", max_steps=32), gpu=True, done_file=OUT / "rollout_sft_b32.jsonl"),
    Exp("budget_64", rollout_cmd("sft", "rollout_sft_b64.jsonl", max_steps=64), gpu=True, done_file=OUT / "rollout_sft_b64.jsonl"),

    # ===== 扩展：temperature 稳健性（policy × temp 0.8 × 3 samples） =====
    Exp("temp_08", rollout_cmd("policy", "rollout_policy_t08.jsonl", temperature=0.8, samples=3),
        gpu=True, done_file=OUT / "rollout_policy_t08.jsonl"),

    # ===== 分析（依赖 GPU 实验） =====
    Exp("agg_matrix", [PY, "-u", str(ROOT / "drift_exp/capability_probe/analyze_matrix.py")],
        deps=["cap_base", "cap_sft", "cap_policy"]),
    Exp("info_gap", [PY, "-u", str(ROOT / "drift_exp/capability_probe/info_gap.py")],
        deps=["agg_matrix"]),
    Exp("adaptive", [PY, "-u", str(ROOT / "drift_exp/capability_probe/adaptive_sim.py"),
                     "--pool", str(PROBE),
                     "--rollouts", str(OUT / "rollout_base.jsonl"),
                     str(OUT / "rollout_sft.jsonl"), str(OUT / "rollout_policy.jsonl")],
        deps=["agg_matrix"]),
    Exp("budget_analysis", [PY, "-u", str(ROOT / "drift_exp/capability_probe/budget_analysis.py")],
        deps=["budget_8", "budget_16", "budget_32", "budget_64", "cap_sft"]),
    Exp("temp_analysis", [PY, "-u", str(ROOT / "drift_exp/capability_probe/temp_analysis.py")],
        deps=["temp_08"]),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"队列启动，共 {len(EXPERIMENTS)} 个实验")

    # 断点续跑：GPU 实验 rollout 已存在则跳过
    for e in EXPERIMENTS:
        if e.rollout_exists():
            e.status = "done"
            log(f"  [跳过] {e.name}（rollout 已存在）")

    while True:
        done_names = {e.name for e in EXPERIMENTS if e.status == "done"}
        running = [e for e in EXPERIMENTS if e.status == "running"]

        # 完成检测：进程退出
        for e in running:
            if e.proc is not None and e.proc.poll() is not None:
                e.status = "done"
                log(f"  [完成] {e.name}（exit={e.proc.returncode}）")

        done_names = {e.name for e in EXPERIMENTS if e.status == "done"}
        free_gpus = n_gpus_free()
        used_gpus = [e.gpu_id for e in EXPERIMENTS if e.status == "running" and e.gpu_id is not None]

        # 找可启动实验
        for e in EXPERIMENTS:
            if e.status != "pending":
                continue
            if not all(d in done_names for d in e.deps):
                continue
            gpu_id = None
            if e.gpu:
                avail = [g for g in free_gpus if g not in used_gpus]
                if not avail:
                    continue
                gpu_id = avail[0]
                used_gpus.append(gpu_id)
            # 启动
            e.status = "running"
            e.gpu_id = gpu_id
            env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/QEvolve",
                       TOKENIZERS_PARALLELISM="false")
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logfile = OUT / f"{e.name}.log"
            lf = open(logfile, "w")
            e.proc = subprocess.Popen(e.cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            log(f"  [启动] {e.name}（GPU {gpu_id if gpu_id is not None else '-'}）")

        # 全完成则退出
        if all(e.status == "done" for e in EXPERIMENTS):
            log("全部实验完成")
            break

        n_pending = sum(1 for e in EXPERIMENTS if e.status == "pending")
        n_running = sum(1 for e in EXPERIMENTS if e.status == "running")
        n_done = sum(1 for e in EXPERIMENTS if e.status == "done")
        log(f"状态: done={n_done} running={n_running} pending={n_pending} | 空闲卡={free_gpus}")
        time.sleep(120)


if __name__ == "__main__":
    main()
