#!/usr/bin/env python3
"""
Active vs Static 诊断模拟（离线，不训练新模型）。

输入：各 checkpoint 的 rollout jsonl（per-probe success）+ probe_pool。
用 per-probe 成功率 θ 作为「已知真值」，蒙特卡洛模拟「用有限 Probe 观测诊断当前 checkpoint」，
比较五种选题策略的最坏情况错误率随预算的变化。

五种策略：
  passive   —— 按被动分布 q0 采样 probe
  random    —— 从 pool 随机采样
  stratified—— 每科目固定比例采样
  static    —— 贪心选择 KL 信息量最大的固定 probe 集合（Optimized Static）
  adaptive  —— 每步根据当前后验选择 KL 最大的下一个 probe（Adaptive）

用法：
  python adaptive_sim.py --pool probe_pool.jsonl \
      --rollouts results/rollout_base.jsonl results/rollout_sft.jsonl results/rollout_policy.jsonl
"""
import argparse
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from itertools import combinations

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")

PASSIVE_Q = {"Algebra": 0.55, "Combinatorics": 0.15, "Geometry": 0.15, "Number Theory": 0.15}


def read_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def bern_kl(p, q):
    p = min(max(p, 1e-6), 1 - 1e-6)
    q = min(max(q, 1e-6), 1 - 1e-6)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


class DiagnosticSim:
    def __init__(self, pool, rollout_paths):
        # probe 元信息
        self.probe_ids = []
        self.subfields = {}
        for r in pool:
            pid = str(r["id"])
            self.probe_ids.append(pid)
            self.subfields[pid] = r.get("subfield", "Other")

        # per-checkpoint per-probe 成功率 θ
        self.checkpoints = []
        self.theta = {}  # ckpt -> {pid: success_rate}
        for path in rollout_paths:
            name = Path(path).stem.replace("rollout_", "")
            succ = defaultdict(list)
            for r in read_jsonl(path):
                pid = str(r.get("id", "")).split("::")[0]
                succ[pid].append(1.0 if r.get("meta", {}).get("success", False) else 0.0)
            self.checkpoints.append(name)
            self.theta[name] = {pid: (sum(v) / len(v) if v else 0.5) for pid, v in succ.items()}
            # 缺失 probe 补 0.5
            for pid in self.probe_ids:
                if pid not in self.theta[name]:
                    self.theta[name][pid] = 0.5

        self.n_ckpt = len(self.checkpoints)

    def kl_pair(self, pid, a, b):
        """probe pid 区分 checkpoint a/b 的对称 KL。"""
        p = self.theta[a][pid]
        q = self.theta[b][pid]
        return (bern_kl(p, q) + bern_kl(q, p)) / 2

    def sample_obs(self, ckpt, pid):
        return 1.0 if random.random() < self.theta[ckpt][pid] else 0.0

    def infer(self, observations):
        """给定观测 [(pid, y)]，用最大似然判断 checkpoint。"""
        loglik = {c: 0.0 for c in self.checkpoints}
        for pid, y in observations:
            for c in self.checkpoints:
                p = self.theta[c][pid]
                loglik[c] += math.log(p) if y == 1 else math.log(1 - p)
        return max(loglik, key=loglik.get)

    # ---- 选题策略 ----
    def pick_passive(self, chosen, obs):
        # 按 q0 按科目采样
        while True:
            pid = random.choice(self.probe_ids)
            sub = self.subfields[pid]
            if random.random() < PASSIVE_Q.get(sub, 0.1):
                if pid not in chosen:
                    return pid

    def pick_random(self, chosen, obs):
        avail = [p for p in self.probe_ids if p not in chosen]
        return random.choice(avail)

    def pick_stratified(self, chosen, obs, per_sub=1):
        # 每科目轮流选一个
        subs = list(PASSIVE_Q.keys())
        for sub in subs:
            avail = [p for p in self.probe_ids if p not in chosen and self.subfields[p] == sub]
            if avail:
                return random.choice(avail)
        return self.pick_random(chosen, obs)

    def pick_static(self, chosen, obs, fixed_set):
        # 用预计算的固定 probe 集合
        for p in fixed_set:
            if p not in chosen:
                return p
        return self.pick_random(chosen, obs)

    def pick_adaptive(self, chosen, obs):
        # 当前后验（用均匀先验，观测更新）
        avail = [p for p in self.probe_ids if p not in chosen]
        if not obs:
            # 无观测：选全局区分度最大的
            return max(avail, key=lambda p: sum(
                self.kl_pair(p, a, b) for a, b in combinations(self.checkpoints, 2)))
        # 有观测：选对当前最不确定的 checkpoint 对区分度最大的
        best, best_score = None, -1
        for p in avail:
            score = sum(self.kl_pair(p, a, b) for a, b in combinations(self.checkpoints, 2))
            if score > best_score:
                best, best_score = p, score
        return best

    def build_static_set(self, budget):
        """贪心选择 KL 信息量最大的固定 probe 集合。"""
        chosen = []
        avail = set(self.probe_ids)
        for _ in range(budget):
            best, best_score = None, -1
            for p in avail:
                score = sum(self.kl_pair(p, a, b) for a, b in combinations(self.checkpoints, 2))
                if score > best_score:
                    best, best_score = p, score
            if best is None:
                break
            chosen.append(best)
            avail.remove(best)
        return chosen

    # ---- 蒙特卡洛模拟 ----
    def run_strategy(self, strategy, budget, n_trials, fixed_set=None):
        """返回最坏情况错误率（每个 checkpoint 作为真值，取最差）。"""
        errors = defaultdict(int)
        trials_per_ckpt = n_trials
        for ckpt in self.checkpoints:
            for _ in range(trials_per_ckpt):
                chosen = set()
                obs = []
                for _ in range(budget):
                    if strategy == "passive":
                        pid = self.pick_passive(chosen, obs)
                    elif strategy == "random":
                        pid = self.pick_random(chosen, obs)
                    elif strategy == "stratified":
                        pid = self.pick_stratified(chosen, obs)
                    elif strategy == "static":
                        pid = self.pick_static(chosen, obs, fixed_set)
                    else:  # adaptive
                        pid = self.pick_adaptive(chosen, obs)
                    chosen.add(pid)
                    y = self.sample_obs(ckpt, pid)
                    obs.append((pid, y))
                pred = self.infer(obs)
                if pred != ckpt:
                    errors[ckpt] += 1
        return {c: errors[c] / trials_per_ckpt for c in self.checkpoints}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "drift_exp/capability_probe/probe_bank/probe_pool.jsonl"))
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--n_trials", type=int, default=200)
    args = ap.parse_args()

    pool = read_jsonl(args.pool)
    sim = DiagnosticSim(pool, args.rollouts)
    print(f"[adaptive_sim] checkpoints: {sim.checkpoints}")
    print(f"[adaptive_sim] probe pool: {len(sim.probe_ids)} 题\n")

    budgets = [3, 5, 10, 20, 40]
    print(f"{'budget':>6} | {'passive':>8} {'random':>8} {'stratified':>10} {'static':>8} {'adaptive':>9}")
    print("-" * 60)
    for B in budgets:
        fixed_set = sim.build_static_set(B)
        res = {}
        for strat in ["passive", "random", "stratified", "static", "adaptive"]:
            errs = sim.run_strategy(strat, B, args.n_trials, fixed_set=fixed_set)
            res[strat] = max(errs.values())  # 最坏情况错误率
        print(f"{B:>6} | {res['passive']:>8.3f} {res['random']:>8.3f} "
              f"{res['stratified']:>10.3f} {res['static']:>8.3f} {res['adaptive']:>9.3f}")

    print("\n解读：最坏情况错误率越低越好。"
          "若 adaptive < static，说明存在自适应收益；"
          "若 adaptive >= static，说明静态分配已足够。")


if __name__ == "__main__":
    main()
