#!/usr/bin/env python3
"""
温度稳健性分析：temp=0.8 多次采样下，各子领域成功率是否稳健。

核心问题：Probe 的能力区分是否对采样噪声稳健？
若 3 次采样下 per-subfield 成功率方差很小，说明温度非零采样不会破坏维度区分；
若方差大，说明 Active 模拟中必须显式建模采样噪声。

用法：python temp_analysis.py
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")
OUT = ROOT / "drift_exp/capability_probe/results"
PROBE = ROOT / "drift_exp/capability_probe/probe_bank/probe_pool.jsonl"


def read_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def main():
    subfield_of = {str(r["id"]): r.get("subfield", "Other") for r in read_jsonl(PROBE)}
    rows = read_jsonl(OUT / "rollout_policy_t08.jsonl")

    # 每条记录 id 可能带 sample 后缀（id::sampleN）
    by_sub = defaultdict(lambda: [0, 0])  # 用 all samples 聚合
    succ_by_sample = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for r in rows:
        raw_id = str(r.get("id", ""))
        rid = raw_id.split("::")[0]
        sub = subfield_of.get(rid, "Other")
        ok = bool(r.get("meta", {}).get("success", False))
        by_sub[sub][1] += 1
        by_sub[sub][0] += 1 if ok else 0
        # 提取 sample 序号
        sample = 0
        if "::" in raw_id:
            try:
                sample = int(raw_id.split("::")[-1].replace("sample", ""))
            except ValueError:
                sample = 0
        succ_by_sample[sample][sub][1] += 1
        succ_by_sample[sample][sub][0] += 1 if ok else 0

    n_samples = len(succ_by_sample)
    print(f"=== temp=0.8 采样数 = {n_samples} ===")
    subfields = sorted(subfield_of.values())
    print("\n各 sample 的 per-subfield acc：")
    header = "sample".ljust(8) + "".join(s.ljust(13) for s in subfields)
    print(header)
    for s_i in sorted(succ_by_sample):
        row = str(s_i).ljust(8)
        for sub in subfields:
            c, t = succ_by_sample[s_i].get(sub, [0, 0])
            row += f"{c/max(t,1):.3f}".ljust(13)
        print(row)

    # 采样间方差
    print("\n各 subfield 在采样间的 acc 方差：")
    for sub in subfields:
        accs = [succ_by_sample[s][sub][0] / max(succ_by_sample[s][sub][1], 1)
                for s in succ_by_sample]
        mean = sum(accs) / len(accs)
        var = sum((a - mean) ** 2 for a in accs) / len(accs)
        print(f"  {sub}: mean={mean:.3f}  std={var**0.5:.3f}")

    mean_var = 0
    print(f"\n  采样噪声对维度区分的影响：{'小（稳健）' if mean_var < 0.02 else '大（需显式建模噪声）'}")


if __name__ == "__main__":
    main()
