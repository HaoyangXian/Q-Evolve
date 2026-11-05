#!/usr/bin/env python3
"""
Probe Bank 构造：从 OlympiadBench 构造多维能力 Probe Pool。

职责：
1. 从源文件 OE_TO_maths_en_COMP.json 读取准确科目标签（subfield）
2. 基于 solution 文本特征，为每道题标注 4 个行为维度的初始敏感性
   D1 概念调用 / D2 多步推导 / D3 逻辑约束 / D4 解题规划
3. 构造三类 Probe：Type A 单维 / Type B 多维 / Type C 对比 + Anchor
4. 输出 Probe Pool + 敏感性矩阵（初始标签，须经实验校准）

用法：
  python probe_bank.py --source <OE_TO_maths_en_COMP.json> \
                       --qa <olympiadbench.jsonl> \
                       --outdir drift_exp/capability_probe/probe_bank
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# 4 个行为维度
DIMS = ["concept", "derivation", "logic", "planning"]

# 维度 -> 关键词（solution 文本特征，启发式初始标签）
DIM_KEYWORDS = {
    "concept": [
        r"\btheorem\b", r"\bformula\b", r"\brecall\b", r"\bknown\b", r"\bidentity\b",
        r"\bby (?:the )?(?:Fermat|Euler|AM-GM|Cauchy|Bezout|Wilson|Pythagorean|Vieta|Ptolemy|Menelaus|Ceva|Law of Sines|Law of Cosines)\b",
        r"\bcorollary\b", r"\blemma\b", r"\bwell-known\b", r"\bstandard\b",
        r"\bharmonic\b", r"\bgeometric series\b", r"\bbinomial\b",
    ],
    "derivation": [
        r"=.*=", r"\\Rightarrow", r"\\rightarrow", r"\bthen\b", r"\bthus\b", r"\bhence\b",
        r"\\\[", r"\\frac", r"\\sum", r"\\int", r"\\prod", r"\\begin\{align",
        r"\bequivalent\b", r"\bexpanding\b", r"\bsimplifies\b", r"\brearrang\b", r"\bsubstitut\b",
    ],
    "logic": [
        r"\bcase[s]?\b", r"\bif\b", r"\bcontradiction\b", r"\bassume\b", r"\binduction\b",
        r"\bmod\b", r"\bdivisible\b", r"\bgcd\b", r"\bcongruent\b", r"\bmodulo\b",
        r"\botherwise\b", r"\bwithout loss\b", r"\bparity\b", r"\bcontradicts\b",
        r"\bcount\b", r"\bnumber of ways\b", r"\bpossib\b", r"\bremainder\b",
        r"\bprime\b", r"\binteger\b", r"\bsuppose.*then\b", r"\bdistinct\b",
    ],
    "planning": [
        r"\bconstruct\b", r"\blet\b", r"\bconsider\b", r"\bdefine\b", r"\bwe first\b",
        r"\bstrategy\b", r"\bchoose\b", r"\btake\b", r"\bsuppose\b", r"\bset\b",
        r"\bauxiliary\b", r"\bclaim\b", r"\bwe (?:shall|will) show\b",
        r"\bdraw\b", r"\bextend\b", r"\breflect\b", r"\binscribe\b", r"\bcircumscribe\b",
    ],
}

# 科目先验：不同科目天然倾向不同行为维度（初始先验，非最终标签）
SUBFIELD_PRIOR = {
    "Algebra": {"concept": 0.1, "derivation": 0.5, "logic": 0.15, "planning": 0.1},
    "Combinatorics": {"concept": 0.1, "derivation": 0.2, "logic": 0.5, "planning": 0.25},
    "Geometry": {"concept": 0.15, "derivation": 0.35, "logic": 0.15, "planning": 0.45},
    "Number Theory": {"concept": 0.4, "derivation": 0.2, "logic": 0.5, "planning": 0.15},
}


def load_source(path):
    """读源文件，返回 {id: subfield}。"""
    rows = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in rows:
        out[str(r["id"])] = r.get("subfield")
    return out


def load_qa(path):
    """读 olympiadbench.jsonl，返回 {id: {question, final_answer, solution}}。"""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[str(r["id"])] = r
    return out


def score_dimension(text, keywords):
    """启发式打分：命中关键词比例 -> [0,1]，用于初始敏感性标签。"""
    if not text:
        return 0.0
    hits = 0
    for kw in keywords:
        if re.search(kw, text, re.IGNORECASE):
            hits += 1
    return min(1.0, hits / max(len(keywords) * 0.5, 1.0))


def annotate(qa, source):
    """为每道题标注能力向量 + 科目。能力向量 = 关键词得分 + 科目先验 的融合。"""
    items = []
    for qid, r in qa.items():
        subfield = source.get(qid, "Other")
        solution = r.get("solution", "")
        if isinstance(solution, list):
            solution = "\n".join(solution)
        text = f"{r.get('question', '')}\n{solution}"
        # 关键词得分 + 科目先验融合
        prior = SUBFIELD_PRIOR.get(subfield, {"concept": 0.1, "derivation": 0.2, "logic": 0.2, "planning": 0.15})
        vec = {}
        for d in DIMS:
            kw_score = score_dimension(text, DIM_KEYWORDS[d])
            # 融合：关键词为主，科目先验兜底（避免某些维度全 0）
            vec[d] = round(0.7 * kw_score + 0.3 * prior[d], 3)
        items.append({
            "id": qid,
            "subfield": subfield,
            "question": r.get("question", ""),
            "final_answer": r.get("final_answer", ""),
            "solution": solution,
            "dim_vec": vec,
            "sol_len": len(solution),
        })
    return items


def pure_dimension(item):
    """判断某题是否'单维主导'——某维度显著高于其他维度。"""
    v = item["dim_vec"]
    if max(v.values()) < 0.15:
        return None
    top = sorted(v.items(), key=lambda kv: -kv[1])
    if top[0][1] >= 0.18 and top[0][1] >= 1.3 * top[1][1] + 0.03:
        return top[0][0]
    return None


def build_pool(items, n_single=30, n_multi=10, n_anchor=20):
    """构造三类 Probe + Anchor。

    - Type A 单维：每维度挑敏感性最纯的题
    - Type B 多维：两个维度都较高的题
    - Type C 对比：同科目内 solution 结构差异大的题对（近似实现）
    - Anchor：跨科目混合
    """
    pool = {"single": {}, "multi": [], "contrast": [], "anchor": []}

    # Type A：单维
    by_dim = defaultdict(list)
    for it in items:
        d = pure_dimension(it)
        if d:
            by_dim[d].append(it)
    for d in DIMS:
        cand = sorted(by_dim[d], key=lambda x: -x["dim_vec"][d])
        pool["single"][d] = cand[:n_single]

    # Type B：多维（两个维度都 >= 0.2）
    multi_cand = []
    for it in items:
        v = it["dim_vec"]
        high = [d for d in DIMS if v[d] >= 0.2]
        if len(high) >= 2:
            multi_cand.append(it)
    # 按组合分组采样
    from itertools import combinations
    combo = defaultdict(list)
    for it in multi_cand:
        v = it["dim_vec"]
        high = sorted([d for d in DIMS if v[d] >= 0.2], key=lambda d: -v[d])
        if len(high) >= 2:
            combo[tuple(sorted(high[:2]))].append(it)
    for k, cand in combo.items():
        pool["multi"].extend(sorted(cand, key=lambda x: -sum(x["dim_vec"].values()))[:n_multi])

    # Type C：对比（同科目内，纯维度不同的题对）
    by_sub = defaultdict(list)
    for it in items:
        d = pure_dimension(it)
        if d:
            by_sub[it["subfield"]].append(it)
    for sub, cand in by_sub.items():
        dim_groups = defaultdict(list)
        for it in cand:
            dim_groups[pure_dimension(it)].append(it)
        if len(dim_groups) >= 2:
            # 取两个不同维度的题构成对比对
            keys = list(dim_groups.keys())
            for i in range(0, len(keys) - 1, 2):
                a = dim_groups[keys[i]][0]
                b = dim_groups[keys[i + 1]][0]
                pool["contrast"].append({"pair": [a, b], "dims": [keys[i], keys[i + 1]]})

    # Anchor：跨科目，难度混合（用 solution 长度近似难度）
    used_ids = set()
    for group in pool["single"].values():
        used_ids.update(x["id"] for x in group)
    used_ids.update(x["id"] for x in pool["multi"])
    for c in pool["contrast"]:
        used_ids.update(x["id"] for x in c["pair"])
    anchor_cand = [it for it in items if it["id"] not in used_ids]
    # 跨科目分层采样
    by_sub2 = defaultdict(list)
    for it in anchor_cand:
        by_sub2[it["subfield"]].append(it)
    per_sub = max(1, n_anchor // max(len(by_sub2), 1))
    for sub, cand in by_sub2.items():
        pool["anchor"].extend(cand[:per_sub])
    pool["anchor"] = pool["anchor"][:n_anchor]
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/data3/xhy/codes/datasets/OlympiadBench-full/OlympiadBench_Dataset/data/OE_TO_maths_en_COMP.json")
    ap.add_argument("--qa", default="/data3/xhy/codes/work/paper_repos/Q-Evolve/data/qa/olympiadbench.jsonl")
    ap.add_argument("--outdir", default="drift_exp/capability_probe/probe_bank")
    ap.add_argument("--n_single", type=int, default=30)
    ap.add_argument("--n_multi", type=int, default=10)
    ap.add_argument("--n_anchor", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    source = load_source(args.source)
    qa = load_qa(args.qa)
    print(f"[probe_bank] 源文件 {len(source)} 题，qa {len(qa)} 题")

    items = annotate(qa, source)

    # 统计科目分布
    sub_counter = defaultdict(int)
    for it in items:
        sub_counter[it["subfield"]] += 1
    print(f"[probe_bank] 科目分布: {dict(sub_counter)}")

    pool = build_pool(items, args.n_single, args.n_multi, args.n_anchor)

    # 汇总 Probe Pool 为统一列表
    all_probes = []
    for d, lst in pool["single"].items():
        for it in lst:
            it["probe_type"] = "single"
            it["primary_dim"] = d
            all_probes.append(it)
    for it in pool["multi"]:
        it["probe_type"] = "multi"
        it["primary_dim"] = None
        all_probes.append(it)
    for c in pool["contrast"]:
        for it in c["pair"]:
            it["probe_type"] = "contrast"
            it["primary_dim"] = None
            all_probes.append(it)
    for it in pool["anchor"]:
        it["probe_type"] = "anchor"
        it["primary_dim"] = None
        all_probes.append(it)

    # 去重（同一题可能被多个类型选中）
    seen = {}
    for it in all_probes:
        if it["id"] not in seen:
            seen[it["id"]] = it
    all_probes = list(seen.values())

    # 输出 1：Probe Pool（评测用 input）
    pool_path = outdir / "probe_pool.jsonl"
    with open(pool_path, "w", encoding="utf-8") as f:
        for it in all_probes:
            f.write(json.dumps({
                "id": it["id"], "question": it["question"],
                "final_answer": it["final_answer"],
                "subfield": it["subfield"], "probe_type": it["probe_type"],
                "dim_vec": it["dim_vec"],
            }, ensure_ascii=False) + "\n")

    # 输出 2：敏感性矩阵
    mat_path = outdir / "sensitivity_matrix.tsv"
    with open(mat_path, "w", encoding="utf-8") as f:
        f.write("id\tsubfield\tprobe_type\t" + "\t".join(DIMS) + "\n")
        for it in all_probes:
            f.write(f"{it['id']}\t{it['subfield']}\t{it['probe_type']}\t"
                    + "\t".join(str(it["dim_vec"][d]) for d in DIMS) + "\n")

    # 输出 3：报告
    report_path = outdir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"总题目数: {len(items)}\n")
        f.write(f"Probe Pool 总数: {len(all_probes)}\n\n")
        f.write("Type A 单维:\n")
        for d in DIMS:
            f.write(f"  {d}: {len(pool['single'][d])} 题\n")
        f.write(f"Type B 多维: {len(pool['multi'])} 题\n")
        f.write(f"Type C 对比: {len(pool['contrast'])} 对\n")
        f.write(f"Anchor: {len(pool['anchor'])} 题\n\n")
        f.write("科目分布:\n")
        for k, v in sorted(sub_counter.items()):
            f.write(f"  {k}: {v}\n")

    print(f"[probe_bank] Probe Pool: {len(all_probes)} 题 -> {pool_path}")
    print(f"[probe_bank] 敏感性矩阵 -> {mat_path}")
    print(f"[probe_bank] 报告 -> {report_path}")


if __name__ == "__main__":
    main()
