"""Small, deterministic answer extractors for QA rollout rewards."""

import re
from collections import Counter
from decimal import Decimal, InvalidOperation


def extract_answer(text: str) -> str:
    text = str(text or "").strip()
    patterns = [
        r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
        r"(?:final answer|answer)\s*[:=]\s*(.+)",
        r"####\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return str(matches[-1]).strip().splitlines()[0].strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[-1] if lines else ""


def has_final_answer(text: str) -> bool:
    """Whether a rollout explicitly emitted a terminal answer marker."""
    return bool(re.search(
        r"<FINAL_ANSWER>\s*.*?\s*</FINAL_ANSWER>|####\s*.+|(?:final answer)\s*[:=]",
        str(text or ""), flags=re.IGNORECASE | re.DOTALL,
    ))


def has_explicit_final_marker(text: str) -> bool:
    """Return true only for the required terminal XML marker pair."""
    return bool(re.search(
        r"<FINAL_ANSWER>\s*.*?\s*</FINAL_ANSWER>",
        str(text or ""), flags=re.IGNORECASE | re.DOTALL,
    ))


_ANSWER_CUE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|would\s+be|should\s+be)\s*[:=]?\s*([^\n]+)",
    flags=re.IGNORECASE,
)


def _candidate_number(value: str):
    value = str(value or "").replace(",", "")
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    return nums[-1] if nums else ""


def answer_candidates(text: str):
    """Return explicit final/cue answer candidates in generation order."""
    text = str(text or "")
    out = []
    for m in re.finditer(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", text, re.I | re.S):
        out.append((m.group(1).strip(), m.start(), m.end()))
    for m in _ANSWER_CUE.finditer(text):
        out.append((m.group(1).strip().rstrip(".。;；"), m.start(), m.end()))
    return sorted(out, key=lambda x: x[1])


def finalize_rollout(text: str, mode: str = "strict"):
    """Normalize a completion into a terminal answer when the model omitted tags.

    ``strict`` preserves the original protocol. ``cue`` uses the last explicit
    answer cue. ``consensus`` uses the most frequent numeric cue, which avoids
    a late one-token loop overwriting a previously correct answer.
    """
    text = str(text or "").strip()
    if mode == "strict" and not has_explicit_final_marker(text):
        return text, extract_answer(text), False
    if has_final_answer(text):
        matches = list(re.finditer(r"<FINAL_ANSWER>\s*.*?</FINAL_ANSWER>", text, re.I | re.S))
        if matches:
            return text[:matches[-1].end()], extract_answer(matches[-1].group(0)), True
        # Preserve the existing acceptance of ``####`` and ``final answer:``.
        return text, extract_answer(text), True
    if mode == "strict":
        return text, extract_answer(text), False
    cues = [(v, s, e) for v, s, e in answer_candidates(text)
            if not re.search(r"<FINAL_ANSWER>", v, re.I)]
    if not cues:
        return text, extract_answer(text), False
    if mode == "consensus":
        keyed = [(_candidate_number(v), v, s, e) for v, s, e in cues]
        keyed = [x for x in keyed if x[0]]
        if keyed:
            counts = Counter(x[0] for x in keyed)
            best = max(counts, key=lambda k: (counts[k], max(x[2] for x in keyed if x[0] == k)))
            chosen = [x for x in keyed if x[0] == best][-1]
            value, start, end = chosen[1], chosen[2], chosen[3]
        else:
            value, start, end = cues[-1]
    else:
        value, start, end = cues[-1]
    line_end = text.find("\n", end)
    cut = len(text) if line_end < 0 else line_end
    value = value.strip()
    return text[:cut].rstrip() + f"\n<FINAL_ANSWER>{value}</FINAL_ANSWER>", value, True


def _clean(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\\boxed\s*\{(.*?)\}", r"\1", value, flags=re.DOTALL)
    value = value.replace("$", "").replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\s+", "", value)
    return value.strip(".。;；,")


def _numeric_value(value: str):
    """Return the last numeric answer embedded in a short answer string.

    GSM8K solutions commonly put prose inside ``<FINAL_ANSWER>`` (for example
    ``Ahito ran 16 miles``), while the gold answer is just ``16``.  Comparing
    the whole string silently labels these correct rollouts as failures.
    """
    value = _clean(value).replace(",", "")
    # Handle the simple fraction form used by some math datasets.
    frac = re.search(r"(?<!\w)([-+]?\d+)\s*/\s*(\d+)(?!\w)", value)
    if frac:
        try:
            return Decimal(frac.group(1)) / Decimal(frac.group(2))
        except (InvalidOperation, ZeroDivisionError):
            pass
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    if not nums:
        return None
    try:
        return Decimal(nums[-1])
    except InvalidOperation:
        return None


def answers_equal(prediction: str, gold: str) -> bool:
    p, g = _clean(extract_answer(prediction)), _clean(extract_answer(gold))
    if not p or not g:
        return False
    pn, gn = _numeric_value(p), _numeric_value(g)
    if pn is not None and gn is not None:
        return pn == gn
    try:
        return Decimal(p.replace(",", "")) == Decimal(g.replace(",", ""))
    except (InvalidOperation, ValueError):
        return p.lower() == g.lower()


def split_steps(text: str):
    """Turn a model completion into prefix-MDP action chunks."""
    text = str(text or "").strip()
    tagged = re.findall(r"<STEP>\s*(.*?)\s*</STEP>", text, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        steps = [x.strip() for x in tagged if x.strip()]
        final = extract_answer(text)
        if has_explicit_final_marker(text) and final and not any("FINAL_ANSWER" in x for x in steps):
            steps.append(f"<FINAL_ANSWER>{final}</FINAL_ANSWER>")
        return steps or [text]

    chunks = [x.strip() for x in re.split(r"\n\s*\n+|(?=\bStep\s+\d+\s*[:.])", text) if x.strip()]
    if len(chunks) <= 1:
        return [text]
    return chunks
