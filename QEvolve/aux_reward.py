#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrospective auxiliary-reward rules for all environments, in one place.

Implements the auxiliary penalty r_aux of  r_hat_t = r_env_t + r_aux_t
(paper Fig. 2, "Retrospective Labeling"). Applied in memory during dataset
preprocessing (see build_traj_dataset), so no intermediate *.assigned.jsonl
files are needed. Only the env-specific patterns / priorities differ.
"""

import re
from typing import Any, Dict, List, Tuple, Optional

# ---- auxiliary penalty values (shared; paper Fig. 2) ----
AUX_REWARD = {
    "format_invalid":   -0.3,   # Thought/Action format invalid
    "env_incompatible": -0.2,   # action not accepted by the environment
    "no_change":        -0.1,   # action produced no change in observation
}

# ---- env-specific rules, all together ----
# no_match_obs : observation substrings that signal an env-incompatible action
# check_repeat : whether to apply the "no_change" (repeated observation) rule
# priority     : order in which rules are checked; first match wins
AUX_RULES: Dict[str, Dict[str, Any]] = {
    "alfworld": {
        "no_match_obs": ["Nothing happens", "Error Input"],
        "check_repeat": True,
        "priority": ["env_incompatible", "format_invalid", "no_change"],
    },
    "sciworld": {
        "no_match_obs": [
            "Observation: No known action matches that input.",
            "Observation: Unknown action.",
            "Observation: Ambiguous request",
        ],
        "check_repeat": True,
        "priority": ["env_incompatible", "format_invalid", "no_change"],
    },
    "webshop": {
        "no_match_obs": ["Observation: Invalid format. The input must contains 'Action:"],
        "check_repeat": False,
        "priority": ["format_invalid", "env_incompatible"],
    },
}

# strict format: "Thought: ... \n Action: ..."  (same regex for every env)
THOUGHT_ACTION_RE = re.compile(r"(?s)^\s*Thought\s*:\s*.+?\n\s*Action\s*:\s*(.+?)\s*$")


def _is_gpt(m: Dict[str, Any]) -> bool:
    return str(m.get("from", "")).lower() in ("gpt", "assistant")

def _text(m: Dict[str, Any]) -> str:
    return str(m.get("value", "") or "")

def _parse_action(gpt_text: str) -> Tuple[bool, str]:
    m = THOUGHT_ACTION_RE.match(gpt_text.strip())
    if not m:
        return False, ""
    return True, (m.group(1) or "").strip()

def _is_wait(action: str) -> bool:
    return "wait" in (action or "").strip().lower()

def _norm_obs(obs_text: str) -> str:
    t = (obs_text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def assign_aux_rewards(convs: List[Dict[str, Any]], task: str) -> List[float]:
    """Per-step auxiliary reward for one trajectory, using the env's rules."""
    cfg = AUX_RULES[task]
    no_match, check_repeat, priority = cfg["no_match_obs"], cfg["check_repeat"], cfg["priority"]

    rewards: List[float] = []
    last_obs_norm: Optional[str] = None
    i, n = 2, len(convs)  # skip the first human/gpt (task prompt) pair
    while i < n:
        if not _is_gpt(convs[i]):
            i += 1
            continue
        fmt_ok, action = _parse_action(_text(convs[i]))
        j = i + 1
        obs_text = _text(convs[j]) if j < n else ""
        obs_norm = _norm_obs(obs_text)

        def fires(rule: str) -> bool:
            if rule == "format_invalid":
                return not fmt_ok
            if rule == "env_incompatible":
                return any(p in obs_text for p in no_match)
            if rule == "no_change":
                return (check_repeat and last_obs_norm is not None
                        and obs_norm == last_obs_norm and not _is_wait(action))
            return False

        r = 0.0
        for rule in priority:
            if fires(rule):
                r = AUX_REWARD[rule]
                break
        rewards.append(float(r))

        if obs_norm:
            last_obs_norm = obs_norm
        i = j + 1 if j > i else i + 1
    return rewards
