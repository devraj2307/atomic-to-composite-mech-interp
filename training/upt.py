import math
from collections import Counter
from typing import List

MAJORITY_STATS = {
    "total_groups"   : 0,
    "no_template"    : 0,  
    "all_identical"  : 0,  
    "weak_consensus" : 0,   
    "rewarded"       : 0,   
}


def extract_answer(text: str) -> str:
    """
    Used to extract the final answer after the 'so, the answer is:' template else returns empty string if template not found.
    """
    marker  = "so, the answer is:"
    lowered = text.lower()
    if marker in lowered:
        prediction = lowered.split(marker)[1].strip()
        prediction = (
            prediction
            .replace("<|endoftext|>", "")
            .replace("<|im_end|>", "")
            .strip()
            .rstrip(".")
        )
        return prediction
    return ""



def reward_fn(
    prompts:     List[str],
    completions: List[str],
    answer:      List[str],   
    **kwargs
) -> List[float]:
    """
    Unsupervised MM-UPT reward via majority voting.

    For each group of G rollouts from the same prompt:
      1. Extract answer from each rollout using rule-based extractor.
      2. y* = most frequent answer among non-empty extractions.
      3. ri = 1.0 if extract(oi) == y*, else 0.0.

    Ground truth labels are ignored entirely — this is unsupervised.

    Three cases produce zero reward for the whole group:
      - All rollouts fail to produce the answer template (no_template).
      - All G rollouts agree on the same answer (all_identical):
          When top_count == G, advantage = (1 - mean) / std = 0 since all
          rewards are equal. GRPO learns nothing. We skip explicitly.
      - Top answer count < G // 2 (weak_consensus):
          Consensus too weak to be a reliable pseudo-label.
          Skipping avoids reinforcing randomly agreed wrong answers.
    """
    assert len(completions) % len(prompts) == 0, (
        f"completions ({len(completions)}) not divisible by "
        f"prompts ({len(prompts)})"
    )
    G       = len(completions) // len(prompts)
    rewards = []

    for group_start in range(0, len(completions), G):
        group     = completions[group_start : group_start + G]
        extracted = [extract_answer(c) for c in group]
        non_empty = [a for a in extracted if a != ""]

        MAJORITY_STATS["total_groups"] += 1

        if len(non_empty) == 0:
            MAJORITY_STATS["no_template"] += 1
            rewards.extend([0.0] * G)
            continue

        top_answer, top_count = Counter(non_empty).most_common(1)[0]

        if top_count == G:
            MAJORITY_STATS["all_identical"] += 1
            rewards.extend([0.0] * G)
            continue

        if top_count < G // 2:
            MAJORITY_STATS["weak_consensus"] += 1
            rewards.extend([0.0] * G)
            continue

        MAJORITY_STATS["rewarded"] += 1
        majority = top_answer
        for ans in extracted:
            rewards.append(1.0 if ans == majority else 0.0)

    return rewards