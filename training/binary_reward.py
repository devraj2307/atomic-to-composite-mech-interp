import re
from typing import List

def extract_answer(text: str) -> str:
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

def reward_fn(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """
    Returns 1.0 if the extracted answer matches ground truth, 0.0 otherwise.
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        predicted = extract_answer(completion)
        expected  = gt.strip().rstrip(".").lower()
        rewards.append(1.0 if predicted == expected else 0.0)
    return rewards