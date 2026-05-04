import json
import math
from typing import List

with open("") as f:
    ENTITY_SET = set(json.load(f))


LAMBDA = 1.0    
K      = 10.0   

CURRENT_STEP = 0
T            = 1000  



def sigmoid_alpha(t: int, total_steps: int, k: float) -> float:
    """
    Annealing schedule for the answer gate weight.
    - At t=0          : alpha ≈ 0.0  →  gate ≈ 1.0  (answer correctness ignored)
    - At t=T/2        : alpha ≈ 0.5  →  gate starts caring about the answer
    - At t=T          : alpha ≈ 1.0  →  gate ≈ hard indicator on answer correctness
    """
    return 1.0 / (1.0 + math.exp(-k * ((t / total_steps) - 0.5)))


def extract_answer(text: str) -> str:
    marker    = "so, the answer is:"
    lowered   = text.lower()
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

def reward_fn(prompts: List[str], completions: List[str], answer: List[str], entities: List[List[str]], **kwargs) -> List[float]:
    """
    Hierarchical entity-aware reward with sigmoid-annealed answer gate.

    R(C, G, a, t) = gate(t) * phi(C, G) * psi(C, G)

    Where:
        gate(t) = alpha(t) * 1[a_hat == a] + (1 - alpha(t))
            Sigmoid-annealed answer correctness gate. Starts at 1.0
            (answer ignored) and decays toward a hard indicator by step T.

        phi(C, G) = exp( -lambda * (|M| - |G|)^2 / (2 * |G|) )
            Gaussian count alignment kernel. Peaks at 1.0 when the number
            of mentioned entities exactly matches the ground-truth count.
            Variance scaled by |G| so tolerance widens for larger entity sets.

        psi(C, G) = |G ∩ M| / |G|
            Recall of ground-truth entities over the model's CoT. Acts as
            the secondary differentiator once count alignment is satisfied.

        M = { e in U | e appears in C }
            Set of known entities (from global universe U) mentioned in CoT.

    Args:
        prompts     : input prompts (unused, required by TRL interface)
        completions : model-generated completions (CoT + answer)
        answer      : ground-truth final answers
        entities    : ground-truth entity lists per datapoint
    
    Returns:
        List of scalar rewards in [0, 1].
    """
    rewards = []
    alpha   = sigmoid_alpha(CURRENT_STEP, T, K)

    for completion, gt_answer, gt_entities in zip(completions, answer, entities):

        C = completion.strip().lower()

        # G: ground-truth entity set for this datapoint
        G = set(e.strip().lower() for e in gt_entities)

        # M: subset of global entity universe mentioned in the CoT
        M = set(e for e in ENTITY_SET if e in C)

        if len(G) == 0:
            rewards.append(1.0 if len(M) == 0 else 0.0)
            continue

        predicted     = extract_answer(completion)
        expected      = gt_answer.strip().rstrip(".").lower()
        answer_correct = 1.0 if predicted == expected else 0.0
        gate          = alpha * answer_correct + (1.0 - alpha)

        phi = math.exp(-LAMBDA * (len(M) - len(G)) ** 2 / (2 * len(G)))

        psi = len(G & M) / len(G)

        rewards.append(gate * phi * psi)

    return rewards