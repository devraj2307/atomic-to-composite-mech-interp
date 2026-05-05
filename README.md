# Unveiling the Synthesis Circuit: Mechanistic Interpretability of RL-Driven Complementary Reasoning

> **Dev Raj** (23121) · **Rishitha Pamu** (23271) · **Aaradhya Pathak** (23005)

## Table of Contents

1. [Overview](#overview)
2. [Key Contributions](#key-contributions)
3. [Task Definition](#task-definition)
4. [Dataset](#dataset)
5. [Stage 1 — Supervised Fine-Tuning (SFT)](#stage-1--supervised-fine-tuning-sft)
6. [Stage 2 — Reinforcement Learning via GRPO](#stage-2--reinforcement-learning-via-grpo)
   - [Reward Functions](#reward-functions)
7. [Stage 3 — Unlabeled Post-Training (MM-UPT)](#stage-3--unlabeled-post-training-mm-upt)
8. [Mechanistic Interpretability](#mechanistic-interpretability)
   - [Logit Lens Analysis](#logit-lens-analysis)
   - [Activation Patching](#activation-patching)
   - [The Synthesis Circuit](#the-synthesis-circuit)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [References](#references)

---

## Overview

This repository replicates and extends the training pipeline from [Cheng et al. (2025)](#references) on **Qwen-2.5** model families (0.5B and 1.5B parameters). The project validates that:

1. **Supervised Fine-Tuning (SFT)** on atomic skills (parametric recall + contextual reading), followed by
2. **Group Relative Policy Optimization (GRPO)** on composite tasks,

enables models to generalize to **unseen relational compositions** — a capability called *Complementary Reasoning (COMP)*.

Beyond behavioral evaluation, this work introduces a **circuit-level mechanistic interpretability** study that causally identifies a sparse, two-stage **"Synthesis Circuit"** localized strictly to **layers 20–28** of the Qwen-2.5-1.5B model. The circuit explains *why* atomic skill decomposition is a necessary prerequisite for RL-driven generalization.

---

## Key Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Pipeline Replication Engineering** | Full replication of the SFT→RL pipeline on Qwen-2.5 (0.5B, 1.5B), with documented failure modes and fixes |
| 2 | **Novel CoT Entity Reward** | A custom Chain-of-Thought reward for GRPO that provides denser gradient signal than sparse binary matching |
| 3 | **MM-UPT** | Self-supervised majority-voting post-training for zero-shot generalization without human-labeled targets |
| 4 | **Synthesis Circuit** | First causal, component-level mechanistic account of why atomic skill decomposition enables RL generalization |

---

## Task Definition

### Reasoning Types

| Type | Description | Example |
|---|---|---|
| **MEM** (Parametric) | Answer requires recalling facts from model weights; no context provided | *"What book did the classmate of Jennifer Lang write?"* |
| **CTX** (Contextual) | Answer is fully derivable from the provided biography paragraph | *"What did Marie Pope major in?"* (with paragraph) |
| **COMP** (Complementary) | Requires both sources simultaneously | *"Who is the spouse of Charles Keith's advisor?"* (advisor from CTX → spouse from MEM) |

### Generalization Levels

| Level | Description |
|---|---|
| **IID** | Exact relational path appeared in training data |
| **Composition** | Individual relations were seen, but this combination is novel |
| **Zero-shot** | Entire relation type is unseen; requires out-of-distribution generalization |

---

## Dataset

Synthetic human biographies with **39 strictly defined relations**, including:
- 8 symmetric relations (e.g., *spouse*, *sibling*)
- 8 pairs of inverse relations (e.g., *husband*/*wife*)

Generated using the **Faker** library to avoid conflicts with model pre-training data. Question-answer pairs are constructed by traversing relational chains of varying lengths from a synthetic knowledge graph.

> Biography entries with empty question/answer fields are used for parametric memory injection during SFT and **must be filtered** before RL training.

### Data Statistics

| Group | Training | I.I.D. | Composition | Zero-shot |
|---|---|---|---|---|
| Parametric | 88,031 | 1,921 | 1,141 | 782 |
| Contextual | 2,651 | 1,910 | 1,320 | 453 |
| Complementary | 180,919 | 2,135 | 1,415 | 918 |

---

## Stage 1 — Supervised Fine-Tuning (SFT)

### Objective

Train on **MEM + CTX** to teach two atomic skills independently:
1. Parametric fact recall (from biography injection entries)
2. Contextual reading comprehension (from QA entries with provided context)

An `SFT(COMP)` baseline is also trained for comparison.

### Loss Function

Completion-only cross-entropy loss, masking all system and user prompt tokens:

$$\mathcal{L} = -\frac{1}{N} \sum_{i \in U} \log P(t_i \mid t_{{<}i})$$

where $\mathcal{U}$ is the set of unmasked (assistant turn) token positions and $N = |\mathcal{U}|$.

Every token in the reasoning chain contributes to the gradient — not just the final answer token.

### Hyperparameters

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Effective batch size | 128 (per device=4, grad accum=32) |
| Learning rate | `3e-4` |
| LR scheduler | Cosine with min LR (`3e-5`) |
| Epochs | 16 (no early stopping) |
| Weight decay | 0.01 |
| Precision | `bfloat16` |
| Optimizer | AdamW (fused) |
| Max sequence length | 2048 |

**Checkpointing:** Loss-threshold snapshots saved at train loss values of `0.30`, `0.15`, `0.05`, and `0.01` for post-hoc selection.

---

## Stage 2 — Reinforcement Learning via GRPO

### Objective

Starting from the converged `SFT(MEM+CTX)` checkpoint, **Group Relative Policy Optimization (GRPO)** trains on COMP to synthesize atomic skills into complementary reasoning.

> **Core Claim:** RL cannot create skills from scratch — it can only combine existing ones. Synthesis is only effective when atomic skills are already reliably present.

### GRPO Objective

Unlike PPO, GRPO eliminates a separate value network by estimating baselines from a group of sampled outputs. For a prompt $q$ with $G$ sampled completions $\{o_1, \ldots, o_G\}$:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta \, D_{\text{KL}}\!\left(\pi_\theta(o_i|q) \,\|\, \pi_{\text{ref}}(o_i|q)\right)\right)\right]$$

where:
- $\rho_i = \dfrac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)}$ is the probability ratio
- $\hat{A}_i$ is the group-relative advantage
- $\beta$ is the KL penalty coefficient

### Reward Functions

#### Baseline: Binary Exact Match

A sparse reward providing no partial credit for intermediate reasoning:

$$r = \begin{cases} 1.0 & \text{if } \text{normalize}(\hat{a}) = \text{normalize}(a^*) \\ 0.0 & \text{otherwise} \end{cases}$$

where $\hat{a}$ is extracted by matching the template `"So, the answer is:"` in the generated text, and normalization lowercases and strips punctuation.

---

#### Novel: Chain-of-Thought (CoT) Entity Reward

A hierarchically composed reward tailored for GRPO on COMP tasks. It explicitly rewards correct intermediate entity retrieval while strictly punishing hallucinated steps.

$$R(\mathcal{M}, G, \hat{a}, t) = \left[\alpha(t)\,\mathbf{1}[\hat{a} = a^*] + (1 - \alpha(t))\right] \cdot \phi(\mathcal{M}, G)^\lambda \cdot \psi(\mathcal{M}, G)$$

The three components are:

**1. Annealed Answer Gate $\alpha(t)$**

Scales from 0 to 1 over training step $t$. Allows the model to receive credit for valid entity paths early in training without the reward being immediately zeroed by a final answer mismatch.

**2. Count Alignment $\phi(\mathcal{M}, G)$**

$$\phi(\mathcal{M}, G) = \exp\!\left(-\frac{(|\mathcal{M}| - |G|)^2}{2|G|}\right)$$

A Gaussian kernel that symmetrically penalizes both over-generation and under-generation of entities. The hyperparameter $\lambda$ controls alignment strictness — a count mismatch exponentially suppresses the total reward.

**3. Entity Recall $\psi(\mathcal{M}, G)$**

$$\psi(\mathcal{M}, G) = \frac{|G \cap \mathcal{M}|}{|G|}$$

Measures the proportion of ground-truth entities $G$ successfully retrieved in the generated set $\mathcal{M}$. Due to exponential suppression from $\phi$, this acts as a differentiator only when the generated entity count closely approximates the ground-truth count.

### GRPO Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `SFT(MEM+CTX)` at loss 0.05 |
| Rollouts per prompt $G$ | 8 |
| Per-device train batch size | 32 |
| Max completion length | 128 |
| Learning rate | `1e-5` |
| LR scheduler | Cosine with min LR (`5e-7`) |
| KL coefficient $\beta$ | 0.001 |
| Epochs | 2 |

---

## Stage 3 — Unlabeled Post-Training (UPT)

### Motivation

Standard GRPO requires ground-truth labels to compute rewards, limiting reinforcement to annotated datasets. **UPT** (Majority-vote based Unlabeled Post-Training) allows the model to act as its own teacher, enabling continued training on *any* unlabeled complementary data — including distributions mimicking the test set.

### Reward Formulation

In standard GRPO, the binary reward requires a known ground truth $a^*$:

$$r_i = \begin{cases} 1 & \text{if } \text{extract}(o_i) = a^* \\ 0 & \text{otherwise} \end{cases}$$

Under UPT, $a^*$ is replaced by a **dynamically generated pseudo-target** based on group consensus. For $G$ rollouts with extracted answers $\mathcal{A} = \{\text{extract}(o_1), \ldots, \text{extract}(o_G)\}$:

$$y^* = \text{mode}(\mathcal{A})$$

The updated reward function is:

$$r_i = \begin{cases} 1 & \text{if } \text{extract}(o_i) = y^* \\ 0 & \text{otherwise} \end{cases}$$

By rewarding agreement with the group majority, the model reinforces its own internal consistency and reasoning paths, bootstrapping performance on entirely novel entity relations without human-annotated targets.

---

## Mechanistic Interpretability

### Motivation

The **Context-to-Memory (CTX→MEM) Handoff** is the core operation under study: resolving a bridge entity from context and using it to retrieve a relational fact from parametric memory.

**Example prompt:**
> *"Tyler Carter works under Taylor Jackson. Who is the classmate of the boss of Tyler Carter?"*

Answering requires:
1. Reading from context: Taylor Jackson is Tyler Carter's boss (CTX hop)
2. Retrieving from parametric memory: who Taylor Jackson's classmate is (MEM hop)

**Hypothesis:** `SFT(MEM+CTX)→RL` training produces a **sparse, interpretable Synthesis Circuit** implementing this handoff; `SFT(COMP)` produces diffuse, shortcut-based representations that fail on novel compositions.

---

### Logit Lens Analysis

The logit lens projects the residual stream at each transformer layer through the final LayerNorm and unembedding matrix, obtaining a layer-wise probability distribution over the vocabulary. The rank and probability of the correct answer token are tracked at the final sequence position across 500 discovery-set prompts.

#### Key Statistics (500 zero-shot composition prompts)

| Model | Final Prob. | Peak Layer | Median Emergence | Never Top-1 |
|---|---|---|---|---|
| `SFT(MEM+CTX)→RL` (Model A) | 0.0240 | 28 | 23.0 | 480/500 |
| `SFT(COMP)` (Model B) | 0.0174 | 28 | 23.0 | 484/500 |

Model A achieves a **38% relative improvement** in final-layer mean probability on the correct answer token.

#### Findings

- **Layers 0–20:** Both models follow nearly identical rank trajectories — domain-general linguistic processing, unaffected by training regime. Both reach a local minimum around rank 800 at layer 20.
- **Layers 21–25:** A transient rank spike (representational reorganization) occurs in both models.
- **Layers 25–28:** Model A's descent is sharper and more abrupt, consistent with a discrete late-layer circuit performing the CTX→MEM handoff.
- **Probability divergence** $\text{Prob}(A) - \text{Prob}(B)$: Near-zero and alternating for layers 0–19; consistently positive and growing from layer 20 onward, peaking at layers 27 and 28. This establishes **layers 20–28 as the primary region of mechanistic interest**.
- **Bridge position analysis:** Uniformly flat signal at the bridge entity's token position for both models across all layers. This rules out a two-stage pre-assembly hypothesis. The CTX→MEM synthesis occurs directly at the **final token position**, with the bridge entity serving as an attended-to key rather than an intermediate storage location.

---

### Activation Patching

Guided by the logit lens, activation patching is performed across layers 20–28. For each attention head $H$ at layer $L$ and each MLP layer, Model B's activation is replaced with the corresponding activation from Model A on the same prompt. The **recovery fraction** is defined as:

$$\text{Recovery Fraction} = \frac{\Delta\text{rank}(\text{patched B}) - \Delta\text{rank}(\text{B})}{\Delta\text{rank}(\text{A}) - \Delta\text{rank}(\text{B})}$$

A component is classified as **critical** if its recovery fraction $\geq 0.10$.

**Prompt partitioning** (500 total prompts):
- **Gap subset** (309 prompts): Model B's rank is worse than Model A's
- **Parity subset** (191 prompts): Models perform similarly

#### MLP Dominance (Gap Subset)

| Layer | MLP Recovery Fraction |
|---|---|
| L20 | 0.218 |
| L21 | ~0 |
| L22 | **1.000** |
| L23 | **1.000** |
| L24 | **1.000** |
| L25 | ~0.05 |
| L26 | 0.868 |
| L27 | 0.744 |
| L28 | 0 (silent) |

The monotonically increasing pattern from L20 through L26 indicates a **cascade of MLP transformations**, each building on the previous. The final answer commitment is distributed across this range rather than concentrated in a single layer.

#### Attention Head Contributions (Gap Subset)

Critical heads are sparse and scattered; no single head dominates:

| Head | Recovery Fraction |
|---|---|
| L22 H10 | 0.34 |
| L25 H4 | 0.31 |
| L25 H3 | 0.26 |
| L21 H6 | 0.24 |
| L23 H0 | 0.22 |

Individual head recovery fractions are substantially lower than MLP recovery fractions at the same layers.

---

### The Synthesis Circuit

#### Two-Stage Structure

| Stage | Layer Range | Mechanism |
|---|---|---|
| **Routing** | L20–L22 | Critical attention heads identify the bridge entity in context and attend to relevant relational structure, preparing the residual stream for retrieval |
| **Retrieval** | L22–L27 | Cascade of MLP transformations progressively surfaces the correct parametric association — the entity related to the bridge entity by the target relation |

The **crossover at layer 22** reveals a clean functional separation: attention-driven routing gives way to MLP-driven parametric memory retrieval.

#### Circuit Summary (Gap Subset, $n = 309$)

| Component | Layer Range | Max Recovery |
|---|---|---|
| MLP (parametric retrieval) | 22–27 | 1.000 (L22, L23, L24) |
| MLP (early preparation) | 20 | 0.218 |
| Attention heads (routing) | 20–27 | 0.340 (L22 H10) |

**Total: 25 critical components — 19 attention heads + 6 MLP layers**, concentrated in layers 20–27 (29% of network depth). Layer 28 is entirely silent — the circuit completes before the final layer.

#### Why SFT(COMP) Fails

`SFT(COMP)` never learns to decompose the task atomically. Without independent exposure to contextual and parametric reasoning as separate skills:
- Attention heads in layers 20–22 **fail to route to the genuine bridge entity**, instead latching onto surface-level lexical patterns
- The downstream MLP cascade receives an imprecise residual stream and **cannot reliably retrieve the correct association**, even if individual MLP weights are otherwise capable

#### Domain Specificity

Evaluation on **GSM8K** mathematical word problems (without retraining) revealed complete **catastrophic forgetting**: Model A produced biographical relation-chain outputs in response to arithmetic problems. This confirms the Synthesis Circuit is a **domain-specific mechanism** shaped by the training distribution, not a general compositional reasoning module.

---

## Results

### SFT: Atomic Skill Learning

| Task Type | Model Size | Overall | I.I.D. | Composition | Zero-shot |
|---|---|---|---|---|---|
| Parametric (MEM) | 1.5B | 66.78% | 87.56% | 75.46% | 3.07% |
| Parametric (MEM) | 0.5B | 51.46% | 65.64% | 60.91% | 2.81% |
| Contextual (CTX) | 1.5B | 75.07% | 84.66% | 72.35% | 42.60% |
| Contextual (CTX) | 0.5B | 75.21% | 87.02% | 73.71% | 29.80% |
| Complementary (COMP) | 1.5B | 13.09% | 10.59% | 14.49% | 16.78% |
| Complementary (COMP) | 0.5B | 13.00% | 10.87% | 12.86% | 18.19% |

*Exact Match accuracy. `SFT(MEM+CTX)` model evaluated on the COMP test set.*

### GRPO and UPT Progression (1.5B Model, COMP Test Set)

| Training Setup | Overall | I.I.D. | Composition | Zero-shot |
|---|---|---|---|---|
| Base: `SFT(MEM+CTX)` at loss 0.05 | 13.09% | 10.59% | 14.49% | 16.78% |
| RL(COMP) CoT reward (12.8k samples) | 22.83% | 21.92% | 24.45% | 22.44% |
| RL(COMP) Binary reward (36k samples) | 38.70% | 42.90% | 38.52% | 29.19% |
| RL(COMP) CoT reward (36k samples) | **39.64%** | **44.31%** | 38.45% | **30.61%** |
| RL(COMP) CoT (36k) + UPT (12.8k) | 40.04% | **45.34%** | 38.30% | 30.39% |
| `SFT(COMP)` (180k) — upper bound | 51.01% | 62.67% | 47.63% | 29.08% |

**Key observations:**
- The CoT entity reward consistently outperforms binary reward
- UPT provides a marginal additional gain, especially on I.I.D. accuracy
- `SFT(COMP)` achieves higher I.I.D. accuracy but comparable zero-shot accuracy to the RL pipeline, consistent with shortcut memorization

---

## Conclusion

This work provides the first causal, component-level explanation for why atomic skill decomposition is a necessary prerequisite for RL-driven generalization in complementary reasoning:

1. **The first 20 layers** are computationally identical between `SFT(MEM+CTX)→RL` and `SFT(COMP)` models — behavioral divergence is entirely localized to **layers 20–28**.

2. A **sparse Synthesis Circuit** of 19 attention heads and 6 MLP layers in layers 20–27 implements the CTX→MEM handoff in two stages:
   - **Routing (L20–L22):** Attention heads identify the bridge entity from context
   - **Retrieval (L22–L27):** MLP cascade performs parametric memory retrieval

3. **MLP dominance** (layers 22–24 each achieving full recovery independently) is consistent with MLPs as implicit key-value stores for relational knowledge.

4. Synthesis occurs **entirely at the final output token position** — the bridge entity serves as an attention key, not an intermediate storage location.

5. The circuit is **domain-specific**: the fine-tuned model undergoes catastrophic forgetting of pre-training capabilities (evidenced by GSM8K failure).

---

## References

1. S. Cheng et al., *"From Atomic to Composite: Reinforcement Learning Enables Generalization in Complementary Reasoning,"* arXiv:2512.01970v2, 2025.
2. Lai Wei et al., *"First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training,"* NeurIPS 2025.
3. Z. Shao et al., *"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models,"* arXiv:2402.03300, 2024.
4. K. Meng et al., *"Locating and Editing Factual Associations in GPT,"* NeurIPS, 2022.
5. Z. Allen-Zhu and Y. Li, *"Physics of Language Models: Part 3.1, Knowledge Storage and Extraction,"* arXiv:2309.14316, 2023.
6. M. Geva et al., *"Transformer Feed-Forward Layers Are Key-Value Memories,"* arXiv:2012.14913, 2021.
7. K. Cobbe et al., *"Training Verifiers to Solve Math Word Problems,"* arXiv:2110.14168, 2021.
