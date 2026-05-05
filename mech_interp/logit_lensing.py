import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils

# Model A here is the SFT + RL model
# Model B here is the SFT trained model on the COMP dataset

@dataclass
class LogitLensConfig:
    model_a_path: str =      
    model_b_path: str =    
    base_model_name: str = "Qwen/Qwen2.5-1.5B"       

    dataset_path: str = "./data/discovery_set.json"   
    max_prompts: int = 500                             

    apply_ln: bool = True           
    target_position: str = "answer" 
    top_k: int = 10                  

    output_dir: str = "./results/logit_lens"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

def load_dataset(path: str, max_prompts: int) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:max_prompts]


def load_hooked_model(
    base_model_name: str,
    checkpoint_path: str,
    device: str
) -> HookedTransformer:
    from transformers import AutoModelForCausalLM

    import os
    from pathlib import Path as _Path
    checkpoint_path = _Path(os.path.abspath(checkpoint_path))
    print(f"  Loading HF weights from: {checkpoint_path}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,             
        torch_dtype=torch.float32,   
        local_files_only=True,       
    )

    print(f"  Wrapping with TransformerLens ...")
    model = HookedTransformer.from_pretrained(
        base_model_name,             
        hf_model=hf_model,           
        fold_ln=False,               
        center_writing_weights=True,
        center_unembed=True,
    )
    model = model.to(device)        
    model.eval()
    model.cfg.default_prepend_bos = False
    print(f"  Model device: {next(model.parameters()).device}")
    return model

def cache_residual_stream(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Run a forward pass and cache the residual stream after every layer.
    """
    names_filter = lambda name: (
        name == "hook_embed"
        or name.endswith("hook_resid_post")
    )

    _, cache = model.run_with_cache(
        tokens,
        names_filter=names_filter,
        return_type=None,  
    )
    return cache

def project_residual_to_logits(
    model: HookedTransformer,
    residual: torch.Tensor,
    apply_ln: bool = True,
) -> torch.Tensor:
    """
    Project one layer's residual stream through the final LayerNorm + unembedding.
    """
    if apply_ln:
        residual = model.ln_final(residual)
    logits = model.unembed(residual)   
    return logits


def get_target_token_id(model: HookedTransformer, token_str: str) -> int:
    token_ids = model.to_tokens(token_str, prepend_bos=False).squeeze()
    if token_ids.ndim > 0:
        return token_ids[0].item()
    return token_ids.item()

@dataclass
class PromptResult:
    """Stores per-layer logit lens results for a single prompt."""
    n_layers: int
    target_token_id: int
    target_logit:    list[float] = field(default_factory=list)
    target_prob:     list[float] = field(default_factory=list)
    target_rank:     list[int]   = field(default_factory=list)
    top_token_ids:   list[list[int]] = field(default_factory=list)  


def analyze_prompt(
    model: HookedTransformer,
    prompt: str,
    answer_token: str,
    bridge_position: Optional[int],
    config: LogitLensConfig,
    track_position: str = "answer_final", 
) -> PromptResult:
    tokens = model.to_tokens(prompt, prepend_bos=True) 
    seq_len = tokens.shape[1]

    if track_position == "bridge" and bridge_position is not None:
        seq_pos = bridge_position
    else:
        seq_pos = seq_len - 1  

    cache = cache_residual_stream(model, tokens)

    target_id = get_target_token_id(model, answer_token)
    n_layers = model.cfg.n_layers

    result = PromptResult(n_layers=n_layers, target_token_id=target_id)

    resid_layers = ["hook_embed"] + [
        f"blocks.{l}.hook_resid_post" for l in range(n_layers)
    ]

    for hook_name in resid_layers:
        resid = cache[hook_name][0]          
        logits = project_residual_to_logits(  
            model, resid, apply_ln=config.apply_ln
        )
        pos_logits = logits[seq_pos]          

        probs = torch.softmax(pos_logits, dim=-1)
        sorted_ids = torch.argsort(pos_logits, descending=True)
        rank = (sorted_ids == target_id).nonzero(as_tuple=True)[0].item()

        result.target_logit.append(pos_logits[target_id].item())
        result.target_prob.append(probs[target_id].item())
        result.target_rank.append(rank)
        result.top_token_ids.append(sorted_ids[:config.top_k].tolist())

    return result

def run_logit_lens_on_dataset(
    model: HookedTransformer,
    dataset: list[dict],
    config: LogitLensConfig,
    model_label: str,
    track_position: str = "answer_final",
) -> dict:
    n_layers = model.cfg.n_layers
    all_probs  = []
    all_ranks  = []
    first_top1 = []

    for item in tqdm(dataset, desc=f"Logit lens [{model_label}]"):
        result = analyze_prompt(
            model,
            prompt=item["clean_prompt"],
            answer_token=item["answer_token"],
            bridge_position=item.get("bridge_position"),
            config=config,
            track_position=track_position,
        )
        all_probs.append(result.target_prob)
        all_ranks.append(result.target_rank)

        try:
            first_top1.append(next(i for i, r in enumerate(result.target_rank) if r == 0))
        except StopIteration:
            first_top1.append(n_layers + 1)  

    all_probs  = np.array(all_probs)  
    all_ranks  = np.array(all_ranks)

    return {
        "all_probs":         all_probs,
        "all_ranks":         all_ranks,
        "mean_prob":         all_probs.mean(axis=0),
        "std_prob":          all_probs.std(axis=0),
        "median_rank":       np.median(all_ranks, axis=0),
        "first_top1_layer":  first_top1,
        "n_layers":          n_layers,
    }

def plot_confidence_curves(
    results_a: dict,
    results_b: dict,
    output_path: str,
    title: str = "Logit Lens: Target Token Probability by Layer",
):
    """
    Mean ± std confidence curves for Model A and Model B side by side.
    The x-axis is layer depth (0 = embedding, N = final layer).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, results, label, color in zip(
        axes,
        [results_a, results_b],
        ["Model A  (SFT_MEM+CTX → RL)", "Model B  (SFT_COMP)"],
        ["steelblue", "tomato"],
    ):
        n = results["n_layers"] + 1
        x = np.arange(n)
        mu = results["mean_prob"]
        sd = results["std_prob"]

        ax.plot(x, mu, color=color, linewidth=2, label=label)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.2, color=color)
        ax.set_xlabel("Layer (0 = embedding)", fontsize=12)
        ax.set_ylabel("Target Token Probability", fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def plot_rank_curves(
    results_a: dict,
    results_b: dict,
    output_path: str,
):
    """
    Median rank of the target token across layers. Log-scale y-axis handles the large dynamic range.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    n = results_a["n_layers"] + 1
    x = np.arange(n)

    ax.semilogy(x, results_a["median_rank"] + 1, color="steelblue",
                linewidth=2, label="Model A (SFT_MEM+CTX → RL)")
    ax.semilogy(x, results_b["median_rank"] + 1, color="tomato",
                linewidth=2, label="Model B (SFT_COMP)", linestyle="--")

    ax.set_xlabel("Layer (0 = embedding)", fontsize=12)
    ax.set_ylabel("Median Rank of Target Token (log scale)", fontsize=12)
    ax.set_title("Target Token Rank Across Layers", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(0, n - 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def plot_heatmap(
    results: dict,
    output_path: str,
    model_label: str,
    max_prompts_display: int = 100,
):
    """
    Heatmap of target token probability: prompts × layers.
    Rows = individual prompts, cols = layers.
    """
    probs = results["all_probs"][:max_prompts_display]
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        probs,
        ax=ax,
        cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=max(1, probs.shape[1] // 10),
        yticklabels=False,
        cbar_kws={"label": "Target Token Probability"},
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Prompt", fontsize=12)
    ax.set_title(
        f"Logit Lens Heatmap — {model_label} (first {max_prompts_display} prompts)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def plot_first_top1_histogram(
    results_a: dict,
    results_b: dict,
    output_path: str,
):
    """
    Histogram of the first layer where each model's target token hits rank 0.
    """
    n = results_a["n_layers"] + 1
    bins = np.arange(0, n + 2) - 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(results_a["first_top1_layer"], bins=bins, alpha=0.6,
            color="steelblue", label="Model A (SFT_MEM+CTX → RL)")
    ax.hist(results_b["first_top1_layer"], bins=bins, alpha=0.6,
            color="tomato", label="Model B (SFT_COMP)")

    ax.set_xlabel("First Layer Where Target Token Hits Rank 0", fontsize=12)
    ax.set_ylabel("Number of Prompts", fontsize=12)
    ax.set_title("Distribution of Answer Emergence Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def plot_divergence(
    results_a: dict,
    results_b: dict,
    output_path: str,
):
    """
    Layer-wise divergence: mean_prob(A) - mean_prob(B).
    """
    diff = results_a["mean_prob"] - results_b["mean_prob"]
    n = results_a["n_layers"] + 1
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, diff, color=np.where(diff > 0, "steelblue", "tomato"), width=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Prob(A) − Prob(B)", fontsize=12)
    ax.set_title(
        "Layer-wise Probability Divergence (Model A − Model B)\n"
        "Peaks indicate candidate Synthesis Circuit layers",
        fontsize=12, fontweight="bold"
    )
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")

def print_summary(results_a: dict, results_b: dict):
    """Print key statistics to guide downstream activation patching decisions."""
    print("\n" + "=" * 60)
    print("LOGIT LENS SUMMARY")
    print("=" * 60)

    for results, label in [(results_a, "Model A (SFT_MEM+CTX → RL)"),
                           (results_b, "Model B (SFT_COMP)")]:
        final_prob = results["mean_prob"][-1]
        peak_layer = int(np.argmax(results["mean_prob"]))
        never_top1 = sum(
            l > results["n_layers"] for l in results["first_top1_layer"]
        )
        median_emergence = float(np.median([
            l for l in results["first_top1_layer"]
            if l <= results["n_layers"]
        ] or [results["n_layers"] + 1]))

        print(f"\n  {label}")
        print(f"    Final layer mean prob :  {final_prob:.4f}")
        print(f"    Peak mean-prob layer  :  {peak_layer}")
        print(f"    Median emergence layer:  {median_emergence:.1f}")
        print(f"    Never reached top-1   :  {never_top1} prompts")

    diff = results_a["mean_prob"] - results_b["mean_prob"]
    peak_divergence_layer = int(np.argmax(diff))
    print(f"\n  Peak divergence at layer  :  {peak_divergence_layer}")
    print(f"  → Prioritise this layer range for activation patching.")
    print("=" * 60 + "\n")


def save_results(results_a: dict, results_b: dict, output_dir: str):
    out = Path(output_dir) / "arrays"
    out.mkdir(parents=True, exist_ok=True)

    for results, prefix in [(results_a, "model_a"), (results_b, "model_b")]:
        np.save(out / f"{prefix}_all_probs.npy",  results["all_probs"])
        np.save(out / f"{prefix}_all_ranks.npy",  results["all_ranks"])
        np.save(out / f"{prefix}_mean_prob.npy",  results["mean_prob"])
        np.save(out / f"{prefix}_median_rank.npy", results["median_rank"])

    print(f"[Saved] Arrays → {out}")


def parse_args() -> LogitLensConfig:
    parser = argparse.ArgumentParser(description="Logit Lens — Complementary Reasoning Study")
    parser.add_argument("--model_a_path",    default="")
    parser.add_argument("--model_b_path",    default="")
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset_path",    default="./data/discovery_set.json")
    parser.add_argument("--max_prompts",     type=int, default=500)
    parser.add_argument("--output_dir",      default="./results/logit_lens")
    parser.add_argument("--target_position", default="answer_final",
                        choices=["answer_final", "bridge"])
    parser.add_argument("--no_ln",           action="store_true",
                        help="Skip final LayerNorm before unembedding (not recommended)")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    return LogitLensConfig(
        model_a_path=args.model_a_path,
        model_b_path=args.model_b_path,
        base_model_name=args.base_model_name,
        dataset_path=args.dataset_path,
        max_prompts=args.max_prompts,
        output_dir=args.output_dir,
        target_position=args.target_position,
        apply_ln=not args.no_ln,
        seed=args.seed,
    )


def main():
    config = parse_args()
    torch.manual_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {config.dataset_path} ...")
    dataset = load_dataset(config.dataset_path, config.max_prompts)
    print(f"  {len(dataset)} prompts loaded.")

    print(f"\nLoading Model A from {config.model_a_path} ...")
    model_a = load_hooked_model(config.base_model_name, config.model_a_path, config.device)

    print(f"Loading Model B from {config.model_b_path} ...")
    model_b = load_hooked_model(config.base_model_name, config.model_b_path, config.device)

    print("\n── Analysing answer-final position ──")
    results_a = run_logit_lens_on_dataset(
        model_a, dataset, config, "Model A", track_position="answer_final"
    )
    results_b = run_logit_lens_on_dataset(
        model_b, dataset, config, "Model B", track_position="answer_final"
    )

    print("\n── Analysing bridge-token position ──")
    results_a_bridge = run_logit_lens_on_dataset(
        model_a, dataset, config, "Model A [bridge]", track_position="bridge"
    )
    results_b_bridge = run_logit_lens_on_dataset(
        model_b, dataset, config, "Model B [bridge]", track_position="bridge"
    )
    out = config.output_dir

    plot_confidence_curves(
        results_a, results_b,
        output_path=f"{out}/confidence_curves_answer.png",
        title="Logit Lens: Target Token Probability by Layer (Answer Position)",
    )
    plot_confidence_curves(
        results_a_bridge, results_b_bridge,
        output_path=f"{out}/confidence_curves_bridge.png",
        title="Logit Lens: Target Token Probability by Layer (Bridge Position)",
    )
    plot_rank_curves(results_a, results_b,
                     output_path=f"{out}/rank_curves.png")
    plot_heatmap(results_a, f"{out}/heatmap_model_a.png", "Model A")
    plot_heatmap(results_b, f"{out}/heatmap_model_b.png", "Model B")
    plot_first_top1_histogram(results_a, results_b,
                              output_path=f"{out}/emergence_histogram.png")
    plot_divergence(results_a, results_b,
                    output_path=f"{out}/divergence.png")

    print_summary(results_a, results_b)
    save_results(results_a, results_b, out)

    print("Done.")


if __name__ == "__main__":
    main()