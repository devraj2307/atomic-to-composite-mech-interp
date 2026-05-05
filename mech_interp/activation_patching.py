import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

@dataclass
class PatchingConfig:
    model_a_path:    str = ""     
    model_b_path: str = "" 
    base_model_name: str = "Qwen/Qwen2.5-1.5B"
    dataset_path:    str = "./data/discovery_set.json"
    output_dir:      str = "./results/patching"

    layers_to_patch: list[int] = field(default_factory=lambda: [20,21,22,23,24, 25, 26, 27, 28])

    patch_heads:     bool = True   
    patch_mlp:       bool = True    
    patch_residual:  bool = False  

    max_prompts:     int  = 500
    device:          str  = "cuda" if torch.cuda.is_available() else "cpu"
    seed:            int  = 42

    critical_threshold: float = 0.1 

def load_hooked_model(
    base_model_name: str,
    checkpoint_path: str,
    device: str,
) -> HookedTransformer:
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
    print(f"  Device: {next(model.parameters()).device}")
    return model

def load_dataset(path: str, max_prompts: int) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:max_prompts]

def get_answer_token_id(model: HookedTransformer, answer: str) -> int:
    for candidate in [" " + answer, answer, " " + answer.split()[0], answer.split()[0]]:
        ids = model.to_tokens(candidate, prepend_bos=False).squeeze()
        if ids.ndim == 0:
            return ids.item()
        if ids.shape[0] == 1:
            return ids[0].item()
    ids = model.to_tokens(" " + answer, prepend_bos=False).squeeze()
    return ids[0].item()


def get_answer_rank(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answer_token_id: int,
) -> int:
    with torch.no_grad():
        logits = model(tokens)                 
    final_logits = logits[0, -1, :]              
    sorted_ids = torch.argsort(final_logits, descending=True)
    rank = (sorted_ids == answer_token_id).nonzero(as_tuple=True)[0].item()
    return int(rank)

def get_hook_names(
    model: HookedTransformer,
    layers: list[int],
    patch_heads: bool,
    patch_mlp: bool,
    patch_residual: bool,
) -> list[str]:
    """
    Build the list of hook names to patch.
    """
    hooks = []
    for L in layers:
        if patch_heads:
            hooks.append(f"blocks.{L}.attn.hook_z")        
        if patch_mlp:
            hooks.append(f"blocks.{L}.hook_mlp_out")
        if patch_residual:
            hooks.append(f"blocks.{L}.hook_resid_post")
    return hooks


def cache_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_names: list[str],
) -> dict[str, torch.Tensor]:
    names_filter = lambda name: name in hook_names
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=names_filter,
            return_type=None,
        )
    return {k: v.detach() for k, v in cache.items()}

def patch_head_and_get_rank(
    model_b: HookedTransformer,
    tokens: torch.Tensor,
    source_cache: dict[str, torch.Tensor],
    layer: int,
    head_idx: int,
    answer_token_id: int,
) -> int:
    hook_name = f"blocks.{layer}.attn.hook_z"
    source_act = source_cache[hook_name]    

    def patch_hook(value, hook):
        value[:, :, head_idx, :] = source_act[:, :, head_idx, :]
        return value

    with torch.no_grad():
        patched_logits = model_b.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, patch_hook)],
            return_type="logits",
        )

    final_logits = patched_logits[0, -1, :]
    sorted_ids = torch.argsort(final_logits, descending=True)
    rank = (sorted_ids == answer_token_id).nonzero(as_tuple=True)[0].item()
    return int(rank)


def patch_mlp_and_get_rank(
    model_b: HookedTransformer,
    tokens: torch.Tensor,
    source_cache: dict[str, torch.Tensor],
    layer: int,
    answer_token_id: int,
) -> int:
    hook_name = f"blocks.{layer}.hook_mlp_out"
    source_act = source_cache[hook_name]  

    def patch_hook(value, hook):
        value[:, :, :] = source_act[:, :, :]
        return value

    with torch.no_grad():
        patched_logits = model_b.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, patch_hook)],
            return_type="logits",
        )

    final_logits = patched_logits[0, -1, :]
    sorted_ids = torch.argsort(final_logits, descending=True)
    rank = (sorted_ids == answer_token_id).nonzero(as_tuple=True)[0].item()
    return int(rank)

@dataclass
class PatchResult:
    layer:           int
    head:            int           
    component:       str           
    mean_rank_improvement: float   
    recovery_fraction:     float   
    n_prompts:       int


def _init_accumulators(
    layers: list[int],
    n_heads: int,
    patch_heads: bool,
    patch_mlp: bool,
) -> dict[tuple, list[float]]:
    acc = {}
    for L in layers:
        if patch_heads:
            for h in range(n_heads):
                acc[(L, h)] = []
        if patch_mlp:
            acc[(L, "mlp")] = []
    return acc


def _patch_prompt(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    prompt: str,
    answer: str,
    hook_names: list[str],
    config: PatchingConfig,
    rank_improvements: dict[tuple, list[float]],
    n_heads: int,
) -> tuple[int, int, str]:
    tokens_a = model_a.to_tokens(prompt, prepend_bos=False)
    tokens_b = model_b.to_tokens(prompt, prepend_bos=False)

    answer_id_a = get_answer_token_id(model_a, answer)
    answer_id_b = get_answer_token_id(model_b, answer)

    rank_a = get_answer_rank(model_a, tokens_a, answer_id_a)
    rank_b = get_answer_rank(model_b, tokens_b, answer_id_b)

    subset = "gap" if rank_b > rank_a else "parity"

    if rank_b <= rank_a:
        return rank_a, rank_b, subset

    gap = rank_b - rank_a
    cache_a = cache_activations(model_a, tokens_a, hook_names)

    for L in config.layers_to_patch:
        if config.patch_heads:
            head_hook = f"blocks.{L}.attn.hook_z"
            if head_hook in cache_a:
                for h in range(n_heads):
                    patched_rank = patch_head_and_get_rank(
                        model_b, tokens_b, cache_a, L, h, answer_id_b
                    )
                    improvement = rank_b - patched_rank
                    rank_improvements[(L, h)].append(improvement / max(gap, 1))

        if config.patch_mlp:
            mlp_hook = f"blocks.{L}.hook_mlp_out"
            if mlp_hook in cache_a:
                patched_rank = patch_mlp_and_get_rank(
                    model_b, tokens_b, cache_a, L, answer_id_b
                )
                improvement = rank_b - patched_rank
                rank_improvements[(L, "mlp")].append(improvement / max(gap, 1))

    return rank_a, rank_b, subset


def _accumulate_to_results(
    rank_improvements: dict[tuple, list[float]],
) -> list[PatchResult]:
    results = []
    for (L, component), improvements in rank_improvements.items():
        if not improvements:
            continue
        mean_improvement  = float(np.mean(improvements))
        recovery_fraction = float(np.clip(mean_improvement, 0, 1))
        if isinstance(component, int):
            results.append(PatchResult(
                layer=L, head=component, component="head",
                mean_rank_improvement=mean_improvement,
                recovery_fraction=recovery_fraction,
                n_prompts=len(improvements),
            ))
        else:
            results.append(PatchResult(
                layer=L, head=-1, component="mlp",
                mean_rank_improvement=mean_improvement,
                recovery_fraction=recovery_fraction,
                n_prompts=len(improvements),
            ))
    return results


def run_patching_experiment(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    dataset: list[dict],
    config: PatchingConfig,
) -> tuple[list[PatchResult], list[PatchResult], list[PatchResult]]:

    n_heads = model_a.cfg.n_heads

    hook_names = get_hook_names(
        model_a,
        config.layers_to_patch,
        config.patch_heads,
        config.patch_mlp,
        config.patch_residual,
    )

    acc_all    = _init_accumulators(config.layers_to_patch, n_heads,
                                    config.patch_heads, config.patch_mlp)
    acc_gap    = _init_accumulators(config.layers_to_patch, n_heads,
                                    config.patch_heads, config.patch_mlp)
    acc_parity = _init_accumulators(config.layers_to_patch, n_heads,
                                    config.patch_heads, config.patch_mlp)

    rank_a_all, rank_b_all = [], []
    n_gap = 0
    n_parity = 0

    for item in tqdm(dataset, desc="Patching prompts"):
        prompt = item["clean_prompt"]
        answer = item["answer"]

        formatted  = prompt
        tokens_a   = model_a.to_tokens(formatted, prepend_bos=False)
        tokens_b   = model_b.to_tokens(formatted, prepend_bos=False)
        answer_id_a = get_answer_token_id(model_a, answer)
        answer_id_b = get_answer_token_id(model_b, answer)

        rank_a = get_answer_rank(model_a, tokens_a, answer_id_a)
        rank_b = get_answer_rank(model_b, tokens_b, answer_id_b)
        rank_a_all.append(rank_a)
        rank_b_all.append(rank_b)

        is_gap = rank_b > rank_a
        if is_gap:
            n_gap += 1
        else:
            n_parity += 1

        if not is_gap:
            gap = max(abs(rank_b - rank_a), 1)
            cache_a = cache_activations(model_a, tokens_a, hook_names) 
            for L in config.layers_to_patch:
                if config.patch_heads:
                    head_hook = f"blocks.{L}.attn.hook_z"
                    if head_hook in cache_a:
                        for h in range(n_heads):
                            patched_rank = patch_head_and_get_rank(
                                model_b, tokens_b, cache_a, L, h, answer_id_b
                            )
                            improvement = rank_b - patched_rank
                            acc_parity[(L, h)].append(improvement / gap)
                if config.patch_mlp:
                    mlp_hook = f"blocks.{L}.hook_mlp_out"
                    if mlp_hook in cache_a:
                        patched_rank = patch_mlp_and_get_rank(
                            model_b, tokens_b, cache_a, L, answer_id_b
                        )
                        improvement = rank_b - patched_rank
                        acc_parity[(L, "mlp")].append(improvement / gap)
            continue

        gap = rank_b - rank_a
        cache_a = cache_activations(model_a, tokens_a, hook_names)

        for L in config.layers_to_patch:
            if config.patch_heads:
                head_hook = f"blocks.{L}.attn.hook_z"
                if head_hook in cache_a:
                    for h in range(n_heads):
                        patched_rank = patch_head_and_get_rank(
                            model_b, tokens_b, cache_a, L, h, answer_id_b
                        )
                        improvement = rank_b - patched_rank
                        frac = improvement / max(gap, 1)
                        acc_gap[(L, h)].append(frac)
                        acc_all[(L, h)].append(frac)

            if config.patch_mlp:
                mlp_hook = f"blocks.{L}.hook_mlp_out"
                if mlp_hook in cache_a:
                    patched_rank = patch_mlp_and_get_rank(
                        model_b, tokens_b, cache_a, L, answer_id_b
                    )
                    improvement = rank_b - patched_rank
                    frac = improvement / max(gap, 1)
                    acc_gap[(L, "mlp")].append(frac)
                    acc_all[(L, "mlp")].append(frac)

    print(f"\n  Median rank Model A : {np.median(rank_a_all):.0f}")
    print(f"  Median rank Model B : {np.median(rank_b_all):.0f}")
    print(f"  Gap prompts         : {n_gap}")
    print(f"  Parity prompts      : {n_parity}")

    return (
        _accumulate_to_results(acc_all),
        _accumulate_to_results(acc_gap),
        _accumulate_to_results(acc_parity),
    )

def plot_head_heatmap(
    results: list[PatchResult],
    n_heads: int,
    layers: list[int],
    output_path: str,
):
    grid = np.zeros((len(layers), n_heads))
    layer_to_row = {L: i for i, L in enumerate(layers)}

    for r in results:
        if r.component == "head":
            grid[layer_to_row[r.layer], r.head] = r.recovery_fraction

    fig, ax = plt.subplots(figsize=(max(12, n_heads // 2), len(layers) + 1))
    sns.heatmap(
        grid,
        ax=ax,
        cmap="Reds",
        vmin=0, vmax=max(grid.max(), 0.01),
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{L}" for L in layers],
        cbar_kws={"label": "Recovery Fraction (0=none, 1=full gap recovered)"},
        linewidths=0.3,
    )

    for r in results:
        if r.component == "head" and r.recovery_fraction >= 0.1:
            row = layer_to_row[r.layer]
            ax.text(
                r.head + 0.5, row + 0.5,
                f"{r.recovery_fraction:.2f}",
                ha="center", va="center",
                fontsize=7, color="black", fontweight="bold",
            )

    ax.set_title(
        "Activation Patching: Head-level Recovery Fraction\n"
        "(Patching Model A → Model B, higher = more critical)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def plot_mlp_bars(
    results: list[PatchResult],
    layers: list[int],
    output_path: str,
):
    mlp_results = {r.layer: r.recovery_fraction
                   for r in results if r.component == "mlp"}

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [mlp_results.get(L, 0) for L in layers]
    bars = ax.bar(
        [f"L{L}" for L in layers],
        vals,
        color=["tomato" if v >= 0.1 else "steelblue" for v in vals],
    )
    ax.axhline(0.1, color="black", linestyle="--", linewidth=1,
               label="Critical threshold (0.1)")
    ax.set_ylabel("Recovery Fraction", fontsize=11)
    ax.set_title("MLP Recovery Fraction by Layer", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def plot_layer_summary(
    results: list[PatchResult],
    layers: list[int],
    output_path: str,
):
    """
    Per-layer summary: max head recovery vs MLP recovery.
    Shows which layers matter most overall.
    """
    max_head_recovery = {}
    mlp_recovery      = {}

    for r in results:
        if r.component == "head":
            max_head_recovery[r.layer] = max(
                max_head_recovery.get(r.layer, 0), r.recovery_fraction
            )
        else:
            mlp_recovery[r.layer] = r.recovery_fraction

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width/2,
           [max_head_recovery.get(L, 0) for L in layers],
           width, label="Max head recovery", color="steelblue")
    ax.bar(x + width/2,
           [mlp_recovery.get(L, 0) for L in layers],
           width, label="MLP recovery", color="tomato")

    ax.axhline(0.1, color="black", linestyle="--", linewidth=1,
               label="Critical threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("Recovery Fraction", fontsize=11)
    ax.set_title(
        "Per-layer Patching Summary\n"
        "Which layers contain critical components?",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")

def print_summary(results: list[PatchResult], threshold: float, label: str = "ALL PROMPTS"):
    print("\n" + "=" * 60)
    print(f"ACTIVATION PATCHING SUMMARY — {label}")
    print("=" * 60)

    critical = [r for r in results if r.recovery_fraction >= threshold]
    critical.sort(key=lambda r: -r.recovery_fraction)

    print(f"\n  Critical components (recovery >= {threshold:.0%}):")
    if not critical:
        print("  None found — circuit may be diffuse (see null hypothesis).")
    else:
        for r in critical:
            comp = f"L{r.layer} H{r.head}" if r.component == "head" else f"L{r.layer} MLP"
            print(f"    {comp:12s}  recovery={r.recovery_fraction:.3f}  "
                  f"n={r.n_prompts}")

    print(f"\n  Total critical heads : "
          f"{sum(1 for r in critical if r.component == 'head')}")
    print(f"  Total critical MLPs  : "
          f"{sum(1 for r in critical if r.component == 'mlp')}")
    print(f"\n  → {'Sparse circuit found.' if len(critical) <= 15 else 'Diffuse — more than 15 components.'}")
    print("=" * 60 + "\n")


def save_results(results: list[PatchResult], output_dir: str, suffix: str = ""):
    out = [asdict(r) for r in results]
    fname = f"patching_results{'_' + suffix if suffix else ''}.json"
    path = f"{output_dir}/{fname}"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Saved] {path}")
    
def parse_args() -> PatchingConfig:
    parser = argparse.ArgumentParser(description="Activation Patching")
    parser.add_argument("--model_a_path",    default="")
    parser.add_argument("--model_b_path",    default="")
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset_path",    default="./data/discovery_set.json")
    parser.add_argument("--output_dir",      default="./results/patching")
    parser.add_argument("--layers", nargs="+", type=int, default=[20,21,22,23,24, 25, 26, 27, 28])
    parser.add_argument("--max_prompts",     type=int, default=500)
    parser.add_argument("--no_heads",        action="store_true")
    parser.add_argument("--no_mlp",          action="store_true")
    parser.add_argument("--patch_residual",  action="store_true")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    return PatchingConfig(
        model_a_path=args.model_a_path,
        model_b_path=args.model_b_path,
        base_model_name=args.base_model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        layers_to_patch=args.layers,
        patch_heads=not args.no_heads,
        patch_mlp=not args.no_mlp,
        patch_residual=args.patch_residual,
        max_prompts=args.max_prompts,
        seed=args.seed,
    )


def main():
    config = parse_args()
    torch.manual_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {config.dataset_path} ...")
    dataset = load_dataset(config.dataset_path, config.max_prompts)
    print(f"  {len(dataset)} prompts loaded.")

    print(f"\nLoading Model A ...")
    model_a = load_hooked_model(config.base_model_name, config.model_a_path, config.device)

    print(f"\nLoading Model B ...")
    model_b = load_hooked_model(config.base_model_name, config.model_b_path, config.device)

    n_heads = model_a.cfg.n_heads
    print(f"\n  n_heads={n_heads}, patching layers={config.layers_to_patch}")
    print(f"  Total head slots to test: {len(config.layers_to_patch) * n_heads}")

    results_all, results_gap, results_parity = run_patching_experiment(
        model_a, model_b, dataset, config
    )

    out = config.output_dir

    for results, suffix, label in [
        (results_gap,    "gap",    "Gap Prompts (B underperforms A)"),
        (results_parity, "parity", "Parity Prompts (models similar)"),
        (results_all,    "all",    "All Prompts"),
    ]:
        sub_out = f"{out}/{suffix}"
        Path(sub_out).mkdir(parents=True, exist_ok=True)

        if config.patch_heads:
            plot_head_heatmap(
                results, n_heads, config.layers_to_patch,
                output_path=f"{sub_out}/head_heatmap.png",
            )
        if config.patch_mlp:
            plot_mlp_bars(
                results, config.layers_to_patch,
                output_path=f"{sub_out}/mlp_recovery.png",
            )
        plot_layer_summary(
            results, config.layers_to_patch,
            output_path=f"{sub_out}/layer_summary.png",
        )
        print_summary(results, config.critical_threshold, label=label)
        save_results(results, sub_out, suffix=suffix)

    print("Done.")


if __name__ == "__main__":
    main()