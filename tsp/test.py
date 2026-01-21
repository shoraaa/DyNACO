#!/usr/bin/env python3
from torch.nn import parallel
import torch
import argparse
import numpy as np
import random
import sys
import time

from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from net import Net
from utils import load_val_dataset, build_pyg_data
from faco import MFACO_TSP
from baselines import get_baseline_tsp

EPS = 1e-10

# ---------------------------
# Visualization Helpers
# ---------------------------
def row_softmax(P: torch.Tensor) -> torch.Tensor:
    return torch.softmax(P.float(), dim=1)

def mean_row_kl(P_prev: torch.Tensor, P_cur: torch.Tensor) -> float:
    p = row_softmax(P_prev)
    q = row_softmax(P_cur)
    kl_row = (p * ((p + EPS).log() - (q + EPS).log())).sum(dim=1)
    return float(kl_row.mean())

def rel_l2_drift(P_prev: torch.Tensor, P_cur: torch.Tensor) -> float:
    a = P_prev.float()
    b = P_cur.float()
    return float((b - a).norm() / (a.norm() + EPS))

def top_set(P: torch.Tensor, frac: float = 0.05) -> set:
    v = P.flatten()
    m = v.numel()
    k = max(1, int(m * frac))
    idx = torch.topk(v, k).indices
    return set(idx.detach().cpu().tolist())

def top_turnover(P_prev: torch.Tensor, P_cur: torch.Tensor, frac: float = 0.05) -> float:
    A = top_set(P_prev, frac)
    B = top_set(P_cur, frac)
    jacc = len(A & B) / max(1, len(A | B))
    return float(1.0 - jacc)

def top1_flip_rate(P_prev: torch.Tensor, P_cur: torch.Tensor) -> float:
    a = P_prev.argmax(dim=1)
    b = P_cur.argmax(dim=1)
    return float((a != b).float().mean())

def safe_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    b = b.detach().reshape(-1).to(device="cpu", dtype=torch.float32)

    # Filter non-finite entries
    mask = torch.isfinite(a) & torch.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]

    # Degenerate variance => undefined correlation
    if float(a.std()) < eps or float(b.std()) < eps:
        return float("nan")

    # More stable than torch.corrcoef for large vectors
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b) / denom)

def top_overlap_frac(a: torch.Tensor, b: torch.Tensor, frac: float = 0.05) -> float:
    a = a.flatten()
    b = b.flatten()
    m = a.numel()
    k = max(1, int(m * frac))
    ai = torch.topk(a, k).indices
    bi = torch.topk(b, k).indices
    inter = len(set(ai.cpu().tolist()).intersection(set(bi.cpu().tolist())))
    return inter / k

def row_top1_match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    # per node: does argmax candidate match?
    return float((a.cpu().argmax(dim=1) == b.cpu().argmax(dim=1)).float().mean())

def infer_instance(model, coords, k_sparse, n_ants, dynamic, args, use_heuristic_only=False, collect_metrics=False):
    if model is not None:
        model.eval()
    
    disable_heuristic_arg = args.disable_heuristic
    if use_heuristic_only:
        disable_heuristic_arg = False # Force enable 1/d for base MFACO
    
    aco = MFACO_TSP(
        n_ants=n_ants,
        coords=coords,
        cand_list_size=k_sparse,
        backup_list_size=k_sparse,
        disable_heuristic=disable_heuristic_arg,
        use_local_search=not args.no_local_search,
        decay=args.rho,
        device=args.device,
        enable_torch_sync=True, 
        smooth_mmas=not args.no_smooth_mmas,
        min_new_edges=args.min_new_edges,
        extend_ls=not args.no_extend_ls,
        normalized_heuristic=not args.no_normalized_heuristic,
    )

    best_seen = float("inf")
    avg_last = None
    H = args.H

    priors, pher_before, probmats = [], [], []
    
    # Store per-step metrics: list of length H
    # We will initialize them with NaNs or 0s
    metrics_log = {
        "cost": [],
        "l2": [], "kl": [], "turnover": [], "flip": [],
        "corr": [], "ov": [], "row_match": []
    }
    
    t_start_total = time.time()
    with torch.no_grad():
        for t in range(H):
            prior_mat = None
            if collect_metrics:
                pher_before.append(aco.pheromone_sparse.detach().cpu().clone())

            if model is not None and not use_heuristic_only:
                pyg_data = build_pyg_data(aco, coords, args.device, dynamic=dynamic)
                heu_vec = model(pyg_data).view(-1)
                prior_mat = heu_vec.view(aco.n, aco.k)
                
                if collect_metrics:
                    priors.append(prior_mat.detach().cpu().clone())
            
            step_best = float("inf")
            for mini_t in range(args.mini_H):
                costs, flats, *rest = aco.sample(require_prob=False, prior=prior_mat)

                avg_last = float(costs.mean())
                best_idx = int(costs.argmin())
                best_cost = float(costs[best_idx])
                best_seen = min(best_seen, best_cost)
                step_best = min(step_best, best_cost)

                aco._update_pheromone_from_flat(flats[best_idx], best_cost)
            
            if collect_metrics:
                # 1. Cost at step H
                metrics_log["cost"].append(step_best)
                
                # 2. Prior Change (requires priors[t] and priors[t-1])
                # Only if we have priors (i.e. model is used)
                is_prior_avail = (len(priors) > 0)
                if is_prior_avail and t > 0:
                     # Compare t-1 and t
                     P_prev, P_cur = priors[t-1], priors[t]
                     metrics_log["l2"].append(rel_l2_drift(P_prev, P_cur))
                     metrics_log["kl"].append(mean_row_kl(P_prev, P_cur))
                     metrics_log["turnover"].append(top_turnover(P_prev, P_cur))
                     metrics_log["flip"].append(top1_flip_rate(P_prev, P_cur))
                else:
                     metrics_log["l2"].append(0.0)
                     metrics_log["kl"].append(0.0)
                     metrics_log["turnover"].append(0.0)
                     metrics_log["flip"].append(0.0)

                # 3. Pheromone-Prior Alignment (requires pher_before[t] and priors[t])
                # pher_before[t] is state at start of step t
                # priors[t] is computed at start of step t
                if is_prior_avail:
                    tau = pher_before[t]
                    pr = priors[t]
                    metrics_log["corr"].append(safe_corr(tau, pr))
                    metrics_log["ov"].append(top_overlap_frac(tau, pr))
                    metrics_log["row_match"].append(row_top1_match_rate(tau, pr))
                else:
                    metrics_log["corr"].append(0.0)
                    metrics_log["ov"].append(0.0)
                    metrics_log["row_match"].append(0.0)

    t_total = time.time() - t_start_total

    if collect_metrics:
        return avg_last, best_seen, t_total, metrics_log
    return avg_last, best_seen, t_total

def main():
    parser = argparse.ArgumentParser(description="MFACO TSP Testing")
    
    # Dataset / Problem
    parser.add_argument("--n_node", type=int, default=100, help="Number of nodes per TSP instance")
    parser.add_argument("--k_sparse", type=int, default=32, help="Candidate list size (k)")
    
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset (.pt) file")
    
    # Model / Training
    parser.add_argument("--checkpoint", type=str, default="none", help="Path to model checkpoint or 'none'")
    parser.add_argument("--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("--H", type=int, default=10, help="ACO steps per instance (H)")
    parser.add_argument("--mini_H", type=int, default=100, help="ACO steps per iteration (mini_H)")
    
    # Checkpoint loaded model typically trained with disable_heuristic=True. 
    # But for "Base MFACO" we want to force enable it.
    parser.add_argument("--disable_heuristic", action="store_true", help="Disable heuristic")
    parser.add_argument("--no_local_search", action="store_true", help="Disable local search")
    parser.add_argument("--no_smooth_mmas", action="store_true", help="Disable smooth MMAS")
    parser.add_argument("--no_extend_ls", action="store_true", help="Extend LS checklist")
    parser.add_argument("--rho", type=float, default=0.5, help="Rho")
    parser.add_argument("--min_new_edges", type=int, default=8, help="Min new edges")
    parser.add_argument("--no_normalized_heuristic", action="store_true", help="Normalize heuristic to [1/k, 1]")
    parser.add_argument("--no_logit_net", action="store_true", help="Use logit network (no sigmoid) and log-space ACO")
    
    # Baseline
    parser.add_argument("--baseline", type=str, choices=['none', 'lkh'], default='lkh', help="Baseline for validation")
    parser.add_argument("--baseline_time_limit", type=float, default=2.0, help="LKH time limit per instance (seconds)")
    parser.add_argument("--baseline_runs", type=int, default=1, help="LKH runs")
    
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--no_dynamic_feats", action="store_true", help="Disable dynamic features")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of metrics")
    parser.add_argument("--visualize_output", type=str, default="visualizations", help="Output directory for visualization plots")
    parser.add_argument("--timed", action="store_true", help="Show performance timings")

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.dataset is not None:
        print(f"Loading dataset from {args.dataset}")
        data = torch.load(args.dataset, map_location="cpu")
        # Handle various formats: list, tensor, dict
        if isinstance(data, dict):
            if "coords" in data:
                val_list = data["coords"]
            else:
                 # Fallback: assume values are instances?
                 # Or maybe the dict IS the instance? Unlikely for .pt dataset file.
                 raise ValueError("Loaded dict dataset must contain 'coords' key")
        elif isinstance(data, (list, torch.Tensor)):
            val_list = data
        else:
            raise ValueError(f"Unknown dataset format: {type(data)}")
            
        # Ensure val_list is iterable (Tensor is iterable, giving slices)
        
        # Infer n_node from first instance
        if len(val_list) > 0:
            first_item = val_list[0]
            if isinstance(first_item, torch.Tensor):
                n_node_inferred = first_item.shape[0]
                if n_node_inferred != args.n_node:
                    print(f"Inferred n_node={n_node_inferred} from dataset (overriding arg {args.n_node})")
                    args.n_node = n_node_inferred
                    
        # Move to device if it is a tensor, else inferred later?
        # infer_instance expects `coords` to be passed.
        # If val_list is tensor on CPU, iterating gives tensor on CPU.
        # infer_instance handles it.
        
    else:
        print(f"Loading validation dataset for n={args.n_node}")
        try:
            val_list = load_val_dataset(args.n_node, args.device)
        except FileNotFoundError:
            print("Validation dataset not found. Generating...")
            from utils import generate_val_dataset
            generate_val_dataset(args.n_node, 16 if args.n_node >= 1000 else 128, args.k_sparse, "cpu")
            val_list = load_val_dataset(args.n_node, args.device)
        
    baseline_values = None
    if args.baseline == 'lkh':
        print(f"Computing/Loading LKH baseline...")
        baseline_values = get_baseline_tsp(val_list, args.n_node, device="cpu", runs=args.baseline_runs, time_limit=args.baseline_time_limit)
        baseline_values = baseline_values.cpu().numpy()
        print(f"Baseline Avg: {baseline_values.mean():.4f}")
        
    # Validation Loop
    dynamic = not args.no_dynamic_feats
    
    model = None
    model = None
    if args.checkpoint != "none":
        print(f"Loading model from {args.checkpoint}")
        
        # Load checkpoint dict
        # Use args.device for map_location (load to current desired device)
        loaded_ckpt = torch.load(args.checkpoint, map_location=args.device)
        
        state_dict = loaded_ckpt
        if isinstance(loaded_ckpt, dict) and "model_state_dict" in loaded_ckpt:
            state_dict = loaded_ckpt["model_state_dict"]
            config = loaded_ckpt.get("config", {})
            
            # Update args from config
            ignored_keys = {"device", "checkpoint", "baseline", "baseline_time_limit", "baseline_runs"}
            for k, v in config.items():
                if k in ignored_keys:
                    continue
                if hasattr(args, k):
                    current_val = getattr(args, k)
                    # Check if overridden explicitely
                    is_explicit = any(arg == f"--{k}" or arg.startswith(f"--{k}=") for arg in sys.argv)
                    
                    if is_explicit:
                        if current_val != v:
                            print(f"WARNING: Overriding checkpoint config {k}={v} with explicit argument {k}={current_val}")
                    else:
                        print(f"Loading {k}={v} from checkpoint")
                        setattr(args, k, v)
        
        # Init model with potentially updated args (e.g. logit_net)
        model = Net(logit_net=not args.no_logit_net).to(args.device)
        model.load_state_dict(state_dict)
        model.eval()
    
    # Storage for results
    results = []
    
    print("\nRunning Evaluation...")
    print(f"{'Idx':<5} {'Opt (LKH)':<10} {'Base MFACO':<12} {'Model MFACO':<12} {'Base Gap':<10} {'Model Gap':<10}")
    
    sum_base_best = 0
    sum_model_best = 0
    sum_base_gap = 0
    sum_model_gap = 0
    n_val = len(val_list)
    
    for i, coords in enumerate(tqdm(val_list, leave=False)):
        # Base MFACO (Heuristic 1/d, No Prior)
        # Assuming we don't visualize Base MFACO specifics unless needed?
        # The request said "Instead of only 1 data instance like in notebook, this will now average of the whole dataset."
        # And "plot the plots in notebook to .pdf file". 
        # Notebook plots "Cost vs H" for Base vs Model.
        # It plots prior changes for Model.
        # So we need metrics for Model. For Base we mainly need Cost trace.
        
        collect = args.visualize
        
        # We need cost history for Base if visualizing
        # infer_instance signature: ..., collect_metrics=False)
        # If collect=True, it returns (avg, best, metrics). Else (avg, best).
        
        base_ret = infer_instance(None, coords, args.k_sparse, args.n_ants, dynamic, args, use_heuristic_only=True, collect_metrics=collect)
        if collect:
            _, base_best, t_base, base_metrics = base_ret
        else:
            _, base_best, t_base = base_ret
        
        # Model MFACO (Prior from Net)
        model_best = float("inf")
        model_metrics = None
        t_model = 0.0
        if model is not None:
             model_ret = infer_instance(model, coords, args.k_sparse, args.n_ants, dynamic, args, use_heuristic_only=False, collect_metrics=collect)
             if collect:
                 _, model_best, t_model, model_metrics = model_ret
             else:
                 _, model_best, t_model = model_ret
        
        if collect:
            # Initialize accumulators on first iteration
            if i == 0:
                H = args.H
                # Keys: cost, l2, kl, turnover, flip, corr, ov, row_match
                # For Base: only cost matters usually, but structure is same.
                agg_base_cost = np.zeros(H)
                
                # For Model: all metrics
            # Accumulate Base Cost
            # base_metrics["cost"] list of floats
            if "cost" in base_metrics:
                agg_base_cost += np.array(base_metrics["cost"])
            
            # Accumulate Model Metrics
            if model_metrics is not None:
                if "agg_model" not in locals():
                     agg_model = {k: np.zeros(H) for k in ["cost", "l2", "kl", "turnover", "flip", "corr", "ov", "row_match"]}

                for k, v in model_metrics.items():
                    if len(v) == H:
                        agg_model[k] += np.array(v)

        if i == 0:
            sum_time_base = 0.0
            sum_time_model = 0.0

        sum_time_base += t_base
        if model is not None:
            sum_time_model += t_model

        
        opt = float(baseline_values[i]) if baseline_values is not None else 0.0
        
        base_gap = (base_best - opt) / opt * 100 if opt > 0 else 0.0
        model_gap = (model_best - opt) / opt * 100 if opt > 0 and model is not None else 0.0
        
        sum_base_best += base_best
        sum_model_best += model_best
        sum_base_gap += base_gap
        sum_model_gap += model_gap
        
        # print(f"{i:<5} {opt:<10.2f} {base_best:<12.2f} {model_best:<12.2f} {base_gap:<10.2f}% {model_gap:<10.2f}%")
        
    avg_base_best = sum_base_best / n_val
    avg_model_best = sum_model_best / n_val
    avg_base_gap = sum_base_gap / n_val
    avg_model_gap = sum_model_gap / n_val
    
    print("\n" + "="*60)
    print(f"Summary (N={args.n_node}, Instances={n_val})")
    if baseline_values is not None:
        print(f"Baseline Avg Cost: {baseline_values.mean():.4f}")
    
    print(f"Base MFACO Avg Cost: {avg_base_best:.4f}")
    if baseline_values is not None:
        print(f"Base MFACO Avg Gap:  {avg_base_gap:.3f}%")
        
    if args.visualize:
        output_dir = Path(args.visualize_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plots to {output_dir}...")
        
        # Average
        avg_base_cost_H = agg_base_cost / n_val
        if baseline_values is not None:
            avg_opt = baseline_values.mean()
        else:
            avg_opt = 1.0 # arbitrary to avoid div/0 if opt is missing
            
        base_gap_H = (avg_base_cost_H - avg_opt) / avg_opt * 100
        
        model_avg = {}
        if model is not None:
            for k, v in agg_model.items():
                model_avg[k] = v / n_val
        
        # 1. Gap vs H
        plt.figure(figsize=(7, 5))
        x = range(1, len(base_gap_H) + 1)
        plt.plot(x, base_gap_H, label="Base MFACO", marker="o")
        if model is not None:
            model_gap_H = (model_avg["cost"] - avg_opt) / avg_opt * 100
            plt.plot(x, model_gap_H, label="Model MFACO", marker="o")
        plt.title("Gap (%) vs H Steps")
        plt.xlabel("H")
        plt.ylabel("Gap (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "gap_vs_h.pdf")
        plt.close()
        
        if model is not None:
            # 2a. Prior Change - Metrics
            plt.figure(figsize=(7, 5))
            x_diff = range(2, args.H + 1)
            plt.plot(x_diff, model_avg["l2"][1:], label="L2 Drift", marker="x")
            plt.plot(x_diff, model_avg["turnover"][1:], label="Turnover", marker="x")
            plt.plot(x_diff, model_avg["flip"][1:], label="Flip Rate", marker="x")
            plt.title("Prior Change Metrics vs H")
            plt.xlabel("H")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "prior_change_metrics.pdf")
            plt.close()

            # 2b. Prior Change - KL
            plt.figure(figsize=(7, 5))
            plt.plot(x_diff, model_avg["kl"][1:], label="KL Div", marker="x", color="tab:orange")
            plt.title("Prior Change KL Divergence vs H")
            plt.xlabel("H")
            plt.ylabel("KL Divergence")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "prior_change_kl.pdf")
            plt.close()
            
            # 3a. Correlation
            plt.figure(figsize=(7, 5))
            plt.plot(x, model_avg["corr"], label="Correlation", marker="s")
            plt.title("Pheromone vs Prior Correlation")
            plt.xlabel("H")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "alignment_corr.pdf")
            plt.close()

            # 4. Cost vs H
            plt.figure(figsize=(7, 5))
            plt.plot(x, avg_base_cost_H, label="Base Cost", marker="o")
            plt.plot(x, model_avg["cost"], label="Model Cost", marker="o")
            plt.axhline(avg_opt, color="k", linestyle="--", label=f"Opt ({avg_opt:.2f})")
            plt.title("Cost vs H")
            plt.xlabel("H")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "cost_vs_h.pdf")
            plt.close()
            
        print(f"Saved separate visualizations to {output_dir}")

    if model is not None:
        print(f"Model MFACO Avg Cost: {avg_model_best:.4f}")
        if baseline_values is not None:
            print(f"Model MFACO Avg Gap:  {avg_model_gap:.3f}%")
            
    if args.timed:
        print("\nPerformance Metrics:")
        print(f"  Base MFACO Total:   {sum_time_base:.6f}s")
        if model is not None:
            print(f"  Model MFACO Total:  {sum_time_model:.6f}s")

    print("="*60)

if __name__ == "__main__":
    main()
