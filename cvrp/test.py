#!/usr/bin/env python3
import torch
import argparse
import numpy as np
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from net import Net
from utils import load_val_dataset, build_pyg_data
from faco import MFACO_CVRP
from baselines import get_baseline_cvrp

EPS = 1e-10
DEMAND_SCALE = 100000

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

def verify_solution(coords, demand, capacity, cost, route0):
    """
    route0: list or ndarray starting and ending with 0, e.g., [0, 1, 2, 0, 3, 0]
    """
    n = len(demand)
    visited = set()
    total_dist = 0.0
    
    cap_int = int(round(capacity * DEMAND_SCALE))
    demand_int = [int(round(d * DEMAND_SCALE)) for d in demand]
    
    current_load_int = 0
    for i in range(len(route0) - 1):
        u, v = int(route0[i]), int(route0[i+1])
        
        # Distance
        du = coords[u]
        dv = coords[v]
        d = np.sqrt(((du - dv)**2).sum())
        total_dist += d
        
        if v == 0:
            # End of a route
            if current_load_int > cap_int:
                raise ValueError(f"Capacity violation: {current_load_int/DEMAND_SCALE} > {capacity}")
            current_load_int = 0
        else:
            # Customer
            if v in visited:
                raise ValueError(f"Node {v} visited more than once")
            visited.add(v)
            current_load_int += demand_int[v]
            
    # Check all customers visited
    if len(visited) != n - 1:
        missing = set(range(1, n)) - visited
        raise ValueError(f"Missing customers: {missing}")
    
    # Check cost
    if abs(total_dist - cost) > 1e-3:
        raise ValueError(f"Cost mismatch: recalculated {total_dist:.6f} vs reported {cost:.6f}")

    return True

def infer_instance(model, coords, demand, capacity, k_sparse, n_ants, dynamic, args, use_heuristic_only=False, collect_metrics=False):
    if model is not None:
        model.eval()

    disable_heuristic_arg = args.disable_heuristic
    if use_heuristic_only:
        disable_heuristic_arg = False # Enable 1/d

    aco = MFACO_CVRP(
        coords=coords,
        demand=demand,
        capacity=float(capacity),
        n_ants=n_ants,
        cand_list_size=k_sparse,
        backup_list_size=k_sparse,
        min_new_edges=args.min_new_edges,
        decay=args.rho,
        p_best=0.05,
        use_local_search=not args.no_local_search,
        disable_heuristic=disable_heuristic_arg,
        extend_ls=args.extend_ls,
        smooth_mmas=not args.no_smooth_mmas,
        device=args.device,
        enable_torch_sync=True,
        normalized_heuristic=not args.no_normalized_heuristic,
        fixed_steps=args.L,
    )

    aco.reset_timings()
    best_seen = float("inf")
    avg_last = None
    H = args.H

    priors, pher_before = [], []
    
    # Store per-step metrics: list of length H
    # We will initialize them with NaNs or 0s
    metrics_log = {
        "cost": [],
        "l2": [], "kl": [], "turnover": [], "flip": [],
        "corr": [], "ov": [], "row_match": []
    }

    with torch.no_grad():
        for t in range(H):
            prior_mat = None
            if collect_metrics:
                pher_before.append(aco.pheromone_sparse.detach().cpu().clone())

            if model is not None and not use_heuristic_only:
                pyg_data = build_pyg_data(aco, coords, demand, args.device, dynamic=dynamic)
                heu_vec = model(pyg_data).view(-1)
                prior_mat = heu_vec.view(aco.n, aco.k) + EPS
                
                if collect_metrics:
                    priors.append(prior_mat.detach().cpu().clone())
            
            for mini_t in range(args.mini_H):
                # Sample
                return_decoded = getattr(args, 'verify', False)
                costs_t, perms, decoded, _, _, _ = aco.sample(require_prob=False, prior=prior_mat, return_decoded=return_decoded)

                if return_decoded:
                    best_idx_t = int(costs_t.argmin().item())
                    try:
                        verify_solution(coords, demand, capacity, float(costs_t[best_idx_t]), decoded[best_idx_t])
                    except ValueError as e:
                        print(f"\n[DEBUG ERROR] Instance verification failed: {e}")
                        sys.exit(1)

                avg_last = float(costs_t.mean().item())
                best_idx = int(costs_t.argmin().item())
                best_cost = float(costs_t[best_idx].item())
                best_seen = min(best_seen, best_cost)
                best_perm = perms[best_idx]

                aco.update_pheromone(best_perm, best_cost)

            if collect_metrics:
                # 1. Cost at step H
                metrics_log["cost"].append(best_seen) # Using best_seen so far or step_best? 
                # TSP implementation used step_best (min of batch), but then updated best_seen.
                # Let's use step_best (best of current iteration) to see progress, assuming local search makes it monotonic-ish?
                # Actually, standard plot is "Cost vs H". Usually best_seen (global best) or current iteration best.
                # TSP implementation used: step_best = min(step_best, best_cost) inside mini_t loop.
                # Here we just want the best cost found at this step t.
                # Let's assume we want the best of these (batch) ants after LS.
                
                # 2. Prior Change (requires priors[t] and priors[t-1])
                is_prior_avail = (len(priors) > 0)
                if is_prior_avail and t > 0:
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

                # 3. Alignment
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

    timings = aco.get_timings() if getattr(args, 'timed', False) else None
    
    if collect_metrics:
        return avg_last, best_seen, timings, metrics_log
        
    return avg_last, best_seen, timings

def main():
    parser = argparse.ArgumentParser(description="MFACO CVRP Testing")
    
    # Dataset / Problem
    parser.add_argument("--n_node", type=int, default=100, help="Number of customers")
    parser.add_argument("--k_sparse", type=int, default=32, help="K")
    
    # Model / Training
    parser.add_argument("--checkpoint", type=str, default="none", help="Checkpoint path or 'none'")
    parser.add_argument("--n_ants", type=int, default=100, help="Ants")
    parser.add_argument("--H", type=int, default=10, help="H")
    parser.add_argument("--mini_H", type=int, default=100, help="Mini H")
    
    parser.add_argument("--disable_heuristic", action="store_true", help="Disable heuristic")
    parser.add_argument("--no_local_search", action="store_true", help="Disable LS")
    parser.add_argument("--no_smooth_mmas", action="store_true", help="Smooth MMAS")
    parser.add_argument("--extend_ls", action="store_true", help="Extend LS")
    parser.add_argument("--rho", type=float, default=0.5, help="Rho")
    parser.add_argument("--min_new_edges", type=int, default=16, help="Min new edges")
    parser.add_argument("--no_normalized_heuristic", action="store_true", help="Normalize heuristic to [1/k, 1]")
    parser.add_argument("--no_logit_net", action="store_true", help="Use logit network (no sigmoid) and log-space ACO")
    
    # Baseline
    parser.add_argument("--baseline", type=str, choices=['none', 'hgs'], default='hgs', help="Baseline")
    parser.add_argument("--baseline_time_limit", type=float, default=0.2, help="HGS time limit (s)")
    
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=1234, help="Seed")
    parser.add_argument("--no_dynamic_feats", action="store_true", help="No dyn feats")
    parser.add_argument("--verify", action="store_true", help="Verify solution correctness")
    parser.add_argument("--timed", action="store_true", help="Show performance timings")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of metrics")
    parser.add_argument("--visualize_output", type=str, default="visualizations_cvrp", help="Output directory for visualization plots")
    parser.add_argument("--L", type=int, default=0, help="Fixed ant trajectory length")

    args = parser.parse_args()
    

    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(f"Loading validation dataset for n={args.n_node}")
    try:
        val_dataset = load_val_dataset(args.n_node, "cpu")
    except FileNotFoundError:
        print("Dataset not found. Generating...")
        # Fallback generation if not exists, similar to logic in other file?
        # Ideally user should have generated it.
        # Calling the gen block from utils if possible or raising error.
        raise FileNotFoundError("Validation dataset not found. Run cvrp/utils.py to generate.")

    baseline_values = None
    if args.baseline == 'hgs':
        print(f"Computing/Loading HGS baseline...")
        baseline_values = get_baseline_cvrp(val_dataset, args.n_node, device="cpu", time_limit=args.baseline_time_limit)
        baseline_values = baseline_values.cpu().numpy()
        print(f"Baseline Avg: {baseline_values.mean():.4f}")

    dynamic = not args.no_dynamic_feats
    
    model = None
    model = None
    if args.checkpoint != "none":
        print(f"Loading model from {args.checkpoint}")
        
        loaded_ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = loaded_ckpt
        
        if isinstance(loaded_ckpt, dict) and "model_state_dict" in loaded_ckpt:
            state_dict = loaded_ckpt["model_state_dict"]
            config = loaded_ckpt.get("config", {})
            
            ignored_keys = {"device", "checkpoint", "baseline", "baseline_time_limit"}
            for k, v in config.items():
                if k in ignored_keys:
                    continue
                if hasattr(args, k):
                    current_val = getattr(args, k)
                    is_explicit = any(arg == f"--{k}" or arg.startswith(f"--{k}=") for arg in sys.argv)
                    
                    if is_explicit:
                        if current_val != v:
                            print(f"WARNING: Overriding checkpoint config {k}={v} with explicit argument {k}={current_val}")
                    else:
                        if current_val != v:
                            print(f"Loading {k}={v} from checkpoint (was {current_val})")
                            setattr(args, k, v)

        model = Net(logit_net=not args.no_logit_net).to(args.device)
        model.load_state_dict(state_dict)
        model.eval()

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    sum_base_best = 0
    sum_model_best = 0
    sum_base_gap = 0
    sum_model_gap = 0
    n_val = len(val_dataset)
    
    total_time_ant = 0.0
    total_time_ls = 0.0
    total_time_split = 0.0

    print("\nRunning Evaluation...")
    print(f"{'Idx':<5} {'Opt (HGS)':<10} {'Base MFACO':<12} {'Model MFACO':<12} {'Base Gap':<10} {'Model Gap':<10}")

    for i, batch in enumerate(tqdm(loader, leave=False)):
        coords = batch[0][0].numpy()
        demand = batch[1][0].numpy()
        capacity = float(batch[2][0])
        
        collect = args.visualize

        # Base MFACO
        base_ret = infer_instance(None, coords, demand, capacity, args.k_sparse, args.n_ants, dynamic, args, use_heuristic_only=True, collect_metrics=collect)
        if collect:
            _, base_best, base_timings, base_metrics = base_ret
        else:
            _, base_best, base_timings = base_ret

        if base_timings:
            total_time_ant += base_timings["time_ant"]
            total_time_ls += base_timings["time_ls"]
            total_time_split += base_timings["time_split"]
            
        # Model MFACO
        model_best = float("inf")
        model_metrics = None
        if model is not None:
             model_ret = infer_instance(model, coords, demand, capacity, args.k_sparse, args.n_ants, dynamic, args, use_heuristic_only=False, collect_metrics=collect)
             if collect:
                 _, model_best, _, model_metrics = model_ret
             else:
                 _, model_best, _ = model_ret
        
        if collect:
            # Initialize accumulators on first iteration
            if i == 0:
                H = args.H
                agg_base_cost = np.zeros(H)
                agg_model = {k: np.zeros(H) for k in ["cost", "l2", "kl", "turnover", "flip", "corr", "ov", "row_match"]}
                
            if "cost" in base_metrics:
                agg_base_cost += np.array(base_metrics["cost"])
            
            if model_metrics is not None:
                for k, v in model_metrics.items():
                    if len(v) == H:
                        agg_model[k] += np.array(v)
        
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
        
    if model is not None:
        print(f"Model MFACO Avg Cost: {avg_model_best:.4f}")
        if baseline_values is not None:
            print(f"Model MFACO Avg Gap:  {avg_model_gap:.3f}%")
            
    if args.timed:
        print("\nPerformance Metrics (Avg per Instance):")
        print(f"  Ant Construction: {total_time_ant / n_val:.6f}s")
        print(f"  Local Search:     {total_time_ls / n_val:.6f}s")
        print(f"  Split Algorithm:  {total_time_split / n_val:.6f}s")
        print(f"  Total Per Instance: {(total_time_ant + total_time_ls + total_time_split) / n_val:.6f}s")

    print("="*60)
    
    if args.visualize:
        output_dir = Path(args.visualize_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plots to {output_dir}...")
        
        # Average
        avg_base_cost_H = agg_base_cost / n_val
        if baseline_values is not None:
            avg_opt = baseline_values.mean()
        else:
            avg_opt = 1.0 
            
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

if __name__ == "__main__":
    main()