#!/usr/bin/env python3
import torch
import argparse
import numpy as np
import random
import sys
import time
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Shared helpers
EPS = 1e-10

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
    if P_prev is None or P_cur is None: return 0.0
    A = top_set(P_prev, frac)
    B = top_set(P_cur, frac)
    jacc = len(A & B) / max(1, len(A | B))
    return float(1.0 - jacc)

def top1_flip_rate(P_prev: torch.Tensor, P_cur: torch.Tensor) -> float:
    if P_prev is None or P_cur is None: return 0.0
    a = P_prev.argmax(dim=1)
    b = P_cur.argmax(dim=1)
    return float((a != b).float().mean())

def safe_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    if a is None or b is None: return 0.0
    a = a.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    b = b.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    mask = torch.isfinite(a) & torch.isfinite(b)
    if mask.sum() < 2: return float("nan")
    a = a[mask]
    b = b[mask]
    if float(a.std()) < eps or float(b.std()) < eps: return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b) / denom)

def top_overlap_frac(a: torch.Tensor, b: torch.Tensor, frac: float = 0.05) -> float:
    if a is None or b is None: return 0.0
    a = a.flatten()
    b = b.flatten()
    m = a.numel()
    k = max(1, int(m * frac))
    ai = torch.topk(a, k).indices
    bi = torch.topk(b, k).indices
    inter = len(set(ai.cpu().tolist()).intersection(set(bi.cpu().tolist())))
    return inter / k

def row_top1_match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None: return 0.0
    return float((a.cpu().argmax(dim=1) == b.cpu().argmax(dim=1)).float().mean())

# CVRP Verification
def verify_solution_cvrp(coords, demand, capacity, cost, route0):
    DEMAND_SCALE = 100000
    n = len(demand)
    visited = set()
    total_dist = 0.0
    cap_int = int(round(capacity * DEMAND_SCALE))
    demand_int = [int(round(d * DEMAND_SCALE)) for d in demand]
    current_load_int = 0
    for i in range(len(route0) - 1):
        u, v = int(route0[i]), int(route0[i+1])
        du = coords[u]
        dv = coords[v]
        d = np.sqrt(((du - dv)**2).sum())
        total_dist += d
        if v == 0:
            if current_load_int > cap_int:
                raise ValueError(f"Capacity violation: {current_load_int/DEMAND_SCALE} > {capacity}")
            current_load_int = 0
        else:
            if v in visited:
                raise ValueError(f"Node {v} visited more than once")
            visited.add(v)
            current_load_int += demand_int[v]
    if len(visited) != n - 1:
        missing = set(range(1, n)) - visited
        raise ValueError(f"Missing customers: {missing}")
    if abs(total_dist - cost) > 1e-3:
        raise ValueError(f"Cost mismatch: recalculated {total_dist:.6f} vs reported {cost:.6f}")
    return True

# -----------------------------------------------------------------------------
# Unified Logic
# -----------------------------------------------------------------------------

def get_modules(problem):
    # Retrieve current working directory
    cwd = os.getcwd()
    problem_path = os.path.join(cwd, problem)
    
    if problem_path not in sys.path:
        sys.path.insert(0, problem_path)

    if problem == 'tsp':
        import net
        import faco
        import utils
        import baselines
        
        Net = net.Net
        MFACO = faco.MFACO_TSP
        load_val = utils.load_val_dataset
        build_pyg = utils.build_pyg_data
        get_base = baselines.get_baseline_tsp
        return Net, MFACO, load_val, build_pyg, get_base, faco.set_faco_cpp_threads
    elif problem == 'cvrp':
        import net
        import faco
        import utils
        import baselines
        
        Net = net.Net
        MFACO = faco.MFACO_CVRP
        load_val = utils.load_val_dataset
        build_pyg = utils.build_pyg_data
        get_base = baselines.get_baseline_cvrp
        return Net, MFACO, load_val, build_pyg, get_base, faco.set_faco_cpp_threads
    else:
        raise ValueError(f"Unknown problem: {problem}")

def infer_instance(problem, MFACOClass, build_pyg_data_fn, model, instance_data, k_sparse, n_ants, dynamic, args, use_heuristic_only=False, collect_metrics=False):
    if model is not None:
        model.eval()

    disable_heuristic_arg = args.disable_heuristic
    if use_heuristic_only:
        disable_heuristic_arg = False 

    # Setup specific args
    if problem == 'tsp':
        coords = instance_data
        kwargs = {
            'n_ants': n_ants,
            'coords': coords,
            'cand_list_size': k_sparse,
            'backup_list_size': k_sparse,
            'disable_heuristic': disable_heuristic_arg,
            'use_local_search': not args.no_local_search,
            'decay': args.rho,
            'device': args.device,
            'enable_torch_sync': True,
            'smooth_mmas': not args.no_smooth_mmas,
            'min_new_edges': args.min_new_edges,
            'extend_ls': not args.no_extend_ls,
            'extend_ls': not args.no_extend_ls,
            'normalized_heuristic': not args.no_normalized_heuristic,
            'fixed_steps': args.L
        }
    else:
        coords, demand, capacity = instance_data
        kwargs = {
            'coords': coords,
            'demand': demand,
            'capacity': float(capacity),
            'n_ants': n_ants,
            'cand_list_size': k_sparse,
            'backup_list_size': k_sparse,
            'min_new_edges': args.min_new_edges,
            'decay': args.rho,
            'p_best': 0.05,
            'use_local_search': not args.no_local_search,
            'disable_heuristic': disable_heuristic_arg,
            'extend_ls': not args.no_extend_ls,
            'smooth_mmas': not args.no_smooth_mmas,
            'device': args.device,
            'enable_torch_sync': True,
            'enable_torch_sync': True,
            'normalized_heuristic': not args.no_normalized_heuristic,
            'fixed_steps': args.L
        }

    aco = MFACOClass(**kwargs)
    if hasattr(aco, 'reset_timings'): aco.reset_timings()

    best_seen = float("inf")
    avg_last = None
    priors, pher_before = [], []
    metrics_log = {k: [] for k in ["cost", "l2", "kl", "turnover", "flip", "corr", "ov", "row_match"]}
    
    with torch.no_grad():
        for t in range(args.H):
            prior_mat = None
            if collect_metrics:
                pher_before.append(aco.pheromone_sparse.detach().cpu().clone())

            if model is not None and not use_heuristic_only:
                if problem == 'tsp':
                    pyg_data = build_pyg_data_fn(aco, coords, args.device, dynamic=dynamic)
                else:
                    pyg_data = build_pyg_data_fn(aco, coords, demand, args.device, dynamic=dynamic)
                    
                heu_vec = model(pyg_data).view(-1)
                prior_mat = heu_vec.view(aco.n, aco.k)
                if problem == 'cvrp': prior_mat += EPS
                
                if collect_metrics:
                    priors.append(prior_mat.detach().cpu().clone())

            for mini_t in range(args.mini_H):
                # Annealing
                current_prior = prior_mat
                if args.anneal_prior and prior_mat is not None:
                     if args.mini_H > 1:
                        ratio = mini_t / (args.mini_H - 1)
                        factor = args.gamma * (1.0 - ratio) + args.min_gamma * ratio
                     else:
                        factor = args.gamma
                     current_prior = prior_mat * factor

                # Sample
                return_decoded = getattr(args, 'verify', False) and (problem == 'cvrp')
                
                if problem == 'tsp':
                    costs_t, flats, _, _, traces, _, _, _ = aco.sample(require_prob=False, prior=current_prior)
                    # TSP sample signature doesn't support return_decoded yet in python wrapper?
                    # TSP python wrapper sample: costs, flats, touched, logps, traces, ...
                else:
                    costs_t, perms, decoded, _, traces, _ = aco.sample(require_prob=False, prior=current_prior, return_decoded=return_decoded)
                    flats = perms

                if return_decoded and problem == 'cvrp':
                     best_idx_t = int(costs_t.argmin().item())
                     try:
                         verify_solution_cvrp(coords, demand, capacity, float(costs_t[best_idx_t]), decoded[best_idx_t])
                     except ValueError as e:
                         print(f"Verification failed: {e}")
                         sys.exit(1)

                avg_last = float(costs_t.mean().item())
                best_idx = int(costs_t.argmin().item())
                best_cost = float(costs_t[best_idx].item())
                best_seen = min(best_seen, best_cost)
                
                if problem == 'tsp':
                    aco._update_pheromone_from_flat(flats[best_idx], best_cost)
                else:
                    aco.update_pheromone(flats[best_idx], best_cost)

            if collect_metrics:
                metrics_log["cost"].append(best_seen)
                
                is_prior_avail = (len(priors) > 0)
                if is_prior_avail and t > 0:
                     P_prev, P_cur = priors[t-1], priors[t]
                     metrics_log["l2"].append(rel_l2_drift(P_prev, P_cur))
                     metrics_log["kl"].append(mean_row_kl(P_prev, P_cur))
                     metrics_log["turnover"].append(top_turnover(P_prev, P_cur))
                     metrics_log["flip"].append(top1_flip_rate(P_prev, P_cur))
                else:
                     for k in ["l2", "kl", "turnover", "flip"]: metrics_log[k].append(0.0)

                if is_prior_avail:
                    tau = pher_before[t]
                    pr = priors[t]
                    metrics_log["corr"].append(safe_corr(tau, pr))
                    metrics_log["ov"].append(top_overlap_frac(tau, pr))
                    metrics_log["row_match"].append(row_top1_match_rate(tau, pr))
                else:
                    for k in ["corr", "ov", "row_match"]: metrics_log[k].append(0.0)

    timings = None
    if hasattr(aco, 'get_timings') and args.timed:
        timings = aco.get_timings()
    
    if collect_metrics:
        return avg_last, best_seen, timings, metrics_log
    return avg_last, best_seen, timings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default=None, choices=['tsp', 'cvrp'])
    parser.add_argument("--n_node", type=int, default=None)
    parser.add_argument("--k_sparse", type=int, default=32)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="none")
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--no_local_search", action="store_true")
    parser.add_argument("--no_smooth_mmas", action="store_true")
    parser.add_argument("--no_extend_ls", action="store_true")
    parser.add_argument("--rho", type=float, default=0.5) # TSP 0.5 default in test? train.py said 0.1? Check.
    parser.add_argument("--min_new_edges", type=int, default=12) # TSP 8, CVRP 16?
    parser.add_argument("--no_normalized_heuristic", action="store_true")
    parser.add_argument("--no_logit_net", action="store_true")
    parser.add_argument("--no_dynamic_feats", action="store_true")
    
    parser.add_argument("--baseline", type=str, default='default')
    parser.add_argument("--baseline_time_limit", type=float, default=2.0)
    parser.add_argument("--baseline_runs", type=int, default=1)
    parser.add_argument("--anneal_prior", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_output", type=str, default="visualizations")
    parser.add_argument("--timed", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--L", type=int, default=0, help="Fixed ant trajectory length")
    parser.add_argument("--threads", type=int, default=16, help="OpenMP threads")

    args = parser.parse_args()

    # Checkpoint Loading logic (Pre-module load)
    ckpt = None
    if args.checkpoint != "none":
        print(f"Loading {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        config = ckpt.get("config", {})
        
        if args.problem is None:
            args.problem = config.get("problem", None)
            if args.problem is None:
                 print("Warning: 'problem' not found in checkpoint config and not provided in args.")
        
        if args.n_node is None:
            args.n_node = config.get("n_node", None)
            if args.n_node is None:
                print("Warning: 'n_node' not found in checkpoint config, defaulting to 100.")
                args.n_node = 100

    # Fallback defaults if still None
    if args.problem is None:
         # Cannot proceed without problem
         raise ValueError("Problem must be specified via --problem or present in checkpoint config.")
         
    if args.n_node is None:
         args.n_node = 100
    
    # Defaults adjustment
    if args.baseline == 'default':
        args.baseline = 'lkh' if args.problem == 'tsp' else 'hgs'

    args.extend_ls = not args.no_extend_ls
    
    # Random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Data
    if args.dataset:
        print(f"Loading {args.dataset}...")
        data = torch.load(args.dataset, map_location="cpu")
        if isinstance(data, dict):
            if "coords" in data: val_list = data["coords"] # TSP format often dict
            else: val_list = data # Maybe directly tensor/list?
        else:
            val_list = data # Tensor or list
    else:
        Net, MFACO, load_val_dataset, build_pyg_data, get_baseline, set_threads_fn = get_modules(args.problem)
        set_threads_fn(args.threads)
        print("Loading validation dataset...")
        try:
             val_list = load_val_dataset(args.n_node, "cpu") # Generalize to CPU load first
        except FileNotFoundError:
            print("Generate data locally first!")
            raise

    # Baseline
    baseline_values = None
    if args.baseline != 'none':
        # Re-import to ensure functions are avail if dataset loaded without top block
        if 'get_baseline' not in locals():
            _, _, _, _, get_baseline, _ = get_modules(args.problem)
        
        print("Computing baseline...")
        if args.problem == 'tsp':
            baseline_values = get_baseline(val_list, args.n_node, "cpu", runs=args.baseline_runs, time_limit=args.baseline_time_limit)
        else:
            baseline_values = get_baseline(val_list, args.n_node, "cpu", time_limit=args.baseline_time_limit)
        baseline_values = baseline_values.cpu().numpy()
        print(f"Baseline mean: {baseline_values.mean()}")

    # Model
    model = None
    if ckpt is not None:
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        config = ckpt.get("config", {})
        
        # Update args from config
        ignored_keys = {"device", "checkpoint", "baseline", "baseline_time_limit", "baseline_runs",
                        "dataset", "visualize", "visualize_output", "timed", "verify"}
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
        
        # Determine net class
        if 'Net' not in locals():
            Net, MFACO, load_val_dataset, build_pyg_data, _, set_threads_fn = get_modules(args.problem)
            set_threads_fn(args.threads)
        
        model = Net(logit_net=not args.no_logit_net).to(args.device)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        # Need MFACO class even if no model
        if 'MFACO' not in locals():
            _, MFACO, _, build_pyg_data, _, set_threads_fn = get_modules(args.problem)
            set_threads_fn(args.threads)

    # Eval Loop
    results = {
        "base_cost": [], "model_cost": [],
        "base_time": [], "model_time": [],
        "base_metrics": {}, "model_metrics": {} 
    }
    
    # Prep iteration
    if args.problem == 'tsp':
        iterable = val_list
    else:
        iterable = torch.utils.data.DataLoader(val_list, batch_size=1, shuffle=False)
        
    print(f"Evaluating {len(val_list)} instances...")
    
    for i, item in enumerate(tqdm(iterable)):
        if args.problem == 'cvrp':
            item = [x[0] if torch.is_tensor(x) else x for x in item]
            if torch.is_tensor(item[0]): item[0] = item[0].numpy()
            if torch.is_tensor(item[1]): item[1] = item[1].numpy()
            if torch.is_tensor(item[2]): item[2] = float(item[2])
            
        # Base
        base_ret = infer_instance(args.problem, MFACO, build_pyg_data, None, item, args.k_sparse, args.n_ants, not args.no_dynamic_feats, args, use_heuristic_only=True, collect_metrics=args.visualize)
        if len(base_ret) == 4: _, base_best, base_time, base_m = base_ret
        else: _, base_best, base_time = base_ret; base_m = None
        
        results["base_cost"].append(base_best)
        
        # Model
        model_best = float("inf")
        model_m = None
        if model:
             mod_ret = infer_instance(args.problem, MFACO, build_pyg_data, model, item, args.k_sparse, args.n_ants, not args.no_dynamic_feats, args, use_heuristic_only=False, collect_metrics=args.visualize)
             if len(mod_ret) == 4: _, model_best, _, model_m = mod_ret
             else: _, model_best, _ = mod_ret
             results["model_cost"].append(model_best)
        
        # Accumulate metrics for vis
        if args.visualize:
            if i == 0:
                 results["base_metrics"] = {k: np.zeros(args.H) for k in (base_m.keys() if base_m else [])}
                 if model_m: results["model_metrics"] = {k: np.zeros(args.H) for k in model_m.keys()}
            
            if base_m:
                for k,v in base_m.items():
                    if k in results["base_metrics"] and len(v)==args.H: results["base_metrics"][k] += np.array(v)
            if model_m:
                 for k,v in model_m.items():
                    if k in results["model_metrics"] and len(v)==args.H: results["model_metrics"][k] += np.array(v)

    # Summary
    base_costs = np.array(results["base_cost"])
    avg_base = base_costs.mean()
    print(f"Base Avg: {avg_base}")
    
    if baseline_values is not None:
        gap = (base_costs - baseline_values) / baseline_values * 100
        print(f"Base Gap: {gap.mean():.4f}%")
        
    if model:
        mod_costs = np.array(results["model_cost"])
        avg_mod = mod_costs.mean()
        print(f"Model Avg: {avg_mod}")
        if baseline_values is not None:
             gap_m = (mod_costs - baseline_values) / baseline_values * 100
             print(f"Model Gap: {gap_m.mean():.4f}%")

    # Visualize
    if args.visualize:
        out = Path(args.visualize_output)
        out.mkdir(parents=True, exist_ok=True)
        N = len(val_list)
        
        # Plot Logic (Simplified from original)
        # Assuming we just dump metrics to plots
        if results["base_metrics"]:
             base_avg = {k: v/N for k,v in results["base_metrics"].items()}
             
             plt.figure()
             plt.plot(base_avg["cost"], label="Base")
             if model and results["model_metrics"]:
                 mod_avg = {k: v/N for k,v in results["model_metrics"].items()}
                 plt.plot(mod_avg["cost"], label="Model")
             plt.legend()
             plt.savefig(out / "cost.pdf")
             plt.close()
             
             # Prior metrics if model
             if model and "l2" in mod_avg:
                 plt.figure()
                 for k in ["l2", "turnover", "flip"]:
                     plt.plot(mod_avg[k], label=k)
                 plt.legend()
                 plt.savefig(out / "prior_changes.pdf")
                 plt.close()

if __name__ == "__main__":
    main()
