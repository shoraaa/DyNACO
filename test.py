#!/usr/bin/env python3
import torch
import argparse
import numpy as np
import random
import sys
import time
import os
import psutil 
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Unified imports
import net
import faco
import utils
import baselines

from net import Net
from baselines import get_baseline

# Import metric helpers from utils
from utils import (
    row_softmax, mean_row_kl, rel_l2_drift, top_set, top_turnover,
    top1_flip_rate, safe_corr, top_overlap_frac, row_top1_match_rate, EPS
)


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




def infer_instance(problem, aco_class, build_fn, model, instance_data, k_sparse, n_ants, dynamic, args, use_heuristic_only=False, collect_metrics=False, metrics_every_step=True):
    if model is not None:
        model.eval()

    disable_heuristic_arg = args.disable_heuristic
    if use_heuristic_only:
        disable_heuristic_arg = False 

    # Determine instance args
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
            'normalized_heuristic': not args.no_normalized_heuristic,
            'fixed_steps': args.L
        }

    aco = aco_class(**kwargs)
    if hasattr(aco, 'reset_timings'): aco.reset_timings()

    best_seen = float("inf")
    avg_last = None
    priors, pher_before = [], []
    metrics_log = {k: [] for k in ["cost", "l2", "kl", "turnover", "flip", "corr", "ov", "row_match", "survival"]}
    
    with torch.no_grad():
        for t in range(args.H):
            do_metrics = collect_metrics and (metrics_every_step or t == args.H - 1)
            
            prior_mat = None
            if do_metrics:
                pher_before.append(aco.pheromone_sparse.detach().cpu().clone())

            if model is not None and not use_heuristic_only:
                if problem == 'tsp':
                    pyg_data = build_fn(aco, coords, args.device, dynamic=dynamic)
                else:
                    pyg_data = build_fn(aco, coords, demand, args.device, dynamic=dynamic)
                    
                heu_vec = model(pyg_data).view(-1)
                prior_mat = heu_vec.view(aco.n, aco.k)
                if problem == 'cvrp': prior_mat += EPS
                
                if do_metrics:
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
                    costs_t, flats, _, _, traces, _, _, _, survival = aco.sample(require_prob=do_metrics, prior=current_prior)
                else:
                    costs_t, perms, decoded, _, traces, _, survival = aco.sample(require_prob=do_metrics, prior=current_prior, return_decoded=return_decoded)
                    flats = perms

                if do_metrics:
                    metrics_log["survival"].append(survival.mean())

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

            if do_metrics:
                metrics_log["cost"].append(best_seen)
                is_prior_avail = (len(priors) > 0)
                
                if is_prior_avail and len(priors) > 1:
                     P_prev, P_cur = priors[-2], priors[-1]
                     metrics_log["l2"].append(rel_l2_drift(P_prev, P_cur))
                     metrics_log["kl"].append(mean_row_kl(P_prev, P_cur))
                     metrics_log["turnover"].append(top_turnover(P_prev, P_cur))
                     metrics_log["flip"].append(top1_flip_rate(P_prev, P_cur))
                else:
                     for k in ["l2", "kl", "turnover", "flip"]: metrics_log[k].append(0.0)

                if is_prior_avail:
                    tau = pher_before[-1] # Match last captured
                    pr = priors[-1]
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
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--min_new_edges", type=int, default=12)
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
    parser.add_argument("--L", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)

    args = parser.parse_args()

    # Args setup
    ckpt = None
    if args.checkpoint != "none":
        print(f"Loading {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        config = ckpt.get("config", {})
        
        if args.problem is None:
            args.problem = config.get("problem", None)
        if args.n_node is None:
            args.n_node = config.get("n_node", None)

    if args.problem is None:
         raise ValueError("Problem must be specified.")
         
    if args.n_node is None:
         args.n_node = 100
    
    if args.baseline == 'default':
        args.baseline = 'lkh' if args.problem == 'tsp' else 'hgs'

    args.extend_ls = not args.no_extend_ls
    
    if args.threads is None:
        args.threads = psutil.cpu_count(logical=False)
    faco.set_faco_cpp_threads(args.threads)
    
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup modules
    if args.problem == 'tsp':
        build_fn = utils.build_pyg_data_tsp
        gen_fn = utils.generate_tsp_instance
        MFACO = faco.MFACO_TSP
    else:
        build_fn = utils.build_pyg_data_cvrp
        gen_fn = utils.gen_cvrp_instance
        MFACO = faco.MFACO_CVRP

    # Dataset
    if args.dataset:
        print(f"Loading {args.dataset}...")
        data = torch.load(args.dataset, map_location="cpu")
        if isinstance(data, dict):
            if "coords" in data: val_list = data["coords"]
            else: val_list = data
        else:
            val_list = data
    else:
        print("Loading validation dataset...")
        val_list = utils.load_val_dataset(args.n_node, problem=args.problem, device='cpu')
        
        if val_list is None:
            print("Generating data...")
            print("Generating 16 instances on fly...")
            val_list = []
            for _ in range(16):
                if args.problem == 'tsp':
                    val_list.append(torch.from_numpy(gen_fn(args.n_node)))
                else:
                    c, d, cap = gen_fn(args.n_node, device='cpu')
                    val_list.append((c.cpu(), d.cpu(), cap))
            
            # Save for reuse
            utils.save_val_dataset(val_list, args.n_node, problem=args.problem)

    # Baseline
    baseline_values = None
    if args.baseline != 'none':
        print("Computing baseline...")
        # Use TensorDataset wrapper for CVRP get_baseline compatibility
        if args.problem == 'cvrp' and isinstance(val_list, list) and not hasattr(val_list, 'tensors'):
            # Convert list of tuples to TensorDataset
            cs = torch.stack([x[0] for x in val_list])
            ds = torch.stack([x[1] for x in val_list])
            caps = torch.stack([torch.tensor(x[2]) for x in val_list])
            ds_wrapper = torch.utils.data.TensorDataset(cs, ds, caps)
            baseline_values = get_baseline(ds_wrapper, problem='cvrp', n_node=args.n_node, time_limit=args.baseline_time_limit)
        else:
            baseline_values = get_baseline(val_list, problem=args.problem, n_node=args.n_node, runs=args.baseline_runs, time_limit=args.baseline_time_limit)
        
        if baseline_values is not None:
             baseline_values = baseline_values.cpu().numpy()
             print(f"Baseline mean: {baseline_values.mean()}")

    # Model
    model = None
    if ckpt is not None:
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        
        # Override args from config logic omitted for brevity, similar to original but using unified flags
        config = ckpt.get("config", {})
        # ... logic to override args if needed ...

        feats = 2 if args.problem == 'tsp' else 4
        model = Net(feats=feats, logit_net=not args.no_logit_net).to(args.device)
        model.load_state_dict(state_dict)
        model.eval()

    # Eval
    results = {
        "base_cost": [], "model_cost": [],
        "base_time": [], "model_time": [],
        "base_metrics": {}, "model_metrics": {} 
    }
    
    iterable = val_list
    if args.problem == 'cvrp' and hasattr(val_list, 'tensors'):
         iterable = torch.utils.data.DataLoader(val_list, batch_size=1, shuffle=False)
    
    print(f"Evaluating {len(val_list)} instances...")
    
    for i, item in enumerate(tqdm(iterable)):
        if args.problem == 'cvrp':
            if isinstance(item, list) and len(item)==1: item = item[0] # DataLoader batch=1
            # item is [coords, demand, cap] tensors if from DataLoader
            # or (coords, demand, cap) tuple if from list
            if isinstance(item, (list, tuple)):
                # If from DataLoader batch=1, we might have (1, N, 2). If from list, (N, 2).
                # Only unbatch if looks like batch dim 
                item = [x[0] if torch.is_tensor(x) and x.dim()==3 else x for x in item] # Unbatch if batched?
                # DataLoader adds batch dim? Yes batch_size=1 -> (1, n, 2).
                if torch.is_tensor(item[0]) and item[0].shape[0] == 1 and item[0].dim() == 3:
                     item[0] = item[0].squeeze(0).numpy()
                     item[1] = item[1].squeeze(0).numpy()
                     item[2] = float(item[2].item())
                elif torch.is_tensor(item[0]): # From list of tensors
                     item[0] = item[0].numpy()
                     item[1] = item[1].numpy()
                     item[2] = float(item[2])
            
        # Base
        tb0 = time.time()
        base_ret = infer_instance(args.problem, MFACO, build_fn, None, item, args.k_sparse, args.n_ants, not args.no_dynamic_feats, args, use_heuristic_only=True, collect_metrics=args.visualize, metrics_every_step=args.visualize)
        tb1 = time.time()
        if len(base_ret) == 4: _, base_best, base_timings, base_m = base_ret
        else: _, base_best, base_timings = base_ret; base_m = None
        
        results["base_cost"].append(base_best)
        results["base_time"].append(tb1 - tb0)
        
        # Model
        if model:
             tm0 = time.time()
             mod_ret = infer_instance(args.problem, MFACO, build_fn, model, item, args.k_sparse, args.n_ants, not args.no_dynamic_feats, args, use_heuristic_only=False, collect_metrics=args.visualize, metrics_every_step=args.visualize)
             tm1 = time.time()
             if len(mod_ret) == 4: _, model_best, _, model_m = mod_ret
             else: _, model_best, _ = mod_ret
             results["model_cost"].append(model_best)
             results["model_time"].append(tm1 - tm0)
        
        if args.visualize:
            if i == 0:
                 results["base_metrics"] = {k: np.zeros(args.H) for k in (base_m.keys() if base_m else [])}
                 if model and model_m: results["model_metrics"] = {k: np.zeros(args.H) for k in model_m.keys()}
            
            if base_m:
                 for k,v in base_m.items():
                    if k in results["base_metrics"] and len(v)==args.H: results["base_metrics"][k] += np.array(v)
            if model and model_m:
                 for k,v in model_m.items():
                    if k in results["model_metrics"] and len(v)==args.H: results["model_metrics"][k] += np.array(v)

    # Summary
    base_costs = np.array(results["base_cost"])
    avg_base = base_costs.mean()
    print(f"Base Avg: {avg_base}")
    print(f"Base Total Time: {np.sum(results['base_time']):.2f}s")
    
    if baseline_values is not None:
        gap = (base_costs - baseline_values) / baseline_values * 100
        print(f"Base Gap: {gap.mean():.4f}%")
        
    if model:
        mod_costs = np.array(results["model_cost"])
        avg_mod = mod_costs.mean()
        print(f"Model Avg: {avg_mod}")
        print(f"Model Total Time: {np.sum(results['model_time']):.2f}s")
        if baseline_values is not None:
             gap_m = (mod_costs - baseline_values) / baseline_values * 100
             print(f"Model Gap: {gap_m.mean():.4f}%")
    
    if args.visualize:
        out = Path(args.visualize_output)
        out.mkdir(parents=True, exist_ok=True)
        N = len(val_list)
        
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
             
             if model and "l2" in mod_avg:
                 plt.figure()
                 for k in ["l2", "turnover", "flip"]:
                     plt.plot(mod_avg[k], label=k)
                 plt.legend()
                 plt.savefig(out / "prior_changes.pdf")
                 plt.close()

if __name__ == "__main__":
    main()
