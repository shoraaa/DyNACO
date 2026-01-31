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
import math

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




def infer_instance(problem, aco_class, build_fn, model, instance_data, k_sparse, n_ants, dynamic, args, use_heuristic_only=False, collect_metrics=False, metrics_every_step=True, inject_step=None):
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
    best_seen = float("inf")
    avg_last = None
    t_neural_total = 0.0
    priors, pher_before = [], []
    metrics_log = {k: [] for k in ["cost", "l2", "kl", "turnover", "flip", "corr", "ov", "row_match", "survival"]}
    metrics_log["snapshots"] = []
    
    with torch.no_grad():
        for t in range(args.H):
            do_metrics = collect_metrics and (metrics_every_step or t == args.H - 1)
            
            prior_mat = None
            if do_metrics:
                pher_before.append(aco.pheromone_sparse.detach().cpu().clone())

            if model is not None and not use_heuristic_only:
                # If inject_step is set, only use model if t >= inject_step
                use_model = True
                if inject_step is not None and t < inject_step:
                    use_model = False
                
                if use_model:
                    if problem == 'tsp':
                        pyg_data = build_fn(aco, coords, args.device, dynamic=dynamic)
                    else:
                        pyg_data = build_fn(aco, coords, demand, args.device, dynamic=dynamic)
                    
                    t_neural_start = time.time()
                    heu_vec = model(pyg_data).view(-1)
                    t_neural_total += time.time() - t_neural_start
                    
                    prior_mat = heu_vec.view(aco.n, aco.k)
                    # if problem == 'cvrp': prior_mat += EPS
                    
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
                
                prior_arg = current_prior.cpu().numpy() if (current_prior is not None and torch.is_tensor(current_prior)) else current_prior

                if problem == 'tsp':
                    costs_t, flats, _, _, traces, _, _, _, survival = aco.sample(require_prob=do_metrics, prior=prior_arg)
                else:
                    costs_t, perms, decoded, _, traces, _, _, _, survival = aco.sample(require_prob=do_metrics, prior=prior_arg, return_decoded=return_decoded)
                    flats = perms

                if do_metrics:
                    metrics_log["survival"].append(survival.mean().item())

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
            
            # Capture snapshots at H/2
            if collect_metrics and t == (args.H // 2):
                 # Pheromone
                 pher = aco.pheromone_sparse.detach().cpu()
                 
                 # Neural Prior (Model Output)
                 neural_prior = None
                 if 'prior_mat' in locals() and prior_mat is not None:
                      neural_prior = prior_mat.detach().cpu()

                 metrics_log["snapshots"].append({
                     "t": t,
                     "pheromone": pher,
                     "neural_prior": neural_prior
                 })


    timings = {}
    if hasattr(aco, 'get_timings') and args.timed:
        t = aco.get_timings()
        timings = {k: v/1000.0 for k, v in t.items()} # ms to s
    
    if args.timed:
        timings["time_neural"] = t_neural_total
    
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
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--min_new_edges", type=int, default=8)
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

    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable warmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.5)
    parser.add_argument("--generate_val", action="store_true", help="Generate test set instead of loading from file")
    parser.add_argument("--save_generated", type=str, default=None, help="Path to save generated test dataset")
    parser.add_argument("--val_size", type=int, default=None, help="Limit validation set size")

    args = parser.parse_args()

    # Args setup
    ckpt = None
    if args.checkpoint != "none":
        print(f"Loading {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        print("Checkpoint Metadata:")
        for k, v in ckpt.items():
            if k not in ["model_state_dict", "optimizer_state_dict", "config"]:
                 print(f"  {k}: {v}")
        if "config" in ckpt:
             print(f"  config: {ckpt['config']}")
        config = ckpt.get("config", {})
        
        # Override args from config if not present in sys.argv
        # Checkpoint/Device/Data args should NOT be overwritten usually
        ignore_args = {
            "checkpoint", "device", "dataset", "visualize", "visualize_output", 
            "timed", "verify", "baseline", "baseline_runs", "baseline_time_limit", 
            "threads", "seed", "save_dir", "wandb_project", "wandb_entity", "no_wandb", "warmup"
        }
        
        print("Restoring config from checkpoint (unless overridden):")
        for k, v in config.items():
            if k in ignore_args: continue
            if not hasattr(args, k): continue
            
            # Simple check: if flag is in sys.argv, user overrode it
            flag_underscore = "--" + k
            flag_hyphen = "--" + k.replace("_", "-")
            
            if (flag_underscore not in sys.argv) and (flag_hyphen not in sys.argv):
                 current_val = getattr(args, k)
                 if current_val != v:
                     print(f"  Override {k}: {current_val} -> {v}")
                     setattr(args, k, v)
        
    if args.problem is None:
         raise ValueError("Problem must be specified (in args or checkpoint).")
         
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
    if args.generate_val:
        # Generate test dataset dynamically
        baseline_solver = args.baseline
        val_list = utils.generate_and_save_dataset(
            problem=args.problem,
            n_node=args.n_node,
            n_instances=args.val_size if args.val_size is not None else 16,
            save_path=args.save_generated,
            baseline_solver=baseline_solver,
            baseline_runs=args.baseline_runs,
            time_limit=args.baseline_time_limit,
            device='cpu'
        )
    elif args.dataset:
        print(f"Loading {args.dataset}...")
        if args.dataset.endswith(".txt") and args.problem == 'tsp':
             val_list = utils.load_tsp_txt_dataset(args.dataset)
        elif args.dataset.endswith(".txt") and args.problem == 'cvrp':
             val_list = utils.load_cvrp_txt_dataset(args.dataset)
        else:
            data = torch.load(args.dataset, map_location="cpu", weights_only=False)
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

    # Limit validation set size if requested
    if args.val_size is not None and val_list is not None:
        if isinstance(val_list, (list, tuple)) or torch.is_tensor(val_list):
            original_len = len(val_list)
            val_list = val_list[:args.val_size]
            print(f"Limited validation dataset from {original_len} to {len(val_list)} instances.")

    # Baseline
    baseline_values = None
    
    # Check if dataset has embedded baseline costs (e.g. from file)
    if isinstance(val_list, list) and len(val_list) > 0:
        if args.problem == 'tsp' and isinstance(val_list[0], tuple) and len(val_list[0]) >= 2:
            try:
                costs = [x[1] for x in val_list]
                # Allow python float/int and numpy scalars. Check strictly positive.
                if all((isinstance(c, (int, float)) or np.issubdtype(type(c), np.number)) and c > 1e-6 for c in costs):
                    print("Using baseline costs from dataset.")
                    baseline_values = np.array(costs)
            except Exception: pass
            
        # CVRP Tuple: (coords, demand, capacity, cost, tour)
        elif args.problem == 'cvrp' and isinstance(val_list[0], tuple) and len(val_list[0]) == 5:
             try:
                 costs = [x[3] for x in val_list]
                 if all((isinstance(c, (int, float)) or np.issubdtype(type(c), np.number)) and c > 1e-6 for c in costs):
                     print("Using baseline costs from dataset.")
                     baseline_values = np.array(costs)
             except Exception: pass

    if args.baseline != 'none' and baseline_values is None:
        print("Computing baseline...")
        # Use TensorDataset wrapper for CVRP get_baseline compatibility
        if args.problem == 'cvrp' and isinstance(val_list, list) and len(val_list) > 0 and not isinstance(val_list[0], tuple):
             # Only if not tuples (i.e. if already tensors)
             pass
        
        # If tuple dataset (text), we need to extract coords/demands for baseline?
        # get_baseline for CVRP expects dataset wrapper.
        # But if we have optimal cost, maybe we don't need baseline?
        # User prompt didn't strictly say so, but usually yes.
        # Let's handle generic case.
        
        if args.problem == 'cvrp' and isinstance(val_list, list) and not hasattr(val_list, 'tensors'):
            if len(val_list)>0 and isinstance(val_list[0], tuple) and len(val_list[0]) == 5:
                 # Text dataset tuple: (coords, demand, capacity, cost, tour)
                 cs = torch.stack([x[0] for x in val_list])
                 ds = torch.stack([x[1] for x in val_list])
                 caps = torch.stack([torch.tensor(x[2]) for x in val_list]) # Capacity is float
                 # opt_costs = [x[3] for x in val_list]
                 ds_wrapper = torch.utils.data.TensorDataset(cs, ds, caps)
                 baseline_values = get_baseline(ds_wrapper, problem='cvrp', n_node=args.n_node, time_limit=args.baseline_time_limit)
            elif len(val_list)>0 and isinstance(val_list[0], tuple) and len(val_list[0]) == 3:
                 # Generated: (c, d, cap)
                 cs = torch.stack([x[0] for x in val_list])
                 ds = torch.stack([x[1] for x in val_list])
                 caps = torch.stack([torch.tensor(x[2]) for x in val_list])
                 ds_wrapper = torch.utils.data.TensorDataset(cs, ds, caps)
                 baseline_values = get_baseline(ds_wrapper, problem='cvrp', n_node=args.n_node, time_limit=args.baseline_time_limit)
        else:
            # Handle potential tuple items in TSP (coords, cost, tour) for baselines
            # Just extract coords for baseline computation if needed
            if args.problem == 'tsp' and isinstance(val_list, list) and len(val_list) > 0 and isinstance(val_list[0], tuple):
                val_list_coords = [x[0] if isinstance(x, tuple) else x for x in val_list]
                baseline_values = get_baseline(val_list_coords, problem=args.problem, n_node=args.n_node, runs=args.baseline_runs, time_limit=args.baseline_time_limit)
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
        "base_cost": [], "model_cost": [], "mix_cost": [],
        "base_time": [], "model_time": [], "mix_time": [],
        "base_metrics": {}, "model_metrics": {}, "mix_metrics": {},
        "base_cost": [], "model_cost": [], "mix_cost": [],
        "base_time": [], "model_time": [], "mix_time": [],
        "base_metrics": {}, "model_metrics": {}, "mix_metrics": {},
        "opt_cost": [], "base_gap": [], "model_gap": [], "mix_gap": [],
        "model_time_breakdown": {
            "neural": [], "sampling": [], "ls": [], "update": []
        }
    }
    
    iterable = val_list
    if args.problem == 'cvrp' and hasattr(val_list, 'tensors'):
         iterable = torch.utils.data.DataLoader(val_list, batch_size=1, shuffle=False)
    
    print(f"Evaluating {len(val_list)} instances...")
    
    sample_snapshots = {}

    for i, item in enumerate(tqdm(iterable)):
        opt_cost = None
        
        # Unpack TSP tuple if present
        if args.problem == 'tsp' and isinstance(item, tuple):
             # (coords, cost, tour)
             coords = item[0]
             if len(item) > 1: opt_cost = item[1]
             item = coords
        
        if args.problem == 'cvrp':
            if isinstance(item, list) and len(item)==1: item = item[0] # DataLoader batch=1
            
            # Unpack CVRP Text Tuple (coords, demand, capacity, cost, tour)
            if isinstance(item, tuple) and len(item) == 5:
                # (coords, demand, capacity, cost, tour)
                coords, demand, capacity, cost, tour = item
                if cost is not None and isinstance(cost, (float, int)) and cost > 0:
                    opt_cost = cost
                
                # Reduce to (coords, demand, capacity) for solver
                item = (coords, demand, capacity)

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
        
        if opt_cost is not None:
             gap = (base_best - opt_cost) / opt_cost
             results["base_gap"].append(gap)
             results["opt_cost"].append(opt_cost)

        # Model
        if model:
             tm0 = time.time()
             mod_ret = infer_instance(args.problem, MFACO, build_fn, model, item, args.k_sparse, args.n_ants, not args.no_dynamic_feats, args, use_heuristic_only=False, collect_metrics=args.visualize, metrics_every_step=args.visualize)
             tm1 = time.time()
             if len(mod_ret) == 4: _, model_best, mod_timings, model_m = mod_ret
             else: _, model_best, mod_timings = mod_ret
             results["model_cost"].append(model_best)
             results["model_time"].append(tm1 - tm0)
             
             if mod_timings:
                 if "time_neural" in mod_timings: results["model_time_breakdown"]["neural"].append(mod_timings["time_neural"])
                 if "time_sampling" in mod_timings: results["model_time_breakdown"]["sampling"].append(mod_timings["time_sampling"])
                 if "time_ls" in mod_timings: results["model_time_breakdown"]["ls"].append(mod_timings["time_ls"])
                 if "time_update" in mod_timings: results["model_time_breakdown"]["update"].append(mod_timings["time_update"])
             
             if opt_cost is not None:
                 gap = (model_best - opt_cost) / opt_cost
                 results["model_gap"].append(gap)
             
             if args.warmup:
                 tmi0 = time.time()
                 inject_step = int(args.H * args.warmup_ratio)
                 mix_ret = infer_instance(args.problem, MFACO, build_fn, model, item, args.k_sparse, args.n_ants, not args.no_dynamic_feats, args, use_heuristic_only=False, collect_metrics=args.visualize, metrics_every_step=args.visualize, inject_step=inject_step)
                 tmi1 = time.time()
                 if len(mix_ret) == 4: _, mix_best, _, mix_m = mix_ret
                 else: _, mix_best, _ = mix_ret
                 results["mix_cost"].append(mix_best)
                 results["mix_time"].append(tmi1 - tmi0)
                 
                 if opt_cost is not None:
                     gap = (mix_best - opt_cost) / opt_cost
                     results["mix_gap"].append(gap)
        
        if args.visualize:
            if i == 0:
                 # Initialize with length from first instance
                 if base_m: 
                     results["base_metrics"] = {k: np.zeros(len(v)) for k,v in base_m.items()}
                 
                 if model and model_m: 
                     results["model_metrics"] = {k: np.zeros(len(v)) for k,v in model_m.items()}
                 
                 if model and args.warmup and mix_m: 
                     results["mix_metrics"] = {k: np.zeros(len(v)) for k,v in mix_m.items()}
            
            if base_m:
                 for k,v in base_m.items():
                    if k == "snapshots": continue
                    if k in results["base_metrics"]:
                         if len(v) == len(results["base_metrics"][k]):
                             results["base_metrics"][k] += np.array(v)
                         else:
                             # Length mismatch fallback (e.g. truncated run?)
                             L = min(len(v), len(results["base_metrics"][k]))
                             results["base_metrics"][k][:L] += np.array(v[:L])
                             
            if model and model_m:
                 for k,v in model_m.items():
                    if k == "snapshots": continue
                    if k in results["model_metrics"]:
                        if len(v) == len(results["model_metrics"][k]):
                            results["model_metrics"][k] += np.array(v)
                        else:
                             L = min(len(v), len(results["model_metrics"][k]))
                             results["model_metrics"][k][:L] += np.array(v[:L])

            if model and args.warmup and mix_m:
                 for k,v in mix_m.items():
                    if k == "snapshots": continue
                    if k in results["mix_metrics"]:
                        if len(v) == len(results["mix_metrics"][k]):
                            results["mix_metrics"][k] += np.array(v)
                        else:
                             L = min(len(v), len(results["mix_metrics"][k]))
                             results["mix_metrics"][k][:L] += np.array(v[:L])

            if args.visualize and i == 0:
                 # Capture snapshots from i=0
                 if base_m and "snapshots" in base_m: sample_snapshots["base"] = base_m["snapshots"]
                 if model and model_m and "snapshots" in model_m: sample_snapshots["model"] = model_m["snapshots"]
                 if model and args.warmup and mix_m and "snapshots" in mix_m: sample_snapshots["mix"] = mix_m["snapshots"]

    print("\n--- Results ---")
    
    # Base
    base_cost_mean = np.mean(results["base_cost"])
    base_time_mean = np.mean(results["base_time"])
    print(f"Base Cost: {base_cost_mean:.4f}, Time: {base_time_mean:.4f}s")
    if results["base_gap"]:
        print(f"Base Gap: {np.mean(results['base_gap']) * 100:.4f}%")
        
    # Model
    if model:
        model_cost_mean = np.mean(results["model_cost"])
        model_time_mean = np.mean(results["model_time"])
        print(f"Model Cost: {model_cost_mean:.4f}, Time: {model_time_mean:.4f}s")
        if results["model_gap"]:
            print(f"Model Gap: {np.mean(results['model_gap']) * 100:.4f}%")
        
        # Timing Breakdown
        breakdown = results["model_time_breakdown"]
        if breakdown["neural"]:
             t_nn = np.sum(breakdown["neural"])
             t_samp = np.sum(breakdown["sampling"])
             t_ls = np.sum(breakdown["ls"])
             t_upd = np.sum(breakdown["update"])
             t_total_calc = t_nn + t_samp + t_ls + t_upd
             if t_total_calc > 1e-9:
                 print(f"Timing Breakdown: NN: {t_nn/t_total_calc*100:.1f}%, Sampling: {t_samp/t_total_calc*100:.1f}%, LS: {t_ls/t_total_calc*100:.1f}%, Update: {t_upd/t_total_calc*100:.1f}%")
            
        if args.warmup:
             mix_cost_mean = np.mean(results["mix_cost"])
             mix_time_mean = np.mean(results["mix_time"])
             print(f"Mix Cost: {mix_cost_mean:.4f}, Time: {mix_time_mean:.4f}s")
             if results["mix_gap"]:
                print(f"Mix Gap: {np.mean(results['mix_gap']) * 100:.4f}%")

    if baseline_values is not None:
         print(f"Baseline Cost: {baseline_values.mean():.4f}")
         # Gap to baseline
         base_gap_bl = (np.mean(results["base_cost"]) - baseline_values.mean()) / baseline_values.mean()
         print(f"Base Gap to Baseline: {base_gap_bl * 100:.4f}%")
         
         if model:
             mod_gap_bl = (np.mean(results["model_cost"]) - baseline_values.mean()) / baseline_values.mean()
             print(f"Model Gap to Baseline: {mod_gap_bl * 100:.4f}%")
             
             if args.warmup:
                  mix_gap_bl = (np.mean(results["mix_cost"]) - baseline_values.mean()) / baseline_values.mean()
                  print(f"Mix Gap to Baseline: {mix_gap_bl * 100:.4f}%")

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
             if model and args.warmup and results["mix_metrics"]:
                 mix_avg = {k: v/N for k,v in results["mix_metrics"].items()}
                 plt.plot(mix_avg["cost"], label="Mix")
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

        if sample_snapshots:
            print("Plotting matrix snapshots...")
            for mode, snaps in sample_snapshots.items():
                for snap in snaps:
                    t = snap["t"]
                    pher = snap.get("pheromone")
                    neural_prior = snap.get("neural_prior")
                    
                    # Columns: Pheromone, [Neural Prior]
                    ncols = 1
                    if neural_prior is not None: ncols += 1
                    
                    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
                    if ncols == 1: axes = [axes]
                    
                    MAX_ROWS = 100
                    
                    # Helper for safe plotting
                    def safe_plot_heatmap(ax, tensor, title, cmap):
                        # Truncate to MAX_ROWS
                        if tensor.shape[0] > MAX_ROWS:
                            tensor = tensor[:MAX_ROWS]
                            title += f" (first {MAX_ROWS} rows)"
                            
                        arr = tensor.numpy()
                        # Handle NaNs/Infs
                        if not np.isfinite(arr).all():
                            arr = np.nan_to_num(arr, nan=0.0, posinf=np.nanmax(arr[np.isfinite(arr)]), neginf=np.nanmin(arr[np.isfinite(arr)]))
                        
                        # Handle constant values to avoid norm errors
                        vmin, vmax = arr.min(), arr.max()
                        if math.isclose(vmin, vmax):
                            vmax = vmin + 1e-6

                        im = ax.imshow(arr, aspect='auto', cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
                        ax.set_title(title)
                        ax.set_xlabel("Neighbor Rank")
                        ax.set_ylabel("Node Index")
                        fig.colorbar(im, ax=ax)

                    # 1. Pheromone
                    safe_plot_heatmap(axes[0], pher, f"{mode} t={t}: Pheromone (tau)", 'viridis')
                    
                    # 2. Neural Prior
                    if neural_prior is not None:
                        safe_plot_heatmap(axes[1], neural_prior, f"{mode} t={t}: Neural Prior (p)", 'inferno')
                    
                    plt.tight_layout()
                    plt.savefig(out / f"matrix_{mode}_t{t}.pdf", dpi=300)
                    plt.close()

if __name__ == "__main__":
    main()
