#!/usr/bin/env python3
import torch
import argparse
import numpy as np
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm

from net import Net
from utils import load_val_dataset, build_pyg_data
from faco import MFACO_TSP
from baselines import get_baseline_tsp

EPS = 1e-10

def infer_instance(model, coords, k_sparse, n_ants, dynamic, args, use_heuristic_only=False):
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
        extend_ls=args.extend_ls,
        normalized_heuristic=not args.no_normalized_heuristic,
    )

    best_seen = float("inf")
    avg_last = None
    H = args.H

    with torch.no_grad():
        for _ in range(H):
            prior_mat = None
            if model is not None and not use_heuristic_only:
                pyg_data = build_pyg_data(aco, coords, args.device, dynamic=dynamic)
                heu_vec = model(pyg_data).view(-1)
                prior_mat = heu_vec.view(aco.n, aco.k)

            for mini_t in range(args.mini_H):
                # If use_heuristic_only, prior_mat is None, so it uses 1/d (tau * 1/d)
                # If model, prior_mat is used (tau * model)
                costs, flats, _, _, _ = aco.sample(require_prob=False, prior=prior_mat)

                avg_last = float(costs.mean())
                best_idx = int(costs.argmin())
                best_cost = float(costs[best_idx])
                best_seen = min(best_seen, best_cost)

                aco._update_pheromone_from_flat(flats[best_idx], best_cost)

    return avg_last, best_seen

def main():
    parser = argparse.ArgumentParser(description="MFACO TSP Testing")
    
    # Dataset / Problem
    parser.add_argument("--n_node", type=int, default=100, help="Number of nodes per TSP instance")
    parser.add_argument("--k_sparse", type=int, default=32, help="Candidate list size (k)")
    
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset (.pt) file")
    
    # Model / Training
    parser.add_argument("--checkpoint", type=str, default="none", help="Path to model checkpoint or 'none'")
    parser.add_argument("--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("--H", type=int, default=20, help="ACO steps per instance (H)")
    parser.add_argument("--mini_H", type=int, default=10, help="ACO steps per iteration (mini_H)")
    
    # Checkpoint loaded model typically trained with disable_heuristic=True. 
    # But for "Base MFACO" we want to force enable it.
    parser.add_argument("--disable_heuristic", action="store_true", help="Disable heuristic")
    parser.add_argument("--no_local_search", action="store_true", help="Disable local search")
    parser.add_argument("--no_smooth_mmas", action="store_true", help="Enable smooth MMAS")
    parser.add_argument("--extend_ls", action="store_true", help="Extend LS checklist")
    parser.add_argument("--rho", type=float, default=0.1, help="Rho")
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
                        if current_val != v:
                            print(f"Loading {k}={v} from checkpoint (was {current_val})")
                            setattr(args, k, v)
        
        # Init model with potentially updated args (e.g. logit_net)
        model = Net(logit_net=args.logit_net).to(args.device)
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
        _, base_best = infer_instance(None, coords, args.k_sparse, args.n_ants, dynamic, args, use_heuristic_only=True)
        
        # Model MFACO (Prior from Net)
        model_best = float("inf")
        if model is not None:
             _, model_best = infer_instance(model, coords, args.k_sparse, args.n_ants, dynamic, args, use_heuristic_only=False)
        
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
            
    print("="*60)

if __name__ == "__main__":
    main()
