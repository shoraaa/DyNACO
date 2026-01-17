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

from net import Net
from utils import load_val_dataset, build_pyg_data
from faco import MFACO_CVRP
from baselines import get_baseline_cvrp

EPS = 1e-10

def infer_instance(model, coords, demand, capacity, k_sparse, n_ants, dynamic, args, use_heuristic_only=False):
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
    )

    best_seen = float("inf")
    avg_last = None
    H = args.H

    with torch.no_grad():
        for _ in range(H):
            for mini_t in range(args.mini_H):
                prior_mat = None
                if model is not None and not use_heuristic_only:
                    pyg_data = build_pyg_data(aco, coords, demand, args.device, dynamic=dynamic)
                    heu_vec = model(pyg_data).view(-1)
                    prior_mat = heu_vec.view(aco.n, aco.k) + EPS

                # Sample
                costs_t, perms, _, _, _ = aco.sample(require_prob=False, prior=prior_mat)

                avg_last = float(costs_t.mean().item())
                best_idx = int(costs_t.argmin().item())
                best_cost = float(costs_t[best_idx].item())
                best_seen = min(best_seen, best_cost)
                best_perm = perms[best_idx]

                aco.update_pheromone(best_perm, best_cost)

    return avg_last, best_seen

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
    parser.add_argument("--min_new_edges", type=int, default=8, help="Min new edges")
    parser.add_argument("--no_normalized_heuristic", action="store_true", help="Normalize heuristic to [1/k, 1]")
    parser.add_argument("--no_logit_net", action="store_true", help="Use logit network (no sigmoid) and log-space ACO")
    
    # Baseline
    parser.add_argument("--baseline", type=str, choices=['none', 'hgs'], default='hgs', help="Baseline")
    parser.add_argument("--baseline_time_limit", type=float, default=0.2, help="HGS time limit (s)")
    
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=1234, help="Seed")
    parser.add_argument("--no_dynamic_feats", action="store_true", help="No dyn feats")

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

    print("\nRunning Evaluation...")
    print(f"{'Idx':<5} {'Opt (HGS)':<10} {'Base MFACO':<12} {'Model MFACO':<12} {'Base Gap':<10} {'Model Gap':<10}")

    for i, batch in enumerate(tqdm(loader, leave=False)):
        coords = batch[0][0].numpy()
        demand = batch[1][0].numpy()
        capacity = float(batch[2][0])

        # Base MFACO
        _, base_best = infer_instance(None, coords, demand, capacity, args.k_sparse, args.n_ants, dynamic, args, use_heuristic_only=True)
        
        # Model MFACO
        model_best = float("inf")
        if model is not None:
             _, model_best = infer_instance(model, coords, demand, capacity, args.k_sparse, args.n_ants, dynamic, args, use_heuristic_only=False)
        
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