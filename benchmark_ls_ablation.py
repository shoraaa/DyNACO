#!/usr/bin/env python3
"""
Ablation study: benchmark which inter-route LS move combinations
provide the best time-quality efficiency for CVRP.

Tests all 7 non-empty subsets of {Relocate, Swap, 2-opt*}.
Uses same methodology as test.py for fair comparison.
"""
import time
import numpy as np
import torch
import psutil
import faco
from faco import MFACO_CVRP
from utils import load_cvrp_txt_dataset
from pathlib import Path

# Configurations to test (all 2^3 - 1 = 7 non-empty combinations)
CONFIGS = [
    {"name": "Relocate only", "use_relocate": True, "use_swap": False, "use_2opt_star": False},
    {"name": "Swap only", "use_relocate": False, "use_swap": True, "use_2opt_star": False},
    {"name": "2-opt* only", "use_relocate": False, "use_swap": False, "use_2opt_star": True},
    {"name": "Relocate + Swap", "use_relocate": True, "use_swap": True, "use_2opt_star": False},
    {"name": "Relocate + 2-opt*", "use_relocate": True, "use_swap": False, "use_2opt_star": True},
    {"name": "Swap + 2-opt*", "use_relocate": False, "use_swap": True, "use_2opt_star": True},
    {"name": "Full (All)", "use_relocate": True, "use_swap": True, "use_2opt_star": True},
]


def run_benchmark(dataset_path: str, n_ants: int = 100, H: int = 10, mini_H: int = 100,
                  k_sparse: int = 32, rho: float = 0.1, min_new_edges: int = 8,
                  n_instances: int = None, threads: int = None):
    """Run benchmark for all configurations using test.py methodology."""
    
    # Set threads
    if threads is None:
        threads = psutil.cpu_count(logical=False)
    faco.set_faco_cpp_threads(threads)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data_list = load_cvrp_txt_dataset(dataset_path)
    if n_instances:
        data_list = data_list[:n_instances]
    print(f"Loaded {len(data_list)} instances")
    
    n = data_list[0][0].shape[0]
    total_iters = H * mini_H
    
    print(f"\n{'='*80}")
    print(f"LS Move Ablation Study: N={n}, n_ants={n_ants}, H={H}, mini_H={mini_H} (total={total_iters})")
    print(f"k_sparse={k_sparse}, rho={rho}, min_new_edges={min_new_edges}, threads={threads}")
    print(f"Testing {len(data_list)} instances")
    print(f"{'='*80}\n")
    
    results = []
    
    for config in CONFIGS:
        name = config["name"]
        costs = []
        baselines = []
        times = []
        
        for inst_idx, (coords, demand, capacity, baseline_cost, _, _) in enumerate(data_list):
            baselines.append(baseline_cost)
            
            # Create solver with same parameters as test.py
            solver = MFACO_CVRP(
                coords=coords,
                demand=demand,
                capacity=float(capacity),
                n_ants=n_ants,
                cand_list_size=k_sparse,
                backup_list_size=max(k_sparse, 64),
                min_new_edges=min_new_edges,
                decay=rho,
                p_best=0.05,
                use_local_search=True,
                disable_heuristic=False,
                extend_ls=True,
                smooth_mmas=True,
                device="cpu",  # No neural network
            )
            
            # Set ablation flags on the C++ solver
            solver.solver.use_relocate = config["use_relocate"]
            solver.solver.use_swap = config["use_swap"]
            solver.solver.use_2opt_star = config["use_2opt_star"]
            
            # Run optimization (same loop structure as test.py)
            start = time.perf_counter()
            for h in range(H):
                for mini_h in range(mini_H):
                    costs_t, routes, _, _, _, _, _, _, _ = solver.sample()
                    # Update pheromone with best ant (same as test.py)
                    best_idx = int(costs_t.argmin().item())
                    best_cost = float(costs_t[best_idx].item())
                    solver.update_pheromone(routes[best_idx], best_cost)
            elapsed = time.perf_counter() - start
            
            costs.append(solver.solver.best_cost)
            times.append(elapsed)
        
        avg_cost = np.mean(costs)
        avg_time = np.mean(times)
        std_cost = np.std(costs)
        
        # Calculate gap
        gaps = [(c - b) / b * 100 for c, b in zip(costs, baselines)]
        avg_gap = np.mean(gaps)
        
        results.append({
            "name": name,
            "avg_cost": avg_cost,
            "std_cost": std_cost,
            "avg_time": avg_time,
            "avg_gap": avg_gap,
        })
        
        print(f"{name:20s} | Cost: {avg_cost:.4f} ± {std_cost:.4f} | Gap: {avg_gap:.2f}% | Time: {avg_time:.2f}s")
    
    # Find best by quality, speed
    print(f"\n{'='*80}")
    best_quality = min(results, key=lambda x: x["avg_gap"])
    best_speed = min(results, key=lambda x: x["avg_time"])
    
    print(f"Best Quality:    {best_quality['name']} (Gap: {best_quality['avg_gap']:.2f}%)")
    print(f"Best Speed:      {best_speed['name']} (Time: {best_speed['avg_time']:.2f}s)")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LS Move Ablation Study (test.py compatible)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CVRP dataset file")
    parser.add_argument("--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("--H", type=int, default=10, help="Outer iterations")
    parser.add_argument("--mini_H", type=int, default=100, help="Inner iterations per outer")
    parser.add_argument("--k_sparse", type=int, default=32, help="Candidate list size")
    parser.add_argument("--rho", type=float, default=0.1, help="Pheromone decay")
    parser.add_argument("--min_new_edges", type=int, default=8, help="Min new edges")
    parser.add_argument("--n_instances", type=int, default=2, help="Number of instances (None=all)")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads")
    args = parser.parse_args()
    
    run_benchmark(
        dataset_path=args.dataset,
        n_ants=args.n_ants,
        H=args.H,
        mini_H=args.mini_H,
        k_sparse=args.k_sparse,
        rho=args.rho,
        min_new_edges=args.min_new_edges,
        n_instances=args.n_instances,
        threads=args.threads,
    )
