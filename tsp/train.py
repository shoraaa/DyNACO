#!/usr/bin/env python3
import time
import torch
import os
import argparse
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm

from test import infer_instance

# Adjust imports assuming script is run from tsp directory
from net import Net
from utils import load_val_dataset, build_pyg_data
from faco import MFACO_TSP
from baselines import get_baseline_tsp
import wandb

# Try to import compiled extension
try:
    import faco_tsp
except ImportError:
    faco_tsp = None
    print("Warning: could not import faco_tsp extension")

# --- Helper Functions from Notebook ---

def _popcount_i64(x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized popcount for int64 tensor x (CUDA-safe).
    Interprets bits in two's complement; for non-negative values this matches uint64 popcount.
    Returns int64 counts in [0, 64].
    """
    if x.dtype != torch.int64:
        raise TypeError(f"_popcount_i64 expects torch.int64, got {x.dtype}")

    # Masks as int64 constants (fit within signed range)
    m1  = x.new_tensor(0x5555555555555555, dtype=torch.int64)
    m2  = x.new_tensor(0x3333333333333333, dtype=torch.int64)
    m4  = x.new_tensor(0x0F0F0F0F0F0F0F0F, dtype=torch.int64)
    h01 = x.new_tensor(0x0101010101010101, dtype=torch.int64)

    # Assumes x is non-negative (your valid_mask should be)
    x = x - ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    return (x * h01) >> 56


def replay_logp_from_cpp_batch_trace(traces, prob_sparse: torch.Tensor):
    device = prob_sparse.device
    k = int(prob_sparse.size(1))
    if k >= 64:
        raise ValueError(f"k={k} >= 64 but trace.valid_mask is 64-bit; store n_valid explicitly or widen mask.")

    # Trace arrays -> torch (non-diff)
    curr = torch.as_tensor(np.asarray(traces.curr_nodes, dtype=np.int64), device=device)
    is_stoch = torch.as_tensor(np.asarray(traces.is_stochastic, dtype=np.uint8), device=device).bool()
    pick = torch.as_tensor(np.asarray(traces.pick_j, dtype=np.int64), device=device)

    # IMPORTANT: keep mask as int64 on GPU to allow indexing
    vm_i64 = torch.as_tensor(np.asarray(traces.valid_mask, dtype=np.uint64), device=device).to(torch.int64)

    # Ant mapping on GPU
    starts_t = torch.as_tensor(np.asarray(traces.starts, dtype=np.int64), device=device, dtype=torch.int64)
    n_ants = int(getattr(traces, "n_ants", int(starts_t.numel() - 1)))
    counts_t = (starts_t[1:1+n_ants] - starts_t[:n_ants]).to(torch.int64)

    ant_idx_all = torch.repeat_interleave(
        torch.arange(n_ants, device=device, dtype=torch.int64),
        counts_t,
    )

    # ndec per ant (stochastic only)
    if bool(is_stoch.any().item()):
        ndec = torch.bincount(ant_idx_all[is_stoch], minlength=n_ants).to(torch.int32)
    else:
        ndec = torch.zeros((n_ants,), device=device, dtype=torch.int32)

    logp = torch.zeros((n_ants,), device=device, dtype=torch.float32)

    # --- Roulette steps ---
    roulette = is_stoch & (pick >= 0)
    if bool(roulette.any().item()):
        idx = roulette.nonzero(as_tuple=False).squeeze(1)
        curr_r = curr[idx]
        pick_r = pick[idx]

        in_range = (pick_r >= 0) & (pick_r < k)
        if not bool(in_range.all().item()):
            bad = pick_r[~in_range][:10].detach().cpu().tolist()
            raise ValueError(f"pick_j out of range (k={k}). Examples: {bad}")

        vm_r = vm_i64[idx]  # int64, CUDA-indexable
        w = prob_sparse[curr_r]  # (D_r, k), differentiable

        # Build validity matrix from bitmask (all int64 ops, then cast)
        bitpos = torch.arange(k, device=device, dtype=torch.int64)
        valid = ((vm_r.unsqueeze(1) >> bitpos) & 1).to(w.dtype)

        denom = (w * valid).sum(dim=1).clamp_min(1e-12)
        numer = w.gather(1, pick_r.unsqueeze(1)).squeeze(1).clamp_min(1e-12)

        chosen_valid = (((vm_r >> pick_r) & 1) != 0)
        if not bool(chosen_valid.all().item()):
            raise ValueError("Trace inconsistency: pick_j not valid under valid_mask for some roulette decisions.")

        lp = torch.log(numer / denom)
        logp.scatter_add_(0, ant_idx_all[idx], lp)

    return logp, ndec



# --- Training Logic ---

EPS = 1e-10

def train_instance(model, optimizer, coords, k_sparse, n_ants, dynamic, args):
    model.train()
    
    # Use args for configuration
    aco = MFACO_TSP(
        n_ants=n_ants,
        coords=coords,
        cand_list_size=k_sparse,
        backup_list_size=k_sparse,
        disable_heuristic=args.disable_heuristic,
        use_local_search=not args.no_local_search,   # Note: args.no_local_search default False -> use_local_search=True
        decay=args.rho,
        device=args.device,
        enable_torch_sync=True,
        smooth_mmas=not args.no_smooth_mmas,
        min_new_edges=args.min_new_edges,
        extend_ls=args.extend_ls,
        normalized_heuristic=not args.no_normalized_heuristic,
    )

    optimizer.zero_grad(set_to_none=True)

    best_seen = float("inf")
    avg_cost_last = None
    
    H = args.H

    for t in tqdm(range(H), desc="ACO Step", leave=False):
        pyg_data = build_pyg_data(aco, coords, args.device, dynamic=dynamic)
        heu_vec = model(pyg_data).view(-1)
        heu_mat = heu_vec.view(aco.n, aco.k) + EPS
        losses = 0
        for mini_t in range(args.mini_H):
            # 1) sample + trace from C++
            costs, flats, _, logps_cpp, traces, costs_raw, flats_raw = aco.sample(require_prob=True, prior=heu_mat)
            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)

            # 2) differentiable prob table for training
            prob_sparse = aco.prob_sparse_torch(prior=heu_mat)
            prob_sparse = prob_sparse.clamp_min(EPS)  # protects log()

            logp_per_ant, ndec_per_ant = replay_logp_from_cpp_batch_trace(traces, prob_sparse)  # (n_ants,)

            if args.debug:
                 # Check discrepancy between C++ log_probs and Python replayed log_probs
                 logps_cpp_t = torch.as_tensor(logps_cpp, device=args.device, dtype=torch.float32)
                 diff = (logp_per_ant - logps_cpp_t).abs()
                 max_diff = diff.max().item()
                 mean_diff = diff.mean().item()
                 if max_diff > 1e-4:
                     print(f"[DEBUG] LogProbe Mismatch! Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
                     top_k_indices = diff.topk(5).indices
                     for idx in top_k_indices:
                         print(f"  Ant {idx}: Py={logp_per_ant[idx]:.6f}, C++={logps_cpp_t[idx]:.6f}, Diff={diff[idx]:.6f}")

            # 3) REINFORCE loss (baseline = mean in batch)
            baseline = costs_t.mean()
            adv = (costs_t - baseline).detach()

            loss = (adv * logp_per_ant / ndec_per_ant).mean()
            



            # 4) pheromone update (best ant this iter)
            best_idx = int(costs_t.argmin().item())
            best_cost_iter = float(costs[best_idx])
            best_seen = min(best_seen, best_cost_iter)

            with torch.no_grad():
                aco._update_pheromone_from_flat(flats[best_idx], best_cost_iter)

            losses += loss
            avg_cost_last = float(costs_t.mean().item())
        losses.backward()

    # optional but often helpful for stability:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    return avg_cost_last, best_seen


def train_epoch(args, epoch, net, optimizer, global_step, dynamic=True):
    sum_avg_cost = 0
    steps = args.steps_per_epoch
    for step in tqdm(range(steps), desc=f"Epoch {epoch}", leave=True):
        coords = np.random.rand(args.n_node, 2).astype(np.float32)
        avg_cost, best_cost = train_instance(net, optimizer, coords, args.k_sparse, args.n_ants, dynamic, args)
        sum_avg_cost += avg_cost
        
        if not args.no_wandb and wandb is not None:
             wandb.log({
                "train/avg_cost": float(avg_cost),
                "train/best_cost": float(best_cost),
                "train/epoch": int(epoch),
            }, step=global_step)
        global_step += 1
    
    epoch_avg_cost = sum_avg_cost / steps
    return global_step, epoch_avg_cost

def validation(args, epoch, net, val_dataset, dynamic=True, baseline_values=None):
    sum_sample_best, sum_aco_best = 0, 0
    
    n = len(val_dataset)
    idx = 0
    sum_gap = 0
    
    for coords in tqdm(val_dataset, desc="Validation", leave=False):
        avg_last, best_seen = infer_instance(net, coords, args.k_sparse, args.n_ants, dynamic, args)
        sum_sample_best += avg_last; sum_aco_best += best_seen
        
        if baseline_values is not None:
            opt = float(baseline_values[idx])
            gap = (best_seen - opt) / opt * 100
            sum_gap += gap
        idx += 1
    
    n_val = len(val_dataset)
    avg_last = sum_sample_best/n_val
    avg_aco_best = sum_aco_best/n_val
    avg_gap = sum_gap/n_val if baseline_values is not None else 0.0
    
    if baseline_values is not None:
        print(f"Validation Epoch {epoch}: Avg Last={avg_last:.4f}, ACO Best={avg_aco_best:.4f}, Gap={avg_gap:.2f}%")
    else:
        print(f"Validation Epoch {epoch}: Avg Last={avg_last:.4f}, ACO Best={avg_aco_best:.4f}")
        
    return avg_last, avg_aco_best, avg_gap

def main():
    parser = argparse.ArgumentParser(description="MFACO TSP Training")
    
    # Dataset / Problem
    parser.add_argument("--n_node", type=int, default=100, help="Number of nodes per TSP instance")
    parser.add_argument("--k_sparse", type=int, default=32, help="Candidate list size (k)")
    
    # Model / Training
    parser.add_argument("--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("--steps_per_epoch", type=int, default=32, help="Steps per epoch")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    # ACO Hyperparameters
    parser.add_argument("--rho", type=float, default=0.5, help="Pheromone decay (rho)")
    parser.add_argument("--min_new_edges", type=int, default=16, help="Min new edges")
    parser.add_argument("--H", type=int, default=10, help="ACO steps per instance (H)")
    parser.add_argument("--mini_H", type=int, default=100, help="ACO steps per iteration (mini_H)")
    parser.add_argument("--disable_heuristic", action="store_true", help="Disable heuristic")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (check log probabilities)")
    parser.add_argument("--no_local_search", action="store_true", help="Disable local search")
    parser.add_argument("--no_smooth_mmas", action="store_true", help="Disable smooth MMAS")
    parser.add_argument("--extend_ls", action="store_true", help="Extend LS checklist")
    parser.add_argument("--no_normalized_heuristic", action="store_true", help="Disable normalized heuristic")
    parser.add_argument("--no_logit_net", action="store_true", help="Disable logit network (no sigmoid) and log-space ACO")

    
    # Logging / Saving
    parser.add_argument("--no_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="claco", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--save_dir", type=str, default="pretrained/tsp", help="Directory to save models")
    parser.add_argument("--no_dynamic_feats", action="store_true", help="Disable dynamic features")
    parser.add_argument("--baseline", type=str, choices=['none', 'lkh'], default='lkh', help="Baseline for validation")
    parser.add_argument("--baseline_runs", type=int, default=1, help="LKH runs per instance")
    parser.add_argument("--baseline_time_limit", type=float, default=10.0, help="LKH time limit per instance (seconds)")
    
    args = parser.parse_args()
    

    # Setup
    PROJECT_ROOT = Path.cwd().resolve()
    # Resolve save dir relative to project root if not absolute
    save_path = Path(args.save_dir)
    # if not save_path.is_absolute():
    #     save_path = PROJECT_ROOT.parent / args.save_dir
    
    if not save_path.exists():
        os.makedirs(save_path, exist_ok=True)

    # JSON Log
    log_path = save_path / f"logs_{args.n_node}.json"

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # WandB
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"tsp{args.n_node}",
            config=vars(args),
            mode="online"
        )
    
    # Model
    net = Net(logit_net=not args.no_logit_net).to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.1)
    
    # Data
    print(f"Loading validation dataset for n={args.n_node}")
    try:
        val_list = load_val_dataset(args.n_node, args.device)
    except FileNotFoundError:
        print(f"Validation dataset for n={args.n_node} not found. Generating...")
        from utils import generate_val_dataset
        # Generate on utils DATA_DIR which seems to be ../data
        generate_val_dataset(args.n_node, 128 if args.n_node < 1000 else 16, args.k_sparse, "cpu") # Generate 128 instances
        val_list = load_val_dataset(args.n_node, args.device)

    baseline_values = None
    if args.baseline == 'lkh':
        print(f"Computing/Loading LKH baseline (runs={args.baseline_runs}, time_limit={args.baseline_time_limit})...")
        baseline_values = get_baseline_tsp(val_list, args.n_node, device="cpu", runs=args.baseline_runs, time_limit=args.baseline_time_limit)
        baseline_values = baseline_values.cpu()
        print(f"Near-optimal avg cost: {baseline_values.mean()}")
    
    # Validation before training
    dynamic = not args.no_dynamic_feats
    # avg_last, avg_aco_best, avg_gap = validation(args, -1, net, val_list, dynamic=dynamic, baseline_values=baseline_values)
    
    # if not args.no_wandb:
    #     log_dict = {
    #         "val/avg_last": float(avg_last),
    #         "val/avg_best": float(avg_aco_best),
    #         "val/epoch": -1,
    #     }
    #     if baseline_values is not None:
    #         log_dict["val/gap"] = float(avg_gap)
    #     wandb.log(log_dict, step=0)

    global_step = 0
    sum_time = 0
    best_val = float("inf")
    
    print("Starting training...")
    for epoch in range(args.epochs):
        start = time.time()
        global_step, train_avg_cost = train_epoch(args, epoch, net, optimizer, global_step, dynamic=dynamic)
        scheduler.step()
        sum_time += time.time() - start
        
        avg_last, avg_aco_best, avg_gap = validation(args, epoch, net, val_list, dynamic=dynamic, baseline_values=baseline_values)
        
        if not args.no_wandb:
            log_dict = {
                "val/avg_last": float(avg_last),
                "val/avg_best": float(avg_aco_best),
                "val/epoch": int(epoch),
            }
            if baseline_values is not None:
                log_dict["val/gap"] = float(avg_gap)
            wandb.log(log_dict, step=global_step)
            
        # JSON Logging
        log_entry = {
            "epoch": int(epoch),
            "train_avg_cost": float(train_avg_cost), 
            "val_avg_last": float(avg_last),
            "val_avg_best": float(avg_aco_best),
            "gap": float(avg_gap) if baseline_values is not None else None,
            "time": time.time() - start
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        if avg_aco_best < best_val:
            best_val = avg_aco_best
            model_path = save_path / f"tsp{args.n_node}_best.pt"
            checkpoint = {
                "model_state_dict": net.state_dict(),
                "config": vars(args),
            }
            torch.save(checkpoint, model_path)
            print(f"Saved best model to {model_path} (score: {best_val:.4f})")
    
    print(f"Total training duration: {sum_time:.2f}s")
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
