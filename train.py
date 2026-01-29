#!/usr/bin/env python3
import time
import torch
import os
import psutil
import argparse
import numpy as np
import random
import json
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional

# Shared helpers
import wandb

# Unified modules
import net
import faco
import utils
import baselines

# Specific class imports
from net import Net
from baselines import get_baseline

# Import metric helpers from utils
from utils import (
    row_softmax, mean_row_kl, rel_l2_drift, top_set, top_turnover,
    top1_flip_rate, safe_corr, top_overlap_frac, row_top1_match_rate, EPS
)

def _popcount_i64(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.int64:
        raise TypeError(f"_popcount_i64 expects torch.int64, got {x.dtype}")
    m1  = x.new_tensor(0x5555555555555555, dtype=torch.int64)
    m2  = x.new_tensor(0x3333333333333333, dtype=torch.int64)
    m4  = x.new_tensor(0x0F0F0F0F0F0F0F0F, dtype=torch.int64)
    h01 = x.new_tensor(0x0101010101010101, dtype=torch.int64)
    x = x - ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    return (x * h01) >> 56

def replay_logp_from_cpp_batch_trace(traces, prob_sparse: torch.Tensor):
    device = prob_sparse.device
    k = int(prob_sparse.size(1))
    
    # Trace arrays -> torch
    curr = torch.as_tensor(np.asarray(traces.curr_nodes, dtype=np.int64), device=device)
    is_stoch = torch.as_tensor(np.asarray(traces.is_stochastic, dtype=np.uint8), device=device).bool()
    pick = torch.as_tensor(np.asarray(traces.pick_j, dtype=np.int64), device=device)
    
    # Mask might be 64-bit
    vm_arr = np.asarray(traces.valid_mask, dtype=np.uint64)
    vm_i64 = torch.as_tensor(vm_arr, device=device).to(torch.int64)

    starts_t = torch.as_tensor(np.asarray(traces.starts, dtype=np.int64), device=device, dtype=torch.int64)
    n_ants = int(getattr(traces, "n_ants", int(starts_t.numel() - 1)))
    counts_t = (starts_t[1:1+n_ants] - starts_t[:n_ants]).to(torch.int64)

    ant_idx_all = torch.repeat_interleave(
        torch.arange(n_ants, device=device, dtype=torch.int64),
        counts_t,
    )

    ndec = torch.bincount(ant_idx_all[is_stoch], minlength=n_ants).to(torch.int32)
    logp = torch.zeros((n_ants,), device=device, dtype=torch.float32)

    roulette = is_stoch & (pick >= 0)
    idx = roulette.nonzero(as_tuple=False).squeeze(1)
    
    if idx.numel() > 0:
        curr_r = curr[idx]
        pick_r = pick[idx]
        vm_r = vm_i64[idx]
        w = prob_sparse[curr_r]

        bitpos = torch.arange(k, device=device, dtype=torch.int64)
        valid = ((vm_r.unsqueeze(1) >> bitpos) & 1).to(w.dtype)

        denom = (w * valid).sum(dim=1).clamp_min(1e-12)
        numer = w.gather(1, pick_r.unsqueeze(1)).squeeze(1).clamp_min(1e-12)

        lp = torch.log(numer / denom)
        logp.scatter_add_(0, ant_idx_all[idx], lp)

    return logp, ndec

def prob_sparse_from_tau_eta_prior(tau_nk, eta_nk, prior_nk, alpha=1.0, beta=1.0, eps=1e-12):
    tau = tau_nk.clamp_min(eps)
    eta = eta_nk.clamp_min(eps)
    w = torch.exp(alpha * torch.log(tau) + beta * torch.log(eta) + prior_nk)
    return w.clamp_min(eps)





def setup_aco(args, instance_data, problem_type):
    if problem_type == 'tsp':
        coords = instance_data
        kwargs = {
            'n_ants': args.n_ants,
            'coords': coords,
            'cand_list_size': args.k_sparse,
            'backup_list_size': args.k_sparse,
            'disable_heuristic': args.disable_heuristic,
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
        pyg_args = (coords, args.device)
        aco = faco.MFACO_TSP(**kwargs)
    else: # cvrp
        coords, demand, capacity = instance_data
        kwargs = {
            'coords': coords,
            'demand': demand,
            'capacity': float(capacity),
            'n_ants': args.n_ants,
            'cand_list_size': args.k_sparse,
            'backup_list_size': args.k_sparse,
            'min_new_edges': args.min_new_edges,
            'decay': args.rho,
            'p_best': 0.05,
            'use_local_search': not args.no_local_search,
            'disable_heuristic': args.disable_heuristic,
            'extend_ls': not args.no_extend_ls, 
            'smooth_mmas': not args.no_smooth_mmas,
            'device': args.device,
            'enable_torch_sync': True,
            'normalized_heuristic': not args.no_normalized_heuristic,
            'fixed_steps': args.L
        }
        pyg_args = (coords, demand, args.device)
        aco = faco.MFACO_CVRP(**kwargs)
    
    return aco, pyg_args


def train_instance_ppo(model, optimizer, instance_data, args):
    model.train()
    
    aco, pyg_args = setup_aco(args, instance_data, args.problem)
    
    # Heuristic view
    if args.problem == 'tsp':
        eta_nk = aco.h_sparse_torch
    else:
        # For CVRP, heuristic is numpy usually, convert to torch
        eta_nk = torch.tensor(aco.heuristic_sparse_np, device=args.device)

    # Build function
    build_fn = utils.build_pyg_data_tsp if args.problem == 'tsp' else utils.build_pyg_data_cvrp

    best_seen = float("inf")
    avg_cost_last = None
    
    metrics = {
        "ndec": [], "loss": [], 
        "entropy": [], "prior_mean": [], "prior_std": [],
        "approx_kl": [], "clip_frac": [], "new_edges": [], "survival": [],
        "prior_l2_drift": [], "prior_kl": [], "prior_turnover": [], "prior_flip": [],
        "prior_eta_corr": [], "grad_var": []
    }

    t_neural_total = 0.0
    t_aco_sampling = 0.0
    t_aco_ls = 0.0
    t_aco_update = 0.0
    t_aco_total = 0.0
    
    prior_prev_outer = None
    priors_history = [] 
    
    for outer in tqdm(range(args.H), desc="Outer", leave=False):
        t0 = time.time()
        pyg_data = build_fn(aco, *pyg_args, dynamic=not args.no_dynamic_feats)
        
        with torch.no_grad():
            prior_old = model(pyg_data).view(-1).view(aco.n, aco.k)
            t_neural_total += time.time() - t0
            
            metrics["prior_mean"].append(prior_old.mean().item())
            metrics["prior_std"].append(prior_old.std().item())
            
            # Track prior drift metrics
            if prior_prev_outer is not None:
                metrics["prior_l2_drift"].append(rel_l2_drift(prior_prev_outer, prior_old))
                metrics["prior_kl"].append(mean_row_kl(prior_prev_outer, prior_old))
                metrics["prior_turnover"].append(top_turnover(prior_prev_outer, prior_old))
                metrics["prior_flip"].append(top1_flip_rate(prior_prev_outer, prior_old))
            
            # Track prior-eta correlation
            metrics["prior_eta_corr"].append(safe_corr(prior_old, eta_nk))
            
            prior_prev_outer = prior_old.detach().clone()
            priors_history.append(prior_old.detach().clone())

        traces_list = []
        costs_list = []
        logp_old_list = []
        ndec_list = []
        tau_list = []

        if hasattr(aco, "reset_timings"):
            aco.reset_timings()

        t_aco_start_outer = time.time()

        for inner in range(args.mini_H):
            current_prior = prior_old
            if args.anneal_prior:
                if args.mini_H > 1:
                    ratio = inner / (args.mini_H - 1)
                    factor = args.gamma * (1.0 - ratio) + args.min_gamma * ratio
                else:
                    factor = args.gamma
                current_prior = prior_old * factor

            if args.problem == 'tsp':
                costs, flats, _, logps_cpp, traces, costs_raw, flats_raw, new_edges, survival = aco.sample(
                    require_prob=True, prior=current_prior
                )
            else: # cvrp
                costs, perms, _, logps, traces, new_edges, survival = aco.sample(
                    require_prob=True, prior=current_prior
                )
                flats = perms 
            
            # Record survival
            metrics["survival"].append(survival.mean().item()) 
            
            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            tau_nk = aco.tau_nk_torch().detach()
            tau_list.append(tau_nk)

            with torch.no_grad():
                prob_old = prob_sparse_from_tau_eta_prior(
                    tau_nk, eta_nk, current_prior,
                    alpha=args.alpha, beta=args.beta, eps=EPS
                )
                logp_old, ndec = replay_logp_from_cpp_batch_trace(traces, prob_old)
                ndec_f = ndec.to(torch.float32).clamp_min(1.0)
                logp_old = (logp_old / ndec_f).detach()

            traces_list.append(traces)
            costs_list.append(costs_t.detach())
            logp_old_list.append(logp_old)
            ndec_list.append(ndec.detach())
            
            metrics["new_edges"].append(new_edges.astype(np.float32).mean())

            metrics["ndec"].append(ndec.float().mean().item())
            entropy = (-logp_old / ndec.float().clamp_min(1.0)).mean().item()
            metrics["entropy"].append(entropy)

            best_idx = int(costs_t.argmin().item())
            best_cost_iter = float(costs[best_idx])
            best_seen = min(best_seen, best_cost_iter)
            
            with torch.no_grad():
                if args.problem == 'tsp':
                    aco._update_pheromone_from_flat(flats[best_idx], best_cost_iter)
                else:
                    aco.update_pheromone(flats[best_idx], best_cost_iter)

            avg_cost_last = float(costs_t.mean().item())

        t_aco_total += time.time() - t_aco_start_outer
        
        if hasattr(aco, "get_timings"):
             timings = aco.get_timings()
             if "time_sampling" in timings: t_aco_sampling += timings["time_sampling"] / 1000.0
             if "time_ls" in timings: t_aco_ls += timings["time_ls"] / 1000.0
             if "time_update" in timings: t_aco_update += timings["time_update"] / 1000.0

        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            
            t0 = time.time()
            prior_new = model(pyg_data).view(-1).view(aco.n, aco.k)
            t_neural_total += time.time() - t0
            
            all_losses = []
            param_kl_list = []
            clip_frac_list = []
            
            for inner in range(args.mini_H):
                # Annealing Calc (Same as above)
                current_prior = prior_new
                if args.anneal_prior:
                    if args.mini_H > 1:
                        ratio = inner / (args.mini_H - 1)
                        factor = args.gamma * (1.0 - ratio) + args.min_gamma * ratio
                    else:
                        factor = args.gamma
                    current_prior = prior_new * factor

                tau_nk = tau_list[inner]
                traces = traces_list[inner]
                costs_t = costs_list[inner]
                logp_old = logp_old_list[inner]
                
                prob_new = prob_sparse_from_tau_eta_prior(
                    tau_nk, eta_nk, current_prior,
                    alpha=args.alpha, beta=args.beta, eps=EPS
                )
                logp_new, ndec_new = replay_logp_from_cpp_batch_trace(traces, prob_new)
                ndec_f = ndec_new.to(torch.float32).clamp_min(1.0)
                logp_new = logp_new / ndec_f
                
                ratio = torch.exp(logp_new - logp_old)
                
                log_ratio = logp_new - logp_old
                approx_kl = (log_ratio.pow(2) * 0.5).mean()
                param_kl_list.append(approx_kl)
                
                clipped = (ratio > 1 + args.ppo_clip) | (ratio < 1 - args.ppo_clip)
                clip_frac = clipped.float().mean()
                clip_frac_list.append(clip_frac)
                
                baseline = costs_t.mean()
                adv = (baseline - costs_t).detach()
                if args.adv_norm:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * adv
                loss = -torch.mean(torch.min(surr1, surr2))
                all_losses.append(loss)
            
            total_loss = torch.stack(all_losses).mean()
            total_loss.backward()
            
            # Compute gradient variance
            grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())
            if grad_norms:
                metrics["grad_var"].append(np.var(grad_norms))
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metrics["loss"].append(total_loss.item())
            metrics["approx_kl"].append(torch.stack(param_kl_list).mean().item())
            metrics["clip_frac"].append(torch.stack(clip_frac_list).mean().item())

    metrics["time_neural"] = [t_neural_total]
    metrics["time_aco"] = [t_aco_total]
    if t_aco_sampling > 0: metrics["time_sampling"] = [t_aco_sampling]
    if t_aco_ls > 0: metrics["time_ls"] = [t_aco_ls]
    if t_aco_update > 0: metrics["time_update"] = [t_aco_update]

    out_metrics = {}
    for k, v in metrics.items():
        if v: out_metrics[k] = np.mean(v)
        else: out_metrics[k] = 0.0
    
    return avg_cost_last, best_seen, out_metrics

def train_epoch(net, optimizer, global_step, epoch, args):
    sum_avg_cost = 0
    steps = args.steps_per_epoch
    
    # Pre-determine gen function and capacity for loop
    gen_func = utils.generate_tsp_instance if args.problem == 'tsp' else utils.gen_cvrp_instance

    for step in tqdm(range(steps), desc="Epoch", leave=True):
        if args.problem == 'tsp':
            instance_data = np.random.rand(args.n_node, 2).astype(np.float32)
        else:
            coords_t, demand_t, capacity = gen_func(args.n_node, device=args.device)
            instance_data = (
                coords_t.detach().cpu().numpy().astype(np.float32),
                demand_t.detach().cpu().numpy().astype(np.float32),
                capacity
            )
        
        # Currently only PPO is verified and refactored fully in this file, REINFORCE can be similar
        # For brevity/focus we use PPO path (default)
        avg_cost, best_cost, metrics = train_instance_ppo(net, optimizer, instance_data, args)
            
        sum_avg_cost += avg_cost
        
        if "time_neural" in metrics: epoch_time_neural = metrics["time_neural"] 
        if "time_aco" in metrics: epoch_time_aco = metrics["time_aco"]

        if not args.no_wandb and wandb.run is not None:
             log_dict = {
                "train/avg_cost": float(avg_cost),
                "train/best_cost": float(best_cost),
                "train/epoch": int(epoch),
             }
             if metrics:
                 for k, v in metrics.items():
                     log_dict[f"train/{k}"] = float(v)
             wandb.log(log_dict, step=global_step)
        global_step += 1
    
    return global_step, sum_avg_cost / steps, epoch_time_neural, epoch_time_aco

def infer_instance(net, instance_data, k, n_ants, dynamic, args, collect_metrics=True):
    """
    Unified inference. 
    tsp -> instance_data = coords
    cvrp -> instance_data = (coords, demand, capacity)
    """
    if args.problem == 'tsp':
        aco, pyg_args = setup_aco(args, instance_data, 'tsp')
        build_fn = utils.build_pyg_data_tsp
    else:
        aco, pyg_args = setup_aco(args, instance_data, 'cvrp')
        build_fn = utils.build_pyg_data_cvrp

    # Inference loop (H steps)
    best_seen = float("inf")
    
    # Initialize metrics
    if collect_metrics:
        metrics_log = {
            'new_edges': [], 'prior_mean': [], 'prior_std': [],
            'prior_l2_drift': [], 'prior_kl': [], 'prior_turnover': [], 'prior_flip': [],
            'prior_eta_corr': [], 'survival': []
        }
    else:
        metrics_log = {}
    
    net.eval()
    
    # Get heuristic for correlation tracking
    if collect_metrics:
        if args.problem == 'tsp':
            eta_nk = aco.h_sparse_torch.to(device=args.device)
        else:
            eta_nk = torch.tensor(aco.heuristic_sparse_np, device=args.device)
    
    prior_prev_outer = None

    timer_sampling = 0
    timer_ls = 0
    timer_update = 0
    
    for outer in range(args.H):
        pyg_data = build_fn(aco, *pyg_args, dynamic=dynamic)
        
        with torch.no_grad():
            heuristics = net(pyg_data).view(-1).view(aco.n, aco.k)
        
        # Track prior metrics
        if collect_metrics:
            metrics_log['prior_mean'].append(heuristics.mean().item())
            metrics_log['prior_std'].append(heuristics.std().item())
            
            # Track prior drift metrics
            if prior_prev_outer is not None:
                metrics_log['prior_l2_drift'].append(rel_l2_drift(prior_prev_outer, heuristics))
                metrics_log['prior_kl'].append(mean_row_kl(prior_prev_outer, heuristics))
                metrics_log['prior_turnover'].append(top_turnover(prior_prev_outer, heuristics))
                metrics_log['prior_flip'].append(top1_flip_rate(prior_prev_outer, heuristics))
            
            # Track prior-eta correlation
            metrics_log['prior_eta_corr'].append(safe_corr(heuristics, eta_nk))
            
            prior_prev_outer = heuristics.detach().clone()
        
        # Inner loop with mini_H iterations (matching training and test.py)
        for inner in range(args.mini_H):
            # Annealing
            current_prior = heuristics
            if args.anneal_prior:
                if args.mini_H > 1:
                    ratio = inner / (args.mini_H - 1)
                    factor = args.gamma * (1.0 - ratio) + args.min_gamma * ratio
                else:
                    factor = args.gamma
                current_prior = heuristics * factor

            if args.problem == 'tsp':
                costs, flats, _, _, _, _, _, new_edges, survival = aco.sample(prior=current_prior.cpu().numpy(), require_prob=False)
            else:
                costs, perms, _, _, _, new_edges, survival = aco.sample(prior=current_prior.cpu().numpy(), require_prob=False)
                flats = perms
            
            if collect_metrics:
                metrics_log['new_edges'].append(new_edges.astype(np.float32).mean())
                metrics_log['survival'].append(survival.mean().item())

            best_idx = np.argmin(costs)
            best_val = costs[best_idx]
            best_seen = min(best_seen, best_val)
            
            if args.problem == 'tsp':
                aco._update_pheromone_from_flat(flats[best_idx], best_val)
            else:
                aco.update_pheromone(flats[best_idx], best_val)
    
    avg_cost = float(np.mean(costs))
    
    # Timings
    timings = {}
    if hasattr(aco, "get_timings"):
        t = aco.get_timings()
        timings = {k: v/1000.0 for k, v in t.items()} # ms to s

    if collect_metrics:
        return avg_cost, best_seen, timings, metrics_log
    
    return avg_cost, best_seen, timings



def validation(net, val_dataset, args, baseline_values=None):
    sum_sample_best, sum_aco_best = 0, 0
    sum_gap = 0
    n_val = len(val_dataset)
    
    if args.problem == 'tsp':
        iterable = val_dataset
    else:
        iterable = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    agg_metrics = {}
    idx = 0
    
    for item in tqdm(iterable, desc="Validating", leave=False):
        if args.problem == 'cvrp':
            item = [x[0] if torch.is_tensor(x) else x for x in item]
            if torch.is_tensor(item[0]): item[0] = item[0].numpy()
            if torch.is_tensor(item[1]): item[1] = item[1].numpy()
            if torch.is_tensor(item[2]): item[2] = float(item[2])
            
        dynamic = not args.no_dynamic_feats
        res = infer_instance(net, item, args.k_sparse, args.n_ants, dynamic, args, collect_metrics=True)
        
        avg, best, timings, metrics = res
        
        sum_sample_best += avg
        sum_aco_best += best
        
        if baseline_values is not None:
            opt = float(baseline_values[idx])
            gap = (best - opt) / opt * 100
            sum_gap += gap
        
        if metrics:
            for k, v in metrics.items():
                if k not in agg_metrics: agg_metrics[k] = []
                if len(v) > 0: agg_metrics[k].append(np.mean(v))
        
        idx += 1
    
    avg_last = sum_sample_best/n_val
    avg_aco_best = sum_aco_best/n_val
    avg_gap = sum_gap/n_val if baseline_values is not None else 0.0
    
    out_metrics = {}
    for k, v in agg_metrics.items():
        if len(v) > 0: out_metrics[k] = np.mean(v)
            
    return avg_last, avg_aco_best, avg_gap, out_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True, choices=['tsp', 'cvrp'])
    parser.add_argument("--n_node", type=int, default=1000)
    parser.add_argument("--k_sparse", type=int, default=32)
    parser.add_argument("--algo", choices=["reinforce", "ppo"], default="ppo")
    
    # PPO
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--adv_norm", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    
    # Training
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ppo_lr", type=float, default=5e-6)
    parser.add_argument("--reinforce_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # ACO
    parser.add_argument("--rho", type=float, default=0.1) 
    parser.add_argument("--min_new_edges", type=int, default=12)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--no_local_search", action="store_true")
    parser.add_argument("--no_smooth_mmas", action="store_true")
    parser.add_argument("--no_extend_ls", action="store_true")
    parser.add_argument("--no_normalized_heuristic", action="store_true")
    parser.add_argument("--no_logit_net", action="store_true")
    
    # Misc
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lga")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="pretrained")
    parser.add_argument("--no_dynamic_feats", action="store_true")
    parser.add_argument("--baseline", type=str, default='default') 
    parser.add_argument("--baseline_runs", type=int, default=1)
    parser.add_argument("--baseline_time_limit", type=float, default=300.0)
    parser.add_argument("--anneal_prior", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.2)
    parser.add_argument("--L", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--threads", type=int, default=None)
    
    args = parser.parse_args()

    # Defaults setup
    if args.lr is None:
        args.lr = args.ppo_lr if args.algo == 'ppo' else args.reinforce_lr
    
    if args.baseline == 'default':
        args.baseline = 'lkh' if args.problem == 'tsp' else 'hgs'
        
    args.extend_ls = not args.no_extend_ls
    
    if args.threads is None:
        args.threads = psutil.cpu_count(logical=False)
    faco.set_faco_cpp_threads(args.threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize Net
    # TSP: feats=2, CVRP: feats=4
    feats = 2 if args.problem == 'tsp' else 4
    net_model = Net(feats=feats, logit_net=not args.no_logit_net).to(args.device)
    
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr)

    # Generate descriptive filename based on key parameters
    model_name = f"{args.problem}_n{args.n_node}_k{args.k_sparse}_ants{args.n_ants}_H{args.H}_miniH{args.mini_H}_rho{args.rho}_mne{args.min_new_edges}_{args.algo}"
    if args.anneal_prior:
        model_name += f"_anneal_g{args.gamma}_mg{args.min_gamma}"
    if args.L > 0:
        model_name += f"_L{args.L}"

    run_id = wandb.util.generate_id()
    if not args.no_wandb:
        run_name = args.run_name if args.run_name else model_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=run_id,
            config=args
        )
    
    # Checkpoints
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load Validation Data & Baselines
    val_dataset = utils.load_val_dataset(args.n_node, problem=args.problem, device='cpu')
    baseline_values = None
    
    if val_dataset is None:
        print("Validation dataset not found. Generating 16 instances on fly...")
        val_dataset = []
        gen_fn = utils.generate_tsp_instance if args.problem == 'tsp' else utils.gen_cvrp_instance
        for _ in range(16):
            if args.problem == 'tsp':
                val_dataset.append(torch.from_numpy(gen_fn(args.n_node)))
            else:
                 c, d, cap = gen_fn(args.n_node, device='cpu')
                 val_dataset.append((c.cpu(), d.cpu(), cap))
        
        # Save it for future reuse
        utils.save_val_dataset(val_dataset, args.n_node, problem=args.problem)
    
    baseline_values = get_baseline(val_dataset, problem=args.problem, n_node=args.n_node, runs=args.baseline_runs, time_limit=args.baseline_time_limit)

    global_step = 0
    best_val_cost = float('inf')
    best_model_state = None
    
    # Create problem-specific save directory
    save_dir = Path(args.save_dir) / args.problem
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        # Train
        global_step, avg_train, t_neural, t_aco = train_epoch(net_model, optimizer, global_step, epoch, args)
        
        # Validate
        if val_dataset is not None:
             avg_last, avg_best, avg_gap, val_metrics = validation(net_model, val_dataset, args, baseline_values)
             print(f"Epoch {epoch}: TrainCost={avg_train:.4f} ValBest={avg_best:.4f} Gap={avg_gap:.2f}%")
             
             # Track best model and save immediately
             if avg_best < best_val_cost:
                 best_val_cost = avg_best
                 best_model_state = {
                     "model_state_dict": net_model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "epoch": epoch,
                     "val_cost": avg_best,
                     "val_gap": avg_gap,
                     "config": vars(args)
                 }
                 # Save best model immediately
                 if args.save_dir:
                     best_path = save_dir / f"{model_name}_best.pt"
                     torch.save(best_model_state, best_path)
                     print(f"Saved new best model to {best_path} (Epoch {epoch}, Val Cost: {avg_best:.4f}, Gap: {avg_gap:.2f}%)")
             
             if not args.no_wandb:
                 log_dict = {
                     "val/avg_last": avg_last,
                     "val/avg_best": avg_best,
                     "val/gap": avg_gap,
                     "val/epoch": epoch,
                     "time/neural_epoch": t_neural,
                     "time/aco_epoch": t_aco
                 }
                 # Add validation metrics
                 if val_metrics:
                     for k, v in val_metrics.items():
                         log_dict[f"val/{k}"] = float(v)
                 wandb.log(log_dict, step=global_step)
        
        # Save latest checkpoint periodically
        if args.save_dir and (epoch + 1) % 5 == 0:
            chkpt = {
                "model_state_dict": net_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": vars(args)
            }
            torch.save(chkpt, save_dir / f"{model_name}_latest.pt")
    
    # Save best model
    if best_model_state is not None and args.save_dir:
        best_path = save_dir / f"{model_name}_best.pt"
        torch.save(best_model_state, best_path)
        print(f"Saved best model to {best_path} (Val Cost: {best_val_cost:.4f})")
    
    # Save final model
    if args.save_dir:
        final_chkpt = {
            "model_state_dict": net_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": args.epochs - 1,
            "config": vars(args)
        }
        final_path = save_dir / f"{model_name}_final.pt"
        torch.save(final_chkpt, final_path)
        print(f"Saved final model to {final_path}")

if __name__ == "__main__":
    main()
