#!/usr/bin/env python3
"""
Training script for Neural-Guided Fast ACO (NGFACO).

This module implements PPO and REINFORCE training for learning neural priors
that guide ant colony optimization for TSP and CVRP problems.
"""

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
from typing import Optional, Dict, List, Any, Tuple, Union
import gc

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

# Import from utils
from utils import (
    row_softmax, mean_row_kl, rel_l2_drift, top_set, top_turnover,
    top1_flip_rate, safe_corr, top_overlap_frac, row_top1_match_rate, EPS,
    Logger, MetricsCollector, get_logger, init_logger
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _popcount_i64(x: torch.Tensor) -> torch.Tensor:
    """Compute population count (number of set bits) for int64 tensors."""
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


def replay_logp_from_cpp_batch_trace(
    traces,
    log_prob_sparse: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replay traces to compute log probabilities using log-space arithmetic.
    
    Args:
        traces: Trace object from C++ ACO containing decision history
        log_prob_sparse: (n, k) tensor of UNNORMALIZED log-probabilities
    
    Returns:
        Tuple of (logp_sum, ndec) where:
            - logp_sum: Sum of log probabilities per ant
            - ndec: Number of decisions per ant
    """
    device = log_prob_sparse.device
    k = int(log_prob_sparse.size(1))
    
    # Trace arrays -> torch
    curr = torch.as_tensor(
        np.asarray(traces.curr_nodes, dtype=np.int64), device=device
    )
    is_stoch = torch.as_tensor(
        np.asarray(traces.is_stochastic, dtype=np.uint8), device=device
    ).bool()
    pick = torch.as_tensor(
        np.asarray(traces.pick_j, dtype=np.int64), device=device
    )
    
    # Mask might be 64-bit
    vm_arr = np.asarray(traces.valid_mask, dtype=np.uint64)
    vm_i64 = torch.as_tensor(vm_arr, device=device).to(torch.int64)

    starts_t = torch.as_tensor(
        np.asarray(traces.starts, dtype=np.int64),
        device=device,
        dtype=torch.int64
    )
    n_ants = int(getattr(traces, "n_ants", int(starts_t.numel() - 1)))
    counts_t = (starts_t[1:1+n_ants] - starts_t[:n_ants]).to(torch.int64)
    
    # Map trace steps to ant index
    ant_idx_all = torch.repeat_interleave(
        torch.arange(n_ants, device=device, dtype=torch.int64),
        counts_t,
    )

    ndec = torch.bincount(ant_idx_all[is_stoch], minlength=n_ants).to(torch.int32)
    logp_sum = torch.zeros((n_ants,), device=device, dtype=torch.float32)

    roulette = is_stoch & (pick >= 0)
    idx = roulette.nonzero(as_tuple=False).squeeze(1)
    
    if idx.numel() > 0:
        curr_r = curr[idx]   # current node
        pick_r = pick[idx]   # chosen neighbor index in sparse list
        vm_r = vm_i64[idx]   # valid mask
        
        # log weights for the current node's candidates: (batch, k)
        log_w = log_prob_sparse[curr_r]
        
        # Determine valid mask as float: 0.0 for valid, -inf for invalid
        bitpos = torch.arange(k, device=device, dtype=torch.int64)
        # valid: (batch, k), 1 if valid, 0 if invalid
        valid_bits = ((vm_r.unsqueeze(1) >> bitpos) & 1)
        
        # log_mask: 0.0 if valid, -inf otherwise
        log_mask_val = torch.zeros_like(log_w)
        log_mask_val.masked_fill_(valid_bits == 0, float('-inf'))
        
        log_w_valid = log_w + log_mask_val
        
        # Log-Sum-Exp for denominator
        log_denom = torch.logsumexp(log_w_valid, dim=1)
        
        # Numerator is just the log_w of the picked choice
        log_numer = log_w.gather(1, pick_r.unsqueeze(1)).squeeze(1)
        
        # log p = log_numer - log_denom
        step_logp = log_numer - log_denom
        
        logp_sum.scatter_add_(0, ant_idx_all[idx], step_logp)

    return logp_sum, ndec

# =============================================================================
# PROBABILITY COMPUTATION
# =============================================================================

def log_prob_sparse_from_tau_eta_prior(
    tau_nk: torch.Tensor,
    eta_nk: torch.Tensor,
    prior_nk: Optional[torch.Tensor],
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute log probabilities from pheromone, heuristic, and neural prior.
    
    Args:
        tau_nk: Pheromone values (n, k)
        eta_nk: Heuristic values (n, k)
        prior_nk: Neural prior logits (n, k), optional
        alpha: Pheromone exponent
        beta: Heuristic exponent
        eps: Small constant for numerical stability
    
    Returns:
        Log weights (unnormalized log probabilities)
    """
    tau = tau_nk.clamp_min(eps)
    eta = eta_nk.clamp_min(eps)
    log_w = alpha * torch.log(tau) + beta * torch.log(eta)
    if prior_nk is not None:
        log_w = log_w + prior_nk
    return log_w





# =============================================================================
# ACO SETUP
# =============================================================================

def setup_aco(
    args: argparse.Namespace,
    instance_data: Any,
    problem_type: str
) -> Tuple[Any, Tuple]:
    """
    Setup ACO solver for the given problem instance.
    
    Args:
        args: Training arguments
        instance_data: Problem instance data (coords for TSP, tuple for CVRP)
        problem_type: 'tsp' or 'cvrp'
    
    Returns:
        Tuple of (aco_solver, pyg_args)
    """
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
            'fixed_steps': args.L,
            'nls': args.nls,
            'T_nls': args.T_nls
        }
        pyg_args = (coords, args.device)
        
        if args.alg == 'mmas':
            aco = faco.ACO_TSP(
                coords=coords,
                n_ants=args.n_ants,
                cand_list_size=args.k_sparse,
                decay=args.rho,
                alpha=args.alpha,
                beta=args.beta,
                p_best=0.05,
                min_max=True,
                device=args.device,
                enable_torch_sync=True
            )
        else:
            aco = faco.MFACO_TSP(**kwargs)
    else:  # cvrp
        coords, demand, capacity = instance_data
        kwargs = {
            'coords': coords,
            'demand': demand,
            'capacity': float(capacity),
            'n_ants': args.n_ants,
            'cand_list_size': args.k_sparse,
            'backup_list_size': max(args.k_sparse, 64),
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
            'fixed_steps': args.L,
            'nls': args.nls,
            'T_nls': args.T_nls
        }
        pyg_args = (coords, demand, args.device)
        
        if args.alg == 'mmas':
            aco = faco.ACO_CVRP(
                coords=coords,
                demand=demand,
                capacity=float(capacity),
                n_ants=args.n_ants,
                cand_list_size=args.k_sparse,
                decay=args.rho,
                alpha=args.alpha,
                beta=args.beta,
                p_best=0.05,
                min_max=True,
                device=args.device,
                enable_torch_sync=True
            )
        else:
            aco = faco.MFACO_CVRP(**kwargs)
    
    return aco, pyg_args


def get_heuristic_tensor(
    aco: Any,
    problem_type: str,
    device: str
) -> torch.Tensor:
    """Get heuristic tensor from ACO solver (unified API)."""
    return aco.h_sparse_torch


def compute_annealing_factor(
    inner: int,
    mini_H: int,
    gamma: float,
    min_gamma: float
) -> float:
    """Compute annealing factor for prior scaling."""
    if mini_H > 1:
        ratio = inner / (mini_H - 1)
        return gamma * (1.0 - ratio) + min_gamma * ratio
    return gamma


def compute_warmup_steps(args: argparse.Namespace, use_train: bool = True) -> int:
    """Compute number of warmup steps based on configuration."""
    warmup_attr = 'train_warmup' if use_train else 'warmup'
    if not getattr(args, warmup_attr, False):
        return 0
    
    # Warmup should use at most half of H, but always leave at least 1 step for neural guidance
    max_limit = int(args.H * getattr(args, 'warmup_ratio', 0.5))
    
    # Ensure we don't warmup for all H steps - need at least 1 step with neural guidance
    max_limit = min(max_limit, args.H - 1)
    
    if max_limit <= 0:
        return 0
    
    return np.random.randint(1, max_limit + 1) if use_train else max_limit



# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def _collect_prior_metrics(
    metrics: MetricsCollector,
    prior: torch.Tensor,
    prior_prev_cpu: Optional[torch.Tensor],
    eta_nk: torch.Tensor,
    simple_train: bool
) -> Optional[torch.Tensor]:
    """Collect metrics for neural prior and return CPU copy for next iteration."""
    metrics.add("prior_mean", prior.mean().item())
    metrics.add("prior_std", prior.std().item())
    
    if simple_train:
        return None
    
    # Track prior-eta correlation
    metrics.add("prior_eta_corr", safe_corr(prior, eta_nk))
    
    # Move to CPU for drift metrics
    prior_cpu = prior.detach().cpu()
    
    if prior_prev_cpu is not None:
        metrics.add("prior_l2_drift", rel_l2_drift(prior_prev_cpu, prior_cpu))
        metrics.add("prior_kl", mean_row_kl(prior_prev_cpu, prior_cpu))
        metrics.add("prior_turnover", top_turnover(prior_prev_cpu, prior_cpu))
        metrics.add("prior_flip", top1_flip_rate(prior_prev_cpu, prior_cpu))
    
    return prior_cpu


def _update_aco_timings(
    metrics: MetricsCollector,
    aco: Any,
    t_sampling: float,
    t_ls: float,
    t_update: float
) -> Tuple[float, float, float]:
    """Extract and accumulate ACO timing information."""
    if hasattr(aco, "get_timings"):
        timings = aco.get_timings()
        if "time_sampling" in timings:
            t_sampling += timings["time_sampling"] / 1000.0
        if "time_ls" in timings:
            t_ls += timings["time_ls"] / 1000.0
        if "time_update" in timings:
            t_update += timings["time_update"] / 1000.0
    return t_sampling, t_ls, t_update


def train_instance_reinforce(
    model: Net,
    optimizer: torch.optim.Optimizer,
    instance_data: Any,
    args: argparse.Namespace
) -> Tuple[float, float, Dict[str, float]]:
    """
    Train on a single instance using REINFORCE algorithm.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        instance_data: Problem instance
        args: Training arguments
    
    Returns:
        Tuple of (avg_cost, best_cost, metrics_dict)
    """
    model.train()
    
    aco, pyg_args = setup_aco(args, instance_data, args.problem)
    eta_nk = get_heuristic_tensor(aco, args.problem, args.device)
    build_fn = (utils.build_pyg_data_tsp if args.problem == 'tsp' 
                else utils.build_pyg_data_cvrp)

    best_seen = float("inf")
    avg_cost_last = None
    
    metrics = MetricsCollector()
    prior_prev_cpu = None
    
    # Timing accumulators
    t_neural_total = 0.0
    t_aco_sampling = 0.0
    t_aco_ls = 0.0
    t_aco_update = 0.0
    t_aco_total = 0.0
    
    warmup_steps = compute_warmup_steps(args, use_train=True)

    for outer in tqdm(range(args.H), desc="Outer", leave=False):
        t0 = time.time()
        pyg_data = build_fn(aco, *pyg_args, dynamic=not args.no_dynamic_feats)
        
        prior_old = None
        if outer >= warmup_steps:
            with torch.no_grad():
                prior_old = model(pyg_data).view(-1).view(aco.n, aco.k)
                t_neural_total += time.time() - t0
                
                prior_prev_cpu = _collect_prior_metrics(
                    metrics, prior_old, prior_prev_cpu, eta_nk, args.simple_train
                )

        tau_list = []
        traces_list = []
        costs_list = []

        if hasattr(aco, "reset_timings"):
            aco.reset_timings()

        t_aco_start_outer = time.time()

        for inner in range(args.mini_H):
            current_prior = prior_old
            if prior_old is not None and args.train_anneal:
                factor = compute_annealing_factor(
                    inner, args.mini_H, args.gamma, args.min_gamma
                )
                current_prior = prior_old * factor

            res = aco.sample(require_prob=True, prior=current_prior, parallel_traced=args.parallel_traced)
            costs, flats, _, _, traces, costs_raw, flats_raw, new_edges, survival = res
            
            if survival is not None:
                metrics.add("survival", survival.mean().item())

            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            
            tau_nk = aco.tau_nk_torch().detach()
            tau_list.append(tau_nk)
            traces_list.append(traces)
            costs_list.append(costs_t.detach())
            
            if new_edges is not None:
                metrics.add("new_edges", new_edges.astype(np.float32).mean())

            best_idx = int(costs_t.argmin().item())
            best_cost_iter = float(costs[best_idx])
            best_seen = min(best_seen, best_cost_iter)
            
            if not args.train_deepaco:
                with torch.no_grad():
                    aco.update_pheromone(flats[best_idx], best_cost_iter)

            avg_cost_last = float(costs_t.mean().item())

        t_aco_total += time.time() - t_aco_start_outer
        t_aco_sampling, t_aco_ls, t_aco_update = _update_aco_timings(
            metrics, aco, t_aco_sampling, t_aco_ls, t_aco_update
        )

        if outer < warmup_steps:
            continue

        optimizer.zero_grad(set_to_none=True)
        
        t0 = time.time()
        prior_new = model(pyg_data).view(-1).view(aco.n, aco.k)
        t_neural_total += time.time() - t0
        
        all_losses = []
        all_entropies = []
        
        for inner in range(args.mini_H):
            current_prior = prior_new
            if args.train_anneal:
                factor = compute_annealing_factor(
                    inner, args.mini_H, args.gamma, args.min_gamma
                )
                current_prior = prior_new * factor

            tau_nk = tau_list[inner]
            traces = traces_list[inner]
            costs_t = costs_list[inner]
            
            log_prob_new = log_prob_sparse_from_tau_eta_prior(
                tau_nk, eta_nk, current_prior,
                alpha=args.alpha, beta=args.beta, eps=EPS
            )
            logp_new, ndec_new = replay_logp_from_cpp_batch_trace(traces, log_prob_new)
            ndec_f = ndec_new.to(torch.float32).clamp_min(1.0)
            logp_new = logp_new / ndec_f
            
            baseline = costs_t.mean()
            adv = (costs_t - baseline).detach()
            
            loss = (logp_new * adv).mean()
            all_losses.append(loss)
            
            entropy = -logp_new.mean()
            all_entropies.append(entropy.detach().item())

        total_loss = torch.stack(all_losses).mean()
        total_loss.backward()
        
        if not args.simple_train:
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            if grad_norms:
                metrics.add("grad_var", np.var(grad_norms))
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        metrics.add("loss", total_loss.item())
        metrics.add("entropy", np.mean(all_entropies))
        metrics.add("ndec", ndec_f.mean().item())

    # Finalize timing metrics
    out_metrics = metrics.get_all_means()
    out_metrics["time_neural"] = t_neural_total
    out_metrics["time_aco"] = t_aco_total
    if t_aco_sampling > 0:
        out_metrics["time_sampling"] = t_aco_sampling
    if t_aco_ls > 0:
        out_metrics["time_ls"] = t_aco_ls
    if t_aco_update > 0:
        out_metrics["time_update"] = t_aco_update
    
    return avg_cost_last, best_seen, out_metrics

def train_instance_ppo(
    model: Net,
    optimizer: torch.optim.Optimizer,
    instance_data: Any,
    args: argparse.Namespace
) -> Tuple[float, float, Dict[str, float]]:
    """
    Train on a single instance using PPO algorithm.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        instance_data: Problem instance
        args: Training arguments
    
    Returns:
        Tuple of (avg_cost, best_cost, metrics_dict)
    """
    model.train()
    
    aco, pyg_args = setup_aco(args, instance_data, args.problem)
    eta_nk = get_heuristic_tensor(aco, args.problem, args.device)
    build_fn = (utils.build_pyg_data_tsp if args.problem == 'tsp' 
                else utils.build_pyg_data_cvrp)

    best_seen = float("inf")
    avg_cost_last = None
    
    metrics = MetricsCollector()
    prior_prev_cpu = None
    
    # Timing accumulators
    t_neural_total = 0.0
    t_aco_sampling = 0.0
    t_aco_ls = 0.0
    t_aco_update = 0.0
    t_aco_total = 0.0
    
    warmup_steps = compute_warmup_steps(args, use_train=True)

    for outer in tqdm(range(args.H), desc="Outer", leave=False):
        t0 = time.time()
        pyg_data = build_fn(aco, *pyg_args, dynamic=not args.no_dynamic_feats)
        
        prior_old = None
        if outer >= warmup_steps:
            with torch.no_grad():
                prior_old = model(pyg_data).view(-1).view(aco.n, aco.k)
                t_neural_total += time.time() - t0
                
                prior_prev_cpu = _collect_prior_metrics(
                    metrics, prior_old, prior_prev_cpu, eta_nk, args.simple_train
                )

        # Storage for PPO update
        traces_list = []
        flats_list = []
        costs_list = []
        logp_old_list = []
        ndec_list = []
        tau_list = []
        costs_raw_t = None
        
        if hasattr(aco, "reset_timings"):
            aco.reset_timings()

        t_aco_start_outer = time.time()

        for inner in range(args.mini_H):
            current_prior = prior_old
            if prior_old is not None and args.train_anneal:
                factor = compute_annealing_factor(
                    inner, args.mini_H, args.gamma, args.min_gamma
                )
                current_prior = prior_old * factor

            # Sample from ACO
            res = aco.sample(require_prob=True, prior=current_prior, parallel_traced=args.parallel_traced)
            costs, flats, _, _, traces, costs_raw, flats_raw, new_edges, survival = res
            
            if survival is not None:
                metrics.add("survival", survival.mean().item())

            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            if costs_raw is not None:
                costs_raw_t = torch.as_tensor(costs_raw, device=args.device, dtype=torch.float32)
            else:
                costs_raw_t = costs_t
            
            tau_nk = aco.tau_nk_torch().detach()

            # Compute old log probabilities
            with torch.no_grad():
                log_prob_old = log_prob_sparse_from_tau_eta_prior(
                    tau_nk, eta_nk, current_prior,
                    alpha=args.alpha, beta=args.beta, eps=EPS
                )
                logp_old, ndec = replay_logp_from_cpp_batch_trace(traces, log_prob_old)
                ndec_f = ndec.to(torch.float32).clamp_min(1.0)
                logp_old = (logp_old / ndec_f).detach()

            # Store for PPO update
            tau_list.append(tau_nk.detach())
            traces_list.append(traces)
            flats_list.append(None)
            costs_list.append(costs_t.detach())
            logp_old_list.append(logp_old)
            ndec_list.append(ndec.detach())
            
            if new_edges is not None:
                metrics.add("new_edges", new_edges.astype(np.float32).mean())

            metrics.add("ndec", ndec.float().mean().item())
            entropy = (-logp_old / ndec.float().clamp_min(1.0)).mean().item()
            metrics.add("entropy", entropy)

            best_idx = int(costs_t.argmin().item())
            best_cost_iter = float(costs[best_idx])
            best_seen = min(best_seen, best_cost_iter)
            
            if not args.train_deepaco:
                with torch.no_grad():
                    aco.update_pheromone(flats[best_idx], best_cost_iter)

            avg_cost_last = float(costs_t.mean().item())

        t_aco_total += time.time() - t_aco_start_outer
        t_aco_sampling, t_aco_ls, t_aco_update = _update_aco_timings(
            metrics, aco, t_aco_sampling, t_aco_ls, t_aco_update
        )

        # Skip PPO update during warmup
        if outer < warmup_steps:
            continue

        # PPO Update loop
        for _ in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            
            t0 = time.time()
            prior_new = model(pyg_data).view(-1).view(aco.n, aco.k)
            t_neural_total += time.time() - t0
            
            all_param_kl = []
            all_clip_frac = []
            all_losses = []
            total_loss_val_epoch = 0.0
            
            for inner in range(args.mini_H):
                current_prior = prior_new
                if args.train_anneal:
                    factor = compute_annealing_factor(
                        inner, args.mini_H, args.gamma, args.min_gamma
                    )
                    current_prior = prior_new * factor

                tau_nk = tau_list[inner]
                traces = traces_list[inner]
                costs_t = costs_list[inner]
                logp_old = logp_old_list[inner]
                
                log_prob_new = log_prob_sparse_from_tau_eta_prior(
                    tau_nk, eta_nk, current_prior,
                    alpha=args.alpha, beta=args.beta, eps=EPS
                )
                logp_new, ndec_new = replay_logp_from_cpp_batch_trace(traces, log_prob_new)
                ndec_f = ndec_new.to(torch.float32).clamp_min(1.0)
                logp_new = logp_new / ndec_f
                
                ratio = torch.exp(logp_new - logp_old)
                
                log_ratio = logp_new - logp_old
                approx_kl = (log_ratio.pow(2) * 0.5).mean()
                all_param_kl.append(approx_kl.detach().item())
                
                clipped = (ratio > 1 + args.ppo_clip) | (ratio < 1 - args.ppo_clip)
                clip_frac = clipped.float().mean()
                all_clip_frac.append(clip_frac.detach().item())
                
                # Advantage calculation
                if args.nls and costs_raw_t is not None:
                    cost_combined = args.nls_beta * costs_t + (1.0 - args.nls_beta) * costs_raw_t
                    baseline = cost_combined.mean()
                    adv = (baseline - cost_combined).detach()
                else:
                    baseline = costs_t.mean()
                    adv = (baseline - costs_t).detach()
                
                if not args.no_adv_norm:
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * adv
                loss = -torch.mean(torch.min(surr1, surr2))
                
                all_losses.append(loss)
                total_loss_val_epoch += loss.item()
            
            # Single backward pass
            total_loss = torch.stack(all_losses).mean()
            total_loss.backward()
            
            # Compute gradient variance
            if not args.simple_train:
                grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                if grad_norms:
                    metrics.add("grad_var", np.var(grad_norms))
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            metrics.add("loss", total_loss_val_epoch / args.mini_H)
            metrics.add("approx_kl", np.mean(all_param_kl))
            metrics.add("clip_frac", np.mean(all_clip_frac))

    # Finalize metrics
    out_metrics = metrics.get_all_means()
    out_metrics["time_neural"] = t_neural_total
    out_metrics["time_aco"] = t_aco_total
    if t_aco_sampling > 0:
        out_metrics["time_sampling"] = t_aco_sampling
    if t_aco_ls > 0:
        out_metrics["time_ls"] = t_aco_ls
    if t_aco_update > 0:
        out_metrics["time_update"] = t_aco_update

    # Cleanup
    del traces_list, flats_list, tau_list, logp_old_list, ndec_list, costs_list
    gc.collect()
    torch.cuda.empty_cache()

    return avg_cost_last, best_seen, out_metrics

# =============================================================================
# EPOCH TRAINING & INFERENCE
# =============================================================================

def train_epoch(
    net: Net,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    args: argparse.Namespace
) -> Tuple[int, float, float, float, float]:
    """
    Train for one epoch across multiple instances.
    
    Args:
        net: Neural network model
        optimizer: Optimizer
        global_step: Current global step
        epoch: Current epoch number
        args: Training arguments
    
    Returns:
        Tuple of (global_step, avg_cost, t_neural, t_aco, epoch_train_time)
    """
    logger = get_logger()
    
    sum_avg_cost = 0
    steps = args.steps_per_epoch
    
    gen_func = (utils.generate_tsp_instance if args.problem == 'tsp' 
                else utils.gen_cvrp_instance)

    epoch_train_time = 0.0
    epoch_time_neural = 0.0
    epoch_time_aco = 0.0
    
    for step in tqdm(range(steps), desc="Epoch", leave=True):
        # Generate instance
        if args.problem == 'tsp':
            instance_data = np.random.rand(args.n_node, 2).astype(np.float32)
        else:
            coords_t, demand_t, capacity = gen_func(args.n_node, device=args.device)
            instance_data = (
                coords_t.detach().cpu().numpy().astype(np.float32),
                demand_t.detach().cpu().numpy().astype(np.float32),
                capacity
            )
        
        # Train on instance
        t_start_instance = time.time()
        if args.algo == 'ppo':
            avg_cost, best_cost, metrics = train_instance_ppo(
                net, optimizer, instance_data, args
            )
        else:
            avg_cost, best_cost, metrics = train_instance_reinforce(
                net, optimizer, instance_data, args
            )
        epoch_train_time += time.time() - t_start_instance
            
        sum_avg_cost += avg_cost
        
        if "time_neural" in metrics:
            epoch_time_neural = metrics["time_neural"]
        if "time_aco" in metrics:
            epoch_time_aco = metrics["time_aco"]

        # Log step metrics
        logger.set_step(global_step)
        logger.log_train_step(avg_cost, best_cost, epoch, metrics, global_step)
        global_step += 1
    
    return global_step, sum_avg_cost / steps, epoch_time_neural, epoch_time_aco, epoch_train_time


def infer_instance(
    net: Net,
    instance_data: Any,
    k: int,
    n_ants: int,
    dynamic: bool,
    args: argparse.Namespace,
    collect_metrics: bool = True
) -> Tuple[float, float, Dict[str, float], Dict[str, List[float]]]:
    """
    Run inference on a single instance.
    
    Args:
        net: Neural network model
        instance_data: Problem instance (coords for TSP, tuple for CVRP)
        k: Sparse neighbor count
        n_ants: Number of ants
        dynamic: Whether to use dynamic features
        args: Arguments
        collect_metrics: Whether to collect detailed metrics
    
    Returns:
        Tuple of (avg_cost, best_cost, timings, metrics_log)
    """
    if args.problem == 'tsp':
        aco, pyg_args = setup_aco(args, instance_data, 'tsp')
        build_fn = utils.build_pyg_data_tsp
    else:
        aco, pyg_args = setup_aco(args, instance_data, 'cvrp')
        build_fn = utils.build_pyg_data_cvrp

    best_seen = float("inf")
    net.eval()
    
    # Initialize metrics
    metrics_log: Dict[str, List[float]] = {}
    if collect_metrics:
        metrics_log = {
            'new_edges': [], 'prior_mean': [], 'prior_std': [],
            'prior_l2_drift': [], 'prior_kl': [], 'prior_turnover': [], 'prior_flip': [],
            'prior_eta_corr': [], 'survival': []
        }
        eta_nk = get_heuristic_tensor(aco, args.problem, args.device)
    
    prior_prev_outer = None
    warmup_steps = compute_warmup_steps(args, use_train=False)

    for outer in range(args.H):
        pyg_data = build_fn(aco, *pyg_args, dynamic=dynamic)
        
        guidance = None
        if outer >= warmup_steps:
            with torch.no_grad():
                guidance = net(pyg_data).view(-1).view(aco.n, aco.k)
        
        # Track prior metrics
        if collect_metrics and guidance is not None:
            metrics_log['prior_mean'].append(guidance.mean().item())
            metrics_log['prior_std'].append(guidance.std().item())
            
            if prior_prev_outer is not None:
                metrics_log['prior_l2_drift'].append(
                    rel_l2_drift(prior_prev_outer, guidance)
                )
                metrics_log['prior_kl'].append(mean_row_kl(prior_prev_outer, guidance))
                metrics_log['prior_turnover'].append(top_turnover(prior_prev_outer, guidance))
                metrics_log['prior_flip'].append(top1_flip_rate(prior_prev_outer, guidance))
            
            metrics_log['prior_eta_corr'].append(safe_corr(guidance, eta_nk))
            prior_prev_outer = guidance.detach().clone()
        
        # Inner loop
        for inner in range(args.mini_H):
            current_prior = guidance
            if guidance is not None and not args.no_anneal:
                factor = compute_annealing_factor(
                    inner, args.mini_H, args.gamma, args.min_gamma
                )
                current_prior = guidance * factor
            
            costs, flats, _, _, _, _, _, new_edges, survival = aco.sample(
                prior=current_prior, require_prob=False
            )
            
            if collect_metrics:
                if new_edges is not None:
                    metrics_log['new_edges'].append(new_edges.astype(np.float32).mean())
                if survival is not None:
                    metrics_log['survival'].append(survival.mean().item())

            best_idx = np.argmin(costs)
            best_val = costs[best_idx]
            best_seen = min(best_seen, best_val)
            
            aco.update_pheromone(flats[best_idx], best_val)
    
    avg_cost = float(np.mean(costs))
    
    # Get timings
    timings = {}
    if hasattr(aco, "get_timings"):
        t = aco.get_timings()
        timings = {k: v / 1000.0 for k, v in t.items()}

    return avg_cost, best_seen, timings, metrics_log

# =============================================================================
# VALIDATION
# =============================================================================

def validation(
    net: Net,
    val_dataset: List[Any],
    args: argparse.Namespace,
    baseline_values: Optional[np.ndarray] = None
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Run validation on a dataset.
    
    Args:
        net: Neural network model
        val_dataset: Validation dataset
        args: Arguments
        baseline_values: Optional baseline costs for gap calculation
    
    Returns:
        Tuple of (avg_last, avg_best, avg_gap, aggregated_metrics)
    """
    logger = get_logger()
    logger.info(f"Validating on {len(val_dataset)} instances...")
    
    # Create validation args with overrides
    val_args = argparse.Namespace(**vars(args))
    if val_args.val_H is not None:
        val_args.H = val_args.val_H
    if val_args.val_mini_H is not None:
        val_args.mini_H = val_args.val_mini_H
    
    net.eval()
    sum_sample_best = 0.0
    sum_aco_best = 0.0
    sum_gap = 0.0
    n_val = len(val_dataset)
    
    if args.problem == 'tsp':
        iterable = val_dataset
    else:
        iterable = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False
        )
    
    agg_metrics: Dict[str, List[float]] = {}
    
    for idx, item in enumerate(tqdm(iterable, desc="Validating", leave=False)):
        # Preprocess item based on problem type
        item = _preprocess_val_item(item, args.problem)
        
        dynamic = not args.no_dynamic_feats
        avg, best, timings, metrics = infer_instance(
            net, item, args.k_sparse, args.n_ants, dynamic, val_args,
            collect_metrics=not args.simple_train
        )
        
        sum_sample_best += avg
        sum_aco_best += best
        
        if baseline_values is not None:
            opt = float(baseline_values[idx])
            gap = (best - opt) / opt * 100
            sum_gap += gap
        
        # Aggregate metrics
        for k, v in metrics.items():
            if k not in agg_metrics:
                agg_metrics[k] = []
            if len(v) > 0:
                agg_metrics[k].append(np.mean(v))
    
    avg_last = sum_sample_best / n_val
    avg_aco_best = sum_aco_best / n_val
    avg_gap = sum_gap / n_val if baseline_values is not None else 0.0
    
    out_metrics = {k: np.mean(v) for k, v in agg_metrics.items() if len(v) > 0}
    
    return avg_last, avg_aco_best, avg_gap, out_metrics


def _preprocess_val_item(item: Any, problem: str) -> Any:
    """Preprocess validation item to standard format."""
    if problem == 'cvrp':
        if isinstance(item, (tuple, list)):
            item = [item[0], item[1], item[2]]
        
        item = [x[0] if torch.is_tensor(x) else x for x in item]
        if torch.is_tensor(item[0]):
            item[0] = item[0].numpy()
        if torch.is_tensor(item[1]):
            item[1] = item[1].numpy()
        if torch.is_tensor(item[2]):
            item[2] = float(item[2])
    else:  # tsp
        if isinstance(item, (tuple, list)):
            item = item[0]
    
    return item

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train neural-guided ACO for TSP/CVRP"
    )
    
    # Problem configuration
    parser.add_argument("--problem", type=str, required=True, 
                        choices=['tsp', 'cvrp'])
    parser.add_argument("--n_node", type=int, default=1000)
    parser.add_argument("--k_sparse", type=int, default=32)
    parser.add_argument("--algo", choices=["reinforce", "ppo"], default="ppo")
    parser.add_argument("--alg", choices=["faco", "mmas"], default="faco",
                        help="Algorithm type")
    
    # PPO hyperparameters
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_clip", type=float, default=0.1)
    parser.add_argument("--no_adv_norm", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    
    # Training configuration
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ppo_lr", type=float, default=5e-6)
    parser.add_argument("--reinforce_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # ACO configuration
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--min_new_edges", type=int, default=12)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--no_local_search", action="store_true")
    parser.add_argument("--no_smooth_mmas", action="store_true")
    parser.add_argument("--no_extend_ls", action="store_true")
    parser.add_argument("--no_normalized_heuristic", action="store_true")

    # Traced sampling parallelism
    # CVRP wrapper defaults to parallel_traced=False (single-thread traced sampling) unless passed explicitly.
    pt_group = parser.add_mutually_exclusive_group()
    pt_group.add_argument(
        "--parallel_traced",
        dest="parallel_traced",
        action="store_true",
        help="Parallelize traced sampling across ants (faster; different RNG usage)",
    )
    pt_group.add_argument(
        "--no_parallel_traced",
        dest="parallel_traced",
        action="store_false",
        help="Force single-thread traced sampling (slower; legacy behavior)",
    )
    parser.set_defaults(parallel_traced=True)

    # Optimization
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="Enable gradient checkpointing for GNN")
    
    # Neural Local Search
    parser.add_argument("--nls", action="store_true",
                        help="Enable Neural Local Search")
    parser.add_argument("--nls_beta", type=float, default=0.5,
                        help="Weight for post-LS cost in advantage")
    parser.add_argument("--T_nls", type=int, default=10,
                        help="Number of NLS iterations")

    parser.add_argument("--no_logit_net", action="store_true")
    
    # Logging and output
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lga")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="pretrained")
    parser.add_argument("--no_dynamic_feats", action="store_true")
    
    # Baseline configuration
    parser.add_argument("--baseline", type=str, default='default')
    parser.add_argument("--baseline_runs", type=int, default=1)
    parser.add_argument("--baseline_time_limit", type=float, default=300.0)
    
    # Validation configuration
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to validation dataset")
    parser.add_argument("--val_size", type=int, default=16,
                        help="Limit validation set size")
    parser.add_argument("--generate_val", action="store_true",
                        help="Generate validation set")
    parser.add_argument("--save_generated", type=str, default=None,
                        help="Path to save generated validation dataset")
    parser.add_argument("--val_H", type=int, default=None)
    parser.add_argument("--val_mini_H", type=int, default=None)
    
    # Warmup and annealing
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Use warmup strategy in validation")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--train_warmup", action="store_true",
                        help="Use warmup strategy in training")
    parser.add_argument("--warmup_ratio", type=float, default=0.5)
    parser.add_argument("--train_anneal", action="store_true",
                        help="Enable annealing during training")
    parser.add_argument("--no_anneal", action="store_true",
                        help="Disable annealing during validation")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.0)
    parser.add_argument("--L", type=int, default=0)
    
    # Miscellaneous
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--simple_train", action="store_true",
                        help="Skip expensive metric calculations")
    parser.add_argument("--train_deepaco", action="store_true",
                        help="Disable pheromone updates during training")
    
    return parser.parse_args()


def setup_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model_name(args: argparse.Namespace) -> str:
    """Generate descriptive model filename based on parameters."""
    name = (f"{args.problem}_n{args.n_node}_k{args.k_sparse}_ants{args.n_ants}"
            f"_H{args.H}_miniH{args.mini_H}_rho{args.rho}"
            f"_mne{args.min_new_edges}_{args.algo}_lr{args.lr}")
    
    if args.train_anneal:
        name += f"_anneal_g{args.gamma}_mg{args.min_gamma}"
    if args.L > 0:
        name += f"_L{args.L}"
    if args.train_warmup:
        name += f"_warmup{args.warmup_ratio}"
    if args.train_deepaco:
        name += "_deepaco"
    
    # Ablation suffixes
    if args.no_dynamic_feats:
        name += "_static"
    if args.no_smooth_mmas:
        name += "_nosmooth"
    if args.disable_heuristic:
        name += "_noheu"
    if args.no_extend_ls:
        name += "_noextls"
    if args.no_normalized_heuristic:
        name += "_nonorm"
    if args.alg == 'mmas':
        name += "_mmas"
    
    return name


def load_validation_data(args: argparse.Namespace, logger: Logger):
    """Load or generate validation dataset and baselines."""
    val_dataset = None
    baseline_values = None
    
    if args.generate_val:
        baseline_solver = (args.baseline if args.baseline != 'default' 
                          else ('lkh' if args.problem == 'tsp' else 'hgs'))
        val_dataset = utils.generate_and_save_dataset(
            problem=args.problem,
            n_node=args.n_node,
            n_instances=args.val_size,
            save_path=args.save_generated,
            baseline_solver=baseline_solver,
            baseline_runs=args.baseline_runs,
            time_limit=args.baseline_time_limit,
            device='cpu'
        )
    elif args.val_dataset:
        logger.info(f"Loading validation dataset from {args.val_dataset}...")
        val_dataset = _load_dataset_from_path(args.val_dataset, args.problem)
    else:
        val_dataset = utils.load_val_dataset(
            args.n_node, problem=args.problem, device='cpu'
        )

    # Extract baseline values if embedded in dataset
    baseline_values = _extract_baseline_from_dataset(val_dataset, args.problem)
    
    # Generate fallback dataset if needed
    if val_dataset is None:
        logger.info("Validation dataset not found. Generating 16 instances...")
        val_dataset = _generate_fallback_dataset(args)
        if not args.val_dataset:
            utils.save_val_dataset(val_dataset, args.n_node, problem=args.problem)
    
    # Limit validation set size
    if args.val_size is not None and val_dataset is not None:
        original_len = len(val_dataset)
        val_dataset = val_dataset[:args.val_size]
        if original_len != len(val_dataset):
            logger.info(f"Limited validation dataset from {original_len} to {len(val_dataset)} instances.")
    
    # Compute baselines if needed
    if (baseline_values is None and args.baseline != 'none' 
        and not getattr(args, 'no_baseline', False)):
        baseline_values = _compute_baselines(val_dataset, args)
    
    return val_dataset, baseline_values


def _load_dataset_from_path(path: str, problem: str):
    """Load dataset from file path."""
    if path.endswith(".txt") and problem == 'tsp':
        return utils.load_tsp_txt_dataset(path)
    elif path.endswith(".txt") and problem == 'cvrp':
        return utils.load_cvrp_txt_dataset(path)
    else:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            return data.get("coords", data)
        return data


def _extract_baseline_from_dataset(val_dataset, problem: str):
    """Extract baseline costs if embedded in dataset."""
    if not isinstance(val_dataset, list) or len(val_dataset) == 0:
        return None
    
    try:
        if problem == 'tsp' and isinstance(val_dataset[0], tuple) and len(val_dataset[0]) >= 2:
            costs = [x[1] for x in val_dataset]
            if all((isinstance(c, (int, float)) or np.issubdtype(type(c), np.number)) 
                   and c > 1e-6 for c in costs):
                return np.array(costs)
        elif problem == 'cvrp' and isinstance(val_dataset[0], tuple) and len(val_dataset[0]) >= 4:
            costs = [x[3] for x in val_dataset]
            if all((isinstance(c, (int, float)) or np.issubdtype(type(c), np.number)) 
                   and c > 1e-6 for c in costs):
                return np.array(costs)
    except Exception:
        pass
    
    return None


def _generate_fallback_dataset(args: argparse.Namespace):
    """Generate fallback validation dataset."""
    val_dataset = []
    gen_fn = (utils.generate_tsp_instance if args.problem == 'tsp' 
              else utils.gen_cvrp_instance)
    
    for _ in range(16):
        if args.problem == 'tsp':
            val_dataset.append(torch.from_numpy(gen_fn(args.n_node)))
        else:
            c, d, cap = gen_fn(args.n_node, device='cpu')
            val_dataset.append((c.cpu(), d.cpu(), cap))
    
    return val_dataset


def _compute_baselines(val_dataset, args: argparse.Namespace):
    """Compute baseline values for validation dataset."""
    logger = get_logger()
    logger.info("Computing baseline values...")
    
    # Extract coords if dataset contains tuples
    if (args.problem == 'tsp' and isinstance(val_dataset, list) 
        and len(val_dataset) > 0 and isinstance(val_dataset[0], tuple)):
        val_dataset_coords = [x[0] for x in val_dataset]
        return get_baseline(
            val_dataset_coords, problem=args.problem, n_node=args.n_node,
            runs=args.baseline_runs, time_limit=args.baseline_time_limit
        )
    
    return get_baseline(
        val_dataset, problem=args.problem, n_node=args.n_node,
        runs=args.baseline_runs, time_limit=args.baseline_time_limit
    )


def save_checkpoint(
    net: Net,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    save_path: Path,
    val_cost: Optional[float] = None,
    val_gap: Optional[float] = None
):
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": vars(args)
    }
    if val_cost is not None:
        checkpoint["val_cost"] = val_cost
    if val_gap is not None:
        checkpoint["val_gap"] = val_gap
    
    torch.save(checkpoint, save_path)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main training entry point."""
    args = parse_args()
    


    # Setup defaults
    if args.lr is None:
        args.lr = args.ppo_lr if args.algo == 'ppo' else args.reinforce_lr
    
    if args.baseline == 'default':
        args.baseline = 'lkh' if args.problem == 'tsp' else 'hgs'
        
    args.extend_ls = not args.no_extend_ls
    
    if args.threads is None:
        args.threads = psutil.cpu_count(logical=False)
    faco.set_faco_cpp_threads(args.threads)

    # Set seeds
    setup_seeds(args.seed)

    # Initialize logger
    logger = init_logger(
        use_wandb=not args.no_wandb,
        log_dir=Path(args.save_dir) / "logs" if args.save_dir else None,
        verbose=True
    )

    # Build model name
    model_name = build_model_name(args)
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.problem / f"n{args.n_node}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
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

    # Initialize model
    feats = 2 if args.problem == 'tsp' else 4
    net_model = Net(
        feats=feats,
        logit_net=not args.no_logit_net,
        grad_checkpointing=args.grad_checkpoint
    ).to(args.device)
    
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr)

    # Load validation data
    val_dataset, baseline_values = load_validation_data(args, logger)
    
    if baseline_values is not None:
        logger.info("Using baseline costs from dataset.")

    # Training loop
    global_step = 0
    best_val_cost = float('inf')
    best_model_state = None
    total_train_time = 0.0
    
    for epoch in range(args.epochs):
        # Train one epoch
        (global_step, avg_train, t_neural, t_aco, 
         epoch_train_time) = train_epoch(
            net_model, optimizer, global_step, epoch, args
        )
        total_train_time += epoch_train_time
        
        # Validate
        if val_dataset is not None:
            avg_last, avg_best, avg_gap, val_metrics = validation(
                net_model, val_dataset, args, baseline_values
            )
            
            logger.log_epoch_summary(epoch, avg_train, avg_best, avg_gap)
            
            # Track and save best model
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
                best_path = save_dir / f"{model_name}_best.pt"
                torch.save(best_model_state, best_path)
                logger.log_model_saved(best_path, epoch, avg_best, avg_gap)
            
            # Log validation metrics
            logger.log_validation(
                avg_last, avg_best, avg_gap, epoch, val_metrics,
                timing={"neural_epoch": t_neural, "aco_epoch": t_aco},
                step=global_step
            )

        # Save "last" checkpoint every epoch
        if args.save_dir:
            save_checkpoint(
                net_model, optimizer, epoch, args,
                save_dir / f"{model_name}_last.pt"
            )
    
    # Save final model
    logger.info(f"Total Train Time: {total_train_time:.2f}s")
    
    if not args.no_wandb:
        wandb.log({"time/total_train_time": total_train_time})

    # No separate "final" checkpoint; "last" is updated each epoch.


if __name__ == "__main__":
    main()
