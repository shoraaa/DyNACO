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
import sys
from typing import Optional

# Shared helpers
import wandb

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

    # Optimization: blind bincount
    ndec = torch.bincount(ant_idx_all[is_stoch], minlength=n_ants).to(torch.int32)

    logp = torch.zeros((n_ants,), device=device, dtype=torch.float32)

    roulette = is_stoch & (pick >= 0)
    
    # Optimization: Blindly index. nonzero() works on empty tensors.
    idx = roulette.nonzero(as_tuple=False).squeeze(1)
    
    # Optional: check size if overhead of empty slicing is high, but usually stochastic is true
    if idx.numel() > 0:
        curr_r = curr[idx]
        pick_r = pick[idx]

        # Debug checks removed
        vm_r = vm_i64[idx]
        w = prob_sparse[curr_r]

        bitpos = torch.arange(k, device=device, dtype=torch.int64)
        valid = ((vm_r.unsqueeze(1) >> bitpos) & 1).to(w.dtype)

        denom = (w * valid).sum(dim=1).clamp_min(1e-12)
        numer = w.gather(1, pick_r.unsqueeze(1)).squeeze(1).clamp_min(1e-12)

        # check validity? skipping for speed/brevity in unified script, logic identical
        lp = torch.log(numer / denom)
        logp.scatter_add_(0, ant_idx_all[idx], lp)

    return logp, ndec

EPS = 1e-12

def prob_sparse_from_tau_eta_prior(tau_nk, eta_nk, prior_nk, alpha=1.0, beta=1.0, eps=1e-12):
    tau = tau_nk.clamp_min(eps)
    eta = eta_nk.clamp_min(eps)
    w = torch.exp(alpha * torch.log(tau) + beta * torch.log(eta) + prior_nk)
    return w.clamp_min(eps)


EPS = 1e-12

def setup_aco(args, instance_data, MFACOClass):
    if args.problem == 'tsp':
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
    
    aco = MFACOClass(**kwargs)
    return aco, pyg_args

# -----------------------------------------------------------------------------
# Unified Training Logic
# -----------------------------------------------------------------------------

def get_modules(problem):
    # Retrieve current working directory
    cwd = os.getcwd()
    problem_path = os.path.join(cwd, problem)
    
    # Inject into sys.path to allow imports like 'import net' to find problem/net.py
    if problem_path not in sys.path:
        sys.path.insert(0, problem_path)

    # Dynamic imports
    if problem == 'tsp':
        import net
        import faco
        import utils
        import baselines
        try:
             import test as test_mod # might collide with top-level test.py?
             # 'test' is likely the standard library or local test.py if in sys.path[0]
             # But we inserted problem_path at 0. So 'test' should be tsp/test.py
             pass
        except ImportError:
             test_mod = None

        Net = net.Net
        MFACO = faco.MFACO_TSP
        load_val = utils.load_val_dataset
        build_pyg = utils.build_pyg_data
        gen_val = utils.generate_val_dataset
        get_base = baselines.get_baseline_tsp
        
        # Wrapper for infer_instance
        # tsp/test.py has infer_instance
        # But we need to be careful if we imported top-level test.py by mistake?
        # If cwd is in sys.path, 'import test' might pick up ./test.py
        # sys.path[0] is problem_path. So import test picks up tsp/test.py IF it exists.
        # tsp/test.py exists.
        from test import infer_instance as infer_tsp
        
        def infer_wrapper(net, instance_data, k, n_ants, dynamic, args, collect_metrics=False):
            return infer_tsp('tsp', MFACO, build_pyg, net, instance_data, k, n_ants, dynamic, args, collect_metrics=collect_metrics)

        return Net, MFACO, load_val, build_pyg, gen_val, get_base, infer_wrapper, faco.set_faco_cpp_threads
    
    elif problem == 'cvrp':
        import net
        import faco
        import utils
        import baselines
        from test import infer_instance as infer_cvrp
        
        Net = net.Net
        MFACO = faco.MFACO_CVRP
        load_val = utils.load_val_dataset
        build_pyg = utils.build_pyg_data
        # CVRP utils might not have gen_instance_for_mfaco exposed or named differently?
        # Checked file earlier: from utils import ..., gen_instance_for_mfaco
        gen_data = utils.gen_instance_for_mfaco
        get_base = baselines.get_baseline_cvrp
        
        def infer_wrapper(net, instance_data, k, n_ants, dynamic, args, collect_metrics=False):
            # coords, demand, capacity = instance_data
            return infer_cvrp('cvrp', MFACO, build_pyg, net, instance_data, k, n_ants, dynamic, args, collect_metrics=collect_metrics)
        
        return Net, MFACO, load_val, build_pyg, gen_data, get_base, infer_wrapper, faco.set_faco_cpp_threads
    else:
        raise ValueError(f"Unknown problem: {problem}")

def train_instance_ppo(NetClass, MFACOClass, build_pyg_data_fn, model, optimizer, instance_data, args):
    model.train()
    
    aco, pyg_args = setup_aco(args, instance_data, MFACOClass)
    eta_nk = aco.h_sparse_torch if args.problem == 'tsp' else aco.heuristic_sparse_np
    if args.problem == 'cvrp':
         eta_nk = torch.tensor(eta_nk, device=args.device) # convert view

    best_seen = float("inf")
    avg_cost_last = None
    
    metrics = {
        "ndec": [], "loss": [], 
        "entropy": [], "prior_mean": [], "prior_std": [],
        "approx_kl": [], "clip_frac": [], "new_edges": []
    }

    t_neural_total = 0.0
    t_aco_sampling = 0.0
    t_aco_ls = 0.0
    t_aco_update = 0.0
    # Fallback if detailed timings not available
    t_aco_total = 0.0
    
    prior_prev_outer = None # For crossover drift/turnover between outer steps
    
    for outer in tqdm(range(args.H), desc="Outer", leave=False):
        t0 = time.time()
        pyg_data = build_pyg_data_fn(aco, *pyg_args, dynamic=not args.no_dynamic_feats)
        
        with torch.no_grad():
            prior_old = model(pyg_data).view(-1).view(aco.n, aco.k)
            t_neural_total += time.time() - t0
            
            metrics["prior_mean"].append(prior_old.mean().item())
            metrics["prior_std"].append(prior_old.std().item())

        traces_list = []
        costs_list = []
        logp_old_list = []
        ndec_list = []
        tau_list = []

        # Reset detailed timings if available
        if hasattr(aco, "reset_timings"):
            aco.reset_timings()

        t_aco_start_outer = time.time()

        for inner in range(args.mini_H):
            # Annealing
            current_prior = prior_old
            if args.anneal_prior:
                # Linear decay from gamma to min_gamma
                if args.mini_H > 1:
                    ratio = inner / (args.mini_H - 1)
                    factor = args.gamma * (1.0 - ratio) + args.min_gamma * ratio
                else:
                    factor = args.gamma
                current_prior = prior_old * factor

            if args.problem == 'tsp':
                costs, flats, _, logps_cpp, traces, costs_raw, flats_raw, new_edges = aco.sample(
                    require_prob=True, prior=current_prior
                )
            else: # cvrp
                costs, perms, _, logps, traces, new_edges = aco.sample(
                    require_prob=True, prior=current_prior
                )
                flats = perms # rename for consistency
            
            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)
            tau_nk = aco.tau_nk_torch().detach()
            tau_list.append(tau_nk)

            # Replay for logp_old
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

            # Update pheromone
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
        
        # Collect timings if available
        if hasattr(aco, "get_timings"):
             timings = aco.get_timings()
             # accumulators in timings dict: time_sampling, time_ls, time_update (ms)
             # Convert to seconds
             if "time_sampling" in timings: t_aco_sampling += timings["time_sampling"] / 1000.0
             if "time_ls" in timings: t_aco_ls += timings["time_ls"] / 1000.0
             if "time_update" in timings: t_aco_update += timings["time_update"] / 1000.0

        # PPO Epochs
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
                
                # Metrics: PPO
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
            
            # Gradient Variance
            # grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
            # if grads:
            #     grad_vec = torch.cat(grads)
            #     metrics["grad_var"].append(grad_vec.std().item())
            
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

def train_instance_reinforce(NetClass, MFACOClass, build_pyg_data_fn, model, optimizer, instance_data, args):
    model.train()
    
    aco, pyg_args = setup_aco(args, instance_data, MFACOClass)
    optimizer.zero_grad(set_to_none=True)
    
    best_seen = float("inf")
    avg_cost_last = None
    
    metrics = {
        "ndec": [], "loss": [],
        "entropy": [], "prior_mean": [], "prior_std": [], "new_edges": []
    }
    
    t_neural_total = 0.0
    t_aco_total = 0.0
    t_aco_sampling = 0.0
    t_aco_ls = 0.0
    t_aco_update = 0.0
    
    prior_prev_outer = None
    
    for t in tqdm(range(args.H), desc="ACO Step", leave=False):
        t0 = time.time()
        pyg_data = build_pyg_data_fn(aco, *pyg_args, dynamic=not args.no_dynamic_feats)
        heu_vec = model(pyg_data).view(-1)
        prior_mat = heu_vec.view(aco.n, aco.k)
        t_neural_total += time.time() - t0
        
        metrics["prior_mean"].append(prior_mat.detach().mean().item())
        metrics["prior_std"].append(prior_mat.detach().std().item())
        if args.problem == 'cvrp': prior_mat += EPS
            
        losses = 0
        
        if hasattr(aco, "reset_timings"): aco.reset_timings()
        t_aco_start_outer = time.time()
        
        for mini_t in range(args.mini_H):
            # Annealing
            current_prior = prior_mat
            if args.anneal_prior:
                if args.mini_H > 1:
                    ratio = mini_t / (args.mini_H - 1)
                    factor = args.gamma * (1.0 - ratio) + args.min_gamma * ratio
                else:
                    factor = args.gamma
                current_prior = prior_mat * factor
                
            if args.problem == 'tsp':
                costs, flats, _, logps_cpp, traces, _, _, new_edges = aco.sample(require_prob=True, prior=current_prior)
            else:
                costs, perms, _, logps_cpp, traces, new_edges = aco.sample(require_prob=True, prior=current_prior)
                flats = perms

            costs_t = torch.as_tensor(costs, device=args.device, dtype=torch.float32)

            prob_sparse = aco.prob_sparse_torch(prior=current_prior).clamp_min(EPS)
            logp_per_ant, ndec_per_ant = replay_logp_from_cpp_batch_trace(traces, prob_sparse)
            
            metrics["new_edges"].append(new_edges.astype(np.float32).mean())
            
            # Metrics
            ndec_avg = ndec_per_ant.float().mean().item()
            metrics["ndec"].append(ndec_avg)
            
            # Entropy
            entropy = (-logp_per_ant / ndec_per_ant.float().clamp_min(1.0)).mean().item()
            metrics["entropy"].append(entropy)
            
            baseline = costs_t.mean()
            adv = (costs_t - baseline).detach()
            loss = (adv * logp_per_ant / ndec_per_ant).mean()

            best_idx = int(costs_t.argmin().item())
            best_cost_iter = float(costs[best_idx])
            best_seen = min(best_seen, best_cost_iter)
            
            with torch.no_grad():
                if args.problem == 'tsp':
                    aco._update_pheromone_from_flat(flats[best_idx], best_cost_iter)
                else:
                    aco.update_pheromone(flats[best_idx], best_cost_iter)

            losses += loss
            avg_cost_last = float(costs_t.mean().item())
        
        t_aco_total += time.time() - t_aco_start_outer
        if hasattr(aco, "get_timings"):
             timings = aco.get_timings()
             if "time_sampling" in timings: t_aco_sampling += timings["time_sampling"] / 1000.0
             if "time_ls" in timings: t_aco_ls += timings["time_ls"] / 1000.0
             if "time_update" in timings: t_aco_update += timings["time_update"] / 1000.0
            
        losses.backward()
        
        # grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        # if grads:
        #     metrics["grad_var"].append(torch.cat(grads).std().item())
            
        metrics["loss"].append(losses.item())
    
    metrics["time_neural"] = [t_neural_total]
    metrics["time_aco"] = [t_aco_total]
    if t_aco_sampling > 0: metrics["time_sampling"] = [t_aco_sampling]
    if t_aco_ls > 0: metrics["time_ls"] = [t_aco_ls]
    if t_aco_update > 0: metrics["time_update"] = [t_aco_update]
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    out_metrics = {}
    for k, v in metrics.items():
        if v: out_metrics[k] = np.mean(v)
        else: out_metrics[k] = 0.0
    return avg_cost_last, best_seen, out_metrics

def train_epoch(problem, NetClass, MFACOClass, build_pyg_data_fn, gen_data_fn, net, optimizer, global_step, epoch, args):
    sum_avg_cost = 0
    steps = args.steps_per_epoch
    for step in tqdm(range(steps), desc="Epoch", leave=True):
        # Generate data
        if problem == 'tsp':
            instance_data = np.random.rand(args.n_node, 2).astype(np.float32)
        else:
            coords_t, demand_t, capacity = gen_data_fn(args.n_node, device=args.device)
            instance_data = (
                coords_t.detach().cpu().numpy().astype(np.float32),
                demand_t.detach().cpu().numpy().astype(np.float32),
                capacity
            )
        
        if args.algo == 'ppo':
            avg_cost, best_cost, metrics = train_instance_ppo(NetClass, MFACOClass, build_pyg_data_fn, net, optimizer, instance_data, args)
        else:
            avg_cost, best_cost, metrics = train_instance_reinforce(NetClass, MFACOClass, build_pyg_data_fn, net, optimizer, instance_data, args)
            
        sum_avg_cost += avg_cost
        
        # Aggregate timings for epoch print/log
        if "time_neural" in metrics:
             epoch_time_neural = metrics["time_neural"] # It's a mean scalar from out_metrics
        if "time_aco" in metrics:
             epoch_time_aco = metrics["time_aco"]

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

def validation(problem, infer_wrapper, net, val_dataset, args, baseline_values=None):
    sum_sample_best, sum_aco_best = 0, 0
    sum_gap = 0
    n_val = len(val_dataset)
    
    # Iterate depending on problem/dataset structure
    if problem == 'tsp':
        iterable = val_dataset
    else:
        # CVRP val_dataset is TensorDataset usually
        iterable = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Aggregators for metrics
    agg_metrics = {}
    
    idx = 0
    for item in tqdm(iterable, desc="Validating", leave=False):
        if problem == 'cvrp':
            item = [x[0] if torch.is_tensor(x) else x for x in item] # Unbatch
            if torch.is_tensor(item[0]): item[0] = item[0].numpy()
            if torch.is_tensor(item[1]): item[1] = item[1].numpy()
            if torch.is_tensor(item[2]): item[2] = float(item[2])
            
        # Unified infer wrapper
        dynamic = not args.no_dynamic_feats
        # TSP: item is coords. CVRP: item is (coords, demand, cap).
        # Enable metrics collection during validation
        res = infer_wrapper(net, item, args.k_sparse, args.n_ants, dynamic, args, collect_metrics=True)
        
        # Unpack result
        metrics = None
        if len(res) == 4: 
            avg, best, timings, metrics = res
        elif len(res) == 3:
            avg, best, timings = res # fallback if test.py not updated or different
        
        sum_sample_best += avg
        sum_aco_best += best
        
        if baseline_values is not None:
            opt = float(baseline_values[idx])
            gap = (best - opt) / opt * 100
            sum_gap += gap
        
        # Aggregate metrics
        if metrics:
            for k, v in metrics.items():
                if k not in agg_metrics: agg_metrics[k] = []
                # v is list of H values. We can just take mean over H here or append all
                # Let's take mean over H for this instance
                if len(v) > 0:
                    agg_metrics[k].append(np.mean(v))
        
        idx += 1
    
    avg_last = sum_sample_best/n_val
    avg_aco_best = sum_aco_best/n_val
    avg_gap = sum_gap/n_val if baseline_values is not None else 0.0
    
    # Finalize metrics
    out_metrics = {}
    for k, v in agg_metrics.items():
        if len(v) > 0:
            out_metrics[k] = np.mean(v)
            
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
    # If not provided, we pick a default based on --algo.
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ppo_lr", type=float, default=5e-6)
    parser.add_argument("--reinforce_lr", type=float, default=1e-4)
    
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # ACO
    parser.add_argument("--rho", type=float, default=0.1) # TSP 0.1, CVRP 0.5 default
    parser.add_argument("--min_new_edges", type=int, default=12)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--no_local_search", action="store_true")
    parser.add_argument("--no_smooth_mmas", action="store_true")
    parser.add_argument("--no_extend_ls", action="store_true") # Defaults to True (enabled) unless flag passed
    parser.add_argument("--no_normalized_heuristic", action="store_true")
    parser.add_argument("--no_logit_net", action="store_true")
    
    # Misc
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lga")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="pretrained")
    parser.add_argument("--no_dynamic_feats", action="store_true")
    parser.add_argument("--baseline", type=str, default='default') # 'lkh' or 'hgs'
    parser.add_argument("--baseline_runs", type=int, default=1)
    parser.add_argument("--baseline_time_limit", type=float, default=10.0)
    parser.add_argument("--anneal_prior", action="store_true", help="Gradually decrease prior influence in mini_H from gamma to min_gamma")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.2)
    parser.add_argument("--L", type=int, default=0, help="Fixed ant trajectory length")
    parser.add_argument("--run_name", type=str, default=None, help="Custom wandb run name")
    parser.add_argument("--threads", type=int, default=16, help="OpenMP threads")
    parser.add_argument("--timed", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_output", type=str, default="visualizations")
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()

    # Algo-specific default learning rate (source of truth)
    if args.lr is None:
        args.lr = args.ppo_lr if args.algo == 'ppo' else args.reinforce_lr
    
    # Compatibility with legacy scripts that expect args.extend_ls
    # Defaults adjustment
    if args.baseline == 'default':
        args.baseline = 'lkh' if args.problem == 'tsp' else 'hgs'

    # Compatibility with legacy scripts that expect args.extend_ls
    args.extend_ls = not args.no_extend_ls

    # Basic setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    save_path = Path(args.save_dir) / args.problem
    os.makedirs(save_path, exist_ok=True)
    log_path = save_path / f"logs_{args.n_node}.json"

    if not args.no_wandb:
        run_name = args.run_name if args.run_name else f"{args.problem}{args.n_node}_{args.algo}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args))

    # Modules
    NetClass, MFACOClass, load_val_dataset_fn, build_pyg_data_fn, gen_data_fn, get_baseline_fn, infer_wrapper, set_threads_fn = get_modules(args.problem)
    set_threads_fn(args.threads)
    
    # Data
    print("Loading validation dataset...")
    try:
        if args.problem == 'tsp':
            val_dataset = load_val_dataset_fn(args.n_node, args.device)
        else:
            val_dataset = load_val_dataset_fn(args.n_node, "cpu")
    except FileNotFoundError:
        print("Generating data...")
        if args.problem == 'tsp':
            gen_data_fn(args.n_node, 16, args.k_sparse, "cpu")
            val_dataset = load_val_dataset_fn(args.n_node, args.device)
        else:
            # Generate CVRP validation data on the fly (list of tuples) to prevent crash
            val_dataset_list = []
            for _ in range(128): # Standard val size
                 val_dataset_list.append(gen_data_fn(args.n_node, "cpu"))
            
            # Convert to TensorDataset for compatibility with baselines
            val_dataset = torch.utils.data.TensorDataset(
                torch.stack([x[0] for x in val_dataset_list]), # coords
                torch.stack([x[1] for x in val_dataset_list]), # demand
                torch.stack([torch.as_tensor(x[2]) for x in val_dataset_list]) # capacity
            )
    
    # Baseline
    baseline_values = None
    if args.baseline != 'none':
        print("Computing baseline...")
        if args.problem == 'tsp':
            baseline_values = get_baseline_fn(val_dataset, args.n_node, "cpu", runs=args.baseline_runs, time_limit=args.baseline_time_limit)
        else:
            baseline_values = get_baseline_fn(val_dataset, args.n_node, "cpu", time_limit=args.baseline_time_limit)
        baseline_values = baseline_values.cpu()
        print(f"Baseline mean: {baseline_values.mean()}")

    # Model
    net = NetClass(logit_net=not args.no_logit_net).to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.1)

    global_step = 0
    best_val = float("inf")
    
    # Validation before training
    avg_last, avg_aco_best, avg_gap, val_metrics = validation(args.problem, infer_wrapper, net, val_dataset, args, baseline_values)
    print(f"Epoch -1: ValLast={avg_last:.4f} ValBest={avg_aco_best:.4f} Gap={avg_gap:.2f}%")
    
    if not args.no_wandb:
         log_dict = {
             "val/avg_best": float(avg_aco_best),
             "val/avg_last": float(avg_last),
             "val/epoch": 0
         }
         if baseline_values is not None:
             log_dict["val/gap"] = float(avg_gap)
         if val_metrics:
             for k, v in val_metrics.items():
                 log_dict[f"val/{k}"] = float(v)
         wandb.log(log_dict, step=0)

    for epoch in range(args.epochs):
        start = time.time()
        global_step, train_avg, t_neural, t_aco = train_epoch(args.problem, NetClass, MFACOClass, build_pyg_data_fn, gen_data_fn, net, optimizer, global_step, epoch, args)
        scheduler.step()
        
        avg_last, avg_aco_best, avg_gap, val_metrics = validation(args.problem, infer_wrapper, net, val_dataset, args, baseline_values)
        
        print(f"Epoch {epoch}: Train={train_avg:.4f} ValLast={avg_last:.4f} ValBest={avg_aco_best:.4f} Gap={avg_gap:.2f}% Time={time.time()-start:.1f}s TimeN={t_neural:.2f}s TimeA={t_aco:.2f}s")
        
        if not args.no_wandb:
             log_dict = {
                 "val/avg_best": float(avg_aco_best),
                 "val/avg_last": float(avg_last),
                 "val/epoch": int(epoch)
             }
             if baseline_values is not None:
                 log_dict["val/gap"] = float(avg_gap)
             if val_metrics:
                 for k, v in val_metrics.items():
                     log_dict[f"val/{k}"] = float(v)
             wandb.log(log_dict, step=global_step)
        
        if avg_aco_best < best_val:
            best_val = avg_aco_best
            torch.save({"model_state_dict": net.state_dict(), "config": vars(args)}, save_path / f"best_{args.n_node}.pt")

    if not args.no_wandb and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
