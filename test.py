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
from datetime import datetime
import hashlib
import csv


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
    top1_flip_rate, safe_corr, top_overlap_frac, row_top1_match_rate, EPS, infer_instance
)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = None
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()


def _clone_args(args: argparse.Namespace, **overrides) -> argparse.Namespace:
    """Make a shallow copy of argparse.Namespace and apply overrides."""
    ns = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _base_cache_path(args: argparse.Namespace, val_list_len: int) -> Path:
    """Compute a deterministic cache path for baseline costs."""
    dataset_tag = "auto"
    dataset_stat = None
    if getattr(args, "generate_val", False):
        dataset_tag = f"generated:{getattr(args, 'save_generated', None) or 'memory'}"
    elif getattr(args, "dataset", None):
        dataset_tag = str(Path(args.dataset).expanduser().resolve())
        try:
            st = os.stat(dataset_tag)
            dataset_stat = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
        except OSError:
            dataset_stat = {"size": None, "mtime_ns": None}

    # Baseline (heuristic-only) still depends on ACO hyperparams and randomness.
    key_payload = {
        "problem": args.problem,
        "alg": getattr(args, "alg", None),
        "n_node": int(args.n_node) if args.n_node is not None else None,
        "k_sparse": int(args.k_sparse),
        "n_ants": int(args.n_ants) if args.n_ants is not None else None,
        "H": int(args.H),
        "mini_H": int(args.mini_H),
        "rho": float(args.rho),
        "min_new_edges": int(args.min_new_edges),
        "no_local_search": bool(args.no_local_search),
        "no_smooth_mmas": bool(args.no_smooth_mmas),
        "no_extend_ls": bool(args.no_extend_ls),
        "no_normalized_heuristic": bool(args.no_normalized_heuristic),
        "disable_heuristic": bool(args.disable_heuristic),
        "L": int(getattr(args, "L", 0)),
        "seed": int(args.seed),
        "dataset": dataset_tag,
        "dataset_stat": dataset_stat,
        "n_instances": int(val_list_len),
    }
    key_str = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path("output") / "base_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"base_{args.problem}_{digest}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default=None, choices=['tsp', 'cvrp'])
    parser.add_argument("--n_node", type=int, default=None)
    parser.add_argument("--k_sparse", type=int, default=32)
    parser.add_argument("--alg", choices=["faco", "mmas"], default="faco", help="Algorithm type")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="none")
    parser.add_argument("--n_ants", type=int, default=100)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=100)
    
    parser.add_argument("--disable_heuristic", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone weight for MMAS")
    parser.add_argument("--beta", type=float, default=1.0, help="Heuristic weight for MMAS")
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
    parser.add_argument("--no_anneal", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min_gamma", type=float, default=0.0)
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
    parser.add_argument("--log", action="store_true", help="Enable logging to file (auto-named)")
    parser.add_argument("--no_baseline", action="store_true", help="Skip pure MFACO baseline calculation")
    parser.add_argument("--rl_data", action="store_true", help="Load TSPLIB/CVRPLIB data instead of standard test set")
    
    # Selective method execution
    parser.add_argument("--run_model_no_anneal", action="store_true", help="Run Model(no_anneal) method")
    parser.add_argument("--run_mix_no_anneal", action="store_true", help="Run Mix(no_anneal) method")
    parser.add_argument("--no_model", action="store_true", help="Skip model loading and run only baseline methods")

    # Per-iteration logging (H * mini_H rows per run)
    parser.add_argument("--iter_log", action="store_true", help="Log mean/best at every mini-iteration to CSV")
    parser.add_argument("--iter_print", action="store_true", help="Print mean/best at every mini-iteration (very verbose)")

    args = parser.parse_args()

    # Args setup
    ckpt = None
    
    # Auto-load checkpoint if not provided and --no_model not set
    if args.checkpoint == "none" and not args.no_model:
        # Build expected checkpoint filename based on current args
        # Pattern: {problem}_n{n_node}_k{k_sparse}_ants{n_ants}_H{H}_miniH{mini_H}_rho{rho}_mne{min_new_edges}_ppo_lr{lr}_best.pt
        import glob
        pretrained_dir = f"pretrained/{args.problem}/n{args.n_node}"
        if os.path.exists(pretrained_dir):
            # Try to find a matching checkpoint
            pattern = f"{args.problem}_n{args.n_node}_*_best.pt"
            matches = glob.glob(os.path.join(pretrained_dir, pattern))
            if matches:
                # Prefer exact match, otherwise use first match
                args.checkpoint = matches[0]
                print(f"Auto-detected checkpoint: {args.checkpoint}")
            else:
                print(f"Warning: No checkpoints found in {pretrained_dir}, running without model")
                args.no_model = True
        else:
            print(f"Warning: Pretrained directory {pretrained_dir} not found, running without model")
            args.no_model = True
    
    if args.checkpoint != "none" and not args.no_model:
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
            "threads", "seed", "save_dir", "wandb_project", "wandb_entity", "no_wandb", "warmup", "no_baseline"
        }
        
        # If model was not trained with annealing, ignore its min_gamma (use ours)
        was_annealed = config.get("train_anneal", False) or config.get("anneal_prior", False)
        if not was_annealed:
             ignore_args.add("min_gamma")
        
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
        
    print("\n" + "="*50)
    print(f"{'Config Parameter':<30} | {'Value':<15}")
    print("-" * 50)
    for k, v in sorted(vars(args).items()):
         print(f"{k:<30} | {str(v):<15}")
    print("="*50 + "\n")
        
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

    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = "nockpt"
        if args.checkpoint != "none":
            ckpt_name = Path(args.checkpoint).stem
        
        data_name = "gen"
        if args.dataset:
            data_name = Path(args.dataset).stem
        
        log_dir = "logs"
        log_name = f"{log_dir}/test_{args.problem}_{args.n_node}_{ckpt_name}_{data_name}_{timestamp}_annealing{str(not args.no_anneal)}.txt"

        csv_dir = Path(log_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_name_base = f"test_{args.problem}_{args.n_node}_{ckpt_name}_{data_name}_{timestamp}_annealing{str(not args.no_anneal)}"
        csv_path_instances = csv_dir / f"{csv_name_base}_instances.csv"
        csv_path_summary = csv_dir / f"{csv_name_base}_summary.csv"
        csv_path_iters = csv_dir / f"{csv_name_base}_iters.csv"
        
        # Logger handles mkdir
        logger = Logger(log_name)
        sys.stdout = logger
        sys.stderr = logger
        print(f"Logging to {log_name}")
        print(f"CSV instances: {csv_path_instances}")
        print(f"CSV summary:   {csv_path_summary}")
        if args.iter_log:
            print(f"CSV iters:     {csv_path_iters}")
    else:
        csv_path_instances = None
        csv_path_summary = None
        csv_path_iters = None

        # If user wants iter logging without --log, still write a CSV.
        if args.iter_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name = "nockpt" if args.checkpoint == "none" else Path(args.checkpoint).stem
            data_name = "gen" if not args.dataset else Path(args.dataset).stem
            out_dir = Path("output") / "iter_logs"
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_name_base = f"test_{args.problem}_{args.n_node}_{ckpt_name}_{data_name}_{timestamp}_annealing{str(not args.no_anneal)}"
            csv_path_iters = out_dir / f"{csv_name_base}_iters.csv"
            print(f"CSV iters:     {csv_path_iters}")
    
    # Setup modules
    if args.problem == 'tsp':
        build_fn = utils.build_pyg_data_tsp
        gen_fn = utils.generate_tsp_instance
        if args.alg == 'mmas':
            MFACO = faco.ACO_TSP
        else:
            MFACO = faco.MFACO_TSP
    else:
        build_fn = utils.build_pyg_data_cvrp
        gen_fn = utils.gen_cvrp_instance
        if args.alg == 'mmas':
            MFACO = faco.ACO_CVRP
        else:
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
        val_list = utils.load_auto_dataset(
            args.n_node, 
            problem=args.problem, 
            rl_data=args.rl_data,
            device='cpu'
        )
        
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
    
    # Hardcode for specific datasets if requested (fix for TSP10000)
    if args.dataset and "tsp10000" in args.dataset.lower() and "test" in args.dataset.lower():
         print("Hardcoding baseline cost to 71.778 for TSP10000 test set.")
         # Assuming val_list has correct length, we create an array of this cost
         if val_list:
             baseline_values = np.full(len(val_list), 71.778)

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
    if ckpt is not None and not args.no_model:
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
        "base_cost": [],
        "base_time": [],
        "base_metrics": {},

        # Neural-guided costs (anneal on/off)
        "model_cost": [],
        "model_cost_no_anneal": [],
        "model_time": [],
        "model_time_no_anneal": [],
        "model_metrics": {},

        # Warmup/mix costs (anneal on/off)
        "mix_cost": [],
        "mix_cost_no_anneal": [],
        "mix_time": [],
        "mix_time_no_anneal": [],
        "mix_metrics": {},

        "opt_cost": [],
        "base_gap": [],
        "model_gap": [],
        "model_gap_no_anneal": [],
        "mix_gap": [],
        "mix_gap_no_anneal": [],
        "model_time_breakdown": {
            "neural": [], "sampling": [], "ls": [], "update": []
        }
    }
    
    iterable = val_list
    if args.problem == 'cvrp' and hasattr(val_list, 'tensors'):
         iterable = torch.utils.data.DataLoader(val_list, batch_size=1, shuffle=False)
    
    print(f"Evaluating {len(val_list)} instances...")

    # Baseline cache (heuristic-only MFACO). Reuse across runs for the same dataset+config.
    cached_base_costs = None
    base_cache_path = None
    if not args.no_baseline:
        base_cache_path = _base_cache_path(args, len(val_list))
        if base_cache_path.exists():
            try:
                obj = torch.load(base_cache_path, map_location="cpu", weights_only=False)
                base_arr = obj.get("base_costs", None) if isinstance(obj, dict) else None
                if base_arr is not None and len(base_arr) == len(val_list):
                    cached_base_costs = [float(x) for x in base_arr]
                    print(f"Loaded cached base costs: {base_cache_path} ({len(cached_base_costs)} instances)")
            except Exception as e:
                print(f"Warning: failed to load base cache {base_cache_path}: {e}")

    if args.iter_log and cached_base_costs is not None:
        # Can't reconstruct per-iteration traces from cached scalar costs.
        print("Iter logging enabled: ignoring cached base costs to record per-iteration trace.")
        cached_base_costs = None
    
    sample_snapshots = {}

    per_instance_rows = []
    opt_costs_summary = []

    iter_csv_f = None
    iter_csv_writer = None
    if args.iter_log and csv_path_iters is not None:
        try:
            iter_csv_f = open(csv_path_iters, "w", newline="")
            iter_csv_writer = csv.DictWriter(
                iter_csv_f,
                fieldnames=[
                    "idx",
                    "name",
                    "method",
                    "anneal",
                    "iter",
                    "t",
                    "mini_t",
                    "mean",
                    "best",
                ],
            )
            iter_csv_writer.writeheader()
        except Exception as e:
            print(f"Warning: failed to open iter CSV {csv_path_iters}: {e}")
            iter_csv_f = None
            iter_csv_writer = None

    for i, item in enumerate(tqdm(iterable)):
        opt_cost = None
        
        # Unpack TSP tuple if present
        name = f"Instance {i}"
        if args.problem == 'tsp' and isinstance(item, tuple):
             # (coords, cost, tour) or (coords, cost, tour, name)
             coords = item[0]
             if len(item) > 1: opt_cost = item[1]
             if len(item) > 3: name = item[3]
             item = coords
        
        if args.problem == 'cvrp':
            if isinstance(item, list) and len(item)==1: item = item[0] # DataLoader batch=1
            
            # Unpack CVRP Text Tuple (coords, demand, capacity, cost, tour) or (..., name)
            if isinstance(item, tuple) and len(item) >= 5:
                # (coords, demand, capacity, cost, tour)
                coords, demand, capacity, cost, tour = item[:5]
                if len(item) > 5: name = item[5]
                
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

            if opt_cost is not None and isinstance(opt_cost, (int, float)) and float(opt_cost) > 1e-12:
                opt_costs_summary.append(float(opt_cost))
            
        # Base
        base_m = None
        base_best = None
        base_iter_stats = None
        if not args.no_baseline:
            if cached_base_costs is not None:
                base_best = float(cached_base_costs[i])
                results["base_cost"].append(base_best)
                results["base_time"].append(0.0)
            else:
                tb0 = time.time()
                base_ret = infer_instance(
                    args.problem, MFACO, build_fn, None, item,
                    args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                    args,
                    use_heuristic_only=True,
                    collect_metrics=args.visualize,
                    metrics_every_step=args.visualize,
                )
                tb1 = time.time()
                _, base_best, base_timings, base_extra = base_ret
                base_m = base_extra.get("metrics")
                base_iter_stats = base_extra.get("iter_stats")
                results["base_cost"].append(base_best)
                results["base_time"].append(tb1 - tb0)

                if args.iter_log and iter_csv_writer is not None and base_iter_stats is not None:
                    for st in base_iter_stats:
                        iter_csv_writer.writerow({
                            "idx": i,
                            "name": name,
                            "method": "Base",
                            "anneal": "-",
                            **st,
                        })
                if args.iter_print and base_iter_stats is not None:
                    for st in base_iter_stats:
                        print(f"{name} Base iter={st['iter']} mean={st['mean']:.6f} best={st['best']:.6f}")
            
            if opt_cost is not None and opt_cost > 1e-6:
                 gap = (base_best - opt_cost) / opt_cost
                 results["base_gap"].append(gap)
                 results["opt_cost"].append(opt_cost)

        # Model
        model_best = None
        model_best_na = None
        mix_best = None
        mix_best_na = None
        if model:
              # Determine which methods to run based on problem type and flags
              # Default: Mix(anneal) for TSP, Model(anneal) for CVRP
              run_model_anneal = (args.problem == 'cvrp')  # Default for CVRP
              run_model_no_anneal = args.run_model_no_anneal
              run_mix_anneal = (args.problem == 'tsp' and args.warmup)  # Default for TSP
              run_mix_no_anneal = args.run_mix_no_anneal and args.warmup
              
              args_anneal = _clone_args(args, no_anneal=False)
              args_noanneal = _clone_args(args, no_anneal=True)

              # Model (anneal ON)
              if run_model_anneal:
                  tm0 = time.time()
                  mod_ret = infer_instance(
                      args.problem, MFACO, build_fn, model, item,
                      args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                      args_anneal,
                      use_heuristic_only=False,
                      collect_metrics=args.visualize,
                      metrics_every_step=args.visualize,
                  )
                  tm1 = time.time()
                  _, model_best, mod_timings, mod_extra = mod_ret
                  model_m = mod_extra.get("metrics")
                  model_iter_stats = mod_extra.get("iter_stats")
                  results["model_cost"].append(model_best)
                  results["model_time"].append(tm1 - tm0)

                  if args.iter_log and iter_csv_writer is not None and model_iter_stats is not None:
                      for st in model_iter_stats:
                          iter_csv_writer.writerow({
                              "idx": i,
                              "name": name,
                              "method": "Model",
                              "anneal": "on",
                              **st,
                          })
                  if args.iter_print and model_iter_stats is not None:
                      for st in model_iter_stats:
                          print(f"{name} Model(anneal) iter={st['iter']} mean={st['mean']:.6f} best={st['best']:.6f}")
                  
                  if mod_timings:
                      if "time_neural" in mod_timings: results["model_time_breakdown"]["neural"].append(mod_timings["time_neural"])
                      if "time_sampling" in mod_timings: results["model_time_breakdown"]["sampling"].append(mod_timings["time_sampling"])
                      if "time_ls" in mod_timings: results["model_time_breakdown"]["ls"].append(mod_timings["time_ls"])
                      if "time_update" in mod_timings: results["model_time_breakdown"]["update"].append(mod_timings["time_update"])
                  
                  if opt_cost is not None and opt_cost > 1e-6:
                      gap = (model_best - opt_cost) / opt_cost
                      results["model_gap"].append(gap)

              # Model (anneal OFF)
              if run_model_no_anneal:
                  tm0 = time.time()
                  mod_ret_na = infer_instance(
                      args.problem, MFACO, build_fn, model, item,
                      args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                      args_noanneal,
                      use_heuristic_only=False,
                      collect_metrics=False,
                      metrics_every_step=False,
                  )
                  tm1 = time.time()
                  _, model_best_na, _, mod_na_extra = mod_ret_na
                  model_na_iter_stats = mod_na_extra.get("iter_stats")
                  results["model_cost_no_anneal"].append(model_best_na)
                  results["model_time_no_anneal"].append(tm1 - tm0)

                  if args.iter_log and iter_csv_writer is not None and model_na_iter_stats is not None:
                      for st in model_na_iter_stats:
                          iter_csv_writer.writerow({
                              "idx": i,
                              "name": name,
                              "method": "Model",
                              "anneal": "off",
                              **st,
                          })
                  if args.iter_print and model_na_iter_stats is not None:
                      for st in model_na_iter_stats:
                          print(f"{name} Model(no_anneal) iter={st['iter']} mean={st['mean']:.6f} best={st['best']:.6f}")
                  
                  if opt_cost is not None and opt_cost > 1e-6:
                      gap_na = (model_best_na - opt_cost) / opt_cost
                      results["model_gap_no_anneal"].append(gap_na)
                          # Mix methods (only if warmup is enabled)
              if args.warmup:
                  inject_step = int(args.H * args.warmup_ratio)
                  
                  # Mix (anneal ON)
                  if run_mix_anneal:
                      tmi0 = time.time()
                      mix_ret = infer_instance(
                          args.problem, MFACO, build_fn, model, item,
                          args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                          args_anneal,
                          use_heuristic_only=False,
                          collect_metrics=args.visualize,
                          metrics_every_step=args.visualize,
                          inject_step=inject_step,
                      )
                      tmi1 = time.time()
                      _, mix_best, _, mix_extra = mix_ret
                      mix_m = mix_extra.get("metrics")
                      mix_iter_stats = mix_extra.get("iter_stats")
                      results["mix_cost"].append(mix_best)
                      results["mix_time"].append(tmi1 - tmi0)

                      if args.iter_log and iter_csv_writer is not None and mix_iter_stats is not None:
                          for st in mix_iter_stats:
                              iter_csv_writer.writerow({
                                  "idx": i,
                                  "name": name,
                                  "method": "Mix",
                                  "anneal": "on",
                                  **st,
                              })
                      if args.iter_print and mix_iter_stats is not None:
                          for st in mix_iter_stats:
                              print(f"{name} Mix(anneal) iter={st['iter']} mean={st['mean']:.6f} best={st['best']:.6f}")
                      
                      if opt_cost is not None and opt_cost > 1e-6:
                          gap = (mix_best - opt_cost) / opt_cost
                          results["mix_gap"].append(gap)

                  # Mix (anneal OFF)
                  if run_mix_no_anneal:
                      tmi0 = time.time()
                      mix_ret_na = infer_instance(
                          args.problem, MFACO, build_fn, model, item,
                          args.k_sparse, args.n_ants, not args.no_dynamic_feats,
                          args_noanneal,
                          use_heuristic_only=False,
                          collect_metrics=False,
                          metrics_every_step=False,
                          inject_step=inject_step,
                      )
                      tmi1 = time.time()
                      _, mix_best_na, _, mix_na_extra = mix_ret_na
                      mix_na_iter_stats = mix_na_extra.get("iter_stats")
                      results["mix_cost_no_anneal"].append(mix_best_na)
                      results["mix_time_no_anneal"].append(tmi1 - tmi0)

                      if args.iter_log and iter_csv_writer is not None and mix_na_iter_stats is not None:
                          for st in mix_na_iter_stats:
                              iter_csv_writer.writerow({
                                  "idx": i,
                                  "name": name,
                                  "method": "Mix",
                                  "anneal": "off",
                                  **st,
                              })
                      if args.iter_print and mix_na_iter_stats is not None:
                          for st in mix_na_iter_stats:
                              print(f"{name} Mix(no_anneal) iter={st['iter']} mean={st['mean']:.6f} best={st['best']:.6f}")
                      
                      if opt_cost is not None and opt_cost > 1e-6:
                          gap_na = (mix_best_na - opt_cost) / opt_cost
                          results["mix_gap_no_anneal"].append(gap_na)
        
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

        # Output cost for each instance
        if True:  # Always print per-instance results
             msg = f"{name}:"
             if opt_cost is not None: msg += f" Opt={opt_cost:.4f}"

             if baseline_values is not None and len(baseline_values) > i:
                 msg += f" BL={float(baseline_values[i]):.4f}"
             
             if results.get("base_cost") and len(results["base_cost"]) > i: 
                 c = results['base_cost'][-1]
                 msg += f" Base={c:.4f}"
                 if opt_cost is not None and opt_cost > 1e-6:
                     g = (c - opt_cost) / opt_cost * 100
                     msg += f" ({g:.2f}%)"
                     
             if results.get("model_cost") and len(results["model_cost"]) > i: 
                 c = results['model_cost'][-1]
                 msg += f" Model={c:.4f}"
                 if opt_cost is not None and opt_cost > 1e-6:
                     g = (c - opt_cost) / opt_cost * 100
                     msg += f" ({g:.2f}%)"

             if results.get("model_cost_no_anneal") and len(results["model_cost_no_anneal"]) > i:
                 c = results['model_cost_no_anneal'][-1]
                 msg += f" ModelNA={c:.4f}"
                 if opt_cost is not None and opt_cost > 1e-6:
                     g = (c - opt_cost) / opt_cost * 100
                     msg += f" ({g:.2f}%)"

             if results.get("mix_cost") and len(results["mix_cost"]) > i: 
                 c = results['mix_cost'][-1]
                 msg += f" Mix={c:.4f}"
                 if opt_cost is not None and opt_cost > 1e-6:
                     g = (c - opt_cost) / opt_cost * 100
                     msg += f" ({g:.2f}%)"

             if results.get("mix_cost_no_anneal") and len(results["mix_cost_no_anneal"]) > i:
                 c = results['mix_cost_no_anneal'][-1]
                 msg += f" MixNA={c:.4f}"
                 if opt_cost is not None and opt_cost > 1e-6:
                     g = (c - opt_cost) / opt_cost * 100
                     msg += f" ({g:.2f}%)"

             tqdm.write(msg)

        # Per-instance CSV logging (always capture if --log)
        if args.log:
            bl_i = float(baseline_values[i]) if (baseline_values is not None and len(baseline_values) > i) else None
            per_instance_rows.append({
                "idx": i,
                "name": name,
                "opt": (float(opt_cost) if opt_cost is not None else None),
                "baseline": bl_i,
                "base": (float(base_best) if base_best is not None else None),
                "model_anneal": (float(model_best) if model_best is not None else None),
                "model_no_anneal": (float(model_best_na) if model_best_na is not None else None),
                "mix_anneal": (float(mix_best) if mix_best is not None else None),
                "mix_no_anneal": (float(mix_best_na) if mix_best_na is not None else None),
            })

    if iter_csv_f is not None:
        try:
            iter_csv_f.close()
        except Exception:
            pass

    print("\n--- Results ---")

    def _mean_std(xs):
        if xs is None or len(xs) == 0:
            return None, None
        arr = np.array(xs, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    def _fmt(x, nd=4, empty="-"):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return empty
        return f"{x:.{nd}f}"

    def _print_table(rows, headers):
        # rows: list[list[str]]
        col_widths = [len(h) for h in headers]
        for r in rows:
            for j, cell in enumerate(r):
                col_widths[j] = max(col_widths[j], len(str(cell)))

        def _line(sep="-"):
            return "+" + "+".join(sep * (w + 2) for w in col_widths) + "+"

        print(_line("-"))
        print("| " + " | ".join(h.ljust(col_widths[j]) for j, h in enumerate(headers)) + " |")
        print(_line("="))
        for r in rows:
            print("| " + " | ".join(str(r[j]).ljust(col_widths[j]) for j in range(len(headers))) + " |")
        print(_line("-"))

    # Save base cache if we computed it this run
    if (not args.no_baseline) and (cached_base_costs is None) and base_cache_path is not None:
        try:
            torch.save({
                "base_costs": list(results["base_cost"]),
                "meta": {
                    "problem": args.problem,
                    "n_instances": len(results["base_cost"]),
                    "created": datetime.now().isoformat(),
                },
            }, base_cache_path)
            print(f"Saved cached base costs: {base_cache_path}")
        except Exception as e:
            print(f"Warning: failed to save base cache {base_cache_path}: {e}")
    
    # Base
    if not args.no_baseline and results["base_cost"]:
        base_cost_mean = np.mean(results["base_cost"])
        base_time_mean = np.mean(results["base_time"])
        base_time_total = np.sum(results["base_time"])
        print(f"Base Cost: {base_cost_mean:.4f}, Time: {base_time_mean:.4f}s, Total Time: {base_time_total:.4f}s")
        if results["base_gap"]:
            print(f"Base Gap: {np.mean(results['base_gap']) * 100:.4f}%")
        
    # Model
    if model and results.get("model_cost") and len(results["model_cost"]) > 0:
        model_cost_mean = np.mean(results["model_cost"])
        model_cost_na_mean = np.mean(results["model_cost_no_anneal"]) if results.get("model_cost_no_anneal") and len(results["model_cost_no_anneal"]) > 0 else float('nan')
        model_time_mean = np.mean(results["model_time"])
        model_time_total = np.sum(results["model_time"])
        model_time_na_mean = np.mean(results["model_time_no_anneal"]) if results.get("model_time_no_anneal") and len(results["model_time_no_anneal"]) > 0 else float('nan')
        model_time_na_total = np.sum(results["model_time_no_anneal"]) if results.get("model_time_no_anneal") and len(results["model_time_no_anneal"]) > 0 else float('nan')
        print(f"Model Cost (anneal): {model_cost_mean:.4f}, Time: {model_time_mean:.4f}s, Total Time: {model_time_total:.4f}s")
        if results.get("model_cost_no_anneal") and len(results["model_cost_no_anneal"]) > 0:
            print(f"Model Cost (no_anneal): {model_cost_na_mean:.4f}, Time: {model_time_na_mean:.4f}s, Total Time: {model_time_na_total:.4f}s")
        if results.get("model_gap") and len(results["model_gap"]) > 0:
            print(f"Model Gap (anneal): {np.mean(results['model_gap']) * 100:.4f}%")
        if results.get("model_gap_no_anneal") and len(results["model_gap_no_anneal"]) > 0:
            print(f"Model Gap (no_anneal): {np.mean(results['model_gap_no_anneal']) * 100:.4f}%")
        
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
            
    if model and args.warmup and results.get("mix_cost") and len(results["mix_cost"]) > 0:
         mix_cost_mean = np.mean(results["mix_cost"])
         mix_cost_na_mean = np.mean(results["mix_cost_no_anneal"]) if results.get("mix_cost_no_anneal") and len(results["mix_cost_no_anneal"]) > 0 else float('nan')
         mix_time_mean = np.mean(results["mix_time"])
         mix_time_total = np.sum(results["mix_time"])
         mix_time_na_mean = np.mean(results["mix_time_no_anneal"]) if results.get("mix_time_no_anneal") and len(results["mix_time_no_anneal"]) > 0 else float('nan')
         mix_time_na_total = np.sum(results["mix_time_no_anneal"]) if results.get("mix_time_no_anneal") and len(results["mix_time_no_anneal"]) > 0 else float('nan')
         print(f"Mix Cost (anneal): {mix_cost_mean:.4f}, Time: {mix_time_mean:.4f}s, Total Time: {mix_time_total:.4f}s")
         if results.get("mix_cost_no_anneal") and len(results["mix_cost_no_anneal"]) > 0:
             print(f"Mix Cost (no_anneal): {mix_cost_na_mean:.4f}, Time: {mix_time_na_mean:.4f}s, Total Time: {mix_time_na_total:.4f}s")
         if results.get("mix_gap") and len(results["mix_gap"]) > 0:
             print(f"Mix Gap (anneal): {np.mean(results['mix_gap']) * 100:.4f}%")
         if results.get("mix_gap_no_anneal") and len(results["mix_gap_no_anneal"]) > 0:
             print(f"Mix Gap (no_anneal): {np.mean(results['mix_gap_no_anneal']) * 100:.4f}%")

    if baseline_values is not None:
         print(f"Baseline Cost: {baseline_values.mean():.4f}")
         # Gap to baseline
         if not args.no_baseline and results.get("base_cost") and len(results["base_cost"]) > 0:
             base_gap_bl = (np.mean(results["base_cost"]) - baseline_values.mean()) / baseline_values.mean()
             print(f"Base Gap to Baseline: {base_gap_bl * 100:.4f}%")
         
         if model and results.get("model_cost") and len(results["model_cost"]) > 0:
             mod_gap_bl = (np.mean(results["model_cost"]) - baseline_values.mean()) / baseline_values.mean()
             print(f"Model Gap to Baseline (anneal): {mod_gap_bl * 100:.4f}%")
             if results.get("model_cost_no_anneal") and len(results["model_cost_no_anneal"]) > 0:
                 mod_gap_bl_na = (np.mean(results["model_cost_no_anneal"]) - baseline_values.mean()) / baseline_values.mean()
                 print(f"Model Gap to Baseline (no_anneal): {mod_gap_bl_na * 100:.4f}%")
             
             if args.warmup:
                  mix_gap_bl = (np.mean(results["mix_cost"]) - baseline_values.mean()) / baseline_values.mean()
                  print(f"Mix Gap to Baseline (anneal): {mix_gap_bl * 100:.4f}%")
                  if results["mix_cost_no_anneal"]:
                      mix_gap_bl_na = (np.mean(results["mix_cost_no_anneal"]) - baseline_values.mean()) / baseline_values.mean()
                      print(f"Mix Gap to Baseline (no_anneal): {mix_gap_bl_na * 100:.4f}%")

    # Summary table (best = lowest mean cost)
    summary_rows = []
    candidates = []  # (mean_cost, method_name)

    # Optimal row (if available). Never considered for BEST.
    if opt_costs_summary:
        opt_mean, opt_std = _mean_std(opt_costs_summary)
        summary_rows.append([
            "Opt", _fmt(opt_mean), _fmt(opt_std), "0.00", "0.00", "opt", "-", ""
        ])

    # Baseline solver row (if available)
    if baseline_values is not None:
        bl_mean, bl_std = _mean_std(baseline_values)
        summary_rows.append([
            "Baseline", _fmt(bl_mean), _fmt(bl_std), "-", "-", "-", "-", ""
        ])

    def _mean_gap_to_baseline(cost_list, baseline_arr):
        if baseline_arr is None or cost_list is None:
            return None
        if len(cost_list) == 0 or len(baseline_arr) == 0:
            return None
        if len(cost_list) != len(baseline_arr):
            return None
        c = np.array(cost_list, dtype=float)
        b = np.array(baseline_arr, dtype=float)
        ok = np.isfinite(c) & np.isfinite(b) & (b > 1e-12)
        if not ok.any():
            return None
        gaps = (c[ok] - b[ok]) / b[ok]
        return gaps

    def _add_method_row(name, cost_list, time_list, gap_list=None):
        m, s = _mean_std(cost_list)
        tm, _ = _mean_std(time_list)

        # Prefer gap to Opt (from txt dataset optimal values) when available.
        gap_ref = "-"
        gap_val = None
        gap_std = None
        
        if gap_list is not None and len(gap_list) > 0:
            gap_arr = np.array(gap_list) * 100.0
            gap_val = float(np.mean(gap_arr))
            gap_std = float(np.std(gap_arr))
            gap_ref = "opt"
        else:
            gaps = _mean_gap_to_baseline(cost_list, baseline_values)
            if gaps is not None:
                gap_arr = gaps * 100.0
                gap_val = float(np.mean(gap_arr))
                gap_std = float(np.std(gap_arr))
                gap_ref = "bl"

        summary_rows.append([
            name,
            _fmt(m),
            _fmt(s),
            _fmt(gap_val, nd=2),
            _fmt(gap_std, nd=2),
            gap_ref,
            _fmt(tm),
            "",
        ])
        if m is not None:
            candidates.append((m, name))

    if (not args.no_baseline) and results.get("base_cost"):
        _add_method_row("Base", results["base_cost"], results.get("base_time", []), results.get("base_gap", []))

    if model:
        # Only add rows for methods that were actually executed (non-empty results)
        if results.get("model_cost") and len(results["model_cost"]) > 0:
            _add_method_row("Model(anneal)", results["model_cost"], results.get("model_time", []), results.get("model_gap", []))

        if results.get("model_cost_no_anneal") and len(results["model_cost_no_anneal"]) > 0:
            _add_method_row("Model(no_anneal)", results["model_cost_no_anneal"], results.get("model_time_no_anneal", []), results.get("model_gap_no_anneal", []))

        if args.warmup:
            if results.get("mix_cost") and len(results["mix_cost"]) > 0:
                _add_method_row("Mix(anneal)", results["mix_cost"], results.get("mix_time", []), results.get("mix_gap", []))

            if results.get("mix_cost_no_anneal") and len(results["mix_cost_no_anneal"]) > 0:
                _add_method_row("Mix(no_anneal)", results["mix_cost_no_anneal"], results.get("mix_time_no_anneal", []), results.get("mix_gap_no_anneal", []))

    if summary_rows:
        best_name = min(candidates, key=lambda x: x[0])[1] if candidates else None
        if best_name is not None:
            for r in summary_rows:
                if r[0] == best_name:
                    r[-1] = "BEST"
                    break

        print("\n--- Summary (mean over instances) ---")
        _print_table(
            summary_rows,
            headers=["Method", "MeanCost", "StdCost", "Gap%", "StdGap%", "GapRef", "MeanTime", "Best"],
        )

        # CSV summary logging
        if args.log and csv_path_summary is not None:
            try:
                with open(csv_path_summary, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["problem", args.problem])
                    writer.writerow(["n_node", args.n_node])
                    writer.writerow(["checkpoint", args.checkpoint])
                    writer.writerow(["dataset", args.dataset])
                    writer.writerow(["seed", args.seed])
                    writer.writerow([])
                    writer.writerow(["Method", "MeanCost", "StdCost", "Gap%", "StdGap%", "GapRef", "MeanTime", "Best"])
                    for r in summary_rows:
                        writer.writerow(r)
            except Exception as e:
                print(f"Warning: failed to write CSV summary: {e}")

    # CSV instances logging (write once at end)
    if args.log and csv_path_instances is not None and per_instance_rows:
        try:
            fieldnames = [
                "idx",
                "name",
                "opt",
                "baseline",
                "base",
                "model_anneal",
                "model_no_anneal",
                "mix_anneal",
                "mix_no_anneal",
            ]
            with open(csv_path_instances, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_instance_rows)
        except Exception as e:
            print(f"Warning: failed to write CSV instances: {e}")

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
