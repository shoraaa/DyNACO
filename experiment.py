#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any

# =============================================================================
# Configuration & Tables
# =============================================================================

# Default Hyperparameters from Paper
DEFAULT_CONFIG = {
    "H": 10,
    "mini_H": 100,  # S in paper
    "n_ants": 100,
    "k_sparse": 32,
    "epochs": 10,           # Standard training length
    "steps_per_epoch": 32,  
    "ppo_epochs": 4,
    "lr": 5e-6,
    "anneal_prior": False,
    "gamma": 1.0,
    "min_gamma": 0.2,
    "save_dir": "experiments_checkpoints"
}

# Problem specific defaults
TSP_CONFIG = {
    "problem": "tsp",
    "rho": 0.1,
    "min_new_edges": 12,
}

CVRP_CONFIG = {
    "problem": "cvrp",
    "rho": 0.1,
    "min_new_edges": 12,
}

# =============================================================================
# Helper Functions
# =============================================================================

def run_command(cmd: List[str], log_file: Path = None, dry_run: bool = False):
    """Executes a shell command."""
    cmd_str = " ".join(cmd)
    print(f"[CMD] {cmd_str}")
    
    if dry_run:
        return 0, "DRY_RUN", ""

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = ""
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                output += line
            process.wait()
            return process.returncode, output, cmd_str
            
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"[ERROR] Command failed with code {result.returncode}")
            print(result.stderr)
        return result.returncode, result.stdout, cmd_str

def parse_metrics(output: str):
    """Parses output from test.py to find Cost, Time, Gap."""
    # Look for "Model Gap: X.XXXX%" or "Gap: X.XX%"
    gap_match = re.search(r"Model Gap: ([-+]?\d*\.\d+|\d+)%", output)
    if not gap_match:
         gap_match = re.search(r"Gap.*: ([-+]?\d*\.\d+|\d+)%", output)
         
    time_match = re.search(r"Time: ([-+]?\d*\.\d+|\d+)s", output)
    
    gap = float(gap_match.group(1)) if gap_match else None
    time_val = float(time_match.group(1)) if time_match else None
    
    return gap, time_val

def get_model_path(config: Dict[str, Any], suffix: str = "_best.pt") -> Path:
    """Constructs model path based on config."""
    # model_name logic from train.py:
    # f"{args.problem}_n{args.n_node}_k{args.k_sparse}_ants{args.n_ants}_H{args.H}_miniH{args.mini_H}_rho{args.rho}_mne{args.min_new_edges}_{args.algo}"
    # algo is ppo by default
    
    algo = config.get("algo", "ppo")
    name = f"{config['problem']}_n{config['n_node']}_k{config['k_sparse']}_ants{config['n_ants']}_H{config['H']}_miniH{config['mini_H']}_rho{config['rho']}_mne{config['min_new_edges']}_{algo}"
    
    if config.get("anneal_prior", False):
        name += f"_anneal_g{config['gamma']}_mg{config['min_gamma']}"
    
    # L arg? Not in default dict but maybe present
    if config.get("L", 0) > 0:
        name += f"_L{config['L']}"
        
    save_dir = Path(config.get("save_dir", "experiments_checkpoints")) / config["problem"]
    return save_dir / (name + suffix)

def train_model(config: Dict[str, Any], dry_run: bool = False, force: bool = False):
    """Runs training if checkpoint doesn't exist."""
    model_path = get_model_path(config)
    if model_path.exists() and not force:
        print(f"[SKIP] Model exists: {model_path}")
        return model_path

    cmd = ["python3", "train.py"]
    
    # Flags mapping
    # Boolean flags need action
    bool_flags = ["anneal_prior", "no_dynamic_feats", "disable_heuristic", "no_local_search", "no_smooth_mmas"]
    
    for k, v in config.items():
        if k in ["save_dir"]: # Handled separately or passed? train.py takes save_dir
             cmd.extend(["--save_dir", str(v)])
             continue
             
        if k in bool_flags:
            if v is True:
                # If key is positive (anneal_prior) and True -> --anneal_prior
                # If key is negative (no_local_search) and True -> --no_local_search
                 cmd.append(f"--{k}")
            # If False, usually default, so do nothing (checks default in train.py)
            continue
            
        cmd.extend([f"--{k}", str(v)])
        
    if dry_run:
        cmd.extend(["--epochs", "1", "--steps_per_epoch", "1"])
        
    log_file = Path("logs") / f"train_{model_path.stem}.log"
    run_command(cmd, log_file, dry_run)
    return model_path

def test_model(model_path: Path, config: Dict[str, Any], dry_run: bool = False):
    """Runs evaluation."""
    cmd = ["python3", "test.py"]
    cmd.extend(["--problem", config["problem"]])
    cmd.extend(["--n_node", str(config["n_node"])])
    cmd.extend(["--checkpoint", str(model_path)])
    
    # Some args need to be passed to test (like H, mini_H) if they are not saved/loaded correctly or to ensure test consistency
    # train.py saves config, but test.py logic loads it.
    # We can pass them to be safe.
    for k in ["H", "mini_H", "n_ants", "k_sparse", "rho", "min_new_edges"]:
        if k in config:
             cmd.extend([f"--{k}", str(config[k])])
             
    if dry_run:
        cmd.extend(["--n_node", "20", "--n_ants", "10"]) # Smaller scale for dry run validation
        
    log_file = Path("logs") / f"test_{model_path.stem}.log"
    code, output, _ = run_command(cmd, log_file, dry_run)
    return parse_metrics(output)


# =============================================================================
# Experiments
# =============================================================================

def run_tsp_experiments(sizes=[1000, 5000, 10000], dry_run=False):
    print("\n=== Running TSP Experiments (Table 2) ===")
    results = {}
    
    for n in sizes:
        print(f"\n--- TSP N={n} ---")
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(TSP_CONFIG)
        cfg["n_node"] = n
        
        # Training
        # Note: Large sizes usually train on smaller sizes or same size?
        # Paper implies training on corresponding size or generating on fly.
        # We will train on N if N <= 1000, else maybe train on 1000? 
        # "Validation uses fixed datasets... Training data is generated on-the-fly"
        # Usually one model per size, or size generalization.
        # Let's assume training on N for now, or 1000 for all?
        # Table 4 in paper (not provided but typical) shows generalization. 
        # But Table 2 shows results for each size. DeepACO trains on same size usually.
        # Let's train on N.
        
        # For very large N (50K), training might be too slow on single GPU if full epochs.
        # For reproduction, let's train on 1000? Or assume sizes.
        # Re-reading paper introduction/experiments:
        # "Training data is generated on-the-fly ... Validation uses fixed datasets"
        # It doesn't explicitly say "Trained on N".
        # But section "Zero-shot size generalization" implies models are trained on specific sizes.
        # We will train on N for 1K. For 5K+, maybe train on N too if feasible.
        
        if n > 1000 and not dry_run:
             print(f"Warning: Training on N={n} might be slow. Consider using N=1000 model for zero-shot if supported.")
        
        model_path = train_model(cfg, dry_run)
        gap, t_val = test_model(model_path, cfg, dry_run)
        results[n] = {"gap": gap, "time": t_val}
        print(f"TSP N={n}: Gap={gap}%, Time={t_val}s")
        
    return results

def run_cvrp_experiments(sizes=[1000, 5000], dry_run=False):
    print("\n=== Running CVRP Experiments (Table 3) ===")
    results = {}
    
    for n in sizes:
        print(f"\n--- CVRP N={n} ---")
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(CVRP_CONFIG)
        cfg["n_node"] = n
        
        model_path = train_model(cfg, dry_run)
        gap, t_val = test_model(model_path, cfg, dry_run)
        results[n] = {"gap": gap, "time": t_val}
        print(f"CVRP N={n}: Gap={gap}%, Time={t_val}s")
        
    return results

def run_ablation_refresh(dry_run=False):
    print("\n=== Running Ablation: Refresh Frequency (Table 5) ===")
    # N=1000 TSP
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    # H values to test. Fixed T = H*S = 1000
    h_s_pairs = [(1, 1000), (5, 200), (10, 100), (20, 50), (50, 20)]
    
    results = {}
    
    for H, S in h_s_pairs:
        print(f"\n--- Ablation H={H}, S={S} ---")
        cfg = base_cfg.copy()
        cfg["H"] = H
        cfg["mini_H"] = S
        
        model_path = train_model(cfg, dry_run)
        gap, t_val = test_model(model_path, cfg, dry_run)
        results[H] = {"gap": gap, "time": t_val}
        print(f"H={H}: Gap={gap}%, Time={t_val}s")
        
    return results

def run_ablation_features(dry_run=False):
    print("\n=== Running Ablation: Features (Table 6) ===")
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    # Feature settings
    # 1. Full (Default)
    # 2. No Dynamic Feats (Static Only) -> --no_dynamic_feats
    
    results = {}
    
    # Full
    print("\n--- Full Features ---")
    model_path = train_model(base_cfg, dry_run)
    gap, _ = test_model(model_path, base_cfg, dry_run)
    results["Full"] = gap
    
    # Static Only
    print("\n--- Static Only ---")
    cfg_static = base_cfg.copy()
    cfg_static["no_dynamic_feats"] = True
    model_path = train_model(cfg_static, dry_run)
    gap, _ = test_model(model_path, cfg_static, dry_run)
    results["Static"] = gap
    
    return results

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce experiments from LGA paper")
    parser.add_argument("--table", type=str, choices=["2", "3", "5", "6", "all"], default="all", help="Which table to reproduce")
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal steps to verify pipeline")
    parser.add_argument("--fast", action="store_true", help="Skip large instances")
    parser.add_argument("--sizes", type=int, nargs="+", help="Specific sizes to run")
    
    args = parser.parse_args()
    
    if args.table in ["2", "all"]:
        sizes = args.sizes if args.sizes else ([1000, 5000, 10000] if not args.fast else [1000])
        run_tsp_experiments(sizes, args.dry_run)
        
    if args.table in ["3", "all"]:
        sizes = args.sizes if args.sizes else ([1000, 5000] if not args.fast else [1000])
        run_cvrp_experiments(sizes, args.dry_run)
        
    if args.table in ["5", "all"]:
        run_ablation_refresh(args.dry_run)
        
    if args.table in ["6", "all"]:
        run_ablation_features(args.dry_run)

if __name__ == "__main__":
    main()
