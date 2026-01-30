#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import time
import re
import json
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

PROGRESS_FILE = "experiment_progress.json"

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

def load_progress() -> Dict[str, Any]:
    """Loads experiment progress from JSON file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[WARNING] Could not decode {PROGRESS_FILE}. Starting fresh.")
            return {}
    return {}

def save_progress(key: str, data: Any):
    """Saves a single experiment result to the progress file."""
    progress = load_progress()
    progress[key] = data
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=4)
    print(f"[PROGRESS] Saved result for '{key}'")

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

def train_model(config: Dict[str, Any], dry_run: bool = False, force: bool = False, wandb_project: str = "lga"):
    """Runs training if checkpoint doesn't exist."""
    model_path = get_model_path(config)
    if model_path.exists() and not force:
        print(f"[SKIP] Model exists: {model_path}")
        return model_path

    cmd = ["python3", "train.py"]
    cmd.extend(["--wandb_project", wandb_project])
    
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
    results = load_progress()
    
    for n in sizes:
        key = f"tsp_n{n}"
        if key in results:
            print(f"[SKIP] {key} already completed: {results[key]}")
            continue

        print(f"\n--- TSP N={n} ---")
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(TSP_CONFIG)
        cfg["n_node"] = n
        
        if n > 1000 and not dry_run:
             print(f"Warning: Training on N={n} might be slow. Consider using N=1000 model for zero-shot if supported.")
        
        model_path = train_model(cfg, dry_run, wandb_project="lga_tsp")
        gap, t_val = test_model(model_path, cfg, dry_run)
        
        result_data = {"gap": gap, "time": t_val}
        save_progress(key, result_data)
        results[key] = result_data # Update local dict mostly for return if needed
        print(f"TSP N={n}: Gap={gap}%, Time={t_val}s")
        
    return results

def run_cvrp_experiments(sizes=[1000, 5000], dry_run=False):
    print("\n=== Running CVRP Experiments (Table 3) ===")
    results = load_progress()
    
    for n in sizes:
        key = f"cvrp_n{n}"
        if key in results:
            print(f"[SKIP] {key} already completed: {results[key]}")
            continue

        print(f"\n--- CVRP N={n} ---")
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(CVRP_CONFIG)
        cfg["n_node"] = n
        
        model_path = train_model(cfg, dry_run, wandb_project="lga_cvrp")
        gap, t_val = test_model(model_path, cfg, dry_run)
        
        result_data = {"gap": gap, "time": t_val}
        save_progress(key, result_data)
        results[key] = result_data
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
    
    results = load_progress()
    
    for H, S in h_s_pairs:
        # Check for default config reuse (H=10, S=100)
        if H == 10 and S == 100:
             default_key = "tsp_n1000"
             if default_key in results:
                 print(f"[REUSE] Using {default_key} for H=10, S=100")
                 results[f"ablation_refresh_H{H}_S{S}"] = results[default_key]
                 continue
             else:
                 # Train as default if not found? Or just proceed as normal but map it?
                 # Better to train as tsp_n1000 so others can reuse it.
                 print(f"[INFO] H=10, S=100 matches Default. Using/Training '{default_key}'...")
                 # Temporarily switch key
                 key = default_key
                 # Proceed to train/test with this key, then map result back
                 
        key = f"ablation_refresh_H{H}_S{S}" if not (H==10 and S==100) else "tsp_n1000"
        
        if key in results:
             print(f"[SKIP] {key} already completed: {results[key]}")
             if key == "tsp_n1000": results[f"ablation_refresh_H10_S100"] = results[key]
             continue

        print(f"\n--- Ablation H={H}, S={S} ---")
        cfg = base_cfg.copy()
        cfg["H"] = H
        cfg["mini_H"] = S
        
        proj = "lga_ablation_refresh"
        if key == "tsp_n1000": proj = "lga_tsp"
        
        model_path = train_model(cfg, dry_run, wandb_project=proj)
        gap, t_val = test_model(model_path, cfg, dry_run)
        
        result_data = {"gap": gap, "time": t_val}
        save_progress(key, result_data)
        results[key] = result_data
        if key == "tsp_n1000": results[f"ablation_refresh_H10_S100"] = result_data
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
    
    results = load_progress()
    
    # Full
    key_full = "ablation_features_full"
    default_key = "tsp_n1000"
    
    # Check reuse
    if default_key in results:
         print(f"[REUSE] Using {default_key} for Full Features")
         results[key_full] = results[default_key]
         results["Full"] = results[default_key]["gap"]
    elif key_full in results: # Fallback if run independently previously
         print(f"[SKIP] {key_full} already completed: {results[key_full]}")
         results["Full"] = results[key_full]["gap"]
    else:
        # Run as tsp_n1000 preference
        print(f"[INFO] Full Features matches Default. Using/Training '{default_key}'...")
        if default_key in results: # Double check
             pass 
        else:
             print("\n--- Full Features (Default) ---")
             model_path = train_model(base_cfg, dry_run, wandb_project="lga_tsp")
             gap, t_val = test_model(model_path, base_cfg, dry_run)
             result_data = {"gap": gap, "time": t_val}
             save_progress(default_key, result_data)
             results[default_key] = result_data
             
        results[key_full] = results[default_key]
        results["Full"] = results[default_key]["gap"]
    
    # Static Only
    key_static = "ablation_features_static"
    if key_static in results:
        print(f"[SKIP] {key_static} already completed: {results[key_static]}")
        results["Static"] = results[key_static]["gap"]
    else:
        print("\n--- Static Only ---")
        cfg_static = base_cfg.copy()
        cfg_static["no_dynamic_feats"] = True
        model_path = train_model(cfg_static, dry_run, wandb_project="lga_ablation_features")
        gap, t_val = test_model(model_path, cfg_static, dry_run)
        
        result_data = {"gap": gap, "time": t_val}
        save_progress(key_static, result_data)
        results["Static"] = gap
    
    return results

def run_ablation_smoothing(dry_run=False):
    print("\n=== Running Ablation: Smoothing (Table 8) ===")
    n = 1000  # Based on context usually N=1000
    base_cfg = DEFAULT_CONFIG.copy()
    base_cfg.update(TSP_CONFIG) # Assuming TSP for ablation unless specified
    base_cfg["n_node"] = n
    
    results = load_progress()
    
    
    # Default (Smooth MMAS)
    key_smooth = "ablation_smoothing_on"
    default_key = "tsp_n1000"

    if default_key in results:
         print(f"[REUSE] Using {default_key} for Smoothing ON")
         results[key_smooth] = results[default_key]
    elif key_smooth in results:
         print(f"[SKIP] {key_smooth} already completed: {results[key_smooth]}")
    else:
        print(f"[INFO] Smoothing ON matches Default. Using/Training '{default_key}'...")
        print("\n--- Smoothing ON (Default) ---")
        model_path = train_model(base_cfg, dry_run, wandb_project="lga_tsp")
        gap, t_val = test_model(model_path, base_cfg, dry_run)
        
        result_data = {"gap": gap, "time": t_val}
        save_progress(default_key, result_data)
        results[key_smooth] = result_data

    # No Smooth MMAS
    key_no_smooth = "ablation_smoothing_off"
    if key_no_smooth in results:
        print(f"[SKIP] {key_no_smooth} already completed: {results[key_no_smooth]}")
    else:
        print("\n--- Smoothing OFF ---")
        cfg_ns = base_cfg.copy()
        cfg_ns["no_smooth_mmas"] = True
        model_path = train_model(cfg_ns, dry_run, wandb_project="lga_ablation_smoothing")
        gap, t_val = test_model(model_path, cfg_ns, dry_run)
        
        result_data = {"gap": gap, "time": t_val}
        save_progress(key_no_smooth, result_data)

def run_ablation_heuristic(dry_run=False):
    print("\n=== Running Ablation: Heuristic (Future) ===")
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    results = load_progress()
    
    # Heuristic ON (Default)
    key_heu_on = "ablation_heuristic_on"
    default_key = "tsp_n1000"
    
    if default_key in results:
         print(f"[REUSE] Using {default_key} for Heuristic ON")
         results[key_heu_on] = results[default_key]
    elif key_heu_on in results:
         print(f"[SKIP] {key_heu_on} already completed: {results[key_heu_on]}")
    else:
        print(f"[INFO] Heuristic ON matches Default. Using/Training '{default_key}'...")
        print("\n--- Heuristic ON (Default) ---")
        model_path = train_model(base_cfg, dry_run, wandb_project="lga_tsp")
        gap, t_val = test_model(model_path, base_cfg, dry_run)
        save_progress(default_key, {"gap": gap, "time": t_val})
        results[key_heu_on] = {"gap": gap, "time": t_val}

    # Heuristic OFF
    key_heu_off = "ablation_heuristic_off"
    if key_heu_off in results:
        print(f"[SKIP] {key_heu_off} already completed: {results[key_heu_off]}")
    else:
        print("\n--- Heuristic OFF ---")
        cfg_nh = base_cfg.copy()
        cfg_nh["disable_heuristic"] = True
        model_path = train_model(cfg_nh, dry_run, wandb_project="lga_ablation_heuristic")
        gap, t_val = test_model(model_path, cfg_nh, dry_run)
        save_progress(key_heu_off, {"gap": gap, "time": t_val})

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce experiments from LGA paper")
    parser.add_argument("--table", type=str, choices=["2", "3", "5", "6", "8", "heuristic", "all"], default="all", help="Which table to reproduce")
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal steps to verify pipeline")
    parser.add_argument("--fast", action="store_true", help="Skip large instances")
    parser.add_argument("--sizes", type=int, nargs="+", help="Specific sizes to run")
    
    args = parser.parse_args()
    
    # if args.table in ["2", "all"]:
    #     sizes = args.sizes if args.sizes else ([1000, 5000, 10000] if not args.fast else [1000])
    #     run_tsp_experiments(sizes, args.dry_run)
        
    # if args.table in ["3", "all"]:
    #     sizes = args.sizes if args.sizes else ([1000, 5000] if not args.fast else [1000])
    #     run_cvrp_experiments(sizes, args.dry_run)
        
    if args.table in ["5", "all"]:
        run_ablation_refresh(args.dry_run)
        
    if args.table in ["6", "all"]:
        run_ablation_features(args.dry_run)
        
    if args.table in ["8", "all"]:
        run_ablation_smoothing(args.dry_run)
        
    if args.table in ["heuristic", "all"]:
        run_ablation_heuristic(args.dry_run)

if __name__ == "__main__":
    main()
