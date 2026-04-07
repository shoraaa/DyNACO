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
    "val_size": 16,
    "warmup": True,
    "train_warmup": False,
    "save_dir": "pretrained"
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
                data = json.load(f)
                # Filter out null entries just in case
                return {k: v for k, v in data.items() if v is not None and v.get("gap") is not None}
        except json.JSONDecodeError:
            print(f"[WARNING] Could not decode {PROGRESS_FILE}. Starting fresh.")
            return {}
    return {}

def save_progress(key: str, data: Any):
    """Saves a single experiment result to the progress file."""
    if data is None or data.get("gap") is None:
        print(f"[PROGRESS] Skipping save for '{key}' (No data)")
        return

    progress = load_progress()
    progress[key] = data
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=4)
    print(f"[PROGRESS] Saved result for '{key}'")

def parse_metrics(output: str):
    """Parses output from test.py to find Cost, Time, Gap."""
    # Look for "Model Cost: ..., Time: ...s" to get specific model stats
    model_stats = re.search(r"Model Cost: ([-+]?\d*\.\d+|\d+).*?Time: ([-+]?\d*\.\d+|\d+)s", output)
    
    # Model Gap
    gap_match = re.search(r"Model Gap: ([-+]?\d*\.\d+|\d+)%", output)
    
    cost = float(model_stats.group(1)) if model_stats else None
    time_val = float(model_stats.group(2)) if model_stats else None
    gap = float(gap_match.group(1)) if gap_match else None
    
    return {"gap": gap, "time": time_val, "cost": cost}

def get_model_path(config: Dict[str, Any], suffix: str = "_best.pt") -> Path:
    """Constructs model path based on config."""
    # model_name logic from train.py:
    # f"{args.problem}_n{args.n_node}_k{args.k_sparse}_ants{args.n_ants}_H{args.H}_miniH{args.mini_H}_rho{args.rho}_mne{args.min_new_edges}_{args.algo}"
    # algo is ppo by default
    
    algo = config.get("algo", "ppo")
    # Match train.py: args.lr is used in filename.
    # train.py defaults lr to ppo_lr (5e-6) if not set. DEFAULT_CONFIG has lr=5e-6.
    lr = config.get("lr", 5e-6)
    name = f"{config['problem']}_n{config['n_node']}_k{config['k_sparse']}_ants{config['n_ants']}_H{config['H']}_miniH{config['mini_H']}_rho{config['rho']}_mne{config['min_new_edges']}_{algo}_lr{lr}"
    
    if config.get("anneal_prior", False):
        name += f"_anneal_g{config['gamma']}_mg{config['min_gamma']}"
    
    # L arg? Not in default dict but maybe present
    if config.get("L", 0) > 0:
        name += f"_L{config['L']}"

    # Ablation suffixes
    if config.get("no_dynamic_feats"): name += "_static"
    if config.get("no_smooth_mmas"): name += "_nosmooth"
    if config.get("disable_heuristic"): name += "_noheu"
    if config.get("no_extend_ls"): name += "_noextls"
    if config.get("no_normalized_heuristic"): name += "_nonorm"
        
    save_dir = Path(config.get("save_dir", "pretrained")) / config["problem"] / f"n{config['n_node']}"
    return save_dir / (name + suffix)

def train_model(config: Dict[str, Any], dry_run: bool = False, force: bool = False, wandb_project: str = "lga", only_test: bool = False):
    """Runs training if checkpoint doesn't exist."""
    model_path = get_model_path(config)
    
    if only_test:
        if model_path.exists():
            print(f"[TEST-ONLY] Found model: {model_path}")
            return model_path
        else:
            print(f"[TEST-ONLY] Model not found, skipping: {model_path}")
            return None

    if model_path.exists() and not force:
        print(f"[SKIP] Model exists: {model_path}")
        return model_path

    cmd = [sys.executable, "train.py"]
    cmd.extend(["--wandb_project", wandb_project])
    
    # Flags mapping
    # Boolean flags need action
    bool_flags = ["anneal_prior", "no_dynamic_feats", "disable_heuristic", "no_local_search", "no_smooth_mmas", "train_warmup", "warmup", "no_extend_ls", "no_normalized_heuristic"]
    
    for k, v in config.items():
        if k in ["save_dir"]: # Handled separately or passed? train.py takes save_dir
             cmd.extend(["--save_dir", str(v)])
             continue
             
        if k in bool_flags:
            if v is True:
                 cmd.append(f"--{k}")
            elif k == "warmup" and v is False:
                 cmd.append("--no-warmup")
            continue
            
        cmd.extend([f"--{k}", str(v)])
        
    if dry_run:
        cmd.extend(["--epochs", "1", "--steps_per_epoch", "1"])
        
    log_file = Path("logs") / f"train_{model_path.stem}.log"
    run_command(cmd, log_file, dry_run)
    return model_path

def test_model(model_path: Path, config: Dict[str, Any], dry_run: bool = False):
    """Runs evaluation."""
    cmd = [sys.executable, "test.py"]
    cmd.extend(["--problem", config["problem"]])
    cmd.extend(["--n_node", str(config["n_node"])])
    cmd.extend(["--checkpoint", str(model_path)])
    
    # Some args need to be passed to test (like H, mini_H) if they are not saved/loaded correctly or to ensure test consistency
    # train.py saves config, but test.py logic loads it.
    # We can pass them to be safe.
    for k in ["H", "mini_H", "n_ants", "k_sparse", "rho", "min_new_edges", "warmup", "warmup_ratio", "val_size"]:
        if k in config:
             if k == "warmup":
                 if config[k] is False: cmd.append("--no-warmup")
                 # if True (default), do nothing or pass --warmup explicitly? Default is True.
             elif config[k] is True:
                 cmd.append(f"--{k}") # For boolean flags like train_warmup if passed roughly? But this lists params mostly.
             else:
                 cmd.extend([f"--{k}", str(config[k])])
    
    # Pass boolean flags for ablations
    # train_model defines these:
    bool_flags = ["anneal_prior", "no_dynamic_feats", "disable_heuristic", "no_local_search", "no_smooth_mmas", "no_extend_ls", "no_normalized_heuristic"]
    for k in bool_flags:
        if config.get(k, False):
            cmd.append(f"--{k}")
             
    if dry_run:
        cmd.extend(["--n_node", "20", "--n_ants", "10"]) # Smaller scale for dry run validation
             
    log_file = Path("logs") / f"test_{model_path.stem}.log"
    code, output, _ = run_command(cmd, log_file, dry_run)
    return parse_metrics(output)


# =============================================================================
# Experiments
# =============================================================================

def run_tsp_experiments(sizes=[1000, 5000, 10000], dry_run=False, only_test=False):
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
        
        if n > 1000 and not dry_run and not only_test:
             print(f"Warning: Training on N={n} might be slow. Consider using N=1000 model for zero-shot if supported.")
        
        model_path = train_model(cfg, dry_run, wandb_project="lga_tsp", only_test=only_test)
        model_path = train_model(cfg, dry_run, wandb_project="lga_tsp", only_test=only_test)
        if model_path is None: continue

        metrics = test_model(model_path, cfg, dry_run)
        
        save_progress(key, metrics)
        results[key] = metrics # Update local dict mostly for return if needed
        print(f"TSP N={n}: {metrics}")
        
    return results

def run_cvrp_experiments(sizes=[1000, 5000], dry_run=False, only_test=False):
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
        
        model_path = train_model(cfg, dry_run, wandb_project="lga_cvrp", only_test=only_test)
        model_path = train_model(cfg, dry_run, wandb_project="lga_cvrp", only_test=only_test)
        if model_path is None: continue

        metrics = test_model(model_path, cfg, dry_run)
        
        save_progress(key, metrics)
        results[key] = metrics
        print(f"CVRP N={n}: {metrics}")
        
    return results

def run_ablation_refresh(problem='tsp', dry_run=False, only_test=False):
    print(f"\n=== Running Ablation: Refresh Frequency (Table 5) [{problem}] ===")
    # N=1000
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
         base_cfg.update(CVRP_CONFIG)
    else:
         base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    # H values to test. Fixed T = H*S = 1000
    h_s_pairs = [(1, 1000), (5, 200), (10, 100)]
    
    results = load_progress()
    
    for H, S in h_s_pairs:
        # Check for default config reuse (H=10, S=100)
        if H == 10 and S == 100:
             default_key = f"{problem}_n{n}"
             if default_key in results:
                 print(f"[REUSE] Using {default_key} for H=10, S=100")
                 results[f"{problem}_ablation_refresh_H{H}_S{S}"] = results[default_key]
                 continue
             else:
                 # Train as default if not found? Or just proceed as normal but map it?
                 # Better to train as default so others can reuse it.
                 print(f"[INFO] H=10, S=100 matches Default. Using/Training '{default_key}'...")
                 # Temporarily switch key
                 key = default_key
                 # Proceed to train/test with this key, then map result back
                 
        key = f"{problem}_ablation_refresh_H{H}_S{S}" if not (H==10 and S==100) else f"{problem}_n{n}"
        
        if key in results:
             print(f"[SKIP] {key} already completed: {results[key]}")
             if key == f"{problem}_n{n}": results[f"{problem}_ablation_refresh_H10_S100"] = results[key]
             continue

        print(f"\n--- Ablation H={H}, S={S} ---")
        cfg = base_cfg.copy()
        cfg["H"] = H
        cfg["mini_H"] = S
        
        proj = f"lga_{problem}_ablation_refresh"
        if key == f"{problem}_n{n}": proj = f"lga_{problem}"
        
        model_path = train_model(cfg, dry_run, wandb_project=proj, only_test=only_test)
        if model_path is None: continue

        metrics = test_model(model_path, cfg, dry_run)
        
        save_progress(key, metrics)
        results[key] = metrics
        if key == "tsp_n1000": results[f"ablation_refresh_H10_S100"] = metrics
        print(f"H={H}: {metrics}")
        
    return results

def run_ablation_features(problem='tsp', dry_run=False, only_test=False):
    print(f"\n=== Running Ablation: Features (Table 6) [{problem}] ===")
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
         base_cfg.update(CVRP_CONFIG)
    else:
         base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    # Feature settings
    # 1. Full (Default)
    # 2. No Dynamic Feats (Static Only) -> --no_dynamic_feats
    
    results = load_progress()
    
    # Full
    key_full = f"{problem}_ablation_features_full"
    default_key = f"{problem}_n{n}"
    
    # Check reuse
    if default_key in results:
         print(f"[REUSE] Using {default_key} for Full Features")
         results[key_full] = results[default_key]
         results["Full"] = results[default_key].get("gap")
    elif key_full in results: # Fallback if run independently previously
         print(f"[SKIP] {key_full} already completed: {results[key_full]}")
         results["Full"] = results[key_full]["gap"]
    else:
        # Run as default preference
        print(f"[INFO] Full Features matches Default. Using/Training '{default_key}'...")
        if default_key in results: # Double check
             pass 
        else:
             print("\n--- Full Features (Default) ---")
             model_path = train_model(base_cfg, dry_run, wandb_project=f"lga_{problem}", only_test=only_test)
             if model_path:
                 metrics = test_model(model_path, base_cfg, dry_run)
                 save_progress(default_key, metrics)
                 results[default_key] = metrics
                 
                 results[key_full] = results[default_key]
                 results["Full"] = results[default_key].get("gap")
    
    # Static Only
    key_static = f"{problem}_ablation_features_static"
    if key_static in results:
        print(f"[SKIP] {key_static} already completed: {results[key_static]}")
        results["Static"] = results[key_static]["gap"]
    else:
        print("\n--- Static Only ---")
        cfg_static = base_cfg.copy()
        cfg_static["no_dynamic_feats"] = True
        model_path = train_model(cfg_static, dry_run, wandb_project=f"lga_{problem}_ablation_features", only_test=only_test)
        if model_path:
            metrics = test_model(model_path, cfg_static, dry_run)
            
            save_progress(key_static, metrics)
            results["Static"] = metrics.get("gap")
    
    return results

def run_ablation_smoothing(problem='tsp', dry_run=False, only_test=False):
    print(f"\n=== Running Ablation: Smoothing (Table 8) [{problem}] ===")
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
         base_cfg.update(CVRP_CONFIG)
    else:
         base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    results = load_progress()
    
    
    # Default (Smooth MMAS)
    key_smooth = f"{problem}_ablation_smoothing_on"
    default_key = f"{problem}_n{n}"

    if default_key in results:
         print(f"[REUSE] Using {default_key} for Smoothing ON")
         results[key_smooth] = results[default_key]
    elif key_smooth in results:
         print(f"[SKIP] {key_smooth} already completed: {results[key_smooth]}")
    else:
        print(f"[INFO] Smoothing ON matches Default. Using/Training '{default_key}'...")
        print("\n--- Smoothing ON (Default) ---")
        model_path = train_model(base_cfg, dry_run, wandb_project=f"lga_{problem}", only_test=only_test)
        if model_path:
            metrics = test_model(model_path, base_cfg, dry_run)
            
            save_progress(default_key, metrics)
            results[key_smooth] = metrics

    # No Smooth MMAS
    key_no_smooth = f"{problem}_ablation_smoothing_off"
    if key_no_smooth in results:
        print(f"[SKIP] {key_no_smooth} already completed: {results[key_no_smooth]}")
    else:
        print("\n--- Smoothing OFF ---")
        cfg_ns = base_cfg.copy()
        cfg_ns["no_smooth_mmas"] = True
        model_path = train_model(cfg_ns, dry_run, wandb_project=f"lga_{problem}_ablation_smoothing", only_test=only_test)
        if model_path:
            metrics = test_model(model_path, cfg_ns, dry_run)
            
            save_progress(key_no_smooth, metrics)

def run_ablation_heuristic(problem='tsp', dry_run=False, only_test=False):
    print(f"\n=== Running Ablation: Heuristic (Future) [{problem}] ===")
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
         base_cfg.update(CVRP_CONFIG)
    else:
         base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    results = load_progress()
    
    # Heuristic ON (Default)
    key_heu_on = f"{problem}_ablation_heuristic_on"
    default_key = f"{problem}_n{n}"
    
    if default_key in results:
         print(f"[REUSE] Using {default_key} for Heuristic ON")
         results[key_heu_on] = results[default_key]
    elif key_heu_on in results:
         print(f"[SKIP] {key_heu_on} already completed: {results[key_heu_on]}")
    else:
        print(f"[INFO] Heuristic ON matches Default. Using/Training '{default_key}'...")
        print("\n--- Heuristic ON (Default) ---")
        model_path = train_model(base_cfg, dry_run, wandb_project=f"lga_{problem}", only_test=only_test)
        if model_path:
            metrics = test_model(model_path, base_cfg, dry_run)
            save_progress(default_key, metrics)
            results[key_heu_on] = metrics

    # Heuristic OFF
    key_heu_off = f"{problem}_ablation_heuristic_off"
    if key_heu_off in results:
        print(f"[SKIP] {key_heu_off} already completed: {results[key_heu_off]}")
    else:
        print("\n--- Heuristic OFF ---")
        cfg_nh = base_cfg.copy()
        cfg_nh["disable_heuristic"] = True
        model_path = train_model(cfg_nh, dry_run, wandb_project=f"lga_{problem}_ablation_heuristic", only_test=only_test)
        if model_path:
            metrics = test_model(model_path, cfg_nh, dry_run)
            save_progress(key_heu_off, metrics)

def run_ablation_warmup(problem='tsp', dry_run=False, only_test=False):
    print(f"\n=== Running Ablation: Warmup Strategy (Table Future) [{problem}] ===")
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
         base_cfg.update(CVRP_CONFIG)
    else:
         base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    results = load_progress()
    
    # Scenarios:
    # 1. Warmup ON (validation only) - Default
    # 2. Warmup OFF
    # 3. Train Warmup (Default Ratio 0.5)
    # 4. Train Warmup (Ratio 0.2)
    # 5. Train Warmup (Ratio 0.8)
    
    # 1. Default (Warmup ON)
    key_default = f"{problem}_n{n}"
    key_warmup_on = f"{problem}_ablation_warmup_val_only"
    
    if key_default in results:
         print(f"[REUSE] Using {key_default} for Warmup ON")
         results[key_warmup_on] = results[key_default]
    else:
         print(f"[INFO] Warmup ON matches Default. Using/Training '{key_default}'...")
         model_path = train_model(base_cfg, dry_run, wandb_project=f"lga_{problem}", only_test=only_test)
         if model_path:
             metrics = test_model(model_path, base_cfg, dry_run)
             save_progress(key_default, metrics)
             results[key_warmup_on] = metrics

    # 2. Warmup OFF
    key_warmup_off = f"{problem}_ablation_warmup_off"
    if key_warmup_off in results:
         print(f"[SKIP] {key_warmup_off} already completed")
    else:
         print("\n--- Warmup OFF ---")
         # We can reuse the default model, but test with --no-warmup
         default_model_path = get_model_path(base_cfg)
         model_exists = default_model_path.exists()
         
         if model_exists or dry_run or (not only_test): # If dry_run, assume it works or we don't care. If not only_test, we might train it above.
             # If only_test and not available, we should skip
             if only_test and not model_exists:
                  print("[TEST-ONLY] Default model not found for Warmup OFF ablation. Skipping.")
             else:
                  print("Testing Default Model with Warmup OFF...")
                  cfg_off = base_cfg.copy()
                  cfg_off["warmup"] = False 
                  metrics = test_model(default_model_path, cfg_off, dry_run)
                  save_progress(key_warmup_off, metrics)
         else:
             print("[ERROR] Default model not found for Warmup OFF ablation. Should have been trained above.")
         
    # 3. Train Warmup (Default Ratio 0.5)
    key_train_warmup = f"{problem}_ablation_train_warmup_r0.5"
    if key_train_warmup in results:
         print(f"[SKIP] {key_train_warmup} already completed")
    else:
         print("\n--- Train Warmup (Ratio 0.5) ---")
         cfg_tw = base_cfg.copy()
         cfg_tw["train_warmup"] = True
         cfg_tw["warmup_ratio"] = 0.5
         model_path = train_model(cfg_tw, dry_run, wandb_project=f"lga_{problem}_ablation_warmup", only_test=only_test)
         if model_path:
             metrics = test_model(model_path, cfg_tw, dry_run)
             save_progress(key_train_warmup, metrics)

    # 4. Train Warmup (Ratio 0.2)
    key_tw_02 = f"{problem}_ablation_train_warmup_r0.2"
    if key_tw_02 in results:
         print(f"[SKIP] {key_tw_02} already completed")
    else:
         print("\n--- Train Warmup (Ratio 0.2) ---")
         cfg_tw = base_cfg.copy()
         cfg_tw["train_warmup"] = True
         cfg_tw["warmup_ratio"] = 0.2
         model_path = train_model(cfg_tw, dry_run, wandb_project=f"lga_{problem}_ablation_warmup", only_test=only_test)
         if model_path:
             metrics = test_model(model_path, cfg_tw, dry_run)
             save_progress(key_tw_02, metrics)
         
    # 5. Train Warmup (Ratio 0.8)
    key_tw_08 = f"{problem}_ablation_train_warmup_r0.8"
    if key_tw_08 in results:
         print(f"[SKIP] {key_tw_08} already completed")
    else:
         print("\n--- Train Warmup (Ratio 0.8) ---")
         cfg_tw = base_cfg.copy()
         cfg_tw["train_warmup"] = True
         cfg_tw["warmup_ratio"] = 0.8
         model_path = train_model(cfg_tw, dry_run, wandb_project=f"lga_{problem}_ablation_warmup", only_test=only_test)
         if model_path:
             metrics = test_model(model_path, cfg_tw, dry_run)
             save_progress(key_tw_08, metrics)

def run_ablation_rl_algo(dry_run=False, only_test=False):
    print("\n=== Running Comparison: PPO vs REINFORCE ===")
    n = 1000
    base_cfg = DEFAULT_CONFIG.copy()
    base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n
    
    results = load_progress()
    
    # 1. PPO (Default) - reuse existing if possible
    key_ppo = "alg_ppo"
    default_key = "tsp_n1000"
    
    if default_key in results:
         print(f"[REUSE] Using {default_key} for PPO")
         results[key_ppo] = results[default_key]
    else:
         print(f"[INFO] PPO matches Default. Using/Training '{default_key}'...")
         # PPO is default algo, so no need to specify "algo": "ppo" usually, but explicitly:
         cfg_ppo = base_cfg.copy()
         cfg_ppo["algo"] = "ppo" 
         model_path = train_model(cfg_ppo, dry_run, wandb_project="lga_tsp", only_test=only_test)
         if model_path:
             metrics = test_model(model_path, cfg_ppo, dry_run)
             save_progress(default_key, metrics)
             results[key_ppo] = metrics

    # 2. REINFORCE
    key_reinforce = "alg_reinforce"
    if key_reinforce in results:
         print(f"[SKIP] {key_reinforce} already completed: {results[key_reinforce]}")
    else:
         print("\n--- REINFORCE ---")
         cfg_reinforce = base_cfg.copy()
         cfg_reinforce["algo"] = "reinforce"
         # REINFORCE typically uses a higher learning rate (1e-4) compared to PPO (5e-6).
         cfg_reinforce["lr"] = 1e-4
         
         model_path = train_model(cfg_reinforce, dry_run, wandb_project="lga_rl_comparison", only_test=only_test)
         if model_path:
             metrics = test_model(model_path, cfg_reinforce, dry_run)
             save_progress(key_reinforce, metrics)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce experiments from LGA paper")
    parser.add_argument("--table", type=str, choices=["2", "3", "5", "6", "8", "heuristic", "warmup", "rl", "all"], default="all", help="Which table to reproduce")
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal steps to verify pipeline")
    parser.add_argument("--fast", action="store_true", help="Skip large instances")
    parser.add_argument("--sizes", type=int, nargs="+", help="Specific sizes to run")
    parser.add_argument("--test", action="store_true", help="Only test existing models, do not train")
    parser.add_argument("--problem", type=str, choices=["tsp", "cvrp"], default=None, help="Specific problem to run experiments for")
    parser.add_argument("--save_dir", type=str, default="experiments", help="Directory to save models")
    
    args = parser.parse_args()
    
    # Update default config with save dir
    DEFAULT_CONFIG["save_dir"] = args.save_dir
    
    # Determine which problems to run ablations for
    # If problem is specified, only run for that problem.
    # If not specified, default to 'tsp' for ablations to match historical behavior (or both? stick to tsp for now as requested context implies it was tsp only)
    ablation_problems = [args.problem] if args.problem else ['tsp']

    # if args.table in ["2", "all"]:
    #     if args.problem is None or args.problem == 'tsp':
    #         sizes = args.sizes if args.sizes else ([1000, 5000, 10000] if not args.fast else [1000])
    #         run_tsp_experiments(sizes, args.dry_run, only_test=args.test)
        
    # if args.table in ["3", "all"]:
    #     if args.problem is None or args.problem == 'cvrp':
    #         sizes = args.sizes if args.sizes else ([1000, 5000] if not args.fast else [1000])
    #         run_cvrp_experiments(sizes, args.dry_run, only_test=args.test)
        
    for p in ablation_problems:
        if args.table in ["5", "all"]:
            run_ablation_refresh(problem=p, dry_run=args.dry_run, only_test=args.test)
            
        if args.table in ["6", "all"]:
            run_ablation_features(problem=p, dry_run=args.dry_run, only_test=args.test)
            
        if args.table in ["8", "all"]:
            run_ablation_smoothing(problem=p, dry_run=args.dry_run, only_test=args.test)
            
        if args.table in ["heuristic", "all"]:
            run_ablation_heuristic(problem=p, dry_run=args.dry_run, only_test=args.test)
            
        if args.table in ["warmup", "all"]:
            run_ablation_warmup(p, args.dry_run, only_test=args.test)

    if args.table in ["rl", "all"]:
        run_ablation_rl_algo(args.dry_run, only_test=args.test)

if __name__ == "__main__":
    main()
