#!/usr/bin/env python3
"""
Reviewer Experiments - Comprehensive experiment pipeline to address reviewer concerns.

This script orchestrates experiments to address the 8 themes of reviewer concerns:
1. SRR necessity and isolation
2. Stabilized pheromone dynamics
3. Hyperparameter sensitivity / robustness
4. Efficiency and scalability
5. Cross-scale / cross-distribution generalization
6. CVRP-specific behavior
7. Theory / mechanism depth
8. Scope and applicability

Usage:
    python reviewer_experiments.py --phase all
    python reviewer_experiments.py --phase srr_training_stability
    python reviewer_experiments.py --phase srr_inference
    python reviewer_experiments.py --phase pheromone_ablation
    python reviewer_experiments.py --phase hyperparam_sensitivity
    python reviewer_experiments.py --phase efficiency_analysis
    python reviewer_experiments.py --phase cross_scale
    python reviewer_experiments.py --phase cvrp_behavior
    python reviewer_experiments.py --phase guidance_transition
    python reviewer_experiments.py --phase scope_applicability
"""

import argparse
import subprocess
import os
import sys
import time
import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# =============================================================================
# Configuration & Constants
# =============================================================================

# Default Hyperparameters from Paper
DEFAULT_CONFIG = {
    "H": 10,
    "mini_H": 100,  # S in paper
    "n_ants": 100,
    "k_sparse": 32,
    "epochs": 10,
    "steps_per_epoch": 32,
    "ppo_epochs": 4,
    "lr": 5e-6,
    "anneal_prior": False,
    "gamma": 1.0,
    "min_gamma": 0.0,
    "val_size": 16,
    "warmup": True,
    "train_warmup": False,
    "save_dir": "reviewer_experiments/models",
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

# Progress file for resumption
PROGRESS_FILE = "reviewer_experiments/experiment_progress.json"

# Results directory
RESULTS_DIR = Path("reviewer_experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_review_scales(problem: str, n_node: int, limit: Optional[int] = None) -> List[int]:
    """Build review-oriented scales anchored on the requested n_node."""
    if n_node <= 0:
        return [n_node]

    if n_node < 100:
        multipliers = [1, 2, 5]
    elif n_node < 1000:
        multipliers = [1, 2, 5, 10]
    else:
        multipliers = [1, 5, 10]
        if problem == "tsp":
            multipliers.extend([50, 100])
        else:
            multipliers.append(50)

    scales = sorted({int(n_node * m) for m in multipliers})
    return scales[:limit] if limit is not None else scales


def get_hyperparam_scales(problem: str, n_node: int) -> List[int]:
    """Use a compact subset of scales for robustness sweeps."""
    return get_review_scales(problem, n_node, limit=2 if n_node < 1000 else 3)


def choose_best_method(summary: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    methods = summary.get("methods", {})
    best_name = None
    best_payload: Dict[str, Any] = {}
    best_cost = None
    for method_name, payload in methods.items():
        cost = payload.get("mean_cost")
        if cost is None:
            continue
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_name = method_name
            best_payload = payload
    return best_name, best_payload


def count_cvrp_routes(route: Any) -> Optional[int]:
    if route is None:
        return None
    route_list = [int(x) for x in route]
    if not route_list:
        return None
    return sum(1 for prev, cur in zip(route_list[:-1], route_list[1:]) if prev == 0 and cur != 0)


def _get_eval_device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_eval_items(problem: str, n_node: int, val_size: int, rl_data: bool = False) -> List[Any]:
    import torch
    import utils

    val_list = utils.load_auto_dataset(n_node, problem=problem, rl_data=rl_data, device="cpu")
    if val_list is None:
        val_list = []
        for _ in range(val_size):
            if problem == "tsp":
                val_list.append(torch.from_numpy(utils.generate_tsp_instance(n_node)))
            else:
                coords, demand, capacity = utils.gen_cvrp_instance(n_node, device="cpu")
                val_list.append((coords.cpu(), demand.cpu(), capacity))
    return list(val_list[:val_size])


def _normalize_eval_item(problem: str, item: Any) -> Any:
    import torch

    if problem == "tsp":
        if isinstance(item, tuple):
            item = item[0]
        if torch.is_tensor(item):
            return item.cpu()
        return item

    if isinstance(item, tuple) and len(item) >= 5:
        item = item[:3]

    if isinstance(item, (list, tuple)):
        coords, demand, capacity = item[:3]
        if torch.is_tensor(coords):
            coords = coords.cpu().numpy()
        if torch.is_tensor(demand):
            demand = demand.cpu().numpy()
        capacity = float(capacity.item()) if torch.is_tensor(capacity) else float(capacity)
        if capacity > 1.0 + 1e-6:
            demand = demand / capacity
            capacity = 1.0
        return (coords, demand, capacity)

    return item


def _load_model_runtime(model_path: Path, problem: str) -> Tuple[Any, Any, Dict[str, Any]]:
    import torch
    import faco
    import utils
    from net import Net

    device = _get_eval_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    config = ckpt.get("config", {})

    feats = 2 if problem == "tsp" else 1
    edge_feats = 6
    if "emb_net.v_lin0.weight" in state_dict:
        feats = state_dict["emb_net.v_lin0.weight"].shape[1]
    if "emb_net.e_lin0.weight" in state_dict:
        edge_feats = state_dict["emb_net.e_lin0.weight"].shape[1]

    model = Net(feats=feats, edge_feats=edge_feats, logit_net=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if problem == "tsp":
        build_fn = utils.build_pyg_data_tsp
        aco_class = faco.MFACO_TSP
    else:
        build_fn = utils.build_pyg_data_cvrp
        aco_class = faco.MFACO_CVRP
    return model, (build_fn, aco_class, device), config


def collect_guidance_diagnostics(
    model_path: Path,
    problem: str,
    n_node: int,
    config: Dict[str, Any],
    val_size: int = 4,
    rl_data: bool = False
) -> Dict[str, Any]:
    import numpy as np
    from utils import infer_instance

    model, runtime, model_config = _load_model_runtime(model_path, problem)
    build_fn, aco_class, device = runtime

    merged = DEFAULT_CONFIG.copy()
    merged.update(CVRP_CONFIG if problem == "cvrp" else TSP_CONFIG)
    merged.update(model_config)
    merged.update(config)
    merged["problem"] = problem
    merged["n_node"] = n_node

    args = argparse.Namespace(
        problem=problem,
        n_node=n_node,
        device=device,
        H=int(merged.get("H", DEFAULT_CONFIG["H"])),
        mini_H=int(merged.get("mini_H", DEFAULT_CONFIG["mini_H"])),
        gamma=float(merged.get("gamma", 1.0)),
        min_gamma=float(merged.get("min_gamma", 0.0)),
        no_anneal=bool(merged.get("no_anneal", False)),
        disable_heuristic=bool(merged.get("disable_heuristic", False)),
        no_local_search=bool(merged.get("no_local_search", False)),
        rho=float(merged.get("rho", 0.1)),
        no_smooth_mmas=bool(merged.get("no_smooth_mmas", False)),
        min_new_edges=int(merged.get("min_new_edges", 12)),
        no_extend_ls=bool(merged.get("no_extend_ls", False)),
        no_normalized_heuristic=bool(merged.get("no_normalized_heuristic", False)),
        L=int(merged.get("L", 0)),
        ls_scope=merged.get("ls_scope", "localized"),
        ls_budget=merged.get("ls_budget", "truncated"),
        ls_max_opt=int(merged.get("ls_max_opt", 0)),
        timed=True,
        verify=(problem == "cvrp"),
        runtime_limit=None,
        iter_log=False,
        iter_print=False,
        stage_metrics=False,
        alpha=float(merged.get("alpha", 1.0)),
        beta=float(merged.get("beta", 0.0)),
    )

    items = _load_eval_items(problem, n_node, val_size=val_size, rl_data=rl_data)
    profiles = []
    prior_means = []
    route_counts = []
    suppressions = []
    enhancements = []
    rebellions = []

    for idx, item in enumerate(items):
        eval_item = _normalize_eval_item(problem, item)
        _, best_seen, _, extra = infer_instance(
            problem,
            aco_class,
            build_fn,
            model,
            eval_item,
            int(merged.get("k_sparse", DEFAULT_CONFIG["k_sparse"])),
            int(merged.get("n_ants", DEFAULT_CONFIG["n_ants"])),
            not bool(merged.get("no_dynamic_feats", False)),
            args,
            use_heuristic_only=False,
            collect_metrics=True,
            metrics_every_step=True,
            seed=int(merged.get("seed", 1234)) + idx,
            ablation_pheromone=bool(merged.get("ablation_pheromone_features", False)),
            ablation_incumbent=bool(merged.get("ablation_incumbent_features", False)),
        )
        metrics = extra.get("metrics", {})
        profile = {
            "best_cost": float(best_seen),
            "cost_curve": list(metrics.get("cost", [])),
            "prior_mean": list(metrics.get("prior_mean", [])),
            "prior_std": list(metrics.get("prior_std", [])),
            "suppression": list(metrics.get("suppression", [])),
            "enhance": list(metrics.get("enhance", [])),
            "rebellion": list(metrics.get("rebellion", [])),
            "history": list(extra.get("history", [])),
            "route_count": count_cvrp_routes(extra.get("best_decoded_route")),
        }
        profiles.append(profile)
        prior_means.extend(profile["prior_mean"])
        suppressions.extend(profile["suppression"])
        enhancements.extend(profile["enhance"])
        rebellions.extend(profile["rebellion"])
        if profile["route_count"] is not None:
            route_counts.append(profile["route_count"])

    return {
        "profiles": profiles,
        "guidance_mean": float(np.mean(prior_means)) if prior_means else 0.0,
        "guidance_std": float(np.std(prior_means)) if prior_means else 0.0,
        "suppression_mean": float(np.mean(suppressions)) if suppressions else 0.0,
        "enhance_mean": float(np.mean(enhancements)) if enhancements else 0.0,
        "rebellion_mean": float(np.mean(rebellions)) if rebellions else 0.0,
        "route_count_mean": float(np.mean(route_counts)) if route_counts else 0.0,
        "route_count_std": float(np.std(route_counts)) if route_counts else 0.0,
    }

# =============================================================================
# Helper Functions
# =============================================================================

def run_command(cmd: List[str], log_file: Path = None, dry_run: bool = False) -> tuple:
    """Executes a shell command and returns (returncode, output, cmd_str)."""
    cmd_str = " ".join(cmd)
    print(f"[CMD] {cmd_str}")

    if dry_run:
        return 0, "DRY_RUN", cmd_str

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
                return {k: v for k, v in data.items() if v is not None}
        except json.JSONDecodeError:
            print(f"[WARNING] Could not decode {PROGRESS_FILE}. Starting fresh.")
            return {}
    return {}


def save_progress(key: str, data: Any):
    """Saves a single experiment result to the progress file."""
    if data is None:
        print(f"[PROGRESS] Skipping save for '{key}' (No data)")
        return

    progress = load_progress()
    progress[key] = data
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=4)
    print(f"[PROGRESS] Saved result for '{key}'")


def parse_metrics(output: str) -> Dict[str, Any]:
    """Parses output from test.py to find Cost, Time, Gap."""
    metrics = {"gap": None, "time": None, "cost": None}

    # Model Cost & Time
    # Matches "Model Cost (...): 123.45, Time: 67.89s" or "Model Cost: 123.45, Time: 67.89s"
    model_stats = re.search(r"Model Cost(?:\s*\([^)]*\))?:\s*([-+]?\d*\.\d+|\d+).*?Time:\s*([-+]?\d*\.\d+|\d+)s", output)
    if model_stats:
        metrics["cost"] = float(model_stats.group(1))
        metrics["time"] = float(model_stats.group(2))

    # Model Gap
    # Matches "Model Gap (...): 1.23%" or "Model Gap: 1.23%"
    gap_match = re.search(r"Model Gap(?:\s*\([^)]*\))?:\s*([-+]?\d*\.\d+|\d+)%", output)
    if gap_match:
        metrics["gap"] = float(gap_match.group(1))

    return metrics


def parse_training_metrics(output: str) -> List[Dict[str, Any]]:
    """Parses training output to extract per-epoch metrics."""
    epochs = []
    # Look for epoch lines like "Epoch 1: TrainCost=..."
    # Match "Epoch 1: TrainCost=23.45" or "Epoch -1: Pure MFACO check..."
    epoch_pattern = re.compile(r"Epoch\s+(-?\d+):\s*TrainCost=([-+]?\d*\.\d+|\d+)")
    for match in epoch_pattern.finditer(output):
        epoch = int(match.group(1))
        cost = float(match.group(2))
        # Skip pre-training validation (Epoch -1)
        if epoch >= 0:
            epochs.append({"epoch": epoch, "avg_cost": cost})
    return epochs


def get_model_path(config: Dict[str, Any], suffix: str = "_best.pt") -> Path:
    """Constructs model path based on config."""
    algo = config.get("algo", "ppo")
    lr = config.get("lr", 5e-6)
    name = f"{config['problem']}_n{config['n_node']}_k{config['k_sparse']}_ants{config['n_ants']}_H{config['H']}_miniH{config['mini_H']}_rho{config['rho']}_mne{config['min_new_edges']}_{algo}_lr{lr}"

    if config.get("anneal_prior", False):
        name += f"_anneal_g{config['gamma']}_mg{config['min_gamma']}"

    if config.get("L", 0) > 0:
        name += f"_L{config['L']}"

    # Ablation suffixes
    if config.get("no_dynamic_feats"):
        name += "_static"
    if config.get("no_smooth_mmas"):
        name += "_nosmooth"
    if config.get("disable_heuristic"):
        name += "_noheu"
    if config.get("no_local_search"):
        name += "_nols"
    if config.get("no_extend_ls"):
        name += "_noextls"
    if config.get("no_normalized_heuristic"):
        name += "_nonorm"

    # LS scope suffixes
    if config.get("ls_scope") != "localized" and config.get("ls_scope") is not None:
        name += f"_ls{config['ls_scope']}"
    if config.get("ls_budget") != "truncated" and config.get("ls_budget") is not None:
        name += f"_{config['ls_budget']}ls"
    if config.get("ls_max_opt", 0) > 0:
        name += f"_lsmax{config['ls_max_opt']}"

    save_dir = Path(config.get("save_dir", "reviewer_experiments/models")) / config["problem"] / f"n{config['n_node']}"
    return save_dir / (name + suffix)


def train_model(config: Dict[str, Any], dry_run: bool = False, force: bool = False,
               wandb_project: str = "reviewer_experiments", only_test: bool = False) -> Optional[Path]:
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
    cmd.extend(["--baseline", "none"])

    # Flags mapping
    bool_flags = ["anneal_prior", "no_dynamic_feats", "disable_heuristic", "no_local_search",
                  "no_smooth_mmas", "train_warmup", "warmup", "no_extend_ls", "no_normalized_heuristic"]

    for k, v in config.items():
        if k in ["save_dir"]:
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

    log_file = Path("reviewer_experiments/logs") / f"train_{model_path.stem}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    run_command(cmd, log_file, dry_run)
    return model_path


def test_model(
    model_path: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
    runtime_limit: Optional[float] = None,
    summary_json: Optional[Path] = None,
    rl_data: bool = False,
    stage_metrics: bool = False
) -> Dict[str, Any]:
    """Runs evaluation."""
    cmd = [sys.executable, "test.py"]
    cmd.extend(["--problem", config["problem"]])
    cmd.extend(["--n_node", str(config["n_node"])])
    cmd.extend(["--checkpoint", str(model_path)])
    cmd.extend(["--baseline", "none"])

    # Pass relevant args to test
    relevant_keys = ["H", "mini_H", "n_ants", "k_sparse", "rho", "min_new_edges", 
                     "warmup", "warmup_ratio", "val_size", "ls_scope", "ls_budget", "ls_max_opt", "no_baseline"]
    for k in relevant_keys:
        if k in config and config[k] is not None:
            if k == "warmup":
                if config[k] is False:
                    cmd.append("--no-warmup")
            elif k == "no_baseline":
                if config[k] is True:
                    cmd.append("--no-baseline")
            elif config[k] is True:
                cmd.append(f"--{k}")
            else:
                cmd.extend([f"--{k}", str(config[k])])

    # Pass boolean flags for ablations
    bool_flags = ["anneal_prior", "no_dynamic_feats", "disable_heuristic", "no_local_search",
                  "no_smooth_mmas", "no_extend_ls", "no_normalized_heuristic"]
    for k in bool_flags:
        if config.get(k, False):
            cmd.append(f"--{k}")

    # Add runtime limit if specified
    if runtime_limit is not None:
        cmd.extend(["--runtime_limit", str(runtime_limit)])
    if summary_json is not None:
        cmd.extend(["--summary_json", str(summary_json)])
    if rl_data:
        cmd.append("--rl_data")
    if stage_metrics:
        cmd.append("--stage_metrics")

    if dry_run:
        cmd.extend(["--n_node", "20", "--n_ants", "10"])

    log_file = Path("reviewer_experiments/logs") / f"test_{model_path.stem}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    code, output, _ = run_command(cmd, log_file, dry_run)
    metrics = parse_metrics(output)
    if summary_json is not None and summary_json.exists():
        with open(summary_json, "r") as f:
            metrics["summary"] = json.load(f)
    return metrics


def append_to_csv(csv_path: Path, data: Dict[str, Any], fieldnames: List[str]):
    """Appends a row to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file with header if it doesn't exist
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Append the row
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data)


# =============================================================================
# Phase 2: SRR Necessity and Isolation Experiments
# =============================================================================

def run_srr_training_stability(problem='tsp', n_node=1000, dry_run=False, only_test=False):
    """
    Phase 2a: SRR Training Stability Comparison

    Demonstrates that SRR enables learning (decreasing cost over epochs),
    while global LS prevents learning (flat cost curve).
    """
    print(f"\n=== Phase 2a: SRR Training Stability [{problem} N={n_node}] ===")

    base_cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
        base_cfg.update(CVRP_CONFIG)
    else:
        base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n_node

    results = load_progress()
    csv_path = RESULTS_DIR / "srr_training_stability.csv"
    fieldnames = ["problem", "n_node", "ls_type", "epoch", "avg_cost", "best_cost", "timestamp"]

    # Test three LS configurations
    ls_configs = [
        ("srr_local", {"ls_scope": "localized", "ls_budget": "truncated"}),
        ("global_ls", {"ls_scope": "global", "ls_budget": "full"}),
        ("truncated_ls", {"ls_scope": "global", "ls_budget": "truncated", "ls_max_opt": n_node // 4}),
    ]

    for ls_type, ls_config in ls_configs:
        key = f"srr_training_{problem}_n{n_node}_{ls_type}"

        if key in results and not dry_run:
            print(f"[SKIP] {key} already completed")
            continue

        print(f"\n--- Training with {ls_type} ---")
        cfg = base_cfg.copy()
        cfg.update(ls_config)

        # Train the model
        model_path = train_model(cfg, dry_run, wandb_project="reviewer_srr_training", only_test=only_test)

        if model_path is None:
            print(f"[SKIP] Model not found for {ls_type}")
            continue

        # Parse training log to extract per-epoch metrics
        log_file = Path("reviewer_experiments/logs") / f"train_{model_path.stem}.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                log_content = f.read()
            epochs = parse_training_metrics(log_content)

            for epoch_data in epochs:
                row = {
                    "problem": problem,
                    "n_node": n_node,
                    "ls_type": ls_type,
                    "epoch": epoch_data["epoch"],
                    "avg_cost": epoch_data["avg_cost"],
                    "best_cost": epoch_data.get("best_cost", ""),
                    "timestamp": datetime.now().isoformat(),
                }
                append_to_csv(csv_path, row, fieldnames)

            # Save progress
            if epochs:
                final_cost = epochs[-1]["avg_cost"]
                save_progress(key, {"ls_type": ls_type, "final_cost": final_cost, "epochs": len(epochs)})
        else:
            print(f"[WARNING] Training log not found: {log_file}")

    print(f"\n[RESULTS] Training stability data saved to {csv_path}")


def run_srr_inference_comparison(problem='tsp', n_node=1000, dry_run=False, only_test=False):
    """
    Phase 2b: SRR Inference Comparison

    Compares SRR vs truncated LS vs full LS with strict time limits.
    """
    print(f"\n=== Phase 2b: SRR Inference Comparison [{problem} N={n_node}] ===")

    base_cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
        base_cfg.update(CVRP_CONFIG)
    else:
        base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = n_node

    results = load_progress()
    csv_path = RESULTS_DIR / "srr_inference_comparison.csv"
    fieldnames = ["problem", "n_node", "ls_type", "time_limit", "cost", "gap", "time", "timestamp"]

    # Time limits to test (in seconds)
    time_limits = [10.0]

    # LS configurations
    ls_configs = [
        ("srr_local", {"ls_scope": "localized", "ls_budget": "truncated"}),
        ("global_ls", {"ls_scope": "global", "ls_budget": "full"}),
        ("truncated_ls", {"ls_scope": "global", "ls_budget": "truncated", "ls_max_opt": n_node // 4}),
    ]

    # Results tracking
    results = load_progress()
    csv_path = RESULTS_DIR / "srr_inference_comparison.csv"
    fieldnames = ["problem", "n_node", "ls_type", "time_limit", "cost", "gap", "time", "timestamp"]

    # Time limits to test (in seconds)
    time_limits = [0.4]

    # LS configurations
    ls_configs = [
        ("srr_local", {"ls_scope": "localized", "ls_budget": "truncated"}),
        ("global_ls", {"ls_scope": "global", "ls_budget": "full"}),
        ("truncated_ls", {"ls_scope": "global", "ls_budget": "truncated", "ls_max_opt": n_node // 4}),
    ]

    # Step 2: Run all configurations with the fixed time limit
    for ls_type, ls_config in ls_configs:
        # Configuration (Standard)
        cfg = base_cfg.copy()
        cfg.update(ls_config)
        cfg["n_node"] = n_node
        
        # Testing Config (Standard H, skip baseline)
        test_cfg = cfg.copy()
        test_cfg["no_baseline"] = True 
        
        model_path = get_model_path(cfg)
        # Also check standard checkpoints directory
        alt_path = Path("checkpoints") / model_path.name
        if not model_path.exists() and alt_path.exists():
            model_path = alt_path

        if not model_path.exists() and not only_test:
            model_path = train_model(cfg, dry_run, wandb_project="reviewer_srr_inference", only_test=only_test)
        
        if model_path is None or not model_path.exists():
            print(f"[SKIP] Model not found for {ls_type}")
            continue

        for time_limit in time_limits:
            key = f"srr_inference_{problem}_n{n_node}_{ls_type}_t{time_limit}"
            if key in results and not dry_run:
                print(f"[SKIP] {key} already completed")
                continue

            print(f"\n--- Testing {ls_type} with time_limit={time_limit}s ---")
            metrics = test_model(model_path, test_cfg, dry_run, runtime_limit=time_limit)

            row = {
                "problem": problem,
                "n_node": n_node,
                "ls_type": ls_type,
                "time_limit": time_limit,
                "cost": metrics.get("cost", ""),
                "gap": metrics.get("gap", ""),
                "time": metrics.get("time", ""),
                "timestamp": datetime.now().isoformat(),
            }
            append_to_csv(csv_path, row, fieldnames)
            save_progress(key, metrics)

    print(f"\n[RESULTS] Inference comparison data saved to {csv_path}")


# =============================================================================
# Phase 3: Stabilized Pheromone Dynamics Ablation
# =============================================================================

def run_pheromone_ablation(problem='tsp', n_node=1000, dry_run=False, only_test=False):
    """
    Phase 3: Stabilized Pheromone Dynamics Ablation

    Broader ablation beyond TSP-5K, testing smooth vs no-smooth MMAS across scales.
    """
    print(f"\n=== Phase 3: Pheromone Dynamics Ablation [{problem}] ===")

    scales = get_review_scales(problem, n_node)

    results = load_progress()
    csv_path = RESULTS_DIR / "pheromone_ablation.csv"
    fieldnames = ["problem", "n_node", "smooth", "cost", "gap", "time", "timestamp"]

    for n_node in scales:
        base_cfg = DEFAULT_CONFIG.copy()
        if problem == 'cvrp':
            base_cfg.update(CVRP_CONFIG)
        else:
            base_cfg.update(TSP_CONFIG)
        base_cfg["n_node"] = n_node

        # Test smooth vs no-smooth
        for smooth in [True, False]:
            key = f"pheromone_{problem}_n{n_node}_smooth{smooth}"

            if key in results and not dry_run:
                print(f"[SKIP] {key} already completed")
                continue

            print(f"\n--- Testing {problem} N={n_node} smooth={smooth} ---")
            cfg = base_cfg.copy()
            cfg["no_smooth_mmas"] = not smooth

            model_path = train_model(cfg, dry_run, wandb_project="reviewer_pheromone", only_test=only_test)

            if model_path is None:
                print(f"[SKIP] Model not found")
                continue

            metrics = test_model(model_path, cfg, dry_run)

            row = {
                "problem": problem,
                "n_node": n_node,
                "smooth": smooth,
                "cost": metrics.get("cost", ""),
                "gap": metrics.get("gap", ""),
                "time": metrics.get("time", ""),
                "timestamp": datetime.now().isoformat(),
            }
            append_to_csv(csv_path, row, fieldnames)
            save_progress(key, metrics)

    print(f"\n[RESULTS] Pheromone ablation data saved to {csv_path}")


# =============================================================================
# Phase 4: Hyperparameter Sensitivity Analysis
# =============================================================================

def run_hyperparam_sensitivity(problem='tsp', n_node=1000, dry_run=False, only_test=False):
    """
    Phase 4: Hyperparameter Sensitivity Analysis

    Analyzes sensitivity to M (perturbation size), K (K-NN size), S (temporal granularity).
    """
    print(f"\n=== Phase 4: Hyperparameter Sensitivity [{problem} N={n_node}] ===")

    results = load_progress()
    csv_path = RESULTS_DIR / "hyperparam_sensitivity.csv"
    fieldnames = ["problem", "n_node", "hyperparam", "value", "cost", "gap", "time", "training_time", "timestamp"]
    scales = get_hyperparam_scales(problem, n_node)

    # Test perturbation size (M) - min_new_edges
    m_values = [4, 8, 12, 16, 20]
    for scale_n in scales:
        base_cfg = DEFAULT_CONFIG.copy()
        if problem == 'cvrp':
            base_cfg.update(CVRP_CONFIG)
        else:
            base_cfg.update(TSP_CONFIG)
        base_cfg["n_node"] = scale_n

        for m in m_values:
            key = f"hyperparam_{problem}_n{scale_n}_M{m}"

            if key in results and not dry_run:
                print(f"[SKIP] {key} already completed")
                continue

            print(f"\n--- Testing N={scale_n}, M={m} (min_new_edges) ---")
            cfg = base_cfg.copy()
            cfg["min_new_edges"] = m

            t0 = time.time()
            model_path = train_model(cfg, dry_run, wandb_project="reviewer_hyperparam", only_test=only_test)
            training_time = time.time() - t0

            if model_path is None:
                print(f"[SKIP] Model not found")
                continue

            metrics = test_model(model_path, cfg, dry_run)

            row = {
                "problem": problem,
                "n_node": scale_n,
                "hyperparam": "M",
                "value": m,
                "cost": metrics.get("cost", ""),
                "gap": metrics.get("gap", ""),
                "time": metrics.get("time", ""),
                "training_time": training_time,
                "timestamp": datetime.now().isoformat(),
            }
            append_to_csv(csv_path, row, fieldnames)
            save_progress(key, metrics)

    # Test K-NN size (K) - k_sparse
    k_values = [16, 24, 32, 48, 64]
    for scale_n in scales:
        base_cfg = DEFAULT_CONFIG.copy()
        if problem == 'cvrp':
            base_cfg.update(CVRP_CONFIG)
        else:
            base_cfg.update(TSP_CONFIG)
        base_cfg["n_node"] = scale_n

        for k in k_values:
            key = f"hyperparam_{problem}_n{scale_n}_K{k}"

            if key in results and not dry_run:
                print(f"[SKIP] {key} already completed")
                continue

            print(f"\n--- Testing N={scale_n}, K={k} (k_sparse) ---")
            cfg = base_cfg.copy()
            cfg["k_sparse"] = k

            t0 = time.time()
            model_path = train_model(cfg, dry_run, wandb_project="reviewer_hyperparam", only_test=only_test)
            training_time = time.time() - t0

            if model_path is None:
                print(f"[SKIP] Model not found")
                continue

            metrics = test_model(model_path, cfg, dry_run)

            row = {
                "problem": problem,
                "n_node": scale_n,
                "hyperparam": "K",
                "value": k,
                "cost": metrics.get("cost", ""),
                "gap": metrics.get("gap", ""),
                "time": metrics.get("time", ""),
                "training_time": training_time,
                "timestamp": datetime.now().isoformat(),
            }
            append_to_csv(csv_path, row, fieldnames)
            save_progress(key, metrics)

    # Test temporal granularity (S) - mini_H (with fixed H*S=1000)
    s_values = [50, 100, 200, 400]
    for scale_n in scales:
        base_cfg = DEFAULT_CONFIG.copy()
        if problem == 'cvrp':
            base_cfg.update(CVRP_CONFIG)
        else:
            base_cfg.update(TSP_CONFIG)
        base_cfg["n_node"] = scale_n

        for s in s_values:
            key = f"hyperparam_{problem}_n{scale_n}_S{s}"

            if key in results and not dry_run:
                print(f"[SKIP] {key} already completed")
                continue

            print(f"\n--- Testing N={scale_n}, S={s} (mini_H) ---")
            cfg = base_cfg.copy()
            cfg["mini_H"] = s
            cfg["H"] = 1000 // s  # Keep H*S = 1000

            t0 = time.time()
            model_path = train_model(cfg, dry_run, wandb_project="reviewer_hyperparam", only_test=only_test)
            training_time = time.time() - t0

            if model_path is None:
                print(f"[SKIP] Model not found")
                continue

            metrics = test_model(model_path, cfg, dry_run)

            row = {
                "problem": problem,
                "n_node": scale_n,
                "hyperparam": "S",
                "value": s,
                "cost": metrics.get("cost", ""),
                "gap": metrics.get("gap", ""),
                "time": metrics.get("time", ""),
                "training_time": training_time,
                "timestamp": datetime.now().isoformat(),
            }
            append_to_csv(csv_path, row, fieldnames)
            save_progress(key, metrics)

    print(f"\n[RESULTS] Hyperparameter sensitivity data saved to {csv_path}")


# =============================================================================
# Phase 5: Efficiency and Scalability Analysis
# =============================================================================

def run_efficiency_analysis(problem='tsp', n_node=1000, dry_run=False, only_test=False):
    """
    Phase 5: Efficiency and Scalability Analysis

    Analyzes parameter count, memory usage, and training time across scales.
    """
    print(f"\n=== Phase 5: Efficiency and Scalability Analysis [{problem}] ===")

    scales = get_review_scales(problem, n_node)

    results = load_progress()
    csv_path = RESULTS_DIR / "efficiency_analysis.csv"
    fieldnames = ["problem", "n_node", "params", "peak_memory_gb", "training_time", "inference_time", "timestamp"]

    for n_node in scales:
        key = f"efficiency_{problem}_n{n_node}"

        if key in results and not dry_run:
            print(f"[SKIP] {key} already completed")
            continue

        print(f"\n--- Analyzing {problem} N={n_node} ---")
        base_cfg = DEFAULT_CONFIG.copy()
        if problem == 'cvrp':
            base_cfg.update(CVRP_CONFIG)
        else:
            base_cfg.update(TSP_CONFIG)
        base_cfg["n_node"] = n_node

        # For efficiency analysis, we can use existing models
        # Check if model exists in checkpoints
        checkpoint_dir = Path("checkpoints")
        checkpoint_pattern = f"{problem}_n{n_node}_*_best.pt"
        checkpoints = list(checkpoint_dir.glob(checkpoint_pattern))

        if checkpoints:
            print(f"[REUSE] Using existing checkpoint: {checkpoints[0]}")
            model_path = checkpoints[0]
        else:
            # Train if not exists
            model_path = train_model(base_cfg, dry_run, wandb_project="reviewer_efficiency", only_test=only_test)

        if model_path is None:
            print(f"[SKIP] Model not found")
            continue

        # Load model to get parameter count
        try:
            import torch
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                params = sum(p.numel() for p in ckpt["model_state_dict"].values())
            else:
                params = sum(p.numel() for p in ckpt.values() if isinstance(p, torch.Tensor))
        except Exception as e:
            print(f"[WARNING] Could not count parameters: {e}")
            params = 0

        # Measure inference time
        cfg = base_cfg.copy()
        t0 = time.time()
        metrics = test_model(model_path, cfg, dry_run)
        inference_time = time.time() - t0

        # Training time (estimated or from log)
        training_time = 0.0
        log_file = Path("reviewer_experiments/logs") / f"train_{model_path.stem}.log"
        if log_file.exists():
            # Try to extract training time from log
            with open(log_file, "r") as f:
                log_content = f.read()
            # Look for total training time (matches train.py: "Total Train Time: 123.45s")
            time_match = re.search(r"Total Train Time:\s*([\d.]+)", log_content)
            if time_match:
                training_time = float(time_match.group(1))

        # Peak memory (placeholder - would need to add profiling to train.py)
        peak_memory_gb = 0.0

        row = {
            "problem": problem,
            "n_node": n_node,
            "params": params,
            "peak_memory_gb": peak_memory_gb,
            "training_time": training_time,
            "inference_time": inference_time,
            "timestamp": datetime.now().isoformat(),
        }
        append_to_csv(csv_path, row, fieldnames)
        save_progress(key, row)

    print(f"\n[RESULTS] Efficiency analysis data saved to {csv_path}")


# =============================================================================
# Phase 6: Cross-scale Generalization Analysis
# =============================================================================

def run_cross_scale_generalization(problem='tsp', n_node=1000, dry_run=False, only_test=False):
    """
    Phase 6: Cross-scale Generalization Analysis

    Tests 1K-trained models on larger scales without retraining.
    """
    print(f"\n=== Phase 6: Cross-scale Generalization [{problem}] ===")

    train_n = n_node
    test_scales = get_review_scales(problem, n_node)

    results = load_progress()
    csv_path = RESULTS_DIR / "cross_scale_generalization.csv"
    fieldnames = ["problem", "train_n", "test_n", "cost", "gap", "time", "guidance_mean", "guidance_std", "timestamp"]

    base_cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
        base_cfg.update(CVRP_CONFIG)
    else:
        base_cfg.update(TSP_CONFIG)
    base_cfg["n_node"] = train_n

    model_path = get_model_path(base_cfg)
    alt_path = Path("checkpoints") / model_path.name
    if not model_path.exists() and alt_path.exists():
        model_path = alt_path
    if not model_path.exists():
        model_path = train_model(base_cfg, dry_run, wandb_project="reviewer_cross_scale", only_test=only_test)
    if model_path is None or not model_path.exists():
        print(f"[ERROR] No n={train_n}-trained model found for {problem}")
        return

    print(f"[INFO] Using n={train_n}-trained model: {model_path}")

    # Load model config
    try:
        import torch
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model_config = ckpt.get("config", {})
    except Exception as e:
        print(f"[WARNING] Could not load model config: {e}")
        model_config = {}

    for test_n in test_scales:
        key = f"cross_scale_{problem}_train{train_n}_test{test_n}"

        if key in results and not dry_run:
            print(f"[SKIP] {key} already completed")
            continue

        print(f"\n--- Testing n={train_n} model on {problem} N={test_n} ---")

        # Test on larger instance
        cfg = DEFAULT_CONFIG.copy()
        if problem == 'cvrp':
            cfg.update(CVRP_CONFIG)
        else:
            cfg.update(TSP_CONFIG)
        cfg.update(model_config)
        cfg["n_node"] = test_n

        summary_path = RESULTS_DIR / "tmp" / f"cross_scale_{problem}_train{train_n}_test{test_n}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = test_model(model_path, cfg, dry_run, summary_json=summary_path)
        guidance_diag = collect_guidance_diagnostics(model_path, problem, test_n, cfg, val_size=min(int(cfg.get("val_size", 16)), 4))

        row = {
            "problem": problem,
            "train_n": train_n,
            "test_n": test_n,
            "cost": metrics.get("cost", ""),
            "gap": metrics.get("gap", ""),
            "time": metrics.get("time", ""),
            "guidance_mean": guidance_diag["guidance_mean"],
            "guidance_std": guidance_diag["guidance_std"],
            "timestamp": datetime.now().isoformat(),
        }
        append_to_csv(csv_path, row, fieldnames)
        save_progress(key, metrics)

    print(f"\n[RESULTS] Cross-scale generalization data saved to {csv_path}")


# =============================================================================
# Phase 7: CVRP-specific Behavior Analysis
# =============================================================================

def run_cvrp_behavior_analysis(n_node=1000, dry_run=False, only_test=False):
    """
    Phase 7: CVRP-specific Behavior Analysis

    Analyzes guidance polarity reversal across scales and capacity constraints.
    """
    print(f"\n=== Phase 7: CVRP Behavior Analysis ===")

    problem = 'cvrp'
    scales = get_review_scales(problem, n_node)

    results = load_progress()
    csv_path = RESULTS_DIR / "cvrp_behavior.csv"
    fieldnames = ["problem", "n_node", "capacity", "n_routes", "guidance_mean", "guidance_std", "cost", "gap", "timestamp"]

    for n_node in scales:
        key = f"cvrp_behavior_n{n_node}"

        if key in results and not dry_run:
            print(f"[SKIP] {key} already completed")
            continue

        print(f"\n--- Analyzing CVRP N={n_node} ---")

        # Find CVRP model
        checkpoint_dir = Path("checkpoints")
        checkpoint_pattern = f"{problem}_n{n_node}_*_best.pt"
        checkpoints = list(checkpoint_dir.glob(checkpoint_pattern))

        if not checkpoints:
            print(f"[SKIP] No CVRP model found for N={n_node}")
            continue

        model_path = checkpoints[0]
        print(f"[INFO] Using model: {model_path}")

        cfg = DEFAULT_CONFIG.copy()
        cfg.update(CVRP_CONFIG)
        cfg["n_node"] = n_node

        metrics = test_model(model_path, cfg, dry_run)
        diag = collect_guidance_diagnostics(model_path, problem, n_node, cfg, val_size=min(int(cfg.get("val_size", 16)), 4))

        row = {
            "problem": problem,
            "n_node": n_node,
            "capacity": 1.0,  # Normalized capacity
            "n_routes": diag["route_count_mean"],
            "guidance_mean": diag["guidance_mean"],
            "guidance_std": diag["guidance_std"],
            "cost": metrics.get("cost", ""),
            "gap": metrics.get("gap", ""),
            "timestamp": datetime.now().isoformat(),
        }
        append_to_csv(csv_path, row, fieldnames)
        save_progress(key, {
            "cost": metrics.get("cost"),
            "gap": metrics.get("gap"),
            "guidance_mean": diag["guidance_mean"],
            "guidance_std": diag["guidance_std"],
            "suppression_mean": diag["suppression_mean"],
            "enhance_mean": diag["enhance_mean"],
            "rebellion_mean": diag["rebellion_mean"],
            "route_count_mean": diag["route_count_mean"],
        })

    print(f"\n[RESULTS] CVRP behavior data saved to {csv_path}")


# =============================================================================
# Phase 8: Guidance Transition Analysis
# =============================================================================

def run_guidance_transition_analysis(problem='tsp', n_node=1000, dry_run=False, only_test=False):
    """
    Phase 8: Guidance Transition Analysis

    Tracks when guidance transitions from enhancement to suppression.
    """
    print(f"\n=== Phase 8: Guidance Transition Analysis [{problem} N={n_node}] ===")

    results = load_progress()
    csv_path = RESULTS_DIR / "guidance_transition.csv"
    fieldnames = [
        "problem", "n_node", "iteration", "outer_step", "guidance_mean", "guidance_std",
        "enhance", "rebellion", "suppression", "cost", "crossover", "timestamp"
    ]

    # Find model
    checkpoint_dir = Path("checkpoints")
    checkpoint_pattern = f"{problem}_n{n_node}_*_best.pt"
    checkpoints = list(checkpoint_dir.glob(checkpoint_pattern))

    if not checkpoints:
        print(f"[ERROR] No model found for {problem} N={n_node}")
        return

    model_path = checkpoints[0]
    print(f"[INFO] Using model: {model_path}")

    cfg = DEFAULT_CONFIG.copy()
    if problem == 'cvrp':
        cfg.update(CVRP_CONFIG)
    else:
        cfg.update(TSP_CONFIG)
    cfg["n_node"] = n_node

    diag = collect_guidance_diagnostics(model_path, problem, n_node, cfg, val_size=min(int(cfg.get("val_size", 16)), 4))
    profiles = diag["profiles"]
    if not profiles:
        print(f"[SKIP] No guidance trajectories collected for {problem} N={n_node}")
        return

    n_steps = max(len(p["prior_mean"]) for p in profiles)
    crossover_iter = None
    for step_idx in range(n_steps):
        guidance_vals = [p["prior_mean"][step_idx] for p in profiles if step_idx < len(p["prior_mean"])]
        std_vals = [p["prior_std"][step_idx] for p in profiles if step_idx < len(p["prior_std"])]
        enhance_vals = [p["enhance"][step_idx] for p in profiles if step_idx < len(p["enhance"])]
        rebellion_vals = [p["rebellion"][step_idx] for p in profiles if step_idx < len(p["rebellion"])]
        suppression_vals = [p["suppression"][step_idx] for p in profiles if step_idx < len(p["suppression"])]
        cost_vals = [p["cost_curve"][step_idx] for p in profiles if step_idx < len(p["cost_curve"])]

        enhance_mean = float(np.mean(enhance_vals)) if enhance_vals else 0.0
        suppression_mean = float(np.mean(suppression_vals)) if suppression_vals else 0.0
        if crossover_iter is None and suppression_mean >= enhance_mean:
            crossover_iter = (step_idx + 1) * int(cfg.get("mini_H", DEFAULT_CONFIG["mini_H"]))

        row = {
            "problem": problem,
            "n_node": n_node,
            "iteration": (step_idx + 1) * int(cfg.get("mini_H", DEFAULT_CONFIG["mini_H"])),
            "outer_step": step_idx + 1,
            "guidance_mean": float(np.mean(guidance_vals)) if guidance_vals else 0.0,
            "guidance_std": float(np.mean(std_vals)) if std_vals else 0.0,
            "enhance": enhance_mean,
            "rebellion": float(np.mean(rebellion_vals)) if rebellion_vals else 0.0,
            "suppression": suppression_mean,
            "cost": float(np.mean(cost_vals)) if cost_vals else 0.0,
            "crossover": crossover_iter is not None and ((step_idx + 1) * int(cfg.get("mini_H", DEFAULT_CONFIG["mini_H"])) >= crossover_iter),
            "timestamp": datetime.now().isoformat(),
        }
        append_to_csv(csv_path, row, fieldnames)

    save_progress(
        f"guidance_{problem}_n{n_node}",
        {
            "guidance_mean": diag["guidance_mean"],
            "guidance_std": diag["guidance_std"],
            "suppression_mean": diag["suppression_mean"],
            "enhance_mean": diag["enhance_mean"],
            "rebellion_mean": diag["rebellion_mean"],
            "crossover_iter": crossover_iter,
        }
    )

    print(f"\n[RESULTS] Guidance transition data saved to {csv_path}")
    print(f"[INFO] Crossover iteration: {crossover_iter}")


# =============================================================================
# Phase 9: Scope and Applicability Analysis
# =============================================================================

def run_scope_applicability_analysis(n_node=1000, dry_run=False, only_test=False):
    """
    Phase 9: Scope and Applicability Analysis

    Tests on non-Euclidean instances and documents applicability to OP/KP/BPP.
    """
    print(f"\n=== Phase 9: Scope and Applicability Analysis ===")

    results = load_progress()
    csv_path = RESULTS_DIR / "scope_applicability.csv"
    fieldnames = ["problem", "instance", "distance_metric", "cost", "gap", "time", "timestamp"]

    for problem in ["tsp", "cvrp"]:
        base_cfg = DEFAULT_CONFIG.copy()
        if problem == "cvrp":
            base_cfg.update(CVRP_CONFIG)
        else:
            base_cfg.update(TSP_CONFIG)
        base_cfg["n_node"] = n_node

        model_path = get_model_path(base_cfg)
        alt_path = Path("checkpoints") / model_path.name
        if not model_path.exists() and alt_path.exists():
            model_path = alt_path
        if not model_path.exists():
            model_path = train_model(base_cfg, dry_run, wandb_project="reviewer_scope", only_test=only_test)
        if model_path is None or not model_path.exists():
            print(f"[SKIP] Model not found for {problem} N={n_node}")
            continue

        for domain_name, rl_data in [("synthetic", False), ("rl_data", True)]:
            summary_path = RESULTS_DIR / "tmp" / f"scope_{problem}_{domain_name}_n{n_node}.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            metrics = test_model(model_path, base_cfg, dry_run, summary_json=summary_path, rl_data=rl_data)
            summary = metrics.get("summary", {})
            _, best_method = choose_best_method(summary)

            row = {
                "problem": problem,
                "instance": f"{domain_name}_n{n_node}",
                "distance_metric": "library_realworld" if rl_data else "synthetic_euclidean",
                "cost": best_method.get("mean_cost", metrics.get("cost", "")),
                "gap": best_method.get("gap_pct", metrics.get("gap", "")),
                "time": best_method.get("mean_time_s", metrics.get("time", "")),
                "timestamp": datetime.now().isoformat(),
            }
            append_to_csv(csv_path, row, fieldnames)

    save_progress(
        f"scope_n{n_node}",
        {
            "n_node": n_node,
            "supported_domains": ["synthetic", "rl_data"],
            "unsupported_problems": ["op", "kp", "bpp"],
        }
    )

    print(f"\n[RESULTS] Scope applicability data saved to {csv_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reviewer Experiments - Comprehensive experiment pipeline"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=[
            "all",
            "srr_training_stability",
            "srr_inference",
            "srr",
            "pheromone_ablation",
            "hyperparam_sensitivity",
            "efficiency_analysis",
            "cross_scale",
            "cvrp_behavior",
            "guidance_transition",
            "scope_applicability",
        ],
        default="all",
        help="Which phase to run"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal steps to verify pipeline")
    parser.add_argument("--test", action="store_true", help="Only test existing models, do not train")
    parser.add_argument("--problem", type=str, choices=["tsp", "cvrp"], default=None, help="Specific problem to run experiments for")
    parser.add_argument("--n_node", type=int, default=1000, help="Problem size for experiments")
    parser.add_argument("--test-debug", action="store_true", help="Run with minimal parameters for quick end-to-end testing")

    args = parser.parse_args()

    # Apply debug parameters if requested
    if args.test_debug:
        print("[DEBUG] Applying minimal parameters for testing...")
        args.n_node = 20
        DEFAULT_CONFIG.update({
            "H": 10,
            "mini_H": 10,
            "n_ants": 10,
            "k_sparse": 10,
            "min_new_edges": 4,
            "epochs": 5,
            "steps_per_epoch": 4,
            "val_size": 2,
        })
        # Override TSP/CVRP specific if needed
        TSP_CONFIG["k_sparse"] = 10
        TSP_CONFIG["min_new_edges"] = 4
        CVRP_CONFIG["k_sparse"] = 10
        CVRP_CONFIG["min_new_edges"] = 4

    # Determine which problems to run
    problems = [args.problem] if args.problem else ["tsp", "cvrp"]

    print("=" * 80)
    print("Reviewer Experiments Pipeline")
    print("=" * 80)
    print(f"Phase: {args.phase}")
    print(f"Problems: {problems}")
    print(f"Dry run: {args.dry_run}")
    print(f"Test only: {args.test}")
    print("=" * 80)

    # Run the requested phase(s)
    if args.phase == "all":
        for problem in problems:
            print(f"\n\n{'=' * 80}")
            print(f"Running ALL phases for {problem.upper()}")
            print(f"{'=' * 80}")

            run_srr_training_stability(problem, args.n_node, args.dry_run, args.test)
            run_srr_inference_comparison(problem, args.n_node, args.dry_run, args.test)
            run_pheromone_ablation(problem, args.n_node, args.dry_run, args.test)
            run_hyperparam_sensitivity(problem, args.n_node, args.dry_run, args.test)
            run_efficiency_analysis(problem, args.n_node, args.dry_run, args.test)
            run_cross_scale_generalization(problem, args.n_node, args.dry_run, args.test)

            if problem == "cvrp":
                run_cvrp_behavior_analysis(args.n_node, args.dry_run, args.test)

            run_guidance_transition_analysis(problem, args.n_node, args.dry_run, args.test)
            run_scope_applicability_analysis(args.n_node, args.dry_run, args.test)

    elif args.phase == "srr" or args.phase == "srr_training_stability":
        for problem in problems:
            run_srr_training_stability(problem, args.n_node, args.dry_run, args.test)

    elif args.phase == "srr_inference":
        for problem in problems:
            run_srr_inference_comparison(problem, args.n_node, args.dry_run, args.test)

    elif args.phase == "pheromone_ablation":
        for problem in problems:
            run_pheromone_ablation(problem, args.n_node, args.dry_run, args.test)

    elif args.phase == "hyperparam_sensitivity":
        for problem in problems:
            run_hyperparam_sensitivity(problem, args.n_node, args.dry_run, args.test)

    elif args.phase == "efficiency_analysis":
        for problem in problems:
            run_efficiency_analysis(problem, args.n_node, args.dry_run, args.test)

    elif args.phase == "cross_scale":
        for problem in problems:
            run_cross_scale_generalization(problem, args.n_node, args.dry_run, args.test)

    elif args.phase == "cvrp_behavior":
        run_cvrp_behavior_analysis(args.n_node, args.dry_run, args.test)

    elif args.phase == "guidance_transition":
        for problem in problems:
            run_guidance_transition_analysis(problem, args.n_node, args.dry_run, args.test)

    elif args.phase == "scope_applicability":
        run_scope_applicability_analysis(args.n_node, args.dry_run, args.test)

    print("\n" + "=" * 80)
    print("Experiments completed!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Progress file: {PROGRESS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
