#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description="Tuning script for MFACO")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True, help="1: Solver, 2: RL Stability, 3: Budget Factorization, 4: Ablations")
    parser.add_argument("--problem", type=str, required=True, choices=['tsp', 'cvrp'])
    parser.add_argument("--n_node", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--budget", type=int, default=1000, help="Total iterations S = H * mini_H")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run per config")
    parser.add_argument("--output_dir", type=str, default="tuning_results")
    return parser

def run_experiment(cmd_args, log_file):
    # Run the command and capture output
    cmd = [sys.executable, "train.py"] + cmd_args
    print(f"Running: {' '.join(cmd)}")
    
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

def parse_result_full(log_file):
    # Extract best valid, last valid, and time
    best_val = float('inf')
    last_time = 0.0
    found = False
    
    with open(log_file, 'r') as f:
        for line in f:
            if "ValBest=" in line:
                # Format: Epoch X: ... ValBest=Y ... Time=Zs
                parts = line.split()
                current_val = float('inf')
                current_time = 0.0
                current_time_n = 0.0
                current_time_a = 0.0
                
                for p in parts:
                    if p.startswith("ValBest="):
                        try:
                            current_val = float(p.split("=")[1])
                        except: pass
                    if p.startswith("Time="):
                        try:
                            current_time = float(p.split("=")[1].replace("s",""))
                        except: pass
                    if p.startswith("TimeN="):
                        try:
                            current_time_n = float(p.split("=")[1].replace("s",""))
                        except: pass
                    if p.startswith("TimeA="):
                        try:
                            current_time_a = float(p.split("=")[1].replace("s",""))
                        except: pass
                
                if current_val < best_val:
                    best_val = current_val
                
                # Update last observed times
                last_time = current_time
                last_time_n = current_time_n
                last_time_a = current_time_a
                
                found = True
    return (best_val, last_time, last_time_n, last_time_a) if found else (None, None, None, None)

def stage1_solver(args, output_dir):
    # Grid search: Vanilla vs Neural
    
    # Grid from previous user edit + safe defaults
    rhos = [0.02, 0.05, 0.1, 0.2]
    n_ants_list = [50, 100] # Reduced slightly for speed, user had [100]
    k_sparses = [16, 32]    # Reduced for speed
    mnes = [8, 12, 16]
    
    # If using full grid, it might be slow.
    # User specific instruction: "neural net can improve config where solver fail"
    
    grid = list(itertools.product(rhos, n_ants_list, k_sparses, mnes))
    print(f"Stage 1: {len(grid)} configs. Comparing Vanilla vs Neural (Short Train).")
    
    results = []
    
    for rho, ants, k, mne in grid:
        config_name = f"rho{rho}_ants{ants}_k{k}_mne{mne}"
        
        # 1. Vanilla Run
        vanilla_log = output_dir / f"{config_name}_vanilla.log"
        cmd_vanilla = [
            "--problem", args.problem,
            "--n_node", str(args.n_node),
            "--rho", str(rho),
            "--n_ants", str(ants),
            "--k_sparse", str(k),
            "--min_new_edges", str(mne),
            "--epochs", "0",        # Validation only (baseline wrapper runs valid -1)
            "--no_logit_net",       # Pure ACO
            "--device", args.device,
            "--seed", str(1000),
            "--run_name", f"{config_name}_vanilla",
            "--wandb_project", f"lga_stage{args.stage}"
        ] + get_defaults(args.problem)
        
        if not vanilla_log.exists():
            run_experiment(cmd_vanilla, vanilla_log)
            
        v_score, v_time, v_tn, v_ta = parse_result_full(vanilla_log)
        
        # 2. Neural Run (Neural Guided)
        neural_log = output_dir / f"{config_name}_neural.log"
        cmd_neural = [
            "--problem", args.problem,
            "--n_node", str(args.n_node),
            "--rho", str(rho),
            "--n_ants", str(ants),
            "--k_sparse", str(k),
            "--min_new_edges", str(mne),
            "--epochs", "5",        # Short training to see potential
            "--steps_per_epoch", "32",
            "--device", args.device,
            "--seed", str(2000),
            "--run_name", f"{config_name}_neural",
            "--wandb_project", f"lga_stage{args.stage}"
        ] + get_defaults(args.problem)
        
        if not neural_log.exists():
            run_experiment(cmd_neural, neural_log)
            
        n_score, n_time, n_tn, n_ta = parse_result_full(neural_log)
        
        # Record
        if v_score is not None and n_score is not None:
            results.append({
                "rho": rho, "n_ants": ants, "k_sparse": k, "min_new_edges": mne,
                "vanilla_score": v_score, "vanilla_time": v_time, "vanilla_tn": v_tn, "vanilla_ta": v_ta,
                "neural_score": n_score, "neural_time": n_time, "neural_tn": n_tn, "neural_ta": n_ta,
                "improvement": (v_score - n_score) / v_score * 100
            })
            print(f"Config {config_name} | V: {v_score:.4f} (T={v_time:.1f}s) | N: {n_score:.4f} (T={n_time:.1f}s, TN={n_tn:.2f}s, TA={n_ta:.2f}s)")
            
            # Incremental Save
            df = pd.DataFrame(results)
            df.to_csv(output_dir / "stage1_full_comparison.csv", index=False)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("neural_score")
        csv_path = output_dir / "stage1_full_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"Stage 1 Done. Best Neural:\n{df.head(1)}")


def parse_result(log_file, metric="ValBest"):
    # Legacy wrapper if needed, or remove calls to it
    best_val, _, _, _ = parse_result_full(log_file)
    return best_val

def get_defaults(problem):
    # Base defaults
    # User requested wandb logging, so we remove --no_wandb
    args = [] 
    if problem == 'cvrp':
        # Align with common sense defaults or user hints
        pass
    return args

def stage2_rl_stability(args, output_dir):
    # Fix solver params (assuming user provides them or we use defaults)
    # Tune: lr, ppo_clip, ppo_epochs, adv_norm
    
    # 1. PPO Sweep
    lrs = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    ppo_clips = [0.1, 0.2, 0.3]
    ppo_epochs_list = [1, 2, 4, 8]
    adv_norms = [True, False] 
    
    # Short run
    epochs = 5
    steps = 32
    
    results = []
    
    ppo_grid = list(itertools.product(lrs, ppo_clips, ppo_epochs_list, adv_norms))
    print(f"Stage 2: Tuninig PPO ({len(ppo_grid)} configs)")

    for lr, clip, ppo_ep, adv in ppo_grid:
        config_name = f"ppo_lr{lr}_clip{clip}_pe{ppo_ep}_adv{adv}"
        run_name = config_name
        scores = []
        for seed in range(args.seeds):
            log_file = output_dir / f"{config_name}_s{seed}.log"
            cmd = [
                "--algo", "ppo",
                "--problem", args.problem,
                "--n_node", str(args.n_node),
                "--lr", str(lr),
                "--ppo_clip", str(clip),
                "--ppo_epochs", str(ppo_ep),
                "--epochs", str(epochs),
                "--steps_per_epoch", str(steps),
                "--device", args.device,
                "--seed", str(2345 + seed),
                "--run_name", run_name
            ]
            if adv: cmd.append("--adv_norm")
            cmd += ["--wandb_project", f"lga_stage{args.stage}"]
            cmd += get_defaults(args.problem)
            
            if not log_file.exists():
                run_experiment(cmd, log_file)
            
            val = parse_result(log_file)
            if val is not None: scores.append(val)
            
        if scores:
            avg_score = sum(scores) / len(scores)
            results.append({
                "algo": "ppo",
                "lr": lr, "ppo_clip": clip, "ppo_epochs": ppo_ep, "adv_norm": adv,
                "score": avg_score
            })
            print(f"Config {config_name}: {avg_score:.4f}")
            
            # Incremental Save
            df = pd.DataFrame(results)
            df.to_csv(output_dir / "stage2_summary.csv", index=False)
            
    # 2. REINFORCE Sweep
    # Only LR matters usually (and maybe batch size but we fix steps/ants)
    reinforce_lrs = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    print(f"Stage 2: Tuninig REINFORCE ({len(reinforce_lrs)} configs)")
    
    for lr in reinforce_lrs:
        config_name = f"reinforce_lr{lr}"
        run_name = config_name
        scores = []
        for seed in range(args.seeds):
            log_file = output_dir / f"{config_name}_s{seed}.log"
            cmd = [
                "--algo", "reinforce",
                "--problem", args.problem,
                "--n_node", str(args.n_node),
                "--lr", str(lr),
                "--epochs", str(epochs),
                "--steps_per_epoch", str(steps),
                "--device", args.device,
                "--seed", str(2345 + seed),
                "--run_name", run_name,
                "--wandb_project", f"lga_stage{args.stage}"
            ]
            cmd += get_defaults(args.problem)
            
            if not log_file.exists():
                run_experiment(cmd, log_file)
                
            val = parse_result(log_file)
            if val is not None: scores.append(val)
            
        if scores:
            avg_score = sum(scores) / len(scores)
            results.append({
                "algo": "reinforce",
                "lr": lr, "ppo_clip": None, "ppo_epochs": None, "adv_norm": None,
                "score": avg_score
            })
            print(f"Config {config_name}: {avg_score:.4f}")

            # Incremental Save
            if results:
                 df = pd.DataFrame(results)
                 df.to_csv(output_dir / "stage2_summary.csv", index=False)

    if not results:
        print("No results collected for Stage 2. Check logs for errors.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("score")
    df.to_csv(output_dir / "stage2_summary.csv", index=False)
    print(f"Stage 2 Done. Best:\n{df.head(1)}")
    
    # Compare Best PPO vs Best REINFORCE
    best_ppo = df[df["algo"]=="ppo"].iloc[0] if not df[df["algo"]=="ppo"].empty else None
    best_reinforce = df[df["algo"]=="reinforce"].iloc[0] if not df[df["algo"]=="reinforce"].empty else None
    
    print("\nComparison:")
    if best_ppo is not None:
        print(f"Best PPO: {best_ppo['score']:.4f} (lr={best_ppo['lr']})")
    if best_reinforce is not None:
        print(f"Best REINFORCE: {best_reinforce['score']:.4f} (lr={best_reinforce['lr']})")

def stage3_budget(args, output_dir):
    # Tune Factorization of S = H * mini_H
    S = args.budget
    # Find factors
    factors = []
    for h in [10, 20, 50, 100]:
        if S % h == 0:
            factors.append((h, S // h))
    
    print(f"Stage 3: Testing factorizations {factors} for S={S}")
    
    results = []
    for H, mini_H in factors:
        config_name = f"H{H}_miniH{mini_H}"
        scores = []
        for seed in range(args.seeds):
            log_file = output_dir / f"{config_name}_s{seed}.log"
            cmd = [
                "--problem", args.problem,
                "--n_node", str(args.n_node),
                "--H", str(H),
                "--mini_H", str(mini_H),
                "--epochs", "5", # Longer run to see effect?
                "--steps_per_epoch", "32",
                "--device", args.device,
                "--seed", str(3456 + seed),
                "--run_name", config_name,
                "--wandb_project", f"lga_stage{args.stage}"
            ] + get_defaults(args.problem)
            
            if not log_file.exists():
                run_experiment(cmd, log_file)
            
            val = parse_result(log_file)
            if val is not None: scores.append(val)
            
        if scores:
            avg_score = sum(scores) / len(scores)
            results.append({"H": H, "mini_H": mini_H, "score": avg_score})
            print(f"Config {config_name}: {avg_score:.4f}")

            # Incremental Save
            df = pd.DataFrame(results)
            df.to_csv(output_dir / "stage3_summary.csv", index=False)

    df = pd.DataFrame(results)
    df = df.sort_values("score")
    df.to_csv(output_dir / "stage3_summary.csv", index=False)
    print(f"Stage 3 Done. Best:\n{df.head(1)}")

def stage4_ablations(args, output_dir):
    # Ablations
    flags = ["no_local_search", "no_smooth_mmas", "no_extend_ls", "no_normalized_heuristic"]
    
    # Baseline (no flags)
    baseline_cmd = []
    
    configs = {"baseline": []}
    for flag in flags:
        configs[flag] = [f"--{flag}"]
        
    results = []
    print(f"Stage 4: {len(configs)} ablations.")
    
    for name, extra_args in configs.items():
        scores = []
        for seed in range(args.seeds):
            log_file = output_dir / f"{name}_s{seed}.log"
            cmd = [
                "--problem", args.problem,
                "--n_node", str(args.n_node),
                "--epochs", "0", # Ablations on SOLVER? Or Net? User said "Tune as ablation, not full sweep" under Solver section.
                # So implies pure solver performance.
                "--no_logit_net",
                "--device", args.device,
                "--seed", str(4567 + seed),
                "--run_name", f"{name}_ablation",
                "--wandb_project", f"lga_stage{args.stage}"
            ] + extra_args + get_defaults(args.problem)
            
            if not log_file.exists():
                run_experiment(cmd, log_file)
                
            val = parse_result(log_file)
            if val is not None: scores.append(val)
            
        if scores:
            avg_score = sum(scores) / len(scores)
            results.append({"config": name, "score": avg_score})
            print(f"Config {name}: {avg_score:.4f}")
            
            # Incremental Save
            df = pd.DataFrame(results)
            df.to_csv(output_dir / "stage4_summary.csv", index=False)
            
    df = pd.DataFrame(results)
    df = df.sort_values("score")
    df.to_csv(output_dir / "stage4_summary.csv", index=False)
    print(f"Stage 4 Done. Best:\n{df.head(1)}")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / f"{args.problem}_{args.n_node}_stage{args.stage}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.stage == 1:
        stage1_solver(args, output_dir)
    elif args.stage == 2:
        stage2_rl_stability(args, output_dir)
    elif args.stage == 3:
        stage3_budget(args, output_dir)
    elif args.stage == 4:
        stage4_ablations(args, output_dir)
