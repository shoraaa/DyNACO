#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def setup_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6)
    })

# Copied from tune.py
def parse_result_full_log(log_file):
    best_val = float('inf')
    last_time = 0.0
    found = False
    last_time_n = 0.0
    last_time_a = 0.0
    
    with open(log_file, 'r') as f:
        for line in f:
            if "ValBest=" in line:
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
                last_time = current_time 
                last_time_n = current_time_n
                last_time_a = current_time_a
                found = True
    return (best_val, last_time, last_time_n, last_time_a) if found else (None, None, None, None)

def reconstruct_from_logs(input_dir):
    print(f"Scanning logs in {input_dir}...")
    logs = list(input_dir.glob("*.log"))
    if not logs:
        return None, 0
    
    # Try to detect stage from first log filename or context
    first = logs[0].name
    stage = 0
    
    # Stage 1: rho0.02_ants50_k16_mne16_vanilla.log
    if "rho" in first and "ants" in first:
        stage = 1
    # Stage 2: ppo_lr... or reinforce_lr...
    elif "ppo" in first or "reinforce" in first:
        stage = 2
    # Stage 3: H..._miniH...
    elif "H" in first and "miniH" in first:
        stage = 3
    # Stage 4: defaults
    else:
        stage = 4
        
    results = []
    
    if stage == 1:
        # We need to pair vanilla and neural
        # Dictionary config -> type -> metrics
        data_map = {}
        for p in logs:
            name = p.stem
            # e.g. rho0.02_ants50_k16_mne16_vanilla
            # extract type
            if name.endswith("_vanilla"):
                ctype = "vanilla"
                base = name[:-8]
            elif name.endswith("_neural"):
                ctype = "neural"
                base = name[:-7]
            else:
                continue
                
            score, t, tn, ta = parse_result_full_log(p)
            if score is None: continue
            
            if base not in data_map: data_map[base] = {}
            data_map[base][ctype] = (score, t, tn, ta)
            
            # Parse params from base string
            # rho0.02_ants50_k16_mne16
            try:
                # Naive parsing
                parts = base.split("_")
                for part in parts:
                    if part.startswith("rho"): rho = float(part[3:])
                    if part.startswith("ants"): ants = int(part[4:])
                    if part.startswith("k"): k = int(part[1:])
                    if part.startswith("mne"): mne = int(part[3:])
                data_map[base]["params"] = (rho, ants, k, mne)
            except: pass

        for base, vals in data_map.items():
            if "vanilla" in vals and "neural" in vals and "params" in vals:
                v_s, v_t, v_tn, v_ta = vals["vanilla"]
                n_s, n_t, n_tn, n_ta = vals["neural"]
                rho, ants, k, mne = vals["params"]
                
                results.append({
                    "rho": rho, "n_ants": ants, "k_sparse": k, "min_new_edges": mne,
                    "vanilla_score": v_s, "vanilla_time": v_t, "vanilla_tn": v_tn, "vanilla_ta": v_ta,
                    "neural_score": n_s, "neural_time": n_t, "neural_tn": n_tn, "neural_ta": n_ta,
                    "improvement": (v_s - n_s) / v_s * 100
                })
                
    elif stage == 2:
        # ppo_lr1e-06_clip0.1_pe1_advTrue_s0.log
        for p in logs:
            name = p.stem
            score, _, _, _ = parse_result_full_log(p)
            if score is None: continue
            
            row = {"score": score}
            if "ppo" in name:
                row["algo"] = "ppo"
                # Parse params
                try:
                    parts = name.split("_")
                    for part in parts:
                        if part.startswith("lr"): row["lr"] = float(part[2:])
                        if part.startswith("clip"): row["ppo_clip"] = float(part[4:])
                except: pass
            elif "reinforce" in name:
                row["algo"] = "reinforce"
                try:
                    parts = name.split("_")
                    for part in parts:
                        if part.startswith("lr"): row["lr"] = float(part[2:])
                except: pass
            results.append(row)
            
    elif stage == 3:
        # H100_miniH10_s0
        for p in logs:
            name = p.stem
            score, _, _, _ = parse_result_full_log(p)
            if score is None: continue
            
            # H100_miniH10
            try:
                parts = name.split("_")
                H = int(parts[0][1:])
                miniH = int(parts[1][5:])
                results.append({"H": H, "mini_H": miniH, "score": score})
            except: pass

    elif stage == 4:
        for p in logs:
            name = p.stem
            score, _, _, _ = parse_result_full_log(p)
            if score is None: continue
            # name is config name usually
             # strip _sSeed
            config = name.split("_s")[0]
            results.append({"config": config, "score": score})

    if results:
        df = pd.DataFrame(results)
        print(f"Reconstructed {len(df)} rows from logs.")
        # Save it for future?
        df.to_csv(input_dir / "reconstructed_summary.csv", index=False)
        return df, stage
    
    return None, 0

def load_csv(input_dir, pattern="*.csv"):
    files = list(input_dir.glob(pattern))
    if not files:
        print(f"No CSV files found in {input_dir} matching {pattern}")
        return None, None
    
    # Heuristic detection of stage based on filename
    for p in files:
        name = p.name
        if "stage1" in name: return p, 1
        if "stage2" in name: return p, 2
        if "stage3" in name: return p, 3
        if "stage4" in name: return p, 4
    
    return files[0], 0 # Unknown

def plot_stage1(csv_path, output_dir):
    print(f"Plotting Stage 1 results from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Vanilla vs Neural Score Scatter
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x="vanilla_score", y="neural_score", hue="rho", size="n_ants", style="mn_new_edges" if "mn_new_edges" in df.columns else "min_new_edges", sizes=(50, 200))
    
    # Identity line
    min_val = min(df["vanilla_score"].min(), df["neural_score"].min())
    max_val = max(df["vanilla_score"].max(), df["neural_score"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Equal Performance")
    
    plt.title("Stage 1: Vanilla vs Neural Score")
    plt.xlabel("Vanilla Score (Lower is Better)")
    plt.ylabel("Neural Score (Lower is Better)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "stage1_vanilla_vs_neural.pdf")
    plt.close()
    
    # 2. Improvement Bar Plot (Top 10)
    df_imp = df.sort_values("improvement", ascending=False).head(20)
    df_imp["config_id"] = df_imp.apply(lambda row: f"rho{row['rho']}_k{row['k_sparse']}", axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x="config_id", y="improvement", hue="n_ants")
    plt.xticks(rotation=45)
    plt.title("Top Configs by % Improvement Neural/Vanilla")
    plt.ylabel("% Improvement")
    plt.tight_layout()
    plt.savefig(output_dir / "stage1_improvement.pdf")
    plt.close()

    # 3. Time vs Score Tradeoff
    plt.figure(figsize=(10, 6))
    # Vanilla
    plt.scatter(df["vanilla_time"], df["vanilla_score"], label="Vanilla", alpha=0.6, marker='o')
    # Neural
    plt.scatter(df["neural_time"], df["neural_score"], label="Neural", alpha=0.6, marker='x')
    
    # Draw arrows
    for i, row in df.iterrows():
        plt.arrow(row["vanilla_time"], row["vanilla_score"], 
                  row["neural_time"]-row["vanilla_time"], row["neural_score"]-row["vanilla_score"],
                  color='gray', alpha=0.2, head_width=0)
        
    plt.title("Time vs Score Tradeoff")
    plt.xlabel("Time (s)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "stage1_time_score.pdf")
    plt.close()
    
    # 4. Neural / ACO Time Ratio Heatmap
    if "neural_tn" in df.columns and "neural_ta" in df.columns:
        df["feat_ratio"] = df["neural_tn"] / (df["neural_ta"] + 1e-6)
        
        # Pivot for heatmap: Rho vs K (avg over others)
        pivot = df.pivot_table(index="rho", columns="k_sparse", values="feat_ratio", aggfunc="mean")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Neural/ACO Time Ratio (Pivot: Rho vs K)")
        plt.tight_layout()
        plt.savefig(output_dir / "stage1_time_ratio.pdf")
        plt.close()

def plot_stage2(csv_path, output_dir):
    print(f"Plotting Stage 2 results from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Score vs LR per Algo
    plt.figure(figsize=(10, 6))
    
    if "algo" in df.columns:
        sns.lineplot(data=df, x="lr", y="score", hue="algo", marker="o")
    else:
        sns.lineplot(data=df, x="lr", y="score", marker="o")
        
    plt.xscale("log")
    plt.title("Stage 2: Learning Rate Sensitivity")
    plt.ylabel("Validation Score (Best)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_lr_sensitivity.pdf")
    plt.close()
    
    # PPO Clips (Boxplot) if PPO exists
    if "ppo_clip" in df.columns and not df["ppo_clip"].isna().all():
        ppo_df = df[df["algo"]=="ppo"] if "algo" in df.columns else df
        if not ppo_df.empty:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=ppo_df, x="ppo_clip", y="score")
            plt.title("PPO Clip Sensitivity")
            plt.tight_layout()
            plt.savefig(output_dir / "stage2_ppo_clip.pdf")
            plt.close()

def plot_stage3(csv_path, output_dir):
    print(f"Plotting Stage 3 results from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Score vs H (assuming S is constant)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="H", y="score", marker="o")
    plt.title("Stage 3: Budget Factorization (Score vs H)")
    plt.xlabel("H (Frequency of Neural Update)")
    plt.ylabel("Validation Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "stage3_budget.pdf")
    plt.close()

def plot_stage4(csv_path, output_dir):
    print(f"Plotting Stage 4 results from {csv_path}")
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="config", y="score")
    plt.title("Stage 4: Ablation Studies")
    plt.ylabel("Validation Score (Lower is Better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "stage4_ablations.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing tuning CSVs")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        return

    out_path = Path(args.output_dir) if args.output_dir else input_path
    out_path.mkdir(exist_ok=True, parents=True)
    
    setup_style()
    
    # Auto-detect CSVs
    csv_file, stage = load_csv(input_path)
    df = None
    
    if csv_file:
        print(f"Loading {csv_file} (Stage {stage})")
        df = pd.read_csv(csv_file)
    else:
        print("No summary CSV found. Attempting reconstruction from logs...")
        df, stage = reconstruct_from_logs(input_path)
    
    if df is not None:
        if stage == 1:
            # Need to save to temp file to reuse plot function or pass DF
            # Modify functions to accept DF? 
            # Or just save to file which was done in reconstruct
            # Reuse csv_path if reconstruct saved it
            csv_path = input_path / "reconstructed_summary.csv"
            if csv_path.exists(): plot_stage1(csv_path, out_path)
        elif stage == 2:
            plot_stage2(input_path / "reconstructed_summary.csv" if not csv_file else csv_file, out_path)
        elif stage == 3:
            plot_stage3(input_path / "reconstructed_summary.csv" if not csv_file else csv_file, out_path)
        elif stage == 4:
            plot_stage4(input_path / "reconstructed_summary.csv" if not csv_file else csv_file, out_path)
    else:
        print("Could not find data to plot.")

if __name__ == "__main__":
    main()
