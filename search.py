#!/usr/bin/env -S ./.venv/bin/python
"""
Hyperparameter Search for min_new_edges=8

This script systematically searches for hyperparameters that make min_new_edges=8
work properly (currently shows anti-training behavior).

Strategy:
1. Baseline: Test with default config and mne=8
2. Single-parameter nudging: Test each hyperparameter individually
3. Grid search: Test combinations of top promising parameters
"""

import argparse
import subprocess
import sys
import itertools
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

def parse_training_log(log_file):
    """
    Parse training log to extract:
    - Best validation cost
    - Final validation cost
    - All validation costs (to detect anti-training)
    - Training time
    """
    val_costs = []
    val_gaps = []
    best_val = float('inf')
    final_val = None
    train_time = 0.0
    
    with open(log_file, 'r') as f:
        for line in f:
            if "ValBest=" in line or "Epoch" in line and ":" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith("ValBest="):
                        try:
                            current_val = float(p.split("=")[1])
                            val_costs.append(current_val)
                            if current_val < best_val:
                                best_val = current_val
                            final_val = current_val
                        except: pass
                    if p.startswith("Gap="):
                        try:
                            gap = float(p.split("=")[1].replace("%", ""))
                            val_gaps.append(gap)
                        except: pass
    
    # Detect anti-training: check if validation cost is increasing over epochs
    anti_training = False
    if len(val_costs) >= 3:
        # Check if last 60% of epochs show increasing trend
        start_idx = int(len(val_costs) * 0.4)
        recent_costs = val_costs[start_idx:]
        if len(recent_costs) >= 2:
            # Simple trend: compare average of first half vs second half
            mid = len(recent_costs) // 2
            first_half_avg = np.mean(recent_costs[:mid])
            second_half_avg = np.mean(recent_costs[mid:])
            if second_half_avg > first_half_avg * 1.02:  # 2% increase threshold
                anti_training = True
    
    # Calculate trend (slope)
    trend = 0.0
    if len(val_costs) >= 2:
        x = np.arange(len(val_costs))
        trend = np.polyfit(x, val_costs, 1)[0]  # Linear trend slope
    
    return {
        'best_val': best_val if best_val != float('inf') else None,
        'final_val': final_val,
        'val_costs': val_costs,
        'val_gaps': val_gaps,
        'anti_training': anti_training,
        'trend': trend,
        'n_epochs': len(val_costs)
    }

def run_training(config, log_file, args):
    """Run training with given configuration"""
    cmd = [sys.executable, "train.py"]
    
    # Add all config parameters
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Log: {log_file}")
    
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    
    return result.returncode == 0

def get_base_config(args):
    """Get base configuration for all experiments"""
    config = {
        'problem': args.problem,
        'n_node': args.n_node,
        'min_new_edges': 8,  # Fixed: we're trying to make this work
        'device': args.device,
        'epochs': args.epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'no_wandb': not args.wandb,
        'seed': args.seed,
    }
    
    # Add wandb project if enabled
    if args.wandb:
        config['wandb_project'] = f'lga_{args.problem}_search_mne8'
    
    return config

def phase1_baseline(args, output_dir):
    """Phase 1: Run baseline with default config and mne=8"""
    print("\n" + "="*80)
    print("PHASE 1: BASELINE RUN (mne=8, n_node=100, default hyperparameters)")
    print("="*80 + "\n")
    
    config = get_base_config(args)
    
    # Add default hyperparameters (matching train.py defaults)
    defaults = {
        # ACO parameters
        'rho': 0.1,
        'n_ants': 100,
        'k_sparse': 32,
        'H': 10,
        'mini_H': 100,
        
        # PPO/RL parameters
        'ppo_lr': 5e-6,
        'ppo_clip': 0.1,
        'ppo_epochs': 4,
        'adv_norm': True,
        
        # Pheromone and heuristic weighting
        'alpha': 1.0,
        'beta': 1.0,
        
        # Prior annealing (disabled by default)
        'gamma': 1.0,
        'min_gamma': 0.2,
        
        # Fixed steps
        'L': 0,
        
        # NLS parameters (NLS disabled by default, so beta/T_nls won't be used unless nls=True)
        'nls_beta': 0.5,
        'T_nls': 10,
    }
    config.update(defaults)
    
    log_file = output_dir / "baseline.log"
    run_name = "baseline_mne8"
    if args.wandb:
        config['run_name'] = run_name
    
    if not log_file.exists() or args.force:
        run_training(config, log_file, args)
    
    result = parse_training_log(log_file)
    
    print(f"\nBaseline Results:")
    print(f"  Best Val: {result['best_val']:.4f}")
    print(f"  Final Val: {result['final_val']:.4f}")
    print(f"  Trend: {result['trend']:.6f} (negative=improving, positive=degrading)")
    print(f"  Anti-training detected: {result['anti_training']}")
    
    return result, defaults

def phase2_single_param_nudging(args, output_dir, baseline_result, defaults):
    """Phase 2: Nudge each hyperparameter individually"""
    print("\n" + "="*80)
    print("PHASE 2: SINGLE PARAMETER NUDGING")
    print("="*80 + "\n")
    
    # Define parameter ranges to test
    param_ranges = {
        # ACO parameters
        'n_ants': [1, 10, 50, 100],
        'k_sparse': [8, 16, 24, 32],
        'H': [1, 2, 5, 10, 20],
        'mini_H': [1, 10, 50, 100],
        
        # PPO/RL parameters
        'ppo_lr': [1e-6, 3e-6, 1e-5, 3e-5],
        'ppo_epochs': [1, 2, 4, 6, 8],
    }
    
    # Boolean flags to test
    flag_tests = {
        # MMAS and LS options
        'no_smooth_mmas': [False, True],
        'no_extend_ls': [False, True],
        'no_local_search': [False, True],
        
        'disable_heuristic': [False, True],
    }
    
    results = []
    
    # Test each parameter range
    for param, values in param_ranges.items():
        print(f"\nTesting parameter: {param}")
        print(f"Values: {values}")
        
        for value in values:
            # Skip if this is the default value
            if param in defaults and defaults[param] == value:
                print(f"  Skipping {param}={value} (baseline)")
                continue
            
            config = get_base_config(args)
            config.update(defaults)
            config[param] = value
            
            config_name = f"{param}_{value}"
            log_file = output_dir / f"nudge_{config_name}.log"
            
            if args.wandb:
                config['run_name'] = f"nudge_{config_name}"
            
            if not log_file.exists() or args.force:
                success = run_training(config, log_file, args)
                if not success:
                    print(f"  WARNING: Training failed for {config_name}")
            
            result = parse_training_log(log_file)
            
            if result['best_val'] is not None:
                improvement = baseline_result['best_val'] - result['best_val']
                improvement_pct = (improvement / baseline_result['best_val']) * 100
                
                results.append({
                    'phase': 'nudge',
                    'param': param,
                    'value': value,
                    'best_val': result['best_val'],
                    'final_val': result['final_val'],
                    'trend': result['trend'],
                    'anti_training': result['anti_training'],
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'n_epochs': result['n_epochs'],
                })
                
                print(f"  {param}={value}: Best={result['best_val']:.4f}, "
                      f"Trend={result['trend']:.6f}, Anti-training={result['anti_training']}, "
                      f"Improvement={improvement_pct:+.2f}%")
    
    # Test boolean flags
    for flag, values in flag_tests.items():
        print(f"\nTesting flag: {flag}")
        
        for value in values:
            if not value:  # Skip False (default behavior)
                continue
            
            config = get_base_config(args)
            config.update(defaults)
            config[flag] = value
            
            config_name = f"{flag}_{value}"
            log_file = output_dir / f"nudge_{config_name}.log"
            
            if args.wandb:
                config['run_name'] = f"nudge_{config_name}"
            
            if not log_file.exists() or args.force:
                success = run_training(config, log_file, args)
                if not success:
                    print(f"  WARNING: Training failed for {config_name}")
            
            result = parse_training_log(log_file)
            
            if result['best_val'] is not None:
                improvement = baseline_result['best_val'] - result['best_val']
                improvement_pct = (improvement / baseline_result['best_val']) * 100
                
                results.append({
                    'phase': 'nudge',
                    'param': flag,
                    'value': value,
                    'best_val': result['best_val'],
                    'final_val': result['final_val'],
                    'trend': result['trend'],
                    'anti_training': result['anti_training'],
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'n_epochs': result['n_epochs'],
                })
                
                print(f"  {flag}={value}: Best={result['best_val']:.4f}, "
                      f"Trend={result['trend']:.6f}, Anti-training={result['anti_training']}, "
                      f"Improvement={improvement_pct:+.2f}%")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('best_val')
        csv_path = output_dir / "phase2_nudging.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*80}")
        print("PHASE 2 SUMMARY - Top 10 configurations:")
        print('='*80)
        print(df.head(10)[['param', 'value', 'best_val', 'trend', 'anti_training', 'improvement_pct']])
        
        # Identify top parameters that reduce anti-training
        non_anti_training = df[~df['anti_training']]
        if len(non_anti_training) > 0:
            print(f"\n{'='*80}")
            print("Configurations that AVOID anti-training:")
            print('='*80)
            print(non_anti_training[['param', 'value', 'best_val', 'trend', 'improvement_pct']])
    
    return results

def phase3_grid_search(args, output_dir, nudging_results, baseline_result, defaults):
    """Phase 3: Grid search on top promising parameters"""
    print("\n" + "="*80)
    print("PHASE 3: GRID SEARCH ON TOP PARAMETERS")
    print("="*80 + "\n")
    
    if not nudging_results:
        print("No results from Phase 2, skipping grid search")
        return []
    
    df = pd.DataFrame(nudging_results)
    
    # Find top N parameters with best improvement
    top_params = {}
    for param in df['param'].unique():
        param_df = df[df['param'] == param]
        best_row = param_df.nsmallest(1, 'best_val').iloc[0]
        top_params[param] = {
            'value': best_row['value'],
            'improvement_pct': best_row['improvement_pct'],
            'anti_training': best_row['anti_training']
        }
    
    # Sort by improvement and select top 3-4
    sorted_params = sorted(top_params.items(), key=lambda x: x[1]['improvement_pct'], reverse=True)
    
    # Select top parameters for grid search, preferring those that avoid anti-training
    selected_params = []
    for param, info in sorted_params[:args.grid_top_n]:
        selected_params.append((param, info['value']))
        print(f"Selected for grid search: {param}={info['value']} "
              f"(improvement={info['improvement_pct']:+.2f}%, anti_training={info['anti_training']})")
    
    if len(selected_params) < 2:
        print("Not enough diverse parameters for grid search")
        return []
    
    # Create grid combinations
    grid_configs = []
    param_names = [p[0] for p in selected_params]
    param_values = [p[1] for p in selected_params]
    
    # Generate all combinations
    for combo in itertools.combinations(range(len(selected_params)), min(len(selected_params), args.grid_combination_size)):
        config = get_base_config(args)
        config.update(defaults)
        
        config_desc = []
        for idx in combo:
            param, value = selected_params[idx]
            config[param] = value
            config_desc.append(f"{param}_{value}")
        
        grid_configs.append({
            'config': config,
            'name': "_".join(config_desc)
        })
    
    # Also test full combination
    full_combo_config = get_base_config(args)
    full_combo_config.update(defaults)
    full_combo_desc = []
    for param, value in selected_params:
        full_combo_config[param] = value
        full_combo_desc.append(f"{param}_{value}")
    
    grid_configs.append({
        'config': full_combo_config,
        'name': "full_" + "_".join(full_combo_desc)
    })
    
    print(f"\nTesting {len(grid_configs)} grid configurations...")
    
    results = []
    for grid_config in grid_configs:
        config = grid_config['config']
        config_name = grid_config['name']
        
        log_file = output_dir / f"grid_{config_name}.log"
        
        if args.wandb:
            config['run_name'] = f"grid_{config_name}"
        
        print(f"\nTesting grid config: {config_name}")
        
        if not log_file.exists() or args.force:
            success = run_training(config, log_file, args)
            if not success:
                print(f"  WARNING: Training failed for {config_name}")
        
        result = parse_training_log(log_file)
        
        if result['best_val'] is not None:
            improvement = baseline_result['best_val'] - result['best_val']
            improvement_pct = (improvement / baseline_result['best_val']) * 100
            
            results.append({
                'phase': 'grid',
                'config_name': config_name,
                'best_val': result['best_val'],
                'final_val': result['final_val'],
                'trend': result['trend'],
                'anti_training': result['anti_training'],
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'n_epochs': result['n_epochs'],
            })
            
            print(f"  Best={result['best_val']:.4f}, Trend={result['trend']:.6f}, "
                  f"Anti-training={result['anti_training']}, Improvement={improvement_pct:+.2f}%")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('best_val')
        csv_path = output_dir / "phase3_grid.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*80}")
        print("PHASE 3 SUMMARY - Grid Search Results:")
        print('='*80)
        print(df[['config_name', 'best_val', 'trend', 'anti_training', 'improvement_pct']])
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Search for hyperparameters that make min_new_edges=8 work")
    
    # Problem settings
    parser.add_argument('--problem', type=str, default='tsp', choices=['tsp', 'cvrp'])
    parser.add_argument('--n_node', type=int, default=1000, help='Number of nodes (fixed at 100 for search)')
    
    # Search settings
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per experiment')
    parser.add_argument('--steps_per_epoch', type=int, default=32, help='Steps per epoch')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=1234)
    
    # Search phases
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline phase')
    parser.add_argument('--skip_nudging', action='store_true', help='Skip nudging phase')
    parser.add_argument('--skip_grid', action='store_true', help='Skip grid search phase')
    
    # Grid search settings
    parser.add_argument('--grid_top_n', type=int, default=4, help='Number of top parameters for grid search')
    parser.add_argument('--grid_combination_size', type=int, default=2, help='Size of parameter combinations')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='search_results')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--force', action='store_true', help='Force re-run even if logs exist')
    parser.add_argument('--test_run', action='store_true', help='Quick test run with minimal configs')
    
    args = parser.parse_args()
    
    # Adjust for test run
    if args.test_run:
        args.epochs = 2
        args.steps_per_epoch = 8
        args.grid_top_n = 2
        args.grid_combination_size = 2
        print("TEST RUN MODE: Using minimal settings")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.problem}_n{args.n_node}_mne8_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save search configuration
    with open(output_dir / "search_config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    all_results = []
    
    # Phase 1: Baseline
    if not args.skip_baseline:
        baseline_result, defaults = phase1_baseline(args, output_dir)
        baseline_info = {
            'phase': 'baseline',
            'param': 'baseline',
            'value': 'default',
            'best_val': baseline_result['best_val'],
            'final_val': baseline_result['final_val'],
            'trend': baseline_result['trend'],
            'anti_training': baseline_result['anti_training'],
            'improvement': 0.0,
            'improvement_pct': 0.0,
            'n_epochs': baseline_result['n_epochs'],
        }
        all_results.append(baseline_info)
    else:
        # Need to define defaults if skipping baseline
        defaults = {
            'rho': 0.1, 'n_ants': 100, 'k_sparse': 16, 'H': 10, 'mini_H': 100,
            'ppo_lr': 5e-6, 'ppo_clip': 0.1, 'ppo_epochs': 4, 'adv_norm': True,
            'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'min_gamma': 0.2, 'L': 0,
            'nls_beta': 0.5, 'T_nls': 10,
        }
        baseline_result = {'best_val': float('inf'), 'trend': 0.0, 'anti_training': True}
    
    # Phase 2: Single parameter nudging
    nudging_results = []
    if not args.skip_nudging:
        nudging_results = phase2_single_param_nudging(args, output_dir, baseline_result, defaults)
        all_results.extend(nudging_results)
    
    # Phase 3: Grid search
    grid_results = []
    if not args.skip_grid and nudging_results:
        grid_results = phase3_grid_search(args, output_dir, nudging_results, baseline_result, defaults)
        all_results.extend(grid_results)
    
    # Save combined results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "all_results.csv", index=False)
        
        print(f"\n{'='*80}")
        print("FINAL SUMMARY - All Results")
        print('='*80)
        print(f"Total configurations tested: {len(df)}")
        print(f"\nBest overall configuration:")
        best = df.nsmallest(1, 'best_val').iloc[0]
        print(f"  Phase: {best['phase']}")
        if 'param' in best:
            print(f"  Parameter: {best['param']}={best['value']}")
        elif 'config_name' in best:
            print(f"  Config: {best['config_name']}")
        print(f"  Best Val: {best['best_val']:.4f}")
        print(f"  Trend: {best['trend']:.6f}")
        print(f"  Anti-training: {best['anti_training']}")
        print(f"  Improvement: {best['improvement_pct']:+.2f}%")
        
        # Show configs that avoid anti-training
        non_anti = df[~df['anti_training']]
        if len(non_anti) > 0:
            print(f"\n{'='*80}")
            print(f"Configurations avoiding anti-training ({len(non_anti)} total):")
            print('='*80)
            print(non_anti.nsmallest(5, 'best_val'))
        
        print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()
