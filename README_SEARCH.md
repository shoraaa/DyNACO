# Hyperparameter Search for min_new_edges=8

## Overview

`search.py` is a systematic hyperparameter search script to find configurations that make `min_new_edges=8` work properly. Currently, mne=8 shows anti-training behavior (validation cost keeps increasing), while mne>=12 trains normally.

## Usage

### Basic Usage

```bash
# Run full search with default settings
python search.py --problem tsp --n_node 100 --epochs 5 --wandb

# Quick test run (shorter epochs, fewer configs)
python search.py --problem tsp --n_node 100 --test_run

# Run on GPU
python search.py --problem tsp --device cuda:0

# Skip certain phases
python search.py --skip_baseline --skip_grid  # Only run nudging phase
```

### Options

- `--problem`: Problem type (tsp or cvrp), default: tsp
- `--n_node`: Number of nodes (fixed at 100 for search), default: 100
- `--epochs`: Epochs per experiment, default: 5
- `--steps_per_epoch`: Steps per epoch, default: 32
- `--device`: Device to use, default: cuda:0
- `--seed`: Random seed, default: 1234

**Search phases:**
- `--skip_baseline`: Skip baseline phase
- `--skip_nudging`: Skip single-parameter nudging phase
- `--skip_grid`: Skip grid search phase

**Grid search settings:**
- `--grid_top_n`: Number of top parameters for grid search, default: 4
- `--grid_combination_size`: Size of parameter combinations, default: 2

**Output settings:**
- `--output_dir`: Output directory, default: search_results
- `--wandb`: Enable wandb logging
- `--force`: Force re-run even if logs exist
- `--test_run`: Quick test run with minimal configs

## Search Strategy

The search has 3 phases:

### Phase 1: Baseline
- Run with default config and mne=8
- Record baseline performance metrics
- Detect if anti-training behavior occurs

### Phase 2: Single-Parameter Nudging
Test each hyperparameter individually while keeping others at default:

**Continuous parameters:**
- `rho` (pheromone decay): [0.05, 0.1, 0.2, 0.3]
- `n_ants`: [50, 100, 150, 200]
- `k_sparse` (candidate list size): [8, 16, 24, 32]
- `H` (outer iterations): [5, 10, 20]
- `mini_H` (inner iterations): [50, 100, 150, 200]
- `ppo_lr` (learning rate): [1e-6, 3e-6, 1e-5, 3e-5]
- `ppo_clip`: [0.05, 0.1, 0.2, 0.3]
- `ppo_epochs`: [2, 4, 6, 8]
- `alpha` (pheromone weight): [0.5, 1.0, 1.5, 2.0]
- `beta` (heuristic weight): [0.5, 1.0, 1.5, 2.0]
- `gamma` (initial prior scale): [0.5, 1.0, 1.5, 2.0]
- `min_gamma` (final prior scale): [0.1, 0.2, 0.3, 0.5]
- `L` (fixed prior update steps): [0, 5, 10, 20]
- `nls_beta` (NLS cost weight): [0.3, 0.5, 0.7, 1.0]
- `T_nls` (NLS iterations): [5, 10, 15, 20]

**Boolean flags:**
- `no_smooth_mmas`: [False, True]
- `no_normalized_heuristic`: [False, True]
- `no_extend_ls`: [False, True]
- `no_local_search`: [False, True]
- `disable_heuristic`: [False, True]
- `no_dynamic_feats`: [False, True]
- `anneal_prior`: [False, True]
- `nls`: [False, True]

### Phase 3: Grid Search
- Select top 3-4 parameters with best improvement from Phase 2
- Test combinations of these parameters
- Focus on configurations that avoid anti-training behavior

## Output

Results are saved to `search_results/{problem}_n{n_node}_mne8_{timestamp}/`:

- `baseline.log`: Baseline training log
- `nudge_*.log`: Individual parameter nudging logs
- `grid_*.log`: Grid search configuration logs
- `phase2_nudging.csv`: All single-parameter nudging results
- `phase3_grid.csv`: Grid search results
- `all_results.csv`: Combined results from all phases
- `search_config.json`: Search configuration used

### CSV Columns

- `phase`: Search phase (baseline, nudge, or grid)
- `param`: Parameter name (for nudging phase)
- `value`: Parameter value tested
- `config_name`: Configuration name (for grid phase)
- `best_val`: Best validation cost achieved
- `final_val`: Final validation cost
- `trend`: Linear trend of validation cost (negative=improving, positive=degrading)
- `anti_training`: Whether anti-training behavior was detected
- `improvement`: Absolute improvement over baseline
- `improvement_pct`: Percentage improvement over baseline
- `n_epochs`: Number of epochs completed

## Anti-Training Detection

The script automatically detects anti-training behavior by:
1. Analyzing validation cost progression across epochs
2. Computing linear trend (slope) of validation costs
3. Checking if the second half of training shows 2%+ increase vs first half
4. Flagging configurations that exhibit increasing validation cost

## Interpreting Results

1. **Look for configurations that avoid anti-training**: These are the most important to fix the mne=8 issue
2. **Check the trend metric**: Negative values indicate improving performance
3. **Sort by best_val**: Lower is better
4. **Review improvement_pct**: Positive values show improvement over baseline

## Example

```bash
# Run a comprehensive search
python search.py --problem tsp --n_node 100 --epochs 10 --wandb

# Check results
cd search_results/tsp_n100_mne8_*/
cat all_results.csv | grep "False" | sort -t',' -k5 -n  # Find configs without anti-training
```

## Tips

- Use `--test_run` first to verify everything works
- Enable `--wandb` to track all experiments online
- Use `--force` to re-run experiments (otherwise cached logs are used)
- Start with fewer epochs (3-5) for quick iteration, then increase for final search
- Focus on configurations where `anti_training=False`
