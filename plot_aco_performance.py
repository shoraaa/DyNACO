import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Plot ACO Performance (Gap%) from CSV logs')
    parser.add_argument('csv_file', type=str, help='Path to the _iters.csv log file')
    parser.add_argument('--output', type=str, default='aco_performance.pdf', help='Output file name')
    parser.add_argument('--start_iter', type=int, default=0, help='Start iteration for plotting')
    parser.add_argument('--end_iter', type=int, default=None, help='End iteration for plotting')
    return parser.parse_args()

def load_optimal_costs(iters_csv_file):
    # Infer instances file path
    # Expected format: X_iters.csv -> X_instances.csv
    base, ext = os.path.splitext(iters_csv_file)
    if base.endswith('_iters'):
        instances_file = base.replace('_iters', '_instances') + ext
    else:
        # Fallback or strict assumption? Let's try to strip and append
        instances_file = base + "_instances" + ext
    
    if not os.path.exists(instances_file):
        print(f"Warning: Instances file not found at {instances_file}. Cannot calculate Gap.")
        return None
        
    opt_costs = {}
    with open(instances_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['idx'])
            if 'opt' in row and row['opt']:
                 opt_costs[idx] = float(row['opt'])
    return opt_costs

def read_data(csv_file, opt_costs, start_iter=0, end_iter=None):
    data = defaultdict(lambda: defaultdict(list))
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            anneal = row['anneal']
            try:
                iteration = int(row['iter'])
                # Filter by iteration
                if iteration < start_iter:
                    continue
                if end_iter is not None and iteration > end_iter:
                     continue
                     
                best_val = float(row['best'])
                idx = int(row['idx'])
            except ValueError:
                continue
                
            # If we have optimal costs, calculate gap. Else skip or log raw?
            # User requested GAP%. If opt missing for an instance, we can't compute gap.
            if opt_costs and idx in opt_costs:
                opt = opt_costs[idx]
                val = (best_val - opt) / opt * 100
            else:
                # Fallback if no opt cost found? Or just skip?
                # If we mix raw cost and gap, plot is meaningless.
                # Assume if opt_costs is provided, we ONLY plot gap.
                if opt_costs is None:
                     val = best_val # Plots raw cost if instances file missing
                else:
                     continue # Skip this data point if we can't compute gap
            
            # Determine category
            if method == 'Base':
                category = 'Base'
            elif method == 'Model':
                if anneal == 'on':
                    category = 'Model'
                elif anneal == 'off':
                    category = 'Model (no anneal)'
                else:
                    category = f'Model ({anneal})' # Fallback
            elif method == 'Mix':
                if anneal == 'on':
                    category = 'Mix (anneal)'
                elif anneal == 'off':
                    category = 'Mix (no anneal)'
                else:
                    category = f'Mix ({anneal})' # Fallback
            else:
                continue # Skip unknown methods
                
            data[category][iteration].append(val)
            
    return data, (opt_costs is not None)

def aggregate_data(raw_data):
    aggregated = {}
    for category, iterations in raw_data.items():
        sorted_iters = sorted(iterations.keys())
        means = []
        stds = []
        
        for it in sorted_iters:
            values = iterations[it]
            means.append(np.mean(values))
            stds.append(np.std(values))
            
        aggregated[category] = {
            'iters': np.array(sorted_iters),
            'means': np.array(means),
            'stds': np.array(stds)
        }
    return aggregated

def plot_performance(data, output_file, is_gap):
    # Style settings from plot_2opt.py
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 18, 'axes.titlesize': 22,
        'axes.labelsize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
        'legend.fontsize': 16, 'lines.linewidth': 4, 'lines.markersize': 0,
        'figure.titlesize': 24
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors/styles for known categories
    styles = {
        'Base': {'color': 'black', 'linestyle': '--', 'label': 'Base'},
        'Model': {'color': '#1f77b4', 'linestyle': '-', 'label': 'Model'},
        'Model (no anneal)': {'color': '#1f77b4', 'linestyle': ':', 'label': 'Model (no anneal)'},
        'Mix (anneal)': {'color': '#ff7f0e', 'linestyle': '-', 'label': 'Mix (anneal)'},
        'Mix (no anneal)': {'color': '#ff7f0e', 'linestyle': ':', 'label': 'Mix (no anneal)'},
    }
    
    # Iterate over specific order if present, else all keys
    order = ['Base', 'Model', 'Model (no anneal)', 'Mix (anneal)', 'Mix (no anneal)']
    
    for category in order:
        if category not in data:
            continue
            
        cat_data = data[category]
        x = cat_data['iters']
        y = cat_data['means']
        y_std = cat_data['stds']
        
        style = styles.get(category, {'label': category})
        
        ax.plot(x, y, **style)
        
        # Fill between for std dev
        if 'color' in style:
            ax.fill_between(x, y - y_std, y + y_std, color=style['color'], alpha=0.15)
        else:
             # Fallback if color not defined in style dict (shouldn't happen for known keys)
             ax.fill_between(x, y - y_std, y + y_std, alpha=0.15)

    ax.set_title('ACO Performance')
    ax.set_xlabel('Iterations')
    if is_gap:
        ax.set_ylabel('Gap (%)')
    else:
        ax.set_ylabel('Objective Cost (Best)')
        
    # Add vertical dotted lines at 0, 100, 200...
    if data:
        all_iters = [cat_data['iters'] for cat_data in data.values() if len(cat_data['iters']) > 0]
        if all_iters:
            max_iter = int(max([it.max() for it in all_iters]))
            min_iter_plot = int(min([it.min() for it in all_iters]))
            
            # Ensure we start drawing lines from optimal points relative to global 0
            # Next multiple of 100 >= min_iter_plot
            start_line = (min_iter_plot // 100) * 100
            if start_line < min_iter_plot:
                 start_line += 100
                 
            for i in range(0, max_iter + 1, 100):
                if i >= min_iter_plot:
                     ax.axvline(x=i, linestyle=':', color='gray', alpha=0.5, linewidth=2)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best') # specific location or best
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    # plt.show() # Commented out for non-interactive environments

def main():
    args = parse_args()
    opt_costs = load_optimal_costs(args.csv_file)
    raw_data, is_gap = read_data(args.csv_file, opt_costs, args.start_iter, args.end_iter)
    agg_data = aggregate_data(raw_data)
    plot_performance(agg_data, args.output, is_gap)

if __name__ == '__main__':
    main()
