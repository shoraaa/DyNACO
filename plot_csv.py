import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Plot CSV files in csv/ directory')
    parser.add_argument('--csv_dir', type=str, default='csv', help='Directory containing csv files')
    parser.add_argument('--out_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--title', type=str, default=None, help='Custom title/caption for the plot')
    return parser.parse_args()

def setup_plotting_style():
    # Style settings from plot_aco_performance.py
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 22, 'axes.titlesize': 26,
        'axes.labelsize': 24, 'xtick.labelsize': 22, 'ytick.labelsize': 22,
        'legend.fontsize': 20, 'lines.linewidth': 4, 'lines.markersize': 0,
        'figure.titlesize': 28
    })

def plot_csv_file(csv_file, out_dir, title=None):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    # Assuming 'Step' is the index/x-axis
    if 'Step' not in df.columns:
        print(f"Skipping {csv_file}: 'Step' column not found.")
        return
    
    x = df['Step']
    
    # Identify value columns
    # Exclude Step
    feature_cols = [c for c in df.columns if c != 'Step']
    
    # Group by base feature name to find mean, min, max
    # Heuristic: MIN/MAX usually end with __MIN or __MAX
    base_features = set()
    for c in feature_cols:
        if c.endswith('__MIN'):
            base_features.add(c[:-5])
        elif c.endswith('__MAX'):
            base_features.add(c[:-5])
        else:
            base_features.add(c)
            
    # Remove base features if they are just suffixes of others (unlikely with __MIN/__MAX scheme)
    
    setup_plotting_style()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(base_features)))
    
    for i, base_name in enumerate(sorted(base_features)):
        
        # Verify if base column exists or if we only have min/max
        # The prompt implies we have value, min, max.
        
        col_mean = base_name
        col_min = f"{base_name}__MIN"
        col_max = f"{base_name}__MAX"
        
        # Check availability
        has_mean = col_mean in df.columns
        has_min = col_min in df.columns
        has_max = col_max in df.columns
        
        if not has_mean and not (has_min and has_max):
            continue
            
        color = colors[i]
        label = base_name
        
        # Simplify label: extract N value from patterns like "tsp_n100000_..."
        import re
        match = re.search(r'_n(\d+)_', label)
        if match:
            clean_label = f"N = {match.group(1)}"
        else:
            clean_label = label.replace(' - train/', ': ').replace('_rho0.5_mne12_ppo_lr5e-06', '')
        
        if has_mean:
            plt.plot(x, df[col_mean], label=clean_label, color=color)
            
        if has_min and has_max:
            plt.fill_between(x, df[col_min], df[col_max], color=color, alpha=0.2)
            
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(title if title else os.path.basename(csv_file))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    
    # Output file
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    out_path = os.path.join(out_dir, f"{base_filename}.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")

def main():
    args = parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    csv_files = glob.glob(os.path.join(args.csv_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.csv_dir}")
        return
        
    print(f"Found {len(csv_files)} CSV files in {args.csv_dir}")
    
    for f in csv_files:
        plot_csv_file(f, args.out_dir, args.title)

if __name__ == '__main__':
    main()
