#!/usr/bin/env python3
import os
import re
import argparse
import glob
import datetime
from collections import defaultdict

# =============================================================================
# Parsing Logic
# =============================================================================

def parse_filename_config(filename):
    """
    Extracts configuration parameters from the filename.
    Expected format: test_{problem}_{dataset}_... or similar variations.
    Returns a dictionary of parsed keys.
    """
    config = {}
    
    # Remove extension and 'test_' prefix if present
    name = os.path.splitext(os.path.basename(filename))[0]
    if name.startswith("test_"):
        name = name[5:]
        
    patterns = {
        'problem': r'(tsp|cvrp)',
        'n_node': r'(?:n|tsp_|cvrp_)(\d+)(?:_|$)',
        'k_sparse': r'k(\d+)',
        'ants': r'ants(\d+)',
        'H': r'H(\d+)(?!_mini)',
        'mini_H': r'miniH(\d+)',
        'rho': r'rho([\d.]+)',
        'mne': r'mne(\d+)',
        'algo': r'(ppo|reinforce)',
        'lr': r'lr([\de.-]+)',
        'anneal': r'annealing(True|False)',
        'timestamp': r'(\d{8}_\d{6})' 
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            val = match.group(1)
            # Convert to number if possible
            try:
                if '.' in val or 'e' in val:
                    val = float(val)
                else:
                    val = int(val)
            except:
                pass
            config[key] = val
            
    return config

def parse_log_content(filepath):
    """
    Parses the content of the log file to extract results.
    """
    metrics = {
        'base_gap': None,
        'model_gap': None,
        'mix_gap': None,
        'time': None,
        'status': 'Incomplete'
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check for completion
        if "--- Results ---" in content or "--- Summary" in content:
            metrics['status'] = 'Done'
        else:
            metrics['status'] = 'Running/Failed'
            # Check last modified time to guess if it's dead
            mtime = os.path.getmtime(filepath)
            if datetime.datetime.now().timestamp() - mtime > 3600: # 1 hour
                metrics['status'] = 'Stalled'

        # Parse Gaps (Model Gap, Base Gap)
        
        # Base Gap
        base_gap = re.search(r"Base Gap(?: to Baseline)?: \s*([-\d.]+)%", content)
        if base_gap:
            metrics['base_gap'] = float(base_gap.group(1))
            
        # Model Gap
        model_gap = re.search(r"Model Gap(?: to Baseline)?(?: \(anneal\))?: \s*([-\d.]+)%", content)
        # Handle nan
        if model_gap:
            val = model_gap.group(1)
            if 'nan' in val.lower():
                 metrics['model_gap'] = float('nan')
            else:
                 metrics['model_gap'] = float(val)

        # Mix Gap
        mix_gap = re.search(r"Mix Gap(?: to Baseline)?(?: \(anneal\))?: \s*([-\d.]+)%", content)
        if mix_gap:
            val = mix_gap.group(1)
            if 'nan' in val.lower():
                 metrics['mix_gap'] = float('nan')
            else:
                 metrics['mix_gap'] = float(val)
                 
        # Time
        model_time = re.search(r"Model Cost.*Time: \s*([\d.]+)s", content)
        if model_time:
            metrics['time'] = float(model_time.group(1))
        
    except Exception as e:
        metrics['error'] = str(e)
        
    return metrics

def get_runs(log_dir):
    # Search recursively for .txt and .log files
    files = glob.glob(os.path.join(log_dir, "**", "*.txt"), recursive=True) + \
            glob.glob(os.path.join(log_dir, "**", "*.log"), recursive=True)
            
    runs = []
    
    for f in files:
        if os.path.isdir(f): continue
        
        # Skip small files (empty logs)
        if os.path.getsize(f) < 50: 
            continue
            
        config = parse_filename_config(f)
        metrics = parse_log_content(f)
        
        run_data = {
            'file': os.path.basename(f),
            'mtime': datetime.datetime.fromtimestamp(os.path.getmtime(f)),
            **config,
            **metrics
        }
        runs.append(run_data)
        
    return runs

# =============================================================================
# Display Logic
# =============================================================================

def format_row(run, cols, col_widths):
    row = []
    for col in cols:
        val = run.get(col, '')
        width = col_widths.get(col, 10)
        
        # Formatting string without color first
        if val is None:
            s_content = "-"
        elif isinstance(val, float):
             # Check for nan
            if val != val: # NaN
                 s_content = "nan"
            else:
                if 'gap' in col:
                    s_content = f"{val:6.2f}%"
                elif 'time' in col:
                    s_content = f"{val:6.1f}s"
                elif 'lr' in col:
                    s_content = f"{val:.0e}"
                else:
                    s_content = f"{val:.2f}"
        elif isinstance(val, datetime.datetime):
            s_content = val.strftime("%m-%d %H:%M")
        else:
            s_content = str(val)
            
        # Truncate content
        if len(s_content) > width:
            s_content = s_content[:width-1] + '.'
            
        # Apply Color
        s_final = s_content
        if isinstance(val, float) and val == val: # Not NaN
            if 'gap' in col:
                if val < 0.05: # Excellent
                    s_final = f"\033[92m{s_content}\033[0m"
                elif val < 1.0: # Good
                    s_final = f"\033[93m{s_content}\033[0m"
        elif 'status' in col:
            if val == 'Done':
                s_final = f"\033[92m{s_content}\033[0m"
            elif val == 'Running/Failed':
                s_final = f"\033[93m{s_content}\033[0m"
            elif val == 'Stalled':
                s_final = f"\033[91m{s_content}\033[0m"

        # Padding based on visible length
        padding = " " * (width - len(s_content))
        row.append(s_final + padding)
        
    return " | ".join(row)

def main():
    parser = argparse.ArgumentParser(description="View training/testing logs")
    parser.add_argument("--dir", default="logs", help="Log directory")
    parser.add_argument("--cols", default="mtime,problem,n_node,base_gap,model_gap,mix_gap,time,status", help="Columns to display")
    parser.add_argument("--sort", default="mtime", help="Sort column")
    parser.add_argument("--reverse", action="store_true", help="Reverse sort order (default is desc for time, asc for others)")
    parser.add_argument("--filter", help="Filter string (e.g. 'n_node=1000,problem=tsp')")
    parser.add_argument("--grep", help="Grep filter for filename")
    parser.add_argument("--limit", type=int, default=20, help="Max rows")
    parser.add_argument("--all", action="store_true", help="Show all rows")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Directory {args.dir} not found.")
        return

    all_runs = get_runs(args.dir)
    runs = all_runs
    
    # Filtering
    if args.filter:
        filters = args.filter.split(',')
        for f in filters:
            if '=' in f:
                k, v = f.split('=')
                # Try to convert v
                try:
                    if '.' in v: v = float(v)
                    else: v = int(v)
                except:
                    pass
                runs = [r for r in runs if r.get(k) == v]
    
    if args.grep:
        runs = [r for r in runs if args.grep in r['file']]
                
    # Sorting
    # Determine default reverse
    if args.sort == 'mtime':
        default_reverse = True
    else:
        default_reverse = False
        
    reverse = default_reverse
    if args.reverse:
        reverse = not reverse # Toggle
    
    # Handle missing sort keys
    runs.sort(key=lambda x: (x.get(args.sort) is not None, x.get(args.sort)), reverse=reverse)
    
    total_matches = len(runs)
    
    # Limit
    if not args.all and len(runs) > args.limit:
        runs = runs[:args.limit]
        shown_count = len(runs)
    else:
        shown_count = total_matches
        
    # Columns
    cols = args.cols.split(',')
    
    # Auto-width
    col_widths = {}
    for col in cols:
        max_len = len(col)
        for r in runs:
            val = r.get(col, '')
            if isinstance(val, float): l = 8
            elif isinstance(val, datetime.datetime): l = 12
            else: l = len(str(val))
            max_len = max(max_len, l)
        col_widths[col] = min(max_len + 2, 30) # Cap at 30 chars
        
    # Header
    header = []
    for col in cols:
        width = col_widths[col]
        padding = " " * (width - len(col))
        header.append(col.upper() + padding)
    print(" | ".join(header))
    print("-" * (sum(col_widths.values()) + 3 * (len(cols) - 1)))
    
    # Rows
    for r in runs:
        print(format_row(r, cols, col_widths))
        
    # Footer
    if shown_count < total_matches:
        print(f"\nShowing {shown_count} of {total_matches} logs. Use --limit N or --all to see more.")
    else:
        print(f"\nShowing {shown_count} logs.")

if __name__ == "__main__":
    main()
