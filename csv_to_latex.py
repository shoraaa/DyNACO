import csv
import math

def format_time(seconds_str):
    if seconds_str in ['N/A', 'OOM', '-']:
        return seconds_str
    try:
        seconds = float(seconds_str)
    except ValueError:
        return seconds_str

    if seconds < 1:
        return "<1s"
    elif seconds < 60:
        return f"{seconds:.1f}s" # changed from .0f to .1f for small values like 15s?
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_obj_gap(obj_str, gap_str):
    if obj_str in ['N/A', 'OOM', '-']:
        return "--" if obj_str == '-' else obj_str
    
    try:
        obj_val = float(obj_str)
        # Check if we should format as "k" (if > 1000)
        # The user's example had 23.1k. The CSV has 23.12.
        # This implies the CSV values might be "23.12" meaning 23.12, and the User's example 23.1k is from a different context or scale.
        # However, checking LKH3 23.12 in CSV vs 23.1k in example.
        # If I assume the CSV values are correctly scaled for the table, I will just print them.
        # But 23.12 is highly unlikely to be the actual tour length for TSP1k (usually ~23000).
        # It's likely the CSV has normalized values / 1000 or similar.
        # I will format based on value magnitude.
        
        # Heuristic: If value < 1000, print as is. If value > 1000, print as Xk? 
        # But CSV has 23.12. 
        # I'll just print 2 decimal places.
        p_obj = f"{obj_val:.2f}"
    except:
        p_obj = obj_str

    if gap_str in ['N/A', 'OOM', '-']:
        return p_obj # No gap?

    try:
        gap_val = float(gap_str)
        p_gap = f"({gap_val:.2f}\%)"
    except:
        p_gap = f"({gap_str})"
    
    return f"{p_obj} {p_gap}"

def get_row_tex(method_name, row_dict, datasets):
    # row_dict: { 'TSP1K_Obj': ..., 'TSP1K_Gap(%)': ..., 'TSP1K_TotalTime(s)': ... }
    
    tex_cols = []
    
    for ds_name in datasets: # ["TSP1K", "TSP5K", ...]
        obj_key = f"{ds_name}_Obj"
        gap_key = f"{ds_name}_Gap(%)"
        time_key = f"{ds_name}_TotalTime(s)"
        
        obj = row_dict.get(obj_key, "-")
        gap = row_dict.get(gap_key, "-")
        time = row_dict.get(time_key, "-")
        
        formatted_obj_gap = format_obj_gap(obj, gap)
        formatted_time = format_time(time)
        
        tex_cols.append(formatted_obj_gap)
        tex_cols.append(formatted_time)
        
    return f"{method_name} & " + " & ".join(tex_cols) + " \\\\"

def main():
    datasets = ["TSP1K", "TSP5K", "TSP10K", "TSP50K", "TSP100K"]
    
    # Groups
    group1 = ["LKH3", "Concorde", "FACO"] # Exact matches or starts with?
    group3_prefix = "Ours" # DyNACO
    
    rows = []
    with open('results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    # Buckets
    g1_rows = []
    g2_rows = [] # Others
    g3_rows = [] # DyNACO
    
    for row in rows:
        method = row['Method']
        
        # Determine group
        # Check explicit group 1
        is_g1 = False
        for g1 in group1:
            if g1 in method: 
                is_g1 = True
                break
        
        if is_g1:
            g1_rows.append(row)
        elif "DyNACO" in method:
            g3_rows.append(row)
        else:
            g2_rows.append(row)
            
    # Calculate best gaps per dataset (excluding LKH3, Concorde)
    best_gaps = {}
    excluded_methods = ["LKH3", "Concorde"]
    
    for ds in datasets:
        min_gap = float('inf')
        for row in rows:
            if row['Method'] in excluded_methods:
                continue
            
            gap_str = row.get(f"{ds}_Gap(%)", "")
            if gap_str and gap_str not in ['N/A', 'OOM', '-']:
                try:
                    gap_val = float(gap_str)
                    if gap_val < min_gap:
                        min_gap = gap_val
                except ValueError:
                    pass
        best_gaps[ds] = min_gap

    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Main Results...}")
    print(r"\label{tab:main_results}")
    print(r"\setlength{\tabcolsep}{3pt}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{l|cc|cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Method}} & \multicolumn{2}{c|}{\textbf{TSP-1K}} & \multicolumn{2}{c|}{\textbf{TSP-5K}} & \multicolumn{2}{c|}{\textbf{TSP-10K}} & \multicolumn{2}{c|}{\textbf{TSP-50K}} & \multicolumn{2}{c}{\textbf{TSP-100K}} \\")
    print(r" & \textbf{Obj. (Gap)} & \textbf{Time} & \textbf{Obj. (Gap)} & \textbf{Time} & \textbf{Obj. (Gap)} & \textbf{Time} & \textbf{Obj. (Gap)} & \textbf{Time} & \textbf{Obj. (Gap)} & \textbf{Time} \\")
    print(r"\midrule")
    
    # Helper to print rows with bolding
    def print_rows(row_list):
        for row in row_list:
            method_name = row['Method']
            # Name formatting
            if "DyNACO" in method_name:
                 if "H=" in method_name:
                    val = method_name.split("H=")[1]
                    name_tex = f"\\textbf{{DyNACO}} ($H={val}$)"
                 elif "greedy" in method_name:
                     new_name = method_name.replace("greedy", "(Greedy)")
                     name_tex = f"\\textbf{{{new_name}}}"
                 else:
                     name_tex = f"\\textbf{{{method_name}}}"
            else:
                 name_tex = method_name
            
            tex_cols = []
            for ds in datasets:
                obj_key = f"{ds}_Obj"
                gap_key = f"{ds}_Gap(%)"
                time_key = f"{ds}_TotalTime(s)"
                
                obj_str = row.get(obj_key, "-")
                gap_str = row.get(gap_key, "-")
                time_str = row.get(time_key, "-")

                # Format Time
                f_time = format_time(time_str)
                
                # Format Obj (Gap)
                if obj_str in ['N/A', 'OOM', '-']:
                     f_obj_gap = "--" if obj_str == '-' else obj_str
                else:
                    try:
                        p_obj = f"{float(obj_str):.2f}"
                    except:
                        p_obj = obj_str
                        
                    if gap_str in ['N/A', 'OOM', '-']:
                        f_obj_gap = p_obj
                    else:
                        try:
                            gap_val = float(gap_str)
                            # Check if best
                            is_best = False
                            if method_name not in excluded_methods:
                                if abs(gap_val - best_gaps[ds]) < 1e-9:
                                    is_best = True
                            
                            if is_best:
                                p_gap = f"(\\textbf{{{gap_val:.2f}\%}})"
                            else:
                                p_gap = f"({gap_val:.2f}\%)"
                        except:
                            p_gap = f"({gap_str})"
                        f_obj_gap = f"{p_obj} {p_gap}"

                tex_cols.append(f_obj_gap)
                tex_cols.append(f_time)

            print(f"{name_tex} & " + " & ".join(tex_cols) + " \\\\")

    # Print G1
    print_rows(g1_rows)
        
    print(r"\midrule")
    
    # Print G2
    print_rows(g2_rows)
        
    print(r"\midrule")
    
    # Print G3 (DyNACO)
    print_rows(g3_rows)
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()
