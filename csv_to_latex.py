import csv
import math
import argparse
import re

def parse_single_time(val):
    val = val.strip()
    if not val:
        return 0.0
    multiplier = 1.0
    if val.endswith('h'):
        multiplier = 3600.0
        val = val[:-1]
    elif val.endswith('m'):
        multiplier = 60.0
        val = val[:-1]
    elif val.endswith('s'):
        multiplier = 1.0
        val = val[:-1]
    
    try:
        return float(val) * multiplier
    except ValueError:
        return None

def parse_seconds(time_val):
    if isinstance(time_val, (int, float)):
        return float(time_val)
    if not isinstance(time_val, str):
        return None
    
    if time_val in ['N/A', 'OOM', '-']:
        return None
        
    parts = time_val.split('+')
    total = 0.0
    for p in parts:
        val = parse_single_time(p)
        if val is None:
            return None
        total += val
    return total

def format_time(seconds_str):
    seconds = parse_seconds(seconds_str)
    if seconds is None:
        return seconds_str

    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def format_method_name(name):
    name = name.replace('"', '')
        
    def sub_func(match):
        prefix = match.group(1)
        sep = match.group(2) if match.group(2) else ""
        number = match.group(3).replace(",", "")
        return f"{prefix}$_{{{number}}}$"

    name = re.sub(r'(I)(=)([\d,]+)', sub_func, name)
    name = re.sub(r'(RRC)()([\d,]+)', sub_func, name)
    name = re.sub(r'(PRC)()([\d,]+)', sub_func, name)
    return name

def process_csv(filename):
    rows = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return [], {}, [], [], []

    group1 = ["LKH3", "Concorde", "FACO", "HGS"] 
    g1_rows = []
    g2_rows = [] 
    g3_rows = [] 

    for row in rows:
        method = row['Method']
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
            
    return rows, g1_rows, g2_rows, g3_rows

def get_best_gaps(rows, datasets, excluded_methods=None):
    if excluded_methods is None:
        excluded_methods = []
    best_gaps = {}
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
    return best_gaps

def print_section_header(datasets):
    header_cols = []
    for ds in datasets:
         if ds.startswith("TSP"):
             disp = ds.replace("TSP", "\\TSP{}")
         elif ds.startswith("CVRP"):
             disp = ds.replace("CVRP", "\\CVRP{}")
         else:
             disp = ds
         header_cols.append(f"\\multicolumn{{2}}{{c}}{{{disp}}}")
    
    print(r"& " + " & ".join(header_cols) + r" \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}")
    print(r"Method & Obj.\ (Gap) & Time & Obj.\ (Gap) & Time & Obj.\ (Gap) & Time & Obj.\ (Gap) & Time & Obj.\ (Gap) & Time \\")

def print_section_rows(row_list, datasets, args, instance_counts, best_gaps, excluded_methods=None):
    if excluded_methods is None:
        excluded_methods = []
    for row in row_list:
        method_name = row['Method']
        name_tex = format_method_name(method_name)
        
        tex_cols = []
        for ds in datasets:
            obj_key = f"{ds}_Obj"
            gap_key = f"{ds}_Gap(%)"
            time_key = f"{ds}_TotalTime(s)"
            
            obj_str = row.get(obj_key, "-")
            gap_str = row.get(gap_key, "-")
            time_str = row.get(time_key, "-")

            if args.total:
                seconds = parse_seconds(time_str)
                if seconds is not None:
                     total_seconds = seconds * instance_counts[ds]
                     f_time = format_time(total_seconds)
                else:
                    f_time = format_time(time_str)
            else:
                f_time = format_time(time_str)
            
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
                        is_best = False
                        if method_name not in excluded_methods:
                            if abs(gap_val - best_gaps[ds]) < 1e-9:
                                is_best = True
                        
                        if is_best:
                            p_obj_fmt = f"\\textbf{{{p_obj}}}"
                            p_gap = f"(\\textbf{{{gap_val:.2f}\\%}})"
                        else:
                            p_obj_fmt = p_obj
                            p_gap = f"({gap_val:.2f}\\%)"
                        
                        f_obj_gap = f"{p_obj_fmt} {p_gap}"
                    except:
                        p_gap = f"({gap_str})"
                        f_obj_gap = f"{p_obj} {p_gap}"

            tex_cols.append(f_obj_gap)
            tex_cols.append(f_time)

        print(f"{name_tex} & " + " & ".join(tex_cols) + " \\\\")

# TSP Config
tsp_datasets = ["TSP1K", "TSP5K", "TSP10K", "TSP50K", "TSP100K"]
tsp_counts = {"TSP1K": 128, "TSP5K": 16, "TSP10K": 16, "TSP50K": 16, "TSP100K": 16}

# CVRP Config
cvrp_datasets = ["CVRP1K", "CVRP5K", "CVRP10K", "CVRP50K", "CVRP100K"]
cvrp_counts = {"CVRP1K": 128, "CVRP5K": 16, "CVRP10K": 16, "CVRP50K": 16, "CVRP100K": 16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", action="store_true", help="Report total time (default is average time per instance)")
    args = parser.parse_args()

    # Load Data
    from recalculate_gaps import recalculate_gaps
    
    # Recalculate gaps first
    print("Recalculating TSP gaps...")
    recalculate_gaps('results.csv', tsp_datasets, 'LKH3')
    print("Recalculating CVRP gaps...")
    recalculate_gaps('result_cvrp.csv', cvrp_datasets, 'HGS')
    
    tsp_exclusions = ["LKH3", "Concorde"]
    tsp_rows_all, tsp_g1, tsp_g2, tsp_g3 = process_csv("results.csv")
    tsp_best = get_best_gaps(tsp_rows_all, tsp_datasets, tsp_exclusions)

    cvrp_exclusions = ["HGS"]
    cvrp_rows_all, cvrp_g1, cvrp_g2, cvrp_g3 = process_csv("result_cvrp.csv")
    cvrp_best = get_best_gaps(cvrp_rows_all, cvrp_datasets, cvrp_exclusions)

    # Start Table
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Comparative results on synthetic \TSP{} and \CVRP{} instances. *: results are cited directly from original publications. OOM: the method exceeded memory limits.}")
    print(r"\label{tab:main-results}")
    print(r"\small")
    print(r"\renewcommand{\arraystretch}{0.8}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lcccccccccc}")
    print(r"\toprule")

    # TSP Section
    print_section_header(tsp_datasets)
    print(r"\midrule")
    print_section_rows(tsp_g1, tsp_datasets, args, tsp_counts, tsp_best, tsp_exclusions)
    print(r"\midrule")
    print_section_rows(tsp_g2, tsp_datasets, args, tsp_counts, tsp_best, tsp_exclusions)
    print(r"\midrule")
    print_section_rows(tsp_g3, tsp_datasets, args, tsp_counts, tsp_best, tsp_exclusions)

    # Sep
    print(r"\midrule")
    print(r"\midrule")
    
    # CVRP Section
    print_section_header(cvrp_datasets)
    print(r"\midrule")
    print_section_rows(cvrp_g1, cvrp_datasets, args, cvrp_counts, cvrp_best, cvrp_exclusions)
    print(r"\midrule")
    print_section_rows(cvrp_g2, cvrp_datasets, args, cvrp_counts, cvrp_best, cvrp_exclusions)
    print(r"\midrule")
    print_section_rows(cvrp_g3, cvrp_datasets, args, cvrp_counts, cvrp_best, cvrp_exclusions)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()
