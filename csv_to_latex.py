import csv
import math
import argparse
import re

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
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

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

def get_best_gaps(rows, datasets):
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

def print_section_rows(row_list, datasets, args, instance_counts, best_gaps):
    excluded_methods = ["LKH3", "Concorde"]
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

            if args.avg and time_str not in ['N/A', 'OOM', '-']:
                try:
                    total_seconds = float(time_str)
                    avg_seconds = total_seconds / instance_counts[ds]
                    f_time = format_time(avg_seconds)
                except ValueError:
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
                            p_gap = f"({gap_val:.2f}\%)"
                        
                        f_obj_gap = f"{p_obj_fmt} {p_gap}"
                    except:
                        p_gap = f"({gap_str})"
                        f_obj_gap = f"{p_obj} {p_gap}"

            tex_cols.append(f_obj_gap)
            tex_cols.append(f_time)

        print(f"{name_tex} & " + " & ".join(tex_cols) + " \\\\")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg", action="store_true", help="Report average time per instance")
    args = parser.parse_args()

    # TSP Config
    tsp_datasets = ["TSP1K", "TSP5K", "TSP10K", "TSP50K", "TSP100K"]
    tsp_counts = {"TSP1K": 128, "TSP5K": 16, "TSP10K": 16, "TSP50K": 16, "TSP100K": 16}

    # CVRP Config
    cvrp_datasets = ["CVRP1K", "CVRP5K", "CVRP10K", "CVRP50K", "CVRP100K"]
    cvrp_counts = {"CVRP1K": 128, "CVRP5K": 16, "CVRP10K": 16, "CVRP50K": 16, "CVRP100K": 16}

    # Load Data
    from recalculate_gaps import recalculate_gaps
    
    # Recalculate gaps first
    print("Recalculating TSP gaps...")
    recalculate_gaps('results.csv', tsp_datasets, 'LKH3')
    print("Recalculating CVRP gaps...")
    recalculate_gaps('result_cvrp.csv', cvrp_datasets, 'HGS')
    
    tsp_rows_all, tsp_g1, tsp_g2, tsp_g3 = process_csv("results.csv")
    tsp_best = get_best_gaps(tsp_rows_all, tsp_datasets)

    cvrp_rows_all, cvrp_g1, cvrp_g2, cvrp_g3 = process_csv("result_cvrp.csv")
    cvrp_best = get_best_gaps(cvrp_rows_all, cvrp_datasets)

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
    print_section_rows(tsp_g1, tsp_datasets, args, tsp_counts, tsp_best)
    print(r"\midrule")
    print_section_rows(tsp_g2, tsp_datasets, args, tsp_counts, tsp_best)
    print(r"\midrule")
    print_section_rows(tsp_g3, tsp_datasets, args, tsp_counts, tsp_best)

    # Sep
    print(r"\midrule")
    print(r"\midrule")
    
    # CVRP Section
    print_section_header(cvrp_datasets)
    print(r"\midrule")
    print_section_rows(cvrp_g1, cvrp_datasets, args, cvrp_counts, cvrp_best)
    print(r"\midrule")
    print_section_rows(cvrp_g2, cvrp_datasets, args, cvrp_counts, cvrp_best)
    print(r"\midrule")
    print_section_rows(cvrp_g3, cvrp_datasets, args, cvrp_counts, cvrp_best)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()
