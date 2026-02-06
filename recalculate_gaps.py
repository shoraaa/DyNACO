import csv

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def recalculate_gaps(filename, datasets, baseline_method):
    rows = []
    fieldnames = []
    
    try:
        # Read CSV
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    # Find baseline row
    baselines = {}
    
    for row in rows:
        if row['Method'] == baseline_method:
            for ds in datasets:
                obj_key = f"{ds}_Obj"
                if is_float(row[obj_key]):
                    baselines[ds] = float(row[obj_key])
            break
            
    if not baselines:
        print(f"Error: Baseline method '{baseline_method}' not found or has invalid values in {filename}.")
        return

    print(f"Baselines for {filename} ({baseline_method}): {baselines}")

    # Recalculate Gaps
    for row in rows:
        method = row['Method']
        for ds in datasets:
            obj_key = f"{ds}_Obj"
            gap_key = f"{ds}_Gap(%)"
            
            # Check if dataset exists in this row (column exists)
            if obj_key not in row:
                continue

            obj_val = row[obj_key]
            
            # Skip if Obj is not a number (e.g. -, N/A, OOM)
            if not is_float(obj_val):
                continue
                
            # Skip if baseline missing for this dataset
            if ds not in baselines:
                continue
                
            current_obj = float(obj_val)
            base_obj = baselines[ds]
            
            # Calculate Gap
            # Formula: (Obj - Base) / Base * 100
            gap = ((current_obj - base_obj) / base_obj) * 100
            
            # Update row
            row[gap_key] = f"{gap:.2f}"
            
    # Write back
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Updated gaps in {filename}")

def main():
    # Helper to run manually if needed
    tsp_datasets = ["TSP1K", "TSP5K", "TSP10K", "TSP50K", "TSP100K"]
    recalculate_gaps('results.csv', tsp_datasets, 'LKH3')
    
    cvrp_datasets = ["CVRP1K", "CVRP5K", "CVRP10K", "CVRP50K", "CVRP100K"]
    recalculate_gaps('result_cvrp.csv', cvrp_datasets, 'HGS')

if __name__ == "__main__":
    main()
