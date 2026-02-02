import csv

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def main():
    rows = []
    fieldnames = []
    
    # Read CSV
    with open('results.csv', 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
            
    # Find LKH3 baselines
    baselines = {}
    datasets = ["TSP1K", "TSP5K", "TSP10K", "TSP50K", "TSP100K"]
    
    for row in rows:
        if row['Method'] == 'LKH3':
            for ds in datasets:
                obj_key = f"{ds}_Obj"
                if is_float(row[obj_key]):
                    baselines[ds] = float(row[obj_key])
            break
            
    if not baselines:
        print("Error: LKH3 row not found or has invalid values.")
        return

    print(f"Baselines: {baselines}")

    # Recalculate Gaps
    for row in rows:
        method = row['Method']
        for ds in datasets:
            obj_key = f"{ds}_Obj"
            gap_key = f"{ds}_Gap(%)"
            
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
            # Keep consistent formatting? CSV usually had variable decimals.
            # Let's use 2 decimals to match previous manual entries strictly
            row[gap_key] = f"{gap:.2f}"
            
    # Write back
    with open('results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print("Updated gaps in results.csv")

if __name__ == "__main__":
    main()
