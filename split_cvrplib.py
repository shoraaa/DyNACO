import os
from pathlib import Path

INPUT_FILE = "data/CVRP/data/test_set/CVRPlib_scale_larger_than1000_Li_X_XXL_n14.txt"
OUTPUT_DIR = "data/CVRP/data/test_set/"

TARGETS = {
    1000: "CVRPlib_1K.txt",
    5000: "CVRPlib_5K.txt",
    10000: "CVRPlib_10K.txt",
    50000: "CVRPlib_50K.txt",
    100000: "CVRPlib_100K.txt"
}

def get_node_count(line):
    # Parsing logic mirrors utils.load_cvrp_txt_dataset roughly
    try:
        # Check for Format 2 (Python list style)
        if line.startswith("['") or line.startswith('["'):
            content = line.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
            parts = [p.strip() for p in content.split(',')]
            
            # Simple heuristic: find customer coordinates segment
            # ... 'depot', x, y, 'customer', x, y, x, y ...
            if 'customer' in parts:
                cust_idx = parts.index('customer')
                
                # Keywords that terminate the customer list
                end_keywords = ['capacity', 'cost', 'demand', 'node_flag', 'end']
                
                cust_end_idx = len(parts)
                for k in end_keywords:
                    if k in parts:
                        idx = parts.index(k)
                        if idx > cust_idx:
                            cust_end_idx = min(cust_end_idx, idx)
                            
                cust_coords_flat = parts[cust_idx+1 : cust_end_idx]
                n_cust = len(cust_coords_flat) // 2
                return n_cust
                
        # Format 1 (Comma separated with keywords)
        elif "depot" in line and "customer" in line:
            parts = [p.strip() for p in line.split(',')]
            
            depot_idx = parts.index('depot')
            cust_idx = parts.index('customer')
            
            end_keywords = ['capacity', 'cost', 'demand', 'node_flag']
            cust_end_idx = len(parts)
            for k in end_keywords:
                if k in parts:
                   idx = parts.index(k)
                   if idx > cust_idx:
                       cust_end_idx = min(cust_end_idx, idx)
                       
            cust_coords_flat = parts[cust_idx+1 : cust_end_idx]
            n_cust = len(cust_coords_flat) // 2
            return n_cust

    except Exception:
        return None
    return None

def main():
    path = Path(INPUT_FILE)
    if not path.exists():
        print(f"Input file not found: {path}")
        return

    bins = {k: [] for k in TARGETS.keys()}
    
    print(f"Reading from {path}...")
    with open(path, 'r') as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        line = line.strip()
        if not line: continue
        
        n = get_node_count(line)
        if n is None:
            # print(f"Skipping unparseable line: {line[:50]}...")
            continue
            
        # Find nearest target
        if n == 30000:
            nearest_target = 50000
        else:
            nearest_target = min(TARGETS.keys(), key=lambda t: abs(t - n))
        
        bins[nearest_target].append(line)
        count += 1

    print(f"Parsed {count} lines.")
    
    for target, filename in TARGETS.items():
        out_path = Path(OUTPUT_DIR) / filename
        content = bins[target]
        print(f"Writing {len(content)} instances to {out_path} (Target {target})")
        
        with open(out_path, 'w') as f:
            for line in content:
                f.write(line + "\n")

if __name__ == "__main__":
    main()
