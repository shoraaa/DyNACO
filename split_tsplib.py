import os
from pathlib import Path

INPUT_FILE = "data/TSP/data/test_set/TSPlib_scale_ge_1K_n33_ascending.txt"
OUTPUT_DIR = "data/TSP/data/test_set/"

TARGETS = {
    1000: "TSPlib_1K.txt",
    5000: "TSPlib_5K.txt",
    10000: "TSPlib_10K.txt",
    50000: "TSPlib_50K.txt",
    100000: "TSPlib_100K.txt"
}

def get_node_count(line):
    # Parsing logic mirrors utils.load_tsp_txt_dataset roughly
    try:
        if line.startswith("['"):
            # TSPlib list format
            content = line.replace('[', '').replace(']', '').replace("'", "")
            parts = [p.strip() for p in content.split(',')]
            # name = parts[0]
            # cost = parts[1]
            # coords start from index 2
            coords_flat = parts[2:]
            return len(coords_flat) // 2
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
            print(f"Skipping unparseable line: {line[:50]}...")
            continue
            
        # Find nearest target
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
