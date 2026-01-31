import utils
import torch
import numpy as np
from pathlib import Path

fpath = "data/TSP/data/test_set/TSPlib_scale_ge_1K_n33_ascending.txt"
path = Path(fpath)

if not path.exists():
    print(f"File not found: {fpath}")
else:
    print(f"Analyzing {path.name}...")
    try:
        data = utils.load_tsp_txt_dataset(str(path))
        print(f"Total Instances: {len(data)}")
        print("-" * 40)
        print(f"{'Index':<5} | {'Nodes (N)':<10} | {'Cost':<15}")
        print("-" * 40)
        
        node_counts = []
        for i, item in enumerate(data):
            coords = item[0]
            cost = item[1]
            n = len(coords)
            node_counts.append(n)
            print(f"{i:<5} | {n:<10} | {cost:<15}")
            
        print("-" * 40)
        print(f"Min Nodes: {min(node_counts)}")
        print(f"Max Nodes: {max(node_counts)}")
        
    except Exception as e:
        print(f"Error parsing file: {e}")
