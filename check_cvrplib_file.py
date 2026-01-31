import utils
import torch
import numpy as np
from pathlib import Path

fpath = "data/CVRP/data/test_set/CVRPlib_scale_larger_than1000_Li_X_XXL_n14.txt"
path = Path(fpath)

if not path.exists():
    print(f"File not found: {fpath}")
else:
    print(f"Analyzing {path.name}...")
    try:
        data = utils.load_cvrp_txt_dataset(str(path))
        print(f"Total Instances: {len(data)}")
        print("-" * 50)
        print(f"{'Index':<5} | {'Customers (N)':<15} | {'Cost':<15}")
        print("-" * 50)
        
        node_counts = []
        for i, item in enumerate(data):
            coords = item[0]
            cost = item[3] # item is (coords, demand, capacity, cost, tour)
            n_cust = len(coords) - 1 # exclude depot
            node_counts.append(n_cust)
            print(f"{i:<5} | {n_cust:<15} | {cost:<15}")
            
        print("-" * 50)
        print(f"Min Customers: {min(node_counts)}")
        print(f"Max Customers: {max(node_counts)}")
        print(f"Sorted Sizes: {sorted(node_counts)}")
        
    except Exception as e:
        print(f"Error parsing file: {e}")
