import utils
import torch
import numpy as np
from pathlib import Path

tsp_files = [
    "data/TSP/data/test_set/test_tsp100_concorde_n10000.txt",
    "data/TSP/data/test_set/MCTS_tsp1000_test_concorde.txt",
    "data/TSP/data/test_set/test_tsp5000_lkh3_n16.txt",
    "data/TSP/data/test_set/MCTS_tsp10000_test_concorde.txt",
    "data/TSP/data/test_set/test_tsp50000_lkh3_n16.txt",
    "data/TSP/data/test_set/test_tsp100000_lkh3_n16.txt"
]

cvrp_files = [
    "data/CVRP/data/test_set/test_cvrp100_hgs_n10000_C50.txt",
    "data/CVRP/data/test_set/test_cvrp1000_hgs_n128_C250.txt",
    "data/CVRP/data/test_set/test_cvrp5000_hgs_n16_C500.txt",
    "data/CVRP/data/test_set/test_cvrp10000_hgs_n16_C1000.txt",
    "data/CVRP/data/test_set/test_cvrp50000_hgs_n16_C2000.txt",
    "data/CVRP/data/test_set/test_cvrp100000_hgs_n16_C2000.txt"
]

print(f"{'Problem':<10} | {'File Basename':<40} | {'Nodes (N)':<10} | {'Instances':<10}")
print("-" * 80)

def check_files(files, problem):
    for fpath in files:
        path = Path(fpath)
        if not path.exists():
            print(f"{problem:<10} | {path.name:<40} | {'NOT FOUND':<10} | {'-':<10}")
            continue
            
        try:
            if problem == "TSP":
                data = utils.load_tsp_txt_dataset(str(path))
                # data item: (coords, cost, tour)
                coords = data[0][0]
                n = len(coords)
            else:
                data = utils.load_cvrp_txt_dataset(str(path))
                # cvrp item: (coords, demand, capacity, cost, tour)
                # coords includes depot, so N customers = len(coords) - 1
                coords = data[0][0]
                n = len(coords) - 1 
                
            print(f"{problem:<10} | {path.name:<40} | {n:<10} | {len(data):<10}")
            
        except Exception as e:
            print(f"{problem:<10} | {path.name:<40} | {'ERROR':<10} | {e}")

check_files(tsp_files, "TSP")
check_files(cvrp_files, "CVRP")
