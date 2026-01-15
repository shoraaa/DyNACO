
import os
import torch
import numpy as np
import hashlib
from pathlib import Path
from torch.utils.data import TensorDataset
from tqdm import tqdm
import time

try:
    import hygese as hgs
except ImportError:
    hgs = None

def solve_with_hgs(coords, demand, capacity, time_limit=2.0, seed=1, verbose=False):
    """
    Solve a single CVRP instance using HGS-CVRP (hygese).
    coords: (n, 2) tensor or numpy array, normalized [0,1]
    demand: (n,) tensor or numpy array, normalized so capacity=1.0 (usually) or raw
    capacity: float, the capacity value matching demand units
    
    Returns: optimal (or best found) cost as float
    """
    if hgs is None:
        raise ImportError("hygese not installed. Please install it to use HGS baseline.")

    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    if torch.is_tensor(demand):
        demand = demand.cpu().numpy()
        
    # HGS expects integer inputs usually, or we can just pass float but hygese might convert.
    # Hygese wrapper typically scales them if we use the right API.
    # Let's use the explicit integer conversion as seen in the notebook for safety.
    
    S = 10000   # Scale for precision (Notebook used 1000, we use 10000 for slightly better precision without overflow)
    
    # HGS requires depot at index 0. 
    # Our data format from utils.py/gen_instance_for_mfaco usually has depot at 0.
    # coords shape (n+1, 2) includes depot at 0? 
    # train.py uses: coords, demand, capacity. 
    # In utils.py: gen_instance_for_mfaco returns: coords (n+1,2), demand (n+1,), capacity (1.0)
    # So yes, index 0 is depot.
    
    dem_i = np.rint(demand * S).astype(int)
    cap_i = int(np.rint(capacity * S))
    
    x = coords[:, 0] * S
    y = coords[:, 1] * S
    
    # Simple heuristic for num_vehicles if not strictly enforced
    # demand[0] is usually 0 (depot demand)
    total_demand = dem_i.sum()
    if cap_i > 0:
        num_vehicles = int(np.ceil(total_demand / cap_i)) + 5 # Buffer
    else:
        num_vehicles = 50 # Fallback
        
    data = {
        "x_coordinates": x,
        "y_coordinates": y,
        "demands": dem_i,
        "service_times": np.zeros(len(dem_i)),
        "vehicle_capacity": cap_i,
        "num_vehicles": num_vehicles,
        "depot": 0,
    }
    
    ap = hgs.AlgorithmParameters(timeLimit=float(time_limit), seed=int(seed))
    # Disable verbose unless asked
    solver = hgs.Solver(parameters=ap, verbose=bool(verbose))
    result = solver.solve_cvrp(data)
    
    # result.cost is integer cost based on scaled coords
    return result.cost / S

def get_dataset_hash(dataset):
    """
    Compute SHA256 hash of the dataset content to uniquely identify it.
    """
    # Assuming dataset is TensorDataset(coords, demand, capacity)
    # We can hash the underlying tensors bytes
    hasher = hashlib.sha256()
    for t in dataset.tensors:
        # Move to cpu and numpy to get bytes
        b = t.cpu().numpy().tobytes()
        hasher.update(b)
    return hasher.hexdigest()[:16] # Short hash

def get_baseline_cvrp(dataset, n_node, device="cpu", time_limit=2.0, verbose=True):
    """
    Compute or load cached HGS baseline optimal values for the dataset.
    Returns: Tensor of optimal costs (N_samples,)
    """
    # Determine cache path
    # We assume standard location data/cvrp/
    # But we can look at where data came from if dataset object had path info (it doesn't usually).
    # We'll use relative path "data/cvrp" assuming script run from project root or cvrp/
    
    # Try to find data dir
    # train.py defines SAVE_DIR etc. using Path.cwd()
    # We will try to locate data/cvrp relative to this file
    this_dir = Path(__file__).parent.resolve()
    data_dir = (this_dir / "../data/cvrp").resolve()
    
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        
    d_hash = get_dataset_hash(dataset)
    cache_file = data_dir / f"valDataset-{n_node}-{d_hash}-hgs.pt"
    
    if cache_file.exists():
        if verbose:
            print(f"Loading HGS baseline from {cache_file}")
        data = torch.load(cache_file, map_location=device, weights_only=False)
        if isinstance(data, dict):
            costs = data["costs"]
            run_time = data["run_time"]
            if verbose:
                print(f"Baseline run time: {run_time:.2f}s")
            return costs
        return data
    
    if verbose:
        print(f"Computing HGS baseline for {len(dataset)} instances (Time limit={time_limit}s)...")
    
    t_start = time.time()
    costs = []
    # dataset is TensorDataset(coords, demand, capacity)
    # Iterate
    # We can use DataLoader or direct indexing
    for i in tqdm(range(len(dataset)), desc="HGS Baseline"):
        coords = dataset.tensors[0][i]
        demand = dataset.tensors[1][i]
        capacity = float(dataset.tensors[2][i])
        
        c = solve_with_hgs(coords, demand, capacity, time_limit=time_limit)
        costs.append(c)
    
    run_time = time.time() - t_start
    costs_t = torch.tensor(costs, dtype=torch.float32, device=device)
    torch.save({"costs": costs_t, "run_time": run_time}, cache_file)
    if verbose:
        print(f"Saved HGS baseline to {cache_file} (run time: {run_time:.2f}s)")
        
    return costs_t
