
import os
import torch
import numpy as np
import hashlib
import tempfile
import subprocess
from pathlib import Path
from tqdm import tqdm
import time

# Optional imports
try:
    import hygese as hgs
except ImportError:
    hgs = None

# Paths
LKH_PATH = Path(__file__).parent / "baselines" / "LKH-3.0.13" / "LKH"

# ----------------- TSP / LKH -----------------

def write_tsplib_euc2d(path, coords_int, name="inst"):
    n = coords_int.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords_int, start=1):
            f.write(f"{i} {int(x)} {int(y)}\n")
        f.write("EOF\n")

def write_lkh_par(path, tsp_path, out_path, lkh_runs=1, seed=1234, time_limit=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {out_path}\n")
        f.write(f"RUNS = {lkh_runs}\n")
        f.write(f"SEED = {seed}\n")
        if time_limit is not None:
            f.write(f"TIME_LIMIT = {time_limit}\n")

def read_tour_file(tour_path):
    tour = []
    in_section = False
    try:
        with open(tour_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "TOUR_SECTION":
                    in_section = True
                    continue
                if not in_section:
                    continue
                if line == "-1" or line == "EOF":
                    break
                tour.append(int(line) - 1) 
    except FileNotFoundError:
        return None
    return tour

def solve_with_lkh(coords, runs=1, seed=1234, scale=1000000, time_limit=None):
    if not LKH_PATH.exists():
        # Try finding LKH in parent dir if not found (dev setup vs installed package)
        alt_path = Path(__file__).parent.parent / "baselines" / "LKH-3.0.13" / "LKH"
        if alt_path.exists():
            lkh_path = alt_path
        else:
            raise FileNotFoundError(f"LKH solver not found at {LKH_PATH} or {alt_path}")
    else:
        lkh_path = LKH_PATH
        
    coords_int = np.rint(coords * scale).astype(np.int64)

    with tempfile.TemporaryDirectory() as td:
        tsp_path  = os.path.join(td, "inst.tsp")
        par_path  = os.path.join(td, "inst.par")
        tour_path = os.path.join(td, "inst.tour")

        write_tsplib_euc2d(tsp_path, coords_int, name="inst")
        write_lkh_par(par_path, tsp_path, tour_path, lkh_runs=runs, seed=seed, time_limit=time_limit)

        subprocess.run([str(lkh_path), par_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        tour = read_tour_file(tour_path)
        if tour is None:
            return float('inf')
            
        coords_ordered = coords[tour]
        coords_rolled = np.roll(coords_ordered, -1, axis=0)
        cost = np.linalg.norm(coords_ordered - coords_rolled, axis=1).sum()
        
        return cost

# ----------------- CVRP / HGS -----------------

def solve_with_hgs(coords, demand, capacity, time_limit=2.0, seed=1, verbose=False):
    if hgs is None:
        raise ImportError("hygese not installed. Please install it to use HGS baseline.")

    if torch.is_tensor(coords): coords = coords.cpu().numpy()
    if torch.is_tensor(demand): demand = demand.cpu().numpy()
        
    S = 10000   
    dem_i = np.rint(demand * S).astype(int)
    cap_i = int(np.rint(capacity * S))
    x = coords[:, 0] * S
    y = coords[:, 1] * S
    
    total_demand = dem_i.sum()
    if cap_i > 0:
        num_vehicles = int(np.ceil(total_demand / cap_i)) + 5 
    else:
        num_vehicles = 50 
        
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
    solver = hgs.Solver(parameters=ap, verbose=bool(verbose))
    result = solver.solve_cvrp(data)
    
    return result.cost / S

# ----------------- Common -----------------

def get_dataset_hash(dataset):
    hasher = hashlib.sha256()
    # If list (TSP utils often returns list of tensors, or CVRP list of tuples)
    if isinstance(dataset, list):
        for item in dataset:
            # Handle tuple format (coords, demand, capacity) for CVRP
            if isinstance(item, (tuple, list)):
                for t in item:
                    if isinstance(t, torch.Tensor):
                        b = t.cpu().numpy().tobytes()
                    elif isinstance(t, (int, float)):
                        b = np.array([t]).tobytes()
                    else:
                        b = np.array(t).tobytes()
                    hasher.update(b)
            else:
                # Tensor directly
                b = item.cpu().numpy().tobytes()
                hasher.update(b)
    # If TensorDataset (CVRP)
    elif hasattr(dataset, "tensors"):
        for t in dataset.tensors:
            b = t.cpu().numpy().tobytes()
            hasher.update(b)
    # If single tensor
    elif isinstance(dataset, torch.Tensor):
         b = dataset.cpu().numpy().tobytes()
         hasher.update(b)
         
    return hasher.hexdigest()[:16]

def get_baseline(dataset, problem='tsp', n_node=100, device="cpu", **kwargs):
    this_dir = Path(__file__).parent.resolve()
    data_dir = (this_dir / "data" / problem).resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        
    d_hash = get_dataset_hash(dataset)
    solver_name = "lkh" if problem == 'tsp' else "hgs"
    cache_file = data_dir / f"valDataset-{n_node}-{d_hash}-{solver_name}.pt"
    
    if cache_file.exists():
        print(f"Loading {solver_name.upper()} baseline from {cache_file}")
        data = torch.load(cache_file, map_location=device, weights_only=False)
        if isinstance(data, dict):
            print(f"Baseline run time: {data['run_time']:.2f}s")
            return data["costs"]
        return data

    print(f"Computing {solver_name.upper()} baseline for {len(dataset)} examples...")
    t_start = time.time()
    costs = []
    
    runs = kwargs.get("runs", 1)
    time_limit = kwargs.get("time_limit", 300) # None for TSP default?
    if problem == 'cvrp' and time_limit is None: time_limit = 300
    
    # Iterate
    length = len(dataset)
    for i in tqdm(range(length), desc=f"{solver_name.upper()} Baseline"):
        if problem == 'tsp':
            coords = dataset[i] if isinstance(dataset, list) else dataset[i] # Handle tensor indexing
            if isinstance(coords, torch.Tensor): coords = coords.cpu().numpy()
            c = solve_with_lkh(coords, runs=runs, time_limit=time_limit)
        else:
            # CVRP: can be TensorDataset or list of tuples
            if hasattr(dataset, "tensors"):
                coords = dataset.tensors[0][i]
                demand = dataset.tensors[1][i]
                capacity = float(dataset.tensors[2][i])
            else:
                # List of tuples (coords, demand, capacity) or more
                item = dataset[i]
                coords, demand, capacity = item[:3] if len(item) >= 3 else item
                if isinstance(capacity, torch.Tensor):
                    capacity = float(capacity.item())
            c = solve_with_hgs(coords, demand, capacity, time_limit=time_limit)
        costs.append(c)
        
    run_time = time.time() - t_start
    costs_t = torch.tensor(costs, dtype=torch.float32, device=device)
    torch.save({"costs": costs_t, "run_time": run_time}, cache_file)
    print(f"Saved baseline to {cache_file}")
    
    return costs_t
