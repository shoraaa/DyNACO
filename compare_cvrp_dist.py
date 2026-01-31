import utils
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

def analyze_distribution(coords, demands, capacity, name):
    n = len(coords) - 1
    # 1. Clustering Measure (Hopkins Statistic or Nearest Neighbor Index)
    # Simple NNI: Mean Nearest Neighbor Distance / Mean Random Distance
    # For Uniform 2D [0,1], Mean Random NN Dist ~ 0.5 / sqrt(N)
    
    locs = coords[1:] # Exclude depot
    dists = squareform(pdist(locs))
    np.fill_diagonal(dists, np.inf)
    min_dists = dists.min(axis=1)
    mean_nn = min_dists.mean()
    expected_nn = 0.5 / np.sqrt(n) # Theoretical for Poisson process in unit square
    
    nni = mean_nn / expected_nn
    # NNI ~ 1 => Random
    # NNI < 1 => Clustered
    # NNI > 1 => Dispersed (Regular)
    
    # 2. Demand Statistics
    avg_dem = demands.float().mean().item()
    std_dem = demands.float().std().item()
    total_dem = demands.float().sum().item()
    avg_fill = total_dem / capacity
    
    print(f"--- {name} ---")
    print(f"N: {n}")
    print(f"Capacity: {capacity}")
    print(f"NNI: {nni:.4f} (1.0=Random, <1.0=Clustered)")
    print(f"Demand: Avg={avg_dem:.2f}, Std={std_dem:.2f}, Total={total_dem:.1f}")
    if capacity > 1.0:
       print(f"theoretical min vehicles: {total_dem/capacity:.1f}")
    else:
       # Normalized capacity case
       print(f"theoretical min vehicles: {total_dem/1.0:.1f}")
    print("")

def main():
    # 1. Validation File
    val_path = "data/CVRP/data/validation_set/validation_cvrp1000_n128_C250.txt"
    try:
        val_data = utils.load_cvrp_txt_dataset(val_path)
        # Check first instance
        coords, demand, capacity, _, _ = val_data[0]
        analyze_distribution(coords, demand, capacity, "Validation File (Instance 0)")
    except Exception as e:
        print(f"Error analyzing validation file: {e}")

    # 2. Training Generation (Random)
    try:
        n = 1000
        # utils.gen_cvrp_instance(n, device, capacity=None)
        # Note: gen_cvrp_instance returns (coords, demand_f, capacity_norm)
        # demand_f is normalized. We want raw to compare?
        # Re-construct raw logic:
        # capacity real = 250 (for n=1000 in utils)
        # demand real = demand_f * 250
        
        coords, demand_f, cap_norm = utils.gen_cvrp_instance(n, 'cpu')
        
        # Determine real capacity used in generation
        if n >= 1000: real_cap = 250
        else: real_cap = 50 
        
        demand_real = demand_f * real_cap
        
        analyze_distribution(coords, demand_real, real_cap, "Training Generation (Random)")
        
    except Exception as e:
        print(f"Error analyzing training generation: {e}")

if __name__ == "__main__":
    main()
