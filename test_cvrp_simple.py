
import torch
import numpy as np
from faco import ACO_CVRP

def test_cvrp():
    print("Testing ACO_CVRP...")
    n = 20
    n_ants = 5
    coords = torch.rand(n, 2)
    demand = torch.rand(n) * 0.2
    demand[0] = 0.0 # Depot
    capacity = 1.0
    
    # Test initialization
    aco = ACO_CVRP(coords, demand, capacity, n_ants=n_ants, device="cpu")
    print("Initialized ACO_CVRP")
    
    # Test sampling
    costs, routes, decoded, logps, traces, costs_raw, perms_raw, new_edges, survival = aco.sample(
        require_prob=True, parallel_traced=False
    )
    
    print(f"Sampled {n_ants} ants.")
    print(f"Costs: {costs}")
    print(f"Route 0 type: {type(routes[0])}, shape: {routes[0].shape}")
    print(f"Route 0: {routes[0]}")
    
    assert len(costs) == n_ants
    assert len(routes) == n_ants
    
    # Check simple property
    # Route should start with 0 and end with 0, or be a valid sequence?
    # Our C++ implementation returns full path [0, c1, c2, 0, c3, ..., 0]
    r = routes[0]
    if r[0] != 0:
        print("Warning: Route does not start with 0")
    
    # Test Pheromone Update
    print("Testing Pheromone Update...")
    best_idx = np.argmin(costs)
    aco.update_pheromone(routes[best_idx], costs[best_idx])
    
    # Check max/min
    tau = aco.pheromone_sparse_np
    print(f"Pheromone Stats: min={tau.min()}, max={tau.max()}, mean={tau.mean()}")
    
    # Test PPO log prob
    print("Testing evaluate_log_prob_torch...")
    # Pad routes to same length for batching
    max_len = max([len(x) for x in routes])
    padded = torch.zeros((n_ants, max_len), dtype=torch.long)
    for i in range(n_ants):
        l = len(routes[i])
        padded[i, :l] = torch.from_numpy(routes[i])
        # Pad with 0? Or just 0s. 0 is depot, so it's a valid node but meaningless if trailed.
    
    log_probs = aco.evaluate_log_prob_torch(padded)
    print(f"Log Probs: {log_probs}")
    
    print("ACO_CVRP Test Passed!")

if __name__ == "__main__":
    test_cvrp()
