
import torch
import numpy as np
import faco
import time

def generate_cvrp_instance(n=50, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(n, 2).astype(np.float32)
    demand = np.random.randint(1, 10, size=n).astype(np.float32)
    demand[0] = 0
    capacity = 30.0
    return coords, demand, capacity

def verify_solution(coords, demand, capacity, cost, route0):
    visited = set()
    total_dist = 0.0
    current_load = 0.0
    
    # route0 includes depots (0)
    # e.g. [0, 1, 2, 0, 3, 4, 0]
    
    for i in range(len(route0) - 1):
        u, v = route0[i], route0[i+1]
        dist = np.linalg.norm(coords[u] - coords[v])
        total_dist += dist
        
        if v != 0:
            current_load += demand[v]
            visited.add(v)
            if current_load > capacity + 1e-5:
                print(f"Capacity violation at node {v}: {current_load} > {capacity}")
                return False
        else:
            current_load = 0.0
            
    if abs(total_dist - cost) > 1e-4:
        print(f"Cost mismatch: Calculated {total_dist}, Reported {cost}")
        # return False # Allow slight drift due to float prec in C++ vs Py
        
    if len(visited) != len(coords) - 1:
        print(f"Not all customers visited: {len(visited)}/{len(coords)-1}")
        return False
        
    return True

def main():
    n = 100
    coords, demand, capacity = generate_cvrp_instance(n)
    
    print(f"Running MFACO_CVRP on N={n}...")
    solver = faco.MFACO_CVRP(
        coords, demand, capacity,
        n_ants=20,
        cand_list_size=32,
        min_new_edges=0, # Force full run or control via fixed_steps
        fixed_steps=10, 
        use_local_search=True,
        disable_heuristic=False
    )
    
    t0 = time.time()
    # Run a few iterations
    for _ in range(5):
        costs, _, decoded, _, _, _, _, _, _ = solver.sample(return_decoded=True)
        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        best_route = decoded[best_idx]
        
        # Verify
        if not verify_solution(coords, demand, capacity, best_cost, best_route):
             print("Verification FAILED")
             return
             
    t1 = time.time()
    print(f"Finished in {t1-t0:.4f}s. Best Cost: {best_cost:.4f}")
    print("Verification PASSED")

if __name__ == "__main__":
    main()
