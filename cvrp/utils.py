import torch
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import TensorDataset

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = (_THIS_DIR / ".." / "data").resolve()  # adjust depth if needed
print(DATA_DIR)

CAPACITY = 50
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_instance(n, device):
    locations = torch.rand(size=(n, 2), device=device)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=(n,), device=device)
    depot = torch.tensor([DEPOT_COOR], device=device)
    all_locations = torch.cat((depot, locations), dim=0)
    all_demands = torch.cat((torch.zeros((1,), device=device), demands), dim=0)
    distances = gen_distance_matrix(all_locations)
    return all_demands, distances # (n+1), (n+1, n+1)

def gen_instance_for_mfaco(n, device, capacity=CAPACITY):
    locations = torch.rand(size=(n, 2), device=device)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=(n,), device=device)

    depot = torch.tensor([DEPOT_COOR], device=device, dtype=locations.dtype)  # (1,2)
    coords = torch.cat((depot, locations), dim=0)                              # (n+1,2)

    demand = torch.cat((torch.zeros((1,), device=device, dtype=demands.dtype), demands), dim=0)  # (n+1,)

    # normalize so capacity = 1.0 (recommended)
    demand_f = demand.float() / capacity
    capacity_norm = 1.0

    return coords, demand_f, capacity_norm


def gen_distance_matrix(cvrp_coordinates):
    n_nodes = len(cvrp_coordinates)
    distances = torch.norm(cvrp_coordinates[:, None] - cvrp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10 # note here
    return distances

def gen_pyg_data(demands, distances, device, k_nearest=None):
    """
    Build PyG Data object.
    
    Args:
        demands: (n,) tensor
        distances: (n, n) tensor  
        device: torch device
        k_nearest: if None, builds fully connected graph (n² edges)
                   if int, builds kNN graph matching FACO's nn_list (n*k edges)
    """
    n = demands.size(0)
    
    if k_nearest is None:
        # Fully connected graph (original behavior)
        nodes = torch.arange(n, device=device)
        u = nodes.repeat(n)
        v = torch.repeat_interleave(nodes, n)
        edge_index = torch.stack((u, v))
        edge_attr = distances.reshape(((n)**2, 1))
    else:
        # kNN graph matching FACO's nn_list structure
        k = min(k_nearest, n - 1)
        
        # For each node, find k nearest neighbors (excluding self)
        # This matches build_nearest_neighbor_lists in faco.py
        u_list = []
        v_list = []
        dist_list = []
        
        for i in range(n):
            # Get distances from node i, set self-distance to inf to exclude
            dists_i = distances[i].clone()
            dists_i[i] = float('inf')
            
            # Get k nearest neighbors
            _, nn_indices = torch.topk(dists_i, k, largest=False)
            
            # Add edges i -> each neighbor
            for j in nn_indices:
                u_list.append(i)
                v_list.append(j.item())
                dist_list.append(distances[i, j].item())
        
        edge_index = torch.tensor([u_list, v_list], dtype=torch.long, device=device)
        edge_attr = torch.tensor(dist_list, dtype=torch.float32, device=device).unsqueeze(1)
    
    x = demands
    pyg_data = Data(x=x.unsqueeze(1), edge_attr=edge_attr, edge_index=edge_index)
    return pyg_data

def load_val_dataset(n, device: str = "cpu"):
    pack = torch.load(f'{DATA_DIR}/cvrp/valDataset-{n}.pt', map_location=device, weights_only=False)
    return pack
    

if __name__ == '__main__':
    # generate val and test datasets, only coords
    import os
    if not os.path.exists(f'{DATA_DIR}/cvrp'):
        os.makedirs(f'{DATA_DIR}/cvrp')
    torch.manual_seed(123456)
    for n in [100, 200, 500]:
        inst_list = []
        inst_coords = []
        inst_demand = []
        inst_capacity = []
        for _ in range(128):
            coords_t, demand_t, capacity = gen_instance_for_mfaco(n, device="cpu")
            inst_coords.append(coords_t)
            inst_demand.append(demand_t)
            inst_capacity.append(torch.tensor(capacity))

        valDataset = TensorDataset(torch.stack(inst_coords).float().cpu(), torch.stack(inst_demand).float().cpu(), torch.stack(inst_capacity).float().cpu())
        torch.save(valDataset, f'{DATA_DIR}/cvrp/valDataset-{n}.pt')

    for n in [100, 200, 500]:
        inst_list = []
        inst_coords = []
        inst_demand = []
        inst_capacity = []
        for _ in range(128):
            coords_t, demand_t, capacity = gen_instance_for_mfaco(n, device="cpu")
            inst_coords.append(coords_t)
            inst_demand.append(demand_t)
            inst_capacity.append(torch.tensor(capacity))

        testDataset = TensorDataset(torch.stack(inst_coords).float().cpu(), torch.stack(inst_demand).float().cpu(), torch.stack(inst_capacity).float().cpu())
        torch.save(testDataset, f'{DATA_DIR}/cvrp/testDataset-{n}.pt')

    for n in [1000, 2000, 5000]:
        inst_list = []
        inst_coords = []
        inst_demand = []
        inst_capacity = []
        for _ in range(128):
            coords_t, demand_t, capacity = gen_instance_for_mfaco(n, device="cpu")
            inst_coords.append(coords_t)
            inst_demand.append(demand_t)
            inst_capacity.append(torch.tensor(capacity))

        valDataset = TensorDataset(torch.stack(inst_coords).float().cpu(), torch.stack(inst_demand).float().cpu(), torch.stack(inst_capacity).float().cpu())
        torch.save(valDataset, f'{DATA_DIR}/cvrp/valDataset-{n}.pt')
    for n in [1000, 2000, 5000]:
        inst_list = []
        inst_coords = []
        inst_demand = []
        inst_capacity = []
        for _ in range(128):
            coords_t, demand_t, capacity = gen_instance_for_mfaco(n, device="cpu")
            inst_coords.append(coords_t)
            inst_demand.append(demand_t)
            inst_capacity.append(torch.tensor(capacity))

        testDataset = TensorDataset(torch.stack(inst_coords).float().cpu(), torch.stack(inst_demand).float().cpu(), torch.stack(inst_capacity).float().cpu())
        torch.save(testDataset, f'{DATA_DIR}/cvrp/testDataset-{n}.pt')
        