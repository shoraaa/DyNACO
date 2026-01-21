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
    for n in [20, 100, 200, 500]:
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


def build_pyg_data_3(aco, coords, demand, device, dynamic: bool):
    """
    Node features:
      x: coords (n,2)
    """
    # coords -> torch
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    if isinstance(coords, np.ndarray):
        coords_t = torch.from_numpy(coords)
    elif isinstance(coords, torch.Tensor):
        coords_t = coords
    else:
        coords_t = torch.as_tensor(coords)
    coords_t = coords_t.to(device=device, dtype=torch.float32)  # (n,2)
    demand_t = torch.as_tensor(demand, device=device, dtype=torch.float32)  # (n,)

    # candidate graph
    nn = torch.as_tensor(np.asarray(aco.nn_list, dtype=np.int64), device=device, dtype=torch.long)  # (n,k)
    n, k = nn.shape
    E = n * k

    src = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(k)  # (E,)
    dst = nn.reshape(-1)                                                         # (E,)
    edge_index = torch.stack([src, dst], dim=0)                                   # (2,E)

    dist = torch.norm(coords_t[src] - coords_t[dst], dim=1).view(n, k).clamp_min(1e-12)
    log_dist = torch.log(dist).view(E, 1)  # (E,1)

    if dynamic:
        tau = torch.as_tensor(np.asarray(aco.pheromone_sparse_np), device=device, dtype=torch.float32).clamp_min(1e-12)
        log_tau = torch.log(tau).view(E, 1)                  # (n,k)

        solver = getattr(aco, "solver", aco)

        try:
            src_perm_np = aco.source_perm
        except AttributeError:
            src_perm_np = solver.source_perm

        src_perm = torch.as_tensor(
            np.asarray(src_perm_np, dtype=np.int64),
            device=device,
            dtype=torch.long,
        )

        in_source = torch.zeros((E,), device=device, dtype=torch.float32)

        if src_perm.numel() > 0:
            pos = torch.full((n,), -1, device=device, dtype=torch.long)
            pos[src_perm] = torch.arange(src_perm.numel(), device=device)

            m = src_perm.numel()
            duv = (pos[src] - pos[dst]).abs()

            undirected_adj = (
                ((duv == 1) | (duv == (m - 1)))
                & (pos[src] >= 0)
                & (pos[dst] >= 0)
            )

            in_source = undirected_adj.to(torch.float32)

        in_source_tour = in_source.view(E, 1)

    else:
        log_tau = torch.zeros((E, 1), device=device, dtype=torch.float32)
        in_source_tour = torch.zeros((E, 1), device=device, dtype=torch.float32)

    edge_attr = torch.cat([log_dist, log_tau, in_source_tour], dim=1)  # (E,3)

    depot_flag = torch.zeros((coords_t.size(0), 1), device=device)
    depot_flag[0, 0] = 1.0
    # CVRP specific node inputs: coords (2), demand (1), depot_flag (1) -> 4 dims
    coords_cat = torch.cat([coords_t, demand_t.unsqueeze(-1), depot_flag], dim=-1)
    return Data(x=coords_cat, edge_index=edge_index, edge_attr=edge_attr)


def build_pyg_data(aco, coords, demand, device: str, dynamic: bool) -> Data:
    """
    Build a PyG graph matching MFACO_CVRP's sparse candidate graph (n,k).

    Node features:
      x: coords (n,2)
    Edge features (E=n*k, 6):
      0 dist_norm
      1 tau_cv
      2 log_tau_rel (dynamic) else 0
      3 is_source_succ (customer tour)
      4 is_source_pred (customer tour)
      5 is_new_edge (not adjacent in source tour among customers)
    """
    if isinstance(coords, np.ndarray):
        coords_t = torch.from_numpy(coords)
    elif isinstance(coords, torch.Tensor):
        coords_t = coords
    else:
        coords_t = torch.as_tensor(coords)
    coords_t = coords_t.to(device=device, dtype=torch.float32)  # (n,2)
    demand_t = torch.as_tensor(demand, device=device, dtype=torch.float32)  # (n,)

    nn = torch.as_tensor(np.asarray(aco.nn_list, dtype=np.int64), device=device, dtype=torch.long)  # (n,k)
    n, k = nn.shape
    E = n * k

    src = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(k)
    dst = nn.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    # 0) dist_norm
    dist = torch.norm(coords_t[src] - coords_t[dst], dim=1).view(n, k)
    dist_mean = dist.mean(dim=1, keepdim=True).clamp_min(1e-12)
    dist_norm = (dist / dist_mean).view(E, 1)

    # 1-2) tau features
    if dynamic:
        tau = torch.as_tensor(np.asarray(aco.pheromone_sparse_np), device=device, dtype=torch.float32).clamp_min(1e-12)
        tau_mean = tau.mean(dim=1, keepdim=True).clamp_min(1e-12)
        tau_rel = (tau / tau_mean).clamp_min(1e-12)
        log_tau_rel = torch.log(tau_rel).clamp(-5.0, 5.0).view(E, 1)
        tau_std = tau.std(dim=1, keepdim=True)
        tau_cv = (tau_std / tau_mean).clamp(0, 10)
        tau_cv_e = tau_cv.repeat_interleave(k, dim=0)
    else:
        log_tau_rel = torch.zeros((E, 1), device=device, dtype=torch.float32)
        tau_cv_e = torch.zeros((E, 1), device=device, dtype=torch.float32)

    # Source-perm features (customers only; depot gets -1)
    # Check if aco.solver exists (if wrapping) or if aco is the solver
    # MFACO_CVRP has .solver
    solver = getattr(aco, 'solver', aco)
    
    # In cvrp/faco.py MFACO_CVRP has source_perm property which returns numpy array
    try:
        src_perm_np = aco.source_perm # property in facade
    except AttributeError:
        src_perm_np = solver.source_perm # C++ object

    src_perm = torch.as_tensor(np.asarray(src_perm_np, dtype=np.int64), device=device, dtype=torch.long)  # (m,)
    
    succ = torch.full((n,), -1, device=device, dtype=torch.long)
    pred = torch.full((n,), -1, device=device, dtype=torch.long)
    if src_perm.numel() > 0:
        succ[src_perm] = torch.roll(src_perm, shifts=-1)
        pred[src_perm] = torch.roll(src_perm, shifts=+1)

    is_source_succ = (dst == succ[src]).to(torch.float32).view(E, 1)
    is_source_pred = (dst == pred[src]).to(torch.float32).view(E, 1)

    pos = torch.full((n,), -1, device=device, dtype=torch.long)
    if src_perm.numel() > 0:
        pos[src_perm] = torch.arange(src_perm.numel(), device=device, dtype=torch.long)
        m = int(src_perm.numel())
        duv = (pos[src] - pos[dst]).abs()
        # Fix logic for undirected adjacency in cycle: delta=1 or delta=m-1
        undirected_adj = ((duv == 1) | (duv == (m - 1))) & (pos[src] >= 0) & (pos[dst] >= 0)
    else:
        undirected_adj = torch.zeros((E,), device=device, dtype=torch.bool)

    is_new_edge = (~undirected_adj).to(torch.float32).view(E, 1)

    edge_attr = torch.cat(
        [dist_norm, tau_cv_e, log_tau_rel, is_source_succ, is_source_pred, is_new_edge],
        dim=1,
    )

    depot_flag = torch.zeros((coords_t.size(0), 1), device=device)
    depot_flag[0, 0] = 1.0
    # CVRP specific node inputs: coords (2), demand (1), depot_flag (1) -> 4 dims
    coords_cat = torch.cat([coords_t, demand_t.unsqueeze(-1), depot_flag], dim=-1)

    return Data(x=coords_cat, edge_index=edge_index, edge_attr=edge_attr)
        