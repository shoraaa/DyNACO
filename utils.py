import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from torch.utils.data import TensorDataset

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = (_THIS_DIR / "data").resolve()

# ----------------- TSP Utils -----------------

def gen_distance_matrix(coords):
    n = len(coords)
    dists = torch.norm(coords[:, None] - coords, dim=2, p=2)
    dists[torch.arange(n), torch.arange(n)] = 1e9
    return dists

def generate_tsp_instance(n):
    return np.random.rand(n, 2).astype(np.float32)

def build_pyg_data_tsp(aco, coords, device, dynamic: bool):
    """
    Build PyG Data for TSP using 2D node features (coords).
    Edge features (6): dist_norm, tau_cv, log_tau_rel, is_source_succ, is_source_pred, is_new_edge
    """
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    coords = coords.to(device=device, dtype=torch.float32) # (n,2)

    nn = aco.nn_torch.to(device=device, dtype=torch.long)
    n, k = nn.shape
    E = n * k

    src = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(k)
    dst = nn.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    # 1. dist_norm
    dist = torch.norm(coords[src] - coords[dst], dim=1).view(n, k)
    dist_mean = dist.mean(dim=1, keepdim=True).clamp_min(1e-12)
    dist_norm = (dist / dist_mean).view(E, 1)

    # 2-3. tau features
    if dynamic:
        tau = aco.pheromone_sparse.detach().to(device=device, dtype=torch.float32)
        tau_mean = tau.mean(dim=1, keepdim=True).clamp_min(1e-12)
        tau_rel = (tau / tau_mean).clamp_min(1e-12)
        log_tau_rel = torch.log(tau_rel).clamp(-5.0, 5.0).view(E, 1)
        tau_std = tau.std(dim=1, keepdim=True)
        tau_cv = (tau_std / tau_mean).clamp(0, 10).repeat_interleave(k, dim=0)

        # Source tour features
        sr = torch.as_tensor(np.asarray(aco.source_route), device=device, dtype=torch.long)
        succ = torch.empty((n,), device=device, dtype=torch.long)
        pred = torch.empty((n,), device=device, dtype=torch.long)
        succ[sr] = torch.roll(sr, shifts=-1)
        pred[sr] = torch.roll(sr, shifts=+1)

        is_source_succ = (dst == succ[src]).to(torch.float32).view(E, 1)
        is_source_pred = (dst == pred[src]).to(torch.float32).view(E, 1)

        pos = torch.empty((n,), device=device, dtype=torch.long)
        pos[sr] = torch.arange(n, device=device, dtype=torch.long)
        duv = (pos[src] - pos[dst]).abs()
        undirected_adj = (duv == 1) | (duv == (n - 1))
        is_new_edge = (~undirected_adj).to(torch.float32).view(E, 1)

    else:
        log_tau_rel = torch.zeros((E, 1), device=device, dtype=torch.float32)
        tau_cv = torch.zeros((E, 1), device=device, dtype=torch.float32)
        is_source_succ = torch.zeros((E, 1), device=device, dtype=torch.float32)
        is_source_pred = torch.zeros((E, 1), device=device, dtype=torch.float32)
        is_new_edge = torch.zeros((E, 1), device=device, dtype=torch.float32)

    edge_attr = torch.cat(
        [dist_norm, tau_cv, log_tau_rel, is_source_succ, is_source_pred, is_new_edge],
        dim=1
    )
    return Data(x=coords, edge_index=edge_index, edge_attr=edge_attr)


# ----------------- CVRP Utils -----------------

CAPACITY = 50
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_cvrp_instance(n, device, capacity=None):
    if capacity is None:
        if n > 1000:
             capacity = 300 # Larger scale
        elif n == 1000:
             capacity = 200 # CVRP1K
        else:
             capacity = 50 # Standard small scale

    locations = torch.rand(size=(n, 2), device=device)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=(n,), device=device)

    depot = torch.tensor([DEPOT_COOR], device=device, dtype=locations.dtype)
    coords = torch.cat((depot, locations), dim=0)
    
    demand = torch.cat((torch.zeros((1,), device=device, dtype=demands.dtype), demands), dim=0)
    demand_f = demand.float() / capacity
    capacity_norm = 1.0

    return coords, demand_f, capacity_norm

def build_pyg_data_cvrp(aco, coords, demand, device, dynamic: bool):
    """
    Build PyG Data for CVRP using 4D node features (coords, demand, depot_flag).
    Edge features (6): dist_norm, tau_cv, log_tau_rel, is_source_succ, is_source_pred, is_new_edge
    """
    if isinstance(coords, np.ndarray):
        coords_t = torch.from_numpy(coords)
    elif isinstance(coords, torch.Tensor):
        coords_t = coords
    else:
        coords_t = torch.as_tensor(coords)
    
    coords_t = coords_t.to(device=device, dtype=torch.float32)
    demand_t = torch.as_tensor(demand, device=device, dtype=torch.float32)

    nn = aco.nn_torch.to(device=device, dtype=torch.long)
    n, k = nn.shape
    E = n * k

    src = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(k)
    dst = nn.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    # 1. dist_norm
    dist = torch.norm(coords_t[src] - coords_t[dst], dim=1).view(n, k)
    dist_mean = dist.mean(dim=1, keepdim=True).clamp_min(1e-12)
    dist_norm = (dist / dist_mean).view(E, 1)

    # 2-3. tau features
    if dynamic:
        tau = aco.pheromone_sparse.detach().to(device=device, dtype=torch.float32)
        tau_mean = tau.mean(dim=1, keepdim=True).clamp_min(1e-12)
        tau_rel = (tau / tau_mean).clamp_min(1e-12)
        log_tau_rel = torch.log(tau_rel).clamp(-5.0, 5.0).view(E, 1)
        tau_std = tau.std(dim=1, keepdim=True)
        tau_cv = (tau_std / tau_mean).clamp(0, 10).repeat_interleave(k, dim=0)
    else:
        log_tau_rel = torch.zeros((E, 1), device=device, dtype=torch.float32)
        tau_cv = torch.zeros((E, 1), device=device, dtype=torch.float32)

    # Source-perm features
    solver = getattr(aco, 'solver', aco)
    try:
        src_perm_np = aco.source_perm
    except AttributeError:
        src_perm_np = solver.source_perm
        
    src_perm = torch.as_tensor(np.asarray(src_perm_np, dtype=np.int64), device=device, dtype=torch.long)
    
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
        undirected_adj = ((duv == 1) | (duv == (m - 1))) & (pos[src] >= 0) & (pos[dst] >= 0)
    else:
        undirected_adj = torch.zeros((E,), device=device, dtype=torch.bool)

    is_new_edge = (~undirected_adj).to(torch.float32).view(E, 1)

    edge_attr = torch.cat(
        [dist_norm, tau_cv, log_tau_rel, is_source_succ, is_source_pred, is_new_edge],
        dim=1
    )

    depot_flag = torch.zeros((coords_t.size(0), 1), device=device)
    depot_flag[0, 0] = 1.0
    coords_cat = torch.cat([coords_t, demand_t.unsqueeze(-1), depot_flag], dim=-1)

    return Data(x=coords_cat, edge_index=edge_index, edge_attr=edge_attr)


# ----------------- Shared/Dataset -----------------

def load_val_dataset(n, problem='tsp', device='cpu'):
    path = f'{DATA_DIR}/{problem}/valDataset-{n}.pt'
    if not Path(path).exists():
        return None
    try:
        if problem == 'tsp':
            pack = torch.load(path, map_location=device)
            return pack["coords"]
        else:
            return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def save_val_dataset(dataset, n, problem='tsp'):
    path_dir = DATA_DIR / problem
    path_dir.mkdir(parents=True, exist_ok=True)
    path = path_dir / f'valDataset-{n}.pt'
    
    # Store in dict format for TSP as load_val_dataset expects "coords" key
    if problem == 'tsp':
        # Assuming dataset is list of tensors or single tensor
        if isinstance(dataset, list):
            coords = torch.stack(dataset)
        else:
            coords = dataset
        torch.save({"coords": coords}, path)
    else:
        # CVRP: save directly (list of tuples or whatever structure)
        torch.save(dataset, path)
    print(f"Saved generated dataset to {path}")

