import torch
from torch_geometric.data import Data
from faco import MFACO_TSP
import numpy as np

from pathlib import Path
import torch

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = (_THIS_DIR / ".." / "data").resolve()  # adjust depth if needed

def gen_distance_matrix(tsp_coordinates):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances
    
def gen_pyg_data(tsp_coordinates, k_sparse):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    n_nodes = len(tsp_coordinates)
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances, 
                                           k=k_sparse, 
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)
    pyg_data = Data(x=tsp_coordinates, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances

from torch_geometric.data import Data
import torch, numpy as np
import numpy as np
import torch
from torch_geometric.data import Data

def build_pyg_data(aco, coords, device, dynamic: bool):
    """
    Node features:
      x: coords (n,2)

    Edge features (edge_attr, E=n*k, 6):
      0 dist_norm
      1 rank_norm
      2 log_tau_rel         (dynamic) else 0
      3 is_source_succ
      4 is_source_pred
      5 is_new_edge         (not adjacent in source, undirected)
    """
    # coords -> torch
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    coords = coords.to(device=device, dtype=torch.float32)  # (n,2)

    # candidate graph
    nn = aco.nn_torch.to(device=device, dtype=torch.long)   # (n,k)
    n, k = nn.shape
    E = n * k

    src = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(k)  # (E,)
    dst = nn.reshape(-1)                                                         # (E,)
    edge_index = torch.stack([src, dst], dim=0)                                   # (2,E)

    # --- 0) dist_norm ---
    dist = torch.norm(coords[src] - coords[dst], dim=1).view(n, k)                # (n,k)
    dist_mean = dist.mean(dim=1, keepdim=True).clamp_min(1e-12)
    dist_norm = (dist / dist_mean).view(E, 1)


    # --- 1-2) log_tau_rel (dynamic only) ---
    if dynamic:
        tau = aco.pheromone_sparse.detach().to(device=device, dtype=torch.float32)   # (n,k)
        tau_mean = tau.mean(dim=1, keepdim=True).clamp_min(1e-12)
        tau_rel = (tau / tau_mean).clamp_min(1e-12)
        log_tau_rel = torch.log(tau_rel).clamp(-5.0, 5.0).view(E, 1)
        tau_std  = tau.std(dim=1, keepdim=True)
        tau_cv   = (tau_std / tau_mean).clamp(0, 10)  # (n,1)
        tau_cv_e = tau_cv.repeat_interleave(k, dim=0) # (E,1)

        # --- MFACO source-tour features ---
        # source_route is length n, a permutation of nodes
        sr = torch.as_tensor(np.asarray(aco.source_route), device=device, dtype=torch.long)  # (n,)

        # succ[u] = next node in source tour, pred[u] = prev node in source tour
        succ = torch.empty((n,), device=device, dtype=torch.long)
        pred = torch.empty((n,), device=device, dtype=torch.long)
        succ[sr] = torch.roll(sr, shifts=-1)
        pred[sr] = torch.roll(sr, shifts=+1)

        # 3) is_source_succ: v == succ[u]
        is_source_succ = (dst == succ[src]).to(torch.float32).view(E, 1)

        # 4) is_source_pred: v == pred[u]
        is_source_pred = (dst == pred[src]).to(torch.float32).view(E, 1)

        # positions in source tour for undirected adjacency test
        pos = torch.empty((n,), device=device, dtype=torch.long)
        pos[sr] = torch.arange(n, device=device, dtype=torch.long)

        duv = (pos[src] - pos[dst]).abs()
        undirected_adj = (duv == 1) | (duv == (n - 1))

        # 5) is_new_edge: not adjacent in source (MFACO "new edge" notion)
        is_new_edge = (~undirected_adj).to(torch.float32).view(E, 1)

    else:
        log_tau_rel = torch.zeros((E, 1), device=device, dtype=torch.float32)
        tau_cv_e = torch.zeros((E, 1), device=device, dtype=torch.float32)
        is_source_succ = torch.zeros((E, 1), device=device, dtype=torch.float32)
        is_source_pred = torch.zeros((E, 1), device=device, dtype=torch.float32)
        is_new_edge = torch.zeros((E, 1), device=device, dtype=torch.float32)

    # combine edge features (E,6)
    edge_attr = torch.cat(
        [dist_norm, tau_cv_e, log_tau_rel, is_source_succ, is_source_pred, is_new_edge],
        dim=1
    )

    return Data(x=coords, edge_index=edge_index, edge_attr=edge_attr)

def build_pyg_data_shit(aco, coords, device, dynamic: bool):
    """
    Node features:
      x: coords (n,2)
    """
    # coords -> torch
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    coords = coords.to(device=device, dtype=torch.float32)  # (n,2)
    node_feats = coords

    # candidate graph
    nn = aco.nn_torch.to(device=device, dtype=torch.long)   # (n,k)
    n, k = nn.shape
    E = n * k

    src = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(k)  # (E,)
    dst = nn.reshape(-1)                                                         # (E,)
    edge_index = torch.stack([src, dst], dim=0)                                   # (2,E)

    dist = torch.norm(coords[src] - coords[dst], dim=1).view(n, k).clamp_min(1e-12)
    dist_mean = dist.mean(dim=1, keepdim=True).clamp_min(1e-12)
    dist_norm = (dist / dist_mean).view(E, 1)

    if dynamic:
        tau = aco.pheromone_sparse.detach().to(device=device, dtype=torch.float32)   # (n,k)
        tau_mean = tau.mean(dim=1, keepdim=True).clamp_min(1e-12)
        tau_norm = (tau / tau_mean).view(E, 1)
        tau_std  = tau.std(dim=1, keepdim=True)
        tau_cv   = (tau_std / tau_mean)


        sr = torch.as_tensor(np.asarray(aco.source_route), device=device, dtype=torch.long)  # (n,)
        pos = torch.empty((n,), device=device, dtype=torch.long)
        pos[sr] = torch.arange(n, device=device, dtype=torch.long)

        duv = (pos[src] - pos[dst]).abs()
        in_source_tour = ((duv == 1) | (duv == (n - 1))).to(torch.float32).view(E, 1)

    else:
        tau_norm = torch.zeros((E, 1), device=device, dtype=torch.float32)
        tau_cv = torch.zeros((n, 1), device=device, dtype=torch.float32)
        in_source_tour = torch.zeros((E, 1), device=device, dtype=torch.float32)

    node_feats = torch.cat([coords, tau_cv], dim=1)
    edge_attr = torch.cat([dist_norm, tau_norm, in_source_tour], dim=1)  # (E,3)

    return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)


def generate_tsp_instance(n: int):
    """Generate a random TSP instance with n nodes in [0,1]^2."""
    coords = np.random.rand(n, 2).astype(np.float32)
    return coords

def build_pyg_from_faco(faco: MFACO_TSP, coords: np.ndarray, device: str = "cpu"):
    """
    Build PyG Data object from FACO's nn_list (after restore).
    
    Uses faco.nn_list instead of snapshot's stored list to ensure consistency
    between the solver's candidate edges and the graph used for neural encoding.
    """
    # Accept either raw coords (np/torch) or a PyG Data object.
    if not isinstance(coords, (np.ndarray, torch.Tensor)) and hasattr(coords, "x"):
        coords = coords.x

    if isinstance(coords, torch.Tensor):
        x = coords.to(device=device, dtype=torch.float32)
        coords_np = coords.detach().cpu().numpy().astype(np.float32)
    else:
        coords_np = np.asarray(coords, dtype=np.float32)
        x = torch.from_numpy(coords_np).to(device=device, dtype=torch.float32)

    n = int(coords_np.shape[0])
    k = faco.k
    
    # Build edge index from faco.nn_list (kNN graph)
    src = []
    dst = []
    edge_attr = []
    
    for u in range(n):
        for j in range(k):
            v = faco.nn_list[u, j]
            assert v != -1, "FACO nn_list contains invalid index."
            assert v < n, "FACO nn_list contains out-of-bounds index."
            if v >= 0:
                src.append(u)
                dst.append(v)
                edge_attr.append(faco.dist_np32[u, v])
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)


def load_val_dataset(n_node, device):
    pack = torch.load(f'{DATA_DIR}/tsp/valDataset-{n_node}.pt', map_location=device)
    coords = pack["coords"].to(device)
    return coords

def generate_val_dataset(n_node, n_instance, k_sparse, device):
    val_instances = []
    n_node = n_node
    for _ in range(n_instance):
        coords = np.random.rand(n_node, 2).astype(np.float32)
        val_instances.append(torch.from_numpy(coords))
    pack = {"coords": torch.stack(val_instances)}
    torch.save(pack, f'{DATA_DIR}/tsp/valDataset-{n_node}.pt')

def load_test_dataset(n_node, device):
    val_list = []
    val_tensor = torch.load(f'{DATA_DIR}/tsp/testDataset-{n_node}.pt')
    for instance in val_tensor:
        instance = instance.to(device)
        val_list.append(instance)
    return val_list


if __name__ == "__main__":
    # generate val and test datasets, only coords
    import os
    if not os.path.exists(f'{DATA_DIR}/tsp'):
        os.makedirs(f'{DATA_DIR}/tsp')
    torch.manual_seed(123456)
    for n in [100, 200, 500]:
        inst_coords = []
        for _ in range(128):
            coords = np.random.rand(n, 2).astype(np.float32)
            coords_t = torch.from_numpy(coords)
            inst_coords.append(coords_t)
        valDataset = {
            "coords": torch.stack(inst_coords).float(),            # (B, n, 2)
        }
        torch.save(valDataset, f'{DATA_DIR}/tsp/valDataset-{n}.pt')
    for n in [100, 200, 500]:
        inst_coords = []
        for _ in range(128):
            coords = np.random.rand(n, 2).astype(np.float32)
            coords_t = torch.from_numpy(coords)
            inst_coords.append(coords_t)
        testDataset = {
            "coords": torch.stack(inst_coords).float(),            # (B, n, 2)
        }
        torch.save(testDataset, f'{DATA_DIR}/tsp/testDataset-{n}.pt')

    # generate 16 instances of 1000, 2000, 5000 nodes
    for n in [1000, 2000, 5000]:
        inst_coords = []
        for _ in range(16):
            coords = np.random.rand(n, 2).astype(np.float32)
            coords_t = torch.from_numpy(coords)
            inst_coords.append(coords_t)
        valDataset = {
            "coords": torch.stack(inst_coords).float(),            # (B, n, 2)
        }
        torch.save(valDataset, f'{DATA_DIR}/tsp/valDataset-{n}.pt')
    for n in [1000, 2000, 5000]:
        inst_coords = []
        for _ in range(16):
            coords = np.random.rand(n, 2).astype(np.float32)
            coords_t = torch.from_numpy(coords)
            inst_coords.append(coords_t)
        testDataset = {
            "coords": torch.stack(inst_coords).float(),            # (B, n, 2)
        }
        torch.save(testDataset, f'{DATA_DIR}/tsp/testDataset-{n}.pt')
    