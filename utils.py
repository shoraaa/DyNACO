import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from torch.utils.data import TensorDataset
import ast
import logging
import sys
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = (_THIS_DIR / "data").resolve()


# =============================================================================
# LOGGING INFRASTRUCTURE
# =============================================================================

class Logger:
    """
    Unified logger for training that handles console output, file logging,
    and wandb integration.
    """
    
    def __init__(
        self,
        name: str = "train",
        use_wandb: bool = True,
        log_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        self.name = name
        self.use_wandb = use_wandb
        self.verbose = verbose
        self.log_dir = log_dir
        self._step = 0
        
        # Setup Python logging
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Console handler
        if not self._logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
            
            # File handler (if log_dir provided)
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_dir / f"{name}.log")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)
    
    def set_step(self, step: int):
        """Set the current global step for wandb logging."""
        self._step = step
    
    def info(self, msg: str):
        """Log info message to console and file."""
        self._logger.info(msg)
    
    def debug(self, msg: str):
        """Log debug message (file only unless verbose)."""
        self._logger.debug(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self._logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self._logger.error(msg)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: str = "",
        step: Optional[int] = None
    ):
        """
        Log metrics to wandb with optional prefix.
        
        Args:
            metrics: Dictionary of metric name -> value
            prefix: Prefix to add to all metric names (e.g., "train/", "val/")
            step: Global step (uses internal step if not provided)
        """
        if not self.use_wandb:
            return
        
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return
        
        step = step if step is not None else self._step
        
        log_dict = {}
        for k, v in metrics.items():
            key = f"{prefix}{k}" if prefix else k
            log_dict[key] = float(v) if v is not None else 0.0
        
        wandb.log(log_dict, step=step)
    
    def log_train_step(
        self,
        avg_cost: float,
        best_cost: float,
        epoch: int,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log a single training step."""
        step = step if step is not None else self._step
        
        log_dict = {
            "train/avg_cost": avg_cost,
            "train/best_cost": best_cost,
            "train/epoch": epoch,
        }
        
        for k, v in metrics.items():
            log_dict[f"train/{k}"] = float(v) if v is not None else 0.0
        
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(log_dict, step=step)
            except ImportError:
                pass
    
    def log_validation(
        self,
        avg_last: float,
        avg_best: float,
        gap: float,
        epoch: int,
        metrics: Dict[str, float],
        timing: Optional[Dict[str, float]] = None,
        step: Optional[int] = None
    ):
        """Log validation results."""
        step = step if step is not None else self._step
        
        log_dict = {
            "val/avg_last": avg_last,
            "val/avg_best": avg_best,
            "val/gap": gap,
            "val/epoch": epoch,
        }
        
        if timing:
            for k, v in timing.items():
                log_dict[f"time/{k}"] = float(v)
        
        for k, v in metrics.items():
            log_dict[f"val/{k}"] = float(v) if v is not None else 0.0
        
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(log_dict, step=step)
            except ImportError:
                pass
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_cost: float,
        val_best: float,
        gap: float
    ):
        """Print epoch summary to console."""
        self.info(
            f"Epoch {epoch}: TrainCost={train_cost:.4f} "
            f"ValBest={val_best:.4f} Gap={gap:.2f}%"
        )
    
    def log_model_saved(self, path: Path, epoch: int, val_cost: float, gap: float):
        """Log model save event."""
        self.info(
            f"Saved new best model to {path} "
            f"(Epoch {epoch}, Val Cost: {val_cost:.4f}, Gap: {gap:.2f}%)"
        )


@dataclass
class MetricsCollector:
    """
    Collects and aggregates metrics during training/inference.
    
    Provides a clean interface for accumulating metrics across
    iterations and computing final aggregates.
    """
    
    _metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def reset(self):
        """Clear all collected metrics."""
        self._metrics.clear()
    
    def add(self, name: str, value: float):
        """Add a single metric value."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)
    
    def add_dict(self, metrics: Dict[str, float]):
        """Add multiple metrics at once."""
        for name, value in metrics.items():
            self.add(name, value)
    
    def get_mean(self, name: str) -> float:
        """Get mean of a specific metric."""
        if name not in self._metrics or len(self._metrics[name]) == 0:
            return 0.0
        return float(np.mean(self._metrics[name]))
    
    def get_all_means(self) -> Dict[str, float]:
        """Get means of all collected metrics."""
        return {k: self.get_mean(k) for k in self._metrics}
    
    def get_last(self, name: str) -> Optional[float]:
        """Get the last value of a specific metric."""
        if name not in self._metrics or len(self._metrics[name]) == 0:
            return None
        return self._metrics[name][-1]
    
    def has(self, name: str) -> bool:
        """Check if a metric exists and has values."""
        return name in self._metrics and len(self._metrics[name]) > 0


# =============================================================================
# GLOBAL LOGGER INSTANCE
# =============================================================================

_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = Logger(use_wandb=False)
    return _logger


def init_logger(
    use_wandb: bool = True,
    log_dir: Optional[Path] = None,
    verbose: bool = True
) -> Logger:
    """Initialize the global logger."""
    global _logger
    _logger = Logger(
        name="train",
        use_wandb=use_wandb,
        log_dir=log_dir,
        verbose=verbose
    )
    return _logger


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

CAPACITY = 250
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_cvrp_instance(n, device, capacity=None):
    """
    Generate a CVRP instance matching Kool et al. (2019) and Hou et al. (2023) style.
    Depot is first node, coordinates are uniform random in [0,1].
    Demands are integers 1-9 normalized by capacity.
    """
    if capacity is None:
        if n >= 50000:
             capacity = 2000
        elif n >= 10000:
             capacity = 1000
        elif n >= 5000:
             capacity = 500
        elif n >= 1000:
             capacity = 250
        else:
             capacity = 50

    # All locations including depot are random uniform
    coords = torch.rand(size=(n + 1, 2), device=device)
    
    # Demands for n customers (depot has 0 demand)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH + 1, size=(n,), device=device)
    demands_normalized = demands.float() / capacity
    all_demands = torch.cat((torch.zeros((1,), device=device), demands_normalized))
    
    # Return coords (n+1, 2), demands (n+1,) normalized, capacity_norm=1.0
    return coords, all_demands, 1.0

def build_pyg_data_cvrp(aco, coords, demand, device, dynamic: bool):
    """
    Build PyG Data for CVRP using 4D node features (coords, demand, depot_flag).
    Edge features (6): dist_norm, tau_cv, log_tau_rel, is_source_succ, is_source_pred, is_new_edge
    
    Note: CVRP source_route is variable-length with depot (0) appearing multiple times:
        [0, x1, x2, 0, x3, x4, x5, 0, ...]
    We handle this by building adjacency from consecutive pairs in the route.
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

    # Source-route features for CVRP
    # CVRP route is variable-length: [0, c1, c2, 0, c3, c4, 0, ...]
    # Depot (0) can have multiple edges - we use directed edge matrices
    try:
        src_route_np = aco.source_route
    except AttributeError:
        src_route_np = aco.source_perm
    
    src_route = torch.as_tensor(np.asarray(src_route_np, dtype=np.int64), device=device, dtype=torch.long)
    
    # Build directed edge matrices from route
    # forward_edge[u,v] = True if (u→v) is in route
    # backward_edge[u,v] = True if (v→u) is in route (i.e., u is successor of v)
    forward_edge = torch.zeros((n, n), device=device, dtype=torch.bool)
    backward_edge = torch.zeros((n, n), device=device, dtype=torch.bool)
    
    if src_route.numel() > 1:
        route_u = src_route[:-1]  # All but last
        route_v = src_route[1:]   # All but first
        # Mark directed edges: u → v means v is successor of u
        forward_edge[route_u, route_v] = True
        backward_edge[route_v, route_u] = True  # v → u means u is predecessor of v
    
    # is_source_succ[e]: for edge (src[e], dst[e]), is dst the successor of src in route?
    # This correctly handles depot having multiple successors (first customer of each vehicle)
    is_source_succ = forward_edge[src, dst].to(torch.float32).view(E, 1)
    
    # is_source_pred[e]: for edge (src[e], dst[e]), is dst the predecessor of src in route?
    # This correctly handles depot having multiple predecessors (last customer of each vehicle)  
    is_source_pred = backward_edge[src, dst].to(torch.float32).view(E, 1)
    
    # is_new_edge: edge not in current route (either direction)
    is_adjacent = forward_edge | backward_edge.T  # Undirected adjacency
    edge_in_route = is_adjacent[src, dst]
    is_new_edge = (~edge_in_route).to(torch.float32).view(E, 1)

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
    # Priority 1: Check for text datasets in data/{PROBLEM}/data/validation_set/
    # e.g., tsp100 in val_tsp100_concorde(n10000?).txt
    val_set_dir = DATA_DIR / problem.upper() / "data" / "validation_set"
    
    if val_set_dir.exists():
        # Find file matching pattern
        candidates = list(val_set_dir.glob("*.txt"))
        target_file = None
        
        # Simple heuristic: filename contains "{problem}{n}" (e.g. tsp100)
        # We search specifically for the number to avoid partial matches (like tsp10 matching tsp100)
        # But given standard names (tsp100, tsp1000), "tsp{n}" usually distinct enough if n is distinct.
        # Let's try to match `{problem}{n}_` or `{problem}{n}` inside name
        pattern = f"{problem.lower()}{n}"
        
        for f in candidates:
             if pattern in f.name.lower():
                 # Avoid matching tsp100 inside tsp1000 by checking next char is not digit?
                 # e.g. "tsp100_" vs "tsp1000_"
                 # Find where pattern starts
                 name = f.name.lower()
                 idx = name.find(pattern)
                 if idx != -1:
                     after = name[idx+len(pattern):]
                     if not after or not after[0].isdigit():
                         target_file = f
                         break
        
        if target_file:
            print(f"Auto-detected validation set: {target_file}")
            if problem == 'tsp':
                return load_tsp_txt_dataset(str(target_file))
            else:
                return load_cvrp_txt_dataset(str(target_file))

    # Priority 2: Fallback to .pt file
    path = f'{DATA_DIR}/{problem}/valDataset-{n}.pt'
    if not Path(path).exists():
        return None
    try:
        if problem == 'tsp':
            pack = torch.load(path, map_location=device, weights_only=False)
            return pack["coords"]
        else:
            return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def calc_tour_length(coords, tour):
    """
    Calculate the length of a tour.
    coords: (N, 2) tensor or numpy array
    tour: (N,) or (N+1,) list/tensor/array of indices. 
    """
    if isinstance(coords, torch.Tensor):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = coords
    
    if isinstance(tour, torch.Tensor):
        tour_np = tour.cpu().numpy()
    else:
        tour_np = np.array(tour)
    
    # Ensure tour is complete loop
    if tour_np[0] != tour_np[-1] and len(tour_np) == len(coords_np):
        tour_np = np.concatenate([tour_np, [tour_np[0]]])
    
    dist = 0.0
    for i in range(len(tour_np) - 1):
        u, v = tour_np[i], tour_np[i+1]
        diff = coords_np[u] - coords_np[v]
        dist += np.sqrt(np.sum(diff**2))
    return dist


def load_tsp_txt_dataset(path):
    """
    Load TSP dataset from text file. Supports MCTS format and TSPlib format.
    Returns a list of (coords, cost, tour) tuples.
    coords: torch.Tensor (N, 2)
    cost: float
    tour: list of int (0-based indices)
    """
    data_list = []
    print(f"Parsing TSP text data from {path}...")
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        try:
            # Check format type
            if "output" in line:
                # MCTS Format: float... output int...
                parts = line.split(" ")
                output_idx = parts.index("output")
                name = parts[0] if len(parts) > 0 else f"Instance_{line_idx}"
                
                # Parse Coords
                coords_flat = [float(x) for x in parts[:output_idx]]
                num_nodes = len(coords_flat) // 2
                coords = torch.tensor(coords_flat).view(num_nodes, 2)
                
                # Parse Tour
                # Tour indices are 1-based in file, convert to 0-based.
                # Sometimes line ends with potential empty strings if split by space naively, but parts usually cleans up ok?
                # Actually earlier log showed: ... output 1 949 709 ... 
                # Let's filter empty strings just in case
                tour_parts = [x for x in parts[output_idx+1:] if x]
                tour = [int(x) - 1 for x in tour_parts]
                
                # Cost
                cost = calc_tour_length(coords, tour)
                
                data_list.append((coords, cost, tour, name))
                
            elif line.startswith("['"):
                # TSPlib Format: ['name', 'cost', flattened_coords...]
                # We can use ast.literal_eval or string manipulation. 
                # Given the format is simple string repr of list, manual parsing might be faster/safer if standard.
                # implementation in sil_test.py used string replace. Let's do similar for robustness.
                
                # Clean up list syntax
                content = line.replace('[', '').replace(']', '').replace("'", "")
                parts = content.split(',')
                parts = [p.strip() for p in parts]
                
                parts = [p.strip() for p in parts]
                
                name = parts[0]
                cost = float(parts[1])
                coords_flat = [float(x) for x in parts[2:]]
                
                num_nodes = len(coords_flat) // 2
                coords = torch.tensor(coords_flat).view(num_nodes, 2)
                
                # Tour is not explicitly in this line, usually.
                # If we need tour verification, we can't do it comfortably without generating it.
                # But we have the optimal cost provided.
                tour = None 
                
                data_list.append((coords, cost, tour, name))

            elif line.startswith("["):
                 # Generated Format: [coords],cost,[tour]
                 # We wrap in [] to make it a list of 3 elements: [[coords], cost, [tour]]
                 try:
                     row_data = ast.literal_eval(f"[{line}]")
                     if len(row_data) == 3:
                         coords_flat, cost, tour = row_data
                         num_nodes = len(coords_flat) // 2
                         coords = torch.tensor(coords_flat).view(num_nodes, 2)
                         name = f"Gen_{line_idx}"
                         data_list.append((coords, cost, tour, name))
                 except Exception:
                     pass
                
            else:
                # Unknown format or header
                # Try simple coords only if lines are just numbers? 
                # For now skip.
                continue
                
        except Exception as e:
            print(f"Error parsing line {line_idx+1}: {e}")
            continue
            
    print(f"Loaded {len(data_list)} instances.")
    return data_list


def load_cvrp_txt_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    print(f"Loading CVRP txt dataset from {path}")
    with open(path, 'r') as f:
        lines = f.readlines()

    data_list = []
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        try:
            # Check for Format 2 (Python list style) first
            if line.startswith("['") or line.startswith('["'):
                # Format 2: ['name', ..., 'depot', ..., 'customer', ..., 'demand', ..., 'capacity', ..., 'cost', ..., 'end']
                content = line.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
                parts = [p.strip() for p in content.split(',')]
                
                try:
                    depot_idx = parts.index('depot')
                    cust_idx = parts.index('customer')
                    cap_idx = parts.index('capacity')
                    if 'cost' in parts:
                        cost_idx = parts.index('cost')
                    else:
                        cost_idx = -1
                    if 'demand' in parts:
                        dem_idx = parts.index('demand')
                    else:
                        dem_idx = -1
                except ValueError:
                    continue 
                
                    continue 
                
                name = parts[0]

                # Depot
                depot_coords = [float(parts[depot_idx+1]), float(parts[depot_idx+2])]
                
                # Customer Coords
                keywords = [depot_idx, cust_idx, cap_idx, cost_idx, dem_idx]
                keywords = [k for k in keywords if k > cust_idx]
                cust_end_idx = min(keywords) if keywords else len(parts)
                
                cust_coords_flat = [float(x) for x in parts[cust_idx+1 : cust_end_idx]]
                num_cust = len(cust_coords_flat) // 2
                
                # Combine depot + customers
                all_coords_flat = depot_coords + cust_coords_flat
                coords = torch.tensor(all_coords_flat).view(num_cust+1, 2)
                
                # Demand
                if dem_idx != -1:
                   keywords = [depot_idx, cust_idx, cap_idx, cost_idx, dem_idx]
                   keywords = [k for k in keywords if k > dem_idx]
                   dem_end_idx = min(keywords) if keywords else len(parts)
                   dem_raw = [float(x) for x in parts[dem_idx+1 : dem_end_idx]]
                   demand = torch.tensor(dem_raw)
                   if len(demand) == num_cust:
                       demand = torch.cat([torch.tensor([0.0]), demand])
                else:
                   demand = None
                
                # Capacity
                capacity = float(parts[cap_idx+1])
                
                # Cost
                cost = float(parts[cost_idx+1]) if cost_idx != -1 else 0.0
                
                tour = None # Format 2 usually doesn't have tour
                
                data_list.append((coords, demand, capacity, cost, tour, name))

            # Format 1 (Comma separated with keywords)
            elif "depot" in line and "customer" in line:
                parts = [p.strip() for p in line.split(',')]
                name = parts[0] if parts else f"Instance_{line_idx}"
                
                try:
                    depot_idx = parts.index('depot')
                    cust_idx = parts.index('customer')
                    cap_idx = parts.index('capacity')
                except ValueError:
                    continue

                dem_idx = parts.index('demand') if 'demand' in parts else -1
                cost_idx = parts.index('cost') if 'cost' in parts else -1
                tour_idx = parts.index('node_flag') if 'node_flag' in parts else -1
                
                # Depot
                depot_coords = [float(parts[depot_idx+1]), float(parts[depot_idx+2])]
                
                # Customer Coords
                keywords = [depot_idx, cust_idx, cap_idx, dem_idx, cost_idx, tour_idx]
                keywords = [k for k in keywords if k > cust_idx]
                cust_end_idx = min(keywords) if keywords else len(parts)
                
                cust_coords_flat = [float(x) for x in parts[cust_idx+1 : cust_end_idx]]
                num_cust = len(cust_coords_flat) // 2
                
                all_coords_flat = depot_coords + cust_coords_flat
                coords = torch.tensor(all_coords_flat).view(num_cust+1, 2)
                
                # Capacity
                capacity = float(parts[cap_idx+1])
                
                # Demand
                if dem_idx != -1:
                    keywords = [depot_idx, cust_idx, cap_idx, dem_idx, cost_idx, tour_idx]
                    keywords = [k for k in keywords if k > dem_idx]
                    dem_end_idx = min(keywords) if keywords else len(parts)
                    dem_raw = [float(x) for x in parts[dem_idx+1 : dem_end_idx]]
                    demand = torch.tensor(dem_raw)
                    if len(demand) == num_cust:
                        demand = torch.cat([torch.tensor([0.0]), demand])
                else:
                    demand = None
                
                # Cost
                cost = float(parts[cost_idx+1]) if cost_idx != -1 else 0.0
                
                # Tour / node_flag
                tour = None
                if tour_idx != -1:
                     keywords = [depot_idx, cust_idx, cap_idx, dem_idx, cost_idx, tour_idx]
                     keywords = [k for k in keywords if k > tour_idx]
                     tour_end = min(keywords) if keywords else len(parts)
                     tour_parts = parts[tour_idx+1 : tour_end]
                     if tour_parts:
                        try:
                            tour = [int(float(x)) for x in tour_parts]
                        except ValueError:
                             pass
                
                data_list.append((coords, demand, capacity, cost, tour, name))

        except Exception as e:
            print(f"Error parsing CVRP line {line_idx+1}: {e}")
            continue

    print(f"Loaded {len(data_list)} CVRP instances.")
    return data_list


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


# ----------------- Metric Helper Functions -----------------

EPS = 1e-10

def row_softmax(P: torch.Tensor) -> torch.Tensor:
    """Apply softmax normalization per row."""
    return torch.softmax(P.float(), dim=1)

def mean_row_kl(P_prev: torch.Tensor, P_cur: torch.Tensor) -> float:
    """Compute mean KL divergence between consecutive prior distributions (row-wise)."""
    p = row_softmax(P_prev)
    q = row_softmax(P_cur)
    kl_row = (p * ((p + EPS).log() - (q + EPS).log())).sum(dim=1)
    return float(kl_row.mean())

def rel_l2_drift(P_prev: torch.Tensor, P_cur: torch.Tensor) -> float:
    """Compute relative L2 drift between consecutive priors."""
    a = P_prev.float()
    b = P_cur.float()
    return float((b - a).norm() / (a.norm() + EPS))

def top_set(P: torch.Tensor, frac: float = 0.05) -> set:
    """Extract indices of top-k elements (k = frac * total elements)."""
    v = P.flatten()
    m = v.numel()
    k = max(1, int(m * frac))
    idx = torch.topk(v, k).indices
    return set(idx.cpu().tolist())

def top_turnover(P_prev: torch.Tensor, P_cur: torch.Tensor, frac: float = 0.05) -> float:
    """Compute turnover rate of top-k elements using Jaccard distance."""
    if P_prev is None or P_cur is None: 
        return 0.0
    A = top_set(P_prev, frac)
    B = top_set(P_cur, frac)
    jacc = len(A & B) / max(1, len(A | B))
    return float(1.0 - jacc)

def top1_flip_rate(P_prev: torch.Tensor, P_cur: torch.Tensor) -> float:
    """Compute rate at which argmax changes per row."""
    if P_prev is None or P_cur is None: 
        return 0.0
    a = P_prev.argmax(dim=1)
    b = P_cur.argmax(dim=1)
    return float((a != b).float().mean())

def safe_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute robust correlation between two tensors (GPU-optimized)."""
    if a is None or b is None: 
        return 0.0
    
    # Keep on original device, convert to float32
    device = a.device
    a = a.detach().reshape(-1).to(dtype=torch.float32)
    b = b.detach().reshape(-1).to(device=device, dtype=torch.float32)
    
    # Filter out non-finite values
    mask = torch.isfinite(a) & torch.isfinite(b)
    if mask.sum() < 2: 
        return float("nan")
    
    a = a[mask]
    b = b[mask]
    
    # Check for zero variance
    a_std = a.std()
    b_std = b.std()
    if float(a_std) < eps or float(b_std) < eps: 
        return float("nan")
    
    # Compute correlation on GPU
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b) / denom)

def top_overlap_frac(a: torch.Tensor, b: torch.Tensor, frac: float = 0.05) -> float:
    """Compute fraction of overlap in top-k elements between two tensors (GPU-optimized)."""
    if a is None or b is None: 
        return 0.0
    
    a_flat = a.flatten()
    b_flat = b.flatten()
    m = a_flat.numel()
    k = max(1, int(m * frac))
    
    # Compute topk on GPU
    ai = torch.topk(a_flat, k).indices
    bi = torch.topk(b_flat, k).indices
    
    # Use GPU for intersection calculation
    # Create boolean masks and compute intersection
    ai_set = torch.zeros(m, dtype=torch.bool, device=a.device)
    bi_set = torch.zeros(m, dtype=torch.bool, device=b.device)
    ai_set[ai] = True
    bi_set[bi] = True
    
    inter = (ai_set & bi_set).sum()
    return float(inter) / k

def row_top1_match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute rate at which argmax matches per row between two tensors."""
    if a is None or b is None: 
        return 0.0
    return float((a.argmax(dim=1) == b.argmax(dim=1)).float().mean())


def generate_and_save_dataset(problem, n_node, n_instances, save_path, baseline_solver='lkh', 
                               baseline_runs=1, time_limit=300.0, device='cpu'):
    """
    Generate a dataset of problem instances, compute baseline costs, and save to file.
    
    Args:
        problem: 'tsp' or 'cvrp'
        n_node: Number of nodes/customers
        n_instances: Number of instances to generate
        save_path: Path to save the dataset (.txt format)
        baseline_solver: 'lkh' for TSP, 'hgs' for CVRP, or 'none'
        baseline_runs: Number of baseline runs per instance
        time_limit: Time limit for baseline solver
        device: Device for generation
        
    Returns:
        dataset: List of generated instances with baseline costs
    """
    from tqdm import tqdm
    
    dataset = []
    
    print(f"Generating {n_instances} {problem.upper()} instances (n={n_node})...")
    
    for i in tqdm(range(n_instances), desc="Generating"):
        if problem == 'tsp':
            coords = generate_tsp_instance(n_node)
            if isinstance(coords, torch.Tensor):
                coords_np = coords.cpu().numpy()
            else:
                coords_np = coords
                
            # Compute baseline
            if baseline_solver != 'none':
                from baselines import solve_with_lkh
                cost, tour = solve_with_lkh(coords_np, runs=baseline_runs, time_limit=time_limit)
            else:
                cost, tour = 0.0, list(range(n_node))
                
            dataset.append((coords_np, cost, tour, f"Gen_{i}"))
            
        elif problem == 'cvrp':
            coords, demand, capacity = gen_cvrp_instance(n_node, device)
            if isinstance(coords, torch.Tensor):
                coords_np = coords.cpu().numpy()
            else:
                coords_np = coords
            if isinstance(demand, torch.Tensor):
                demand_np = demand.cpu().numpy()
            else:
                demand_np = demand
            capacity_val = float(capacity)
            
            # Compute baseline
            if baseline_solver != 'none':
                from baselines import solve_with_hgs
                # Need to denormalize demand for HGS
                # gen_cvrp_instance returns normalized demand (demand/capacity)
                # HGS expects integer demands
                # Use same capacity logic as gen_cvrp_instance
                if n_node >= 50000:
                    real_cap = 2000
                elif n_node >= 10000:
                    real_cap = 1000
                elif n_node >= 5000:
                    real_cap = 500
                elif n_node >= 1000:
                    real_cap = 250  
                else:
                    real_cap = 50
                    
                # demand_np[1:] contains normalized customer demands, [0] is depot (0)
                demand_int = (demand_np * real_cap).astype(np.int32)
                cost = solve_with_hgs(coords_np, demand_int, real_cap, 
                                         time_limit=time_limit)
                tour = []  # HGS doesn't return tour in this implementation
            else:
                cost, tour = 0.0, list(range(n_node + 1))
                
            dataset.append((coords_np, demand_np, capacity_val, cost, tour, f"Gen_{i}"))
    
    # Save to file
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving dataset to {save_path}...")
        with open(save_path, 'w') as f:
            for item in dataset:
                if problem == 'tsp':
                    coords, cost, tour, name = item
                    # Format: coords_flat, cost, tour
                    coords_flat = coords.flatten().tolist()
                    line = f"{coords_flat},{cost},{tour}\n" # Name not saved in this simplistic generated format but okay
                else:
                    coords, demand, capacity, cost, tour, name = item
                    coords_flat = coords.flatten().tolist()
                    demand_flat = demand.flatten().tolist()
                    line = f"{coords_flat},{demand_flat},{capacity},{cost},{tour}\n"
                f.write(line)
                
        print(f"Saved {len(dataset)} instances.")
        
    return dataset

