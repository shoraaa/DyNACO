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
import time
import math
import faco


import random

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = (_THIS_DIR / "data").resolve()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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

def build_pyg_data_tsp(aco, coords, device, dynamic: bool, simple_features: bool = False):
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

    if simple_features:
        # Simple features: dist_norm, log_tau_rel, is_in_route (3 features)
        # Consistent with CVRP
        is_in_route = (is_source_succ > 0.5) | (is_source_pred > 0.5)
        is_in_route = is_in_route.to(torch.float32).view(E, 1)
        
        edge_attr = torch.cat(
            [dist_norm, log_tau_rel, is_in_route],
            dim=1
        )
    else:
        # Full features (6 features)
        edge_attr = torch.cat(
            [dist_norm, tau_cv, log_tau_rel, is_source_succ, is_source_pred, is_new_edge],
            dim=1
        )
    return Data(x=coords, edge_index=edge_index, edge_attr=edge_attr)


# ----------------- CVRP Utils -----------------

CAPACITY = 250
DEMAND_LOW = 1
DEMAND_HIGH = 9

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
        tau_rel = (tau / tau_mean).clamp_min(1e-12).view(E, 1)
        log_tau_rel = torch.log(tau_rel).clamp(-5.0, 5.0).view(E, 1)
        # Simplified: removed tau_cv
    else:
        tau_rel = torch.ones((E, 1), device=device, dtype=torch.float32)
        log_tau_rel = torch.zeros((E, 1), device=device, dtype=torch.float32)

    # Source-route features for CVRP
    # CVRP route is variable-length: [0, c1, c2, 0, c3, c4, 0, ...]
    # Depot (0) can have multiple edges - we use directed edge matrices
    src_route_np = aco.source_route
    
    src_route = torch.as_tensor(np.asarray(src_route_np, dtype=np.int64), device=device, dtype=torch.long)
    
    # Build directed edge matrices from route
    # forward_edge[u,v] = True if (u→v) is in route
    # backward_edge[u,v] = True if (v→u) is in route (i.e., u is successor of v)
    # Sparse implementation for N=100k scaling
    # We only need to check is_source_succ for edges in (src, dst)
    # is_source_succ[e] = True if edge (src[e], dst[e]) is in route
    
    is_source_succ = torch.zeros((E, 1), device=device, dtype=torch.float32)
    is_source_pred = torch.zeros((E, 1), device=device, dtype=torch.float32)
    
    if src_route.numel() > 1:
        route_u = src_route[:-1]
        route_v = src_route[1:]
        
        # Optimized for N=100k using simple array lookups (O(1)) instead of hashing
        # succ[u] = v implies u->v exists in route.
        # Since customers have unique successor, we can use an array.
        # Depot (0) has multiple successors, handled separately.
        
        # Initialize arrays with -1
        succ_arr = torch.full((n,), -1, device=device, dtype=torch.long)
        pred_arr = torch.full((n,), -1, device=device, dtype=torch.long)
        
        # Fill for non-depot nodes (customers have unique succ/pred)
        # Note: In CVRP, route is 0 -> c1 ... -> 0 -> c2 ...
        # For u!=0, u->v is unique.
        mask_cust_u = (route_u != 0)
        succ_arr[route_u[mask_cust_u]] = route_v[mask_cust_u]
        
        mask_cust_v = (route_v != 0)
        pred_arr[route_v[mask_cust_v]] = route_u[mask_cust_v]
        
        # Identify depot connections
        depot_succs = route_v[route_u == 0] # Nodes v where 0->v
        depot_preds = route_u[route_v == 0] # Nodes u where u->0
        
        # 1. Check Successors: is dst == succ[src]?
        # Case A: src != 0 (Customer)
        mask_src_cust = (src != 0)
        # Check against array. Note: if succ_arr is -1, it won't match valid dst (>=0)
        is_source_succ[mask_src_cust] = (dst[mask_src_cust] == succ_arr[src[mask_src_cust]]).float().view(-1, 1)
        
        # Case B: src == 0 (Depot)
        mask_src_depot = (src == 0)
        if mask_src_depot.any():
            # Check if dst is in the set of depot successors
            is_match = torch.isin(dst[mask_src_depot], depot_succs)
            is_source_succ[mask_src_depot] = is_match.float().view(-1, 1)
            
        # 2. Check Predecessors: is dst == pred[src]? (meaning dst->src in route)
        # Case A: src != 0 (Customer)
        # Note: mask_src_cust is same
        is_source_pred[mask_src_cust] = (dst[mask_src_cust] == pred_arr[src[mask_src_cust]]).float().view(-1, 1)
        
        # Case B: src == 0 (Depot)
        if mask_src_depot.any():
            # Check if dst is in the set of depot predecessors
            is_match = torch.isin(dst[mask_src_depot], depot_preds)
            is_source_pred[mask_src_depot] = is_match.float().view(-1, 1)  
    # Remove redundant assignment (which referred to removed backward_edge)
    # is_source_pred = backward_edge[src, dst].to(torch.float32).view(E, 1) # REMOVED
    
    # is_new_edge: edge not in current route (either direction)
    # Edge is in route if it is a successor OR predecessor edge (undirected check)
    # Simplified: Merge is_source_succ/pred into is_in_route
    is_in_route = (is_source_succ > 0.5) | (is_source_pred > 0.5)
    is_in_route = is_in_route.to(torch.float32).view(E, 1)

    edge_attr = torch.cat(
        [dist_norm, tau_rel, is_in_route],
        dim=1
    )

    depot_flag = torch.zeros((coords_t.size(0), 1), device=device)
    depot_flag[0, 0] = 1.0
    coords_cat = torch.cat([coords_t, demand_t.unsqueeze(-1), depot_flag], dim=-1)

    return Data(x=coords_cat, edge_index=edge_index, edge_attr=edge_attr)


# ----------------- Shared/Dataset -----------------

def load_auto_dataset(n, problem='tsp', data_source='test_set', rl_data=False, device='cpu'):
    """
    Load dataset automatically based on problem size (n) and source/mode.
    defaults to data/{problem}/data/test_set
    python test.py --n_node 1000 --rl_data -> loads TSPlib_1K.txt
    python test.py --n_node 1000           -> loads MCTS_tsp1000... or test_tsp1000...
    """
    # 1. Base Directory
    # Default to test_set
    base_dir = DATA_DIR / problem.upper() / "data" / data_source
    if not base_dir.exists():
        # Fallback to validation_set if test_set not found (legacy behavior?)
        # Or just fail? Let's try validation_set as fallback if default 'test_set' missing
        if data_source == 'test_set':
             alt_dir = DATA_DIR / problem.upper() / "data" / "validation_set"
             if alt_dir.exists():
                 print(f"Warning: {base_dir} not found, falling back to {alt_dir}")
                 base_dir = alt_dir
             else:
                 print(f"Dataset directory {base_dir} not found.")
                 return None
        else:
             print(f"Dataset directory {base_dir} not found.")
             return None

    # 2. Hardcoded File Search
    target_filename = None
    
    # helper for approximate search
    candidates = list(base_dir.glob("*.txt"))
    candidates_names = [f.name for f in candidates]

    # RL Data (TSPlib / CVRPlib)
    if rl_data:
        # Map N to suffix
        # 1000 -> 1K
        # 10000 -> 10K
        # 50000 -> 50K
        # 100000 -> 100K
        suffix = None
        if n == 1000: suffix = "1K"
        elif n == 5000: suffix = "5K"
        elif n == 10000: suffix = "10K"
        elif n == 50000: suffix = "50K"
        elif n == 100000: suffix = "100K"
        
        if problem == 'tsp':
            if suffix:
                pat = f"TSPlib_{suffix}.txt" 
                # Verify existence? Or find match
                # Some files might be slightly different named?
                # Check exact match first
                if pat in candidates_names:
                    target_filename = pat
            
            # Fallback scan:
            if target_filename is None:
                 # Try to find *TSPlib*
                 # Filter by N maybe? 
                 # Just returning first TSPlib related?
                 matches = [f for f in candidates if "TSPlib" in f.name]
                 # If we have matches, maybe try to match numeric? 
                 # For now just pick one?
                 # If N provided, filtering by N is better.
                 # Re-scan for pattern like "{n}K" or "{n}"?
                 pass

        elif problem == 'cvrp':
            if suffix:
                pat = f"CVRPlib_{suffix}.txt"
                if pat in candidates_names:
                    target_filename = pat
            if target_filename is None:
                # CVRPlib fallback
                pass

    else:
        # Standard Test Set (MCTS / HGS / LKH generated)
        # Strictly hardcoded maps to avoid substring matching issues (e.g. 1000 matching 10000)
        
        if problem == 'tsp':
            # Priority: MCTS_{problem}{n} -> test_{problem}{n}
            if n == 10000:
                 target_filename = "MCTS_tsp10000_test_concorde.txt"
            elif n == 50000:
                 target_filename = "test_tsp50000_lkh3_n16.txt"
            elif n == 100000:
                 target_filename = "test_tsp100000_lkh3_n16.txt"
            elif n == 5000:
                 target_filename = "test_tsp5000_lkh3_n16.txt"
            elif n == 100:
                 target_filename = "test_tsp100_concorde_n10000.txt"
            elif n == 1000:
                 target_filename = "MCTS_tsp1000_test_concorde.txt"
        
        elif problem == 'cvrp':
             # Explicit mapping based on file inspection
             if n == 100:
                 target_filename = "test_cvrp100_hgs_n10000_C50.txt"
             elif n == 1000:
                 target_filename = "test_cvrp1000_hgs_n128_C250.txt"
             elif n == 5000:
                 target_filename = "test_cvrp5000_hgs_n16_C500.txt"
             elif n == 10000:
                 target_filename = "test_cvrp10000_hgs_n16_C1000.txt"
             elif n == 50000:
                 target_filename = "test_cvrp50000_hgs_n16_C2000.txt"
             elif n == 100000:
                 target_filename = "test_cvrp100000_hgs_n16_C2000.txt"


        if data_source == 'validation_set':
            # Validation Set (validation_{problem}{n}...)
            
            if problem == 'tsp':
                if n == 100:
                    target_filename = "validation_TSP100_n10000.txt"
                elif n == 1000:
                    target_filename = "validation_TSP1000_n128.txt"
                elif n == 5000:
                    target_filename = "validation_TSP5000_n16.txt"
                elif n == 10000:
                    target_filename = "validation_TSP10000_n16.txt"
                elif n == 50000:
                    target_filename = "validation_TSP50000_n4.txt"
                elif n == 100000:
                    target_filename = "validation_TSP100000_n4.txt"
            
            elif problem == 'cvrp':
                if n == 100:
                    target_filename = "validation_cvrp100_lkh3_n10000_C50.txt"
                elif n == 1000:
                    target_filename = "validation_cvrp1000_n128_C250.txt"
                elif n == 5000:
                    target_filename = "validation_cvrp5000_n16_C500.txt"
                elif n == 10000:
                    target_filename = "validation_cvrp10000_n16_C1000.txt"
                elif n == 50000:
                    target_filename = "validation_cvrp50000_n4_C2000.txt"
                elif n == 100000:
                    target_filename = "validation_cvrp100000_n4_C2000.txt"


    # 3. Load if found
    if target_filename:
        full_path = base_dir / target_filename
        print(f"Auto-detected dataset: {full_path}")
        if problem == 'tsp':
            return load_tsp_txt_dataset(str(full_path))
        else:
            return load_cvrp_txt_dataset(str(full_path))
    else:
        # Fallback to old heuristic scanning if hardcode didn't match
        # (Original logic roughly)
        print(f"No strict match for N={n} (rl_data={rl_data}). Scanning for partial match...")
        
        # Original scanning logic adapted
        pattern = f"{problem.lower()}{n}"
        best_cand = None
        for f in candidates:
             name = f.name.lower()
             if pattern in name:
                 # check digit boundary?
                 idx = name.find(pattern)
                 if idx != -1:
                     after = name[idx+len(pattern):]
                     if not after or not after[0].isdigit():
                         best_cand = f
                         break
        
        if best_cand:
            print(f"Fallback detected: {best_cand}")
            if problem == 'tsp':
                return load_tsp_txt_dataset(str(best_cand))
            else:
                return load_cvrp_txt_dataset(str(best_cand))

    # Priority 2: Fallback to .pt file (unchanged)
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


def verify_solution_cvrp(coords, demand, capacity, cost, route0):
    DEMAND_SCALE = 100000
    n = len(demand)
    visited = set()
    total_dist = 0.0
    cap_int = int(round(capacity * DEMAND_SCALE))
    demand_int = [int(round(d * DEMAND_SCALE)) for d in demand]
    current_load_int = 0
    for i in range(len(route0) - 1):
        u, v = int(route0[i]), int(route0[i+1])
        du = coords[u]
        dv = coords[v]
        d = np.sqrt(((du - dv)**2).sum())
        total_dist += d
        if v == 0:
            if current_load_int > cap_int:
                raise ValueError(f"Capacity violation: {current_load_int/DEMAND_SCALE} > {capacity}")
            current_load_int = 0
        else:
            if v in visited:
                raise ValueError(f"Node {v} visited more than once")
            visited.add(v)
            current_load_int += demand_int[v]
    if len(visited) != n - 1:
        missing = set(range(1, n)) - visited
        raise ValueError(f"Missing customers: {missing}")
    if abs(total_dist - cost) > 1e-3:
        raise ValueError(f"Cost mismatch: recalculated {total_dist:.6f} vs reported {cost:.6f}")
    return True


def infer_instance(problem, aco_class, build_fn, model, instance_data, k_sparse, n_ants, dynamic, args, use_heuristic_only=False, collect_metrics=False, metrics_every_step=True, inject_step=None):
    if model is not None:
        model.eval()

    disable_heuristic_arg = args.disable_heuristic
    if use_heuristic_only:
        disable_heuristic_arg = False 

    # Determine instance args
    if problem == 'tsp':
        coords = instance_data
        n = len(coords)
        if n_ants is None:
             n_ants = int(math.ceil(4 * math.sqrt(n) / 64) * 64)
             
        kwargs = {
            'n_ants': n_ants,
            'coords': coords,
            'cand_list_size': k_sparse,
            'backup_list_size': k_sparse,
            'disable_heuristic': disable_heuristic_arg,
            'use_local_search': not args.no_local_search,
            'decay': args.rho,
            'device': args.device,
            'enable_torch_sync': True,
            'smooth_mmas': not args.no_smooth_mmas,
            'min_new_edges': args.min_new_edges,
            'extend_ls': not args.no_extend_ls,
            'normalized_heuristic': not args.no_normalized_heuristic,
            'fixed_steps': args.L
        }
    else:
        coords, demand, capacity = instance_data
        n = len(coords) - 1 # n customers
        if n_ants is None:
             n_ants = int(math.ceil(4 * math.sqrt(n) / 64) * 64)

        kwargs = {
            'coords': coords,
            'demand': demand,
            'capacity': float(capacity),
            'n_ants': n_ants,
            'cand_list_size': k_sparse,
            'backup_list_size': max(k_sparse, 64),
            'min_new_edges': args.min_new_edges,
            'decay': args.rho,
            'p_best': 0.05,
            'use_local_search': not args.no_local_search,
            'disable_heuristic': disable_heuristic_arg,
            'extend_ls': not args.no_extend_ls, 
            'smooth_mmas': not args.no_smooth_mmas,
            'device': args.device,
            'enable_torch_sync': True,
            'normalized_heuristic': not args.no_normalized_heuristic,
            'fixed_steps': args.L
        }

    # Normalize coordinates for model input (scale to [0, 1] while preserving aspect ratio)
    norm_coords = coords
    if model is not None:
        if torch.is_tensor(coords):
             c_min = coords.min(dim=0)[0]
             c_max = coords.max(dim=0)[0]
             c_diff = c_max - c_min
             scale = c_diff.max()
             if scale < 1e-6: scale = 1.0
             norm_coords = (coords - c_min) / scale
        else:
             c_min = coords.min(axis=0)
             c_max = coords.max(axis=0)
             c_diff = c_max - c_min
             scale = c_diff.max()
             if scale < 1e-6: scale = 1.0
             norm_coords = (coords - c_min) / scale

    
    # Filter kwargs for MMAS classes
    is_mmas = (aco_class == faco.ACO_TSP or aco_class == faco.ACO_CVRP)
    if is_mmas:
        # MMAS classes don't accept these MFACO-specific parameters
        mmas_kwargs = {
            'coords': kwargs['coords'],
            'n_ants': kwargs['n_ants'],
            'cand_list_size': kwargs['cand_list_size'],
            'decay': kwargs['decay'],
            'p_best': kwargs.get('p_best', 0.05),
            'device': kwargs['device'],
            'enable_torch_sync': kwargs['enable_torch_sync'],
        }
        # Add alpha, beta if available
        if hasattr(args, 'alpha'):
            mmas_kwargs['alpha'] = args.alpha
        if hasattr(args, 'beta'):
            mmas_kwargs['beta'] = args.beta
        # CVRP-specific
        if problem == 'cvrp':
            mmas_kwargs['demand'] = kwargs['demand']
            mmas_kwargs['capacity'] = kwargs['capacity']
        kwargs = mmas_kwargs
    
    
    aco = aco_class(**kwargs)
    
    # Seeding C++ backend
    # We generate a unique seed for this instance from the global numpy state
    # This ensures determinism if global seed is set, but uniqueness across instances
    instance_seed = np.random.randint(0, 2**63 - 1)
    if hasattr(aco, 'seed_rng'):
        aco.seed_rng(instance_seed)

    if hasattr(aco, 'reset_timings'): aco.reset_timings()

    best_seen = float("inf")
    avg_last = None
    t_neural_total = 0.0
    priors, pher_before = [], []
    metrics_log = {k: [] for k in ["cost", "l2", "kl", "turnover", "flip", "corr", "ov", "row_match", "survival"]}
    metrics_log["snapshots"] = []

    collect_iter_stats = bool(getattr(args, "iter_log", False) or getattr(args, "iter_print", False))
    iter_stats = [] if collect_iter_stats else None
    
    history = []
    t_start_total_infer = time.time()

    
    # Setup tqdm bar if iter_print is requested
    pbar = None
    if getattr(args, "iter_print", False):
         from tqdm import tqdm
         total_iters = args.H * args.mini_H
         pbar = tqdm(total=total_iters, desc="Inference", leave=False)
    
    with torch.no_grad():
        for t in range(args.H):
            do_metrics = collect_metrics and (metrics_every_step or t == args.H - 1)
            
            prior_mat = None
            if do_metrics:
                pher_before.append(aco.pheromone_sparse.detach().cpu().clone())

            if model is not None and not use_heuristic_only:
                # If inject_step is set, only use model if t >= inject_step
                use_model = True
                if inject_step is not None and t < inject_step:
                    use_model = False
                
                if use_model:
                    if problem == 'tsp':
                        pyg_data = build_fn(aco, norm_coords, args.device, dynamic=dynamic)
                    else:
                        pyg_data = build_fn(aco, norm_coords, demand, args.device, dynamic=dynamic)
                    
                    t_neural_start = time.time()
                    heu_vec = model(pyg_data).view(-1)
                    t_neural_total += time.time() - t_neural_start
                    
                    prior_mat = heu_vec.view(aco.n, aco.k)
                    
                    if do_metrics:
                        priors.append(prior_mat.detach().cpu().clone())

            for mini_t in range(args.mini_H):
                # Annealing
                current_prior = prior_mat
                if not args.no_anneal and prior_mat is not None:
                     if args.mini_H > 1:
                        ratio = mini_t / (args.mini_H - 1)
                        factor = args.gamma * (1.0 - ratio) + args.min_gamma * ratio
                     else:
                        factor = args.gamma
                     current_prior = prior_mat * factor

                # Sample
                return_decoded = getattr(args, 'verify', False) and (problem == 'cvrp')
                
                prior_arg = current_prior.cpu().numpy() if (current_prior is not None and torch.is_tensor(current_prior)) else current_prior

                if problem == 'tsp':
                    costs_t, flats, _, _, traces, _, _, _, survival = aco.sample(require_prob=do_metrics, prior=prior_arg, parallel_traced=True)
                else:
                    costs_t, routes, decoded, _, traces, _, _, _, survival = aco.sample(require_prob=do_metrics, prior=prior_arg, return_decoded=return_decoded, parallel_traced=True)
                    flats = routes

                if do_metrics:
                    metrics_log["survival"].append(survival.mean().item())

                if return_decoded and problem == 'cvrp':
                     best_idx_t = int(costs_t.argmin().item())
                     try:
                         rt = decoded[best_idx_t] if decoded is not None else flats[best_idx_t]
                         verify_solution_cvrp(coords, demand, capacity, float(costs_t[best_idx_t]), rt)
                     except ValueError as e:
                         print(f"Verification failed: {e}")
                         sys.exit(1)

                avg_last = float(costs_t.mean().item())
                best_idx = int(costs_t.argmin().item())
                best_cost = float(costs_t[best_idx].item())
                best_seen = min(best_seen, best_cost)

                if collect_iter_stats:
                    iter_idx = t * int(args.mini_H) + int(mini_t)
                    iter_stats.append({
                        "iter": int(iter_idx),
                        "t": int(t),
                        "mini_t": int(mini_t),
                        "mean": float(avg_last),
                        "best": float(best_seen),
                    })
                
                if problem == 'tsp':
                    aco._update_pheromone_from_flat(flats[best_idx], best_cost)
                else:
                    aco.update_pheromone(flats[best_idx], best_cost)

                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix({
                        "best": f"{best_seen:.4f}", 
                        "mean": f"{avg_last:.4f}"
                    })
                    pbar.set_postfix({
                        "best": f"{best_seen:.4f}", 
                        "mean": f"{avg_last:.4f}"
                    })
                    pbar.update(1)

            # Record history at step t (1-based index is t+1)
            # t is 0-based index of H loop
            # Added t_neural_total to history
            history.append((t + 1, time.time() - t_start_total_infer, best_seen, t_neural_total))

            if do_metrics:
                metrics_log["cost"].append(best_seen)
                is_prior_avail = (len(priors) > 0)
                
                if is_prior_avail and len(priors) > 1:
                     P_prev, P_cur = priors[-2], priors[-1]
                     metrics_log["l2"].append(rel_l2_drift(P_prev, P_cur))
                     metrics_log["kl"].append(mean_row_kl(P_prev, P_cur))
                     metrics_log["turnover"].append(top_turnover(P_prev, P_cur))
                     metrics_log["flip"].append(top1_flip_rate(P_prev, P_cur))
                else:
                     for k in ["l2", "kl", "turnover", "flip"]: metrics_log[k].append(0.0)

                if is_prior_avail:
                    tau = pher_before[-1] # Match last captured
                    pr = priors[-1]
                    metrics_log["corr"].append(safe_corr(tau, pr))
                    metrics_log["ov"].append(top_overlap_frac(tau, pr))
                    metrics_log["row_match"].append(row_top1_match_rate(tau, pr))
                else:
                    for k in ["corr", "ov", "row_match"]: metrics_log[k].append(0.0)
            
            # Capture snapshots at H/2
            if collect_metrics and t == (args.H // 2):
                 # Pheromone
                 pher = aco.pheromone_sparse.detach().cpu()
                 
                 # Neural Prior (Model Output)
                 neural_prior = None
                 if 'prior_mat' in locals() and prior_mat is not None:
                      neural_prior = prior_mat.detach().cpu()

                 metrics_log["snapshots"].append({
                     "t": t,
                     "pheromone": pher,
                     "neural_prior": neural_prior
                 })

    if pbar is not None:
        pbar.close()

    timings = {}
    if hasattr(aco, 'get_timings') and args.timed:
        t = aco.get_timings()
        timings = {k: v/1000.0 for k, v in t.items()} # ms to s
    
    if args.timed:
        timings["time_neural"] = t_neural_total
    
    extra = {}
    if collect_metrics:
        extra["metrics"] = metrics_log
    if collect_iter_stats:
        extra["iter_stats"] = iter_stats
    
    extra["history"] = history
    return avg_last, best_seen, timings, extra
