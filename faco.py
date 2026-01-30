"""
MFACO Solver - Unified interface for TSP and CVRP with C++ backend.

Usage:
    from faco import MFACO_TSP, MFACO_CVRP
    
    # TSP
    solver = MFACO_TSP(coords, n_ants=20, ...)
    costs, flats, ... = solver.sample()
    
    # CVRP
    solver = MFACO_CVRP(coords, demand, capacity, n_ants=20, ...)
    costs, perms, ... = solver.sample()
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
from typing import Optional, List, Tuple, Any

# Ensure src is in path to find C++ extension if not installed globally
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    import faco_opt
except ImportError:
    # Try importing from current directory if compiled in-place at root
    try:
        import faco_opt
    except ImportError:
        # Check if it is in src but import failed
        raise ImportError(
            "C++ backend 'faco_opt' not found. Please build the C++ extension in src/."
        )

def set_faco_cpp_threads(n_threads: int) -> None:
    """Set OpenMP thread count for the C++ backend."""
    faco_opt.set_num_threads(int(n_threads))

def _as_numpy_f32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x)

def _as_numpy_i32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.int32)
    return np.ascontiguousarray(x)

class MFACO_TSP:
    """
    Unified MFACO TSP solver wrapping C++ backend.
    """
    
    def __init__(
        self,
        coords: torch.Tensor,
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 32,
        min_new_edges: int = 8,
        decay: float = 0.1,
        alpha: float = 1.0,
        p_best: float = 0.05,
        use_local_search: bool = True,
        extend_ls: bool = False,
        smooth_mmas: bool = False,
        enable_torch_sync: bool = True,
        device: str = "cuda",
        disable_heuristic: bool = False,
        normalized_heuristic: bool = False,
        fixed_steps: int = 0,
        nls: bool = False,
        T_nls: int = 10,
        **kwargs
    ):
        self.device = device
        self.disable_heuristic = bool(disable_heuristic)
        self.extend_ls = bool(extend_ls)
        self.smooth_mmas = bool(smooth_mmas)
        self.normalized_heuristic = bool(normalized_heuristic)
        self.fixed_steps = int(fixed_steps)

        # Handle PyG Data object
        if not isinstance(coords, torch.Tensor) and hasattr(coords, "x"):
            coords = coords.x
        
        coords_np = _as_numpy_f32(coords)
        if coords_np.ndim != 2 or coords_np.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords_np.shape}")

        self._coords_np = coords_np

        self._cpp = faco_opt.MFACO_TSP(
            coords_np,
            n_ants,
            cand_list_size,
            backup_list_size,
            min_new_edges,
            decay,
            alpha,
            p_best,
            use_local_search,
            self.disable_heuristic,
            self.extend_ls,
            self.smooth_mmas,
            self.fixed_steps,
            nls,
            int(T_nls)
        )
        
        if self.normalized_heuristic and not self.disable_heuristic:
            # Normalize heuristic
            h = np.asarray(self._cpp.heuristic_sparse_np) 
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            np.copyto(h, h_norm)

        # Torch buffers
        self._pheromone_sparse = torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(device)
        self._h_sparse_torch = torch.from_numpy(self._cpp.heuristic_sparse_np.copy()).to(device)
        self._nn_torch = torch.from_numpy(self._cpp.nn_list.copy()).to(device).long()
        self._enable_torch_sync = enable_torch_sync

    # Delegate properties
    @property
    def n(self) -> int: return self._cpp.n
    @property
    def n_ants(self) -> int: return self._cpp.n_ants
    @property
    def k(self) -> int: return self._cpp.k
    @property
    def bl(self) -> int: return self._cpp.bl
    @property
    def source_cost(self) -> float: return self._cpp.source_cost
    @property
    def best_cost(self) -> float: return self._cpp.best_cost
    @property
    def tau_min(self) -> float: return self._cpp.tau_min
    @property
    def tau_max(self) -> float: return self._cpp.tau_max
    
    @property
    def source_route(self) -> np.ndarray: return np.asarray(self._cpp.source_route)
    @property
    def best_route(self) -> np.ndarray: return np.asarray(self._cpp.best_route)
    
    @property
    def pheromone_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.pheromone_sparse_np)
    @pheromone_sparse_np.setter
    def pheromone_sparse_np(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        self._cpp.set_pheromone(arr)
        if self.enable_torch_sync:
            self._pheromone_sparse.copy_(torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(self.device))
    
    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)
    @property
    def backup_list(self) -> np.ndarray: return np.asarray(self._cpp.backup_list)
    # @property
    # def nn_pos(self) -> np.ndarray: return np.asarray(self._cpp.nn_pos)
    @property
    def source_positions(self) -> np.ndarray: return np.asarray(self._cpp.source_positions)

    @property 
    def h_sparse_torch(self) -> torch.Tensor: return self._h_sparse_torch
    @property
    def nn_torch(self) -> torch.Tensor: return self._nn_torch
    @property
    def pheromone_sparse(self) -> torch.Tensor: return self._pheromone_sparse

    def seed_rng(self, seed: int) -> None:
        self._cpp.seed_rng(seed)

    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = True,
    ):
        """
        Sample from C++ backend.
        """
        # Prepare prior
        if isinstance(prior, torch.Tensor):
            prior = prior.detach().to("cpu", dtype=torch.float32).numpy()
        elif prior is not None:
            prior = np.asarray(prior, dtype=np.float32)

        # Handle shapes for convenience
        if prior is not None:
            arr = prior
            n, k = self.n, self.k
            if arr.ndim == 3 and arr.shape[2] == 1: arr = arr[:, :, 0]
            if arr.ndim == 2 and arr.shape == (n, n):
                nn = np.asarray(self.nn_list, dtype=np.int64)
                u = np.arange(n, dtype=np.int64)[:, None]
                arr = arr[u, nn]
            if arr.ndim == 2 and arr.shape == (n*k, 1): arr = arr.reshape(n, k)
            elif arr.ndim == 1 and arr.size == n*k: arr = arr.reshape(n, k)
            prior = arr

        costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival = self._cpp.sample(
            invtemp, require_prob, prior, parallel_traced
        )

        if require_prob and self._enable_torch_sync:
            self.sync_pheromone_to_torch()
        
        # Convert survival to torch tensor if needed, or keep as numpy/array
        if isinstance(survival, np.ndarray):
            survival = torch.from_numpy(survival).to(self.device)

        return costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival

    def _update_pheromone_from_flat(self, best_flat: np.ndarray, best_cost: float) -> None:
        self._cpp._update_pheromone_from_flat(best_flat.astype(np.int32), float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()

    def sync_pheromone_to_torch(self) -> None:
        phe_np = np.asarray(self._cpp.pheromone_sparse_np)
        self._pheromone_sparse.copy_(torch.from_numpy(phe_np).to(self.device))
        
    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()

    def prob_sparse_torch(self, invtemp: float = 1.0, prior: torch.Tensor = None) -> torch.Tensor:
        EPS = 1e-10
        tau = self._pheromone_sparse.clamp_min(EPS)
        logit = self._cpp.alpha * torch.log(tau)
        
        if not self.disable_heuristic:
            h = self._h_sparse_torch.clamp_min(EPS)
            if invtemp != 1.0:
                 logit = logit + float(invtemp) * torch.log(h)
            else:
                 logit = logit + torch.log(h)
        
        if prior is not None:
             logit = logit + prior
             
        return torch.exp(logit)


class MFACO_CVRP:
    """
    Unified MFACO CVRP solver wrapping C++ backend.
    """
    
    def __init__(
        self,
        coords,          
        demand,          
        capacity: float,
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 32,
        min_new_edges: int = 8,
        decay: float = 0.9,
        alpha: float = 1.0,
        p_best: float = 0.05,
        use_local_search: bool = True,
        disable_heuristic: bool = False,
        extend_ls: bool = False,
        smooth_mmas: bool = False,
        device: str = "cpu",
        enable_torch_sync: bool = True,
        normalized_heuristic: bool = False,
        fixed_steps: int = 0,
        nls: bool = False,
        T_nls: int = 10,
        **kwargs
    ):
        coords_np = _as_numpy_f32(coords)
        demand_np = _as_numpy_f32(demand)
        if demand_np.ndim != 1:
            raise ValueError("demand must be 1D")
        demand_np[0] = 0.0
        
        self.solver = faco_opt.MFACO_CVRP(
            coords_np,
            demand_np,
            float(capacity),
            int(n_ants),
            int(cand_list_size),
            int(backup_list_size),
            int(min_new_edges),
            float(decay),
            float(alpha),
            float(p_best),
            bool(use_local_search),
            bool(disable_heuristic),
            bool(extend_ls),
            bool(smooth_mmas),
            int(fixed_steps),
            bool(nls),
            int(T_nls),
        )
        self.device = device
        self.enable_torch_sync = enable_torch_sync
        self.alpha = alpha
        
        if normalized_heuristic and not disable_heuristic:
            h = np.asarray(self.solver.heuristic_sparse_np)
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            np.copyto(h, h_norm)
        
        self.disable_heuristic = disable_heuristic
        self._pheromone_sparse = torch.from_numpy(self.solver.pheromone_sparse_np.copy()).to(device)
        self._heuristic_sparse = torch.from_numpy(self.solver.heuristic_sparse_np.copy()).to(device)

    # Properties
    @property
    def n(self): return self.solver.n
    @property
    def m(self): return self.solver.m
    @property
    def k(self): return self.solver.k
    @property
    def n_ants(self): return self.solver.n_ants

    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self.solver.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self.solver.nn_list)
    @property
    def backup_list(self) -> np.ndarray: return np.asarray(self.solver.backup_list)
    # @property
    # def nn_pos(self) -> np.ndarray: return np.asarray(self.solver.nn_pos)
    
    @property
    def pheromone_sparse(self) -> torch.Tensor:
        return self._pheromone_sparse

    def seed_rng(self, seed: int):
        self.solver.seed_rng(int(seed))

    def sample(self, require_prob: bool = False, prior=None, parallel_traced: bool = False, return_decoded: bool = False):
        if prior is not None:
            prior = _as_numpy_f32(prior)
        costs, perms, decoded, logps, traces, costs_raw, perms_raw, new_edges_count, survival = self.solver.sample(
            require_prob, prior, parallel_traced, return_decoded
        )

        if isinstance(survival, np.ndarray):
             survival = torch.from_numpy(survival).to(self.device)

        return costs, perms, decoded, logps, traces, costs_raw, perms_raw, new_edges_count, survival

    def update_pheromone(self, best_perm, best_cost: float):
        p = _as_numpy_i32(best_perm)
        self.solver.update_pheromone_from_perm(p, float(best_cost))
        if self.enable_torch_sync:
            self.sync_pheromone_to_torch()

    def reset_timings(self):
        self.solver.reset_timings()

    def get_timings(self):
        return self.solver.get_timings()

    def sync_pheromone_to_torch(self):
        phe_np = np.asarray(self.solver.pheromone_sparse_np)
        phe_cpu = torch.from_numpy(phe_np)
        self._pheromone_sparse.copy_(phe_cpu.to(self.device))

    def prob_sparse_torch(self, prior: torch.Tensor = None, invtemp: float = 1.0) -> torch.Tensor:
        EPS = 1e-10
        tau = self._pheromone_sparse.clamp_min(EPS)
        logit = self.alpha * torch.log(tau)
        
        if not self.disable_heuristic:
            h = self._heuristic_sparse.clamp_min(EPS)
            if invtemp != 1.0:
                 logit = logit + float(invtemp) * torch.log(h)
            else:
                 logit = logit + torch.log(h)
        if prior is not None:
             logit = logit + prior
        return torch.exp(logit)

    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()
    
    @property
    def nn_torch(self) -> torch.Tensor:
        if not hasattr(self, '_nn_torch'):
             self._nn_torch = torch.from_numpy(self.solver.nn_list.copy()).to(self.device).long()
        return self._nn_torch
