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

    def update_pheromone(self, best_flat, best_cost: float) -> None:
        """Update pheromone with best solution (unified API)."""
        self._cpp._update_pheromone_from_flat(np.asarray(best_flat, dtype=np.int32), float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()

    def _update_pheromone_from_flat(self, best_flat: np.ndarray, best_cost: float) -> None:
        """Alias for update_pheromone (backward compatibility)."""
        self.update_pheromone(best_flat, best_cost)

    def sync_pheromone_to_torch(self) -> None:
        phe_np = np.asarray(self._cpp.pheromone_sparse_np)
        self._pheromone_sparse.copy_(torch.from_numpy(phe_np).to(self.device))
    
    @property
    def enable_torch_sync(self) -> bool:
        return self._enable_torch_sync
    
    def reset_timings(self) -> None:
        """Reset timing counters (unified API with MFACO_CVRP)."""
        if hasattr(self._cpp, 'reset_timings'):
            self._cpp.reset_timings()
    
    def get_timings(self) -> dict:
        """Get timing counters (unified API with MFACO_CVRP)."""
        if hasattr(self._cpp, 'get_timings'):
            return self._cpp.get_timings()
        return {}
        
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


class ACO_TSP:
    """
    Standard MMAS ACO TSP solver.
    """
    def __init__(
        self,
        coords: torch.Tensor,
        n_ants: int,
        cand_list_size: int = 32,
        decay: float = 0.9,
        alpha: float = 1.0, 
        beta: float = 1.0, # Heuristic weight
        p_best: float = 0.05,
        min_max: bool = True,
        device: str = "cuda",
        enable_torch_sync: bool = True,
        **kwargs
    ):
        self.device = device
        
        # Handle PyG Data object
        if not isinstance(coords, torch.Tensor) and hasattr(coords, "x"):
            coords = coords.x
        
        coords_np = _as_numpy_f32(coords)
        if coords_np.ndim != 2 or coords_np.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords_np.shape}")

        self._cpp = faco_opt.ACO_TSP(
            coords_np,
            int(n_ants),
            int(cand_list_size),
            float(decay),
            float(alpha),
            float(beta),
            float(p_best),
            bool(min_max)
        )
        
        # Torch buffers
        self._pheromone_sparse = torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(device)
        self._h_sparse_torch = torch.from_numpy(self._cpp.heuristic_sparse_np.copy()).to(device)
        self._nn_torch = torch.from_numpy(self._cpp.nn_list.copy()).to(device).long()
        self._enable_torch_sync = enable_torch_sync
        
        self.disable_heuristic = False # For compatibility

    # Delegate properties
    @property
    def n(self) -> int: return self._cpp.n
    @property
    def n_ants(self) -> int: return self._cpp.n_ants
    @property
    def k(self) -> int: return self._cpp.k
    
    @property
    def best_cost(self) -> float: return self._cpp.best_cost
    @property
    def tau_min(self) -> float: return self._cpp.tau_min
    @property
    def tau_max(self) -> float: return self._cpp.tau_max
    
    @property
    def best_route(self) -> np.ndarray: return np.asarray(self._cpp.best_route)

    @property
    def source_route(self) -> np.ndarray: return self.best_route
    
    @property
    def pheromone_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.pheromone_sparse_np)
    
    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)

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
        # Prepare prior
        if isinstance(prior, torch.Tensor):
            prior = prior.detach().to("cpu", dtype=torch.float32).numpy()
        elif prior is not None:
            prior = np.asarray(prior, dtype=np.float32)

        # Handle shapes
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

        # Returns: costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival
        # But my binding returns simpler tuple.
        # Wait, binding returns tuple of size 9 with Nones.
        # So I can just return what self._cpp.sample returns.
        
        ret = self._cpp.sample(require_prob, prior, parallel_traced)
        
        if require_prob and self._enable_torch_sync:
            self.sync_pheromone_to_torch()
            
        return ret 

    def update_pheromone(self, best_flat: np.ndarray, best_cost: float) -> None:
        self._cpp.update_pheromone(best_flat.astype(np.int32), float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()

    def _update_pheromone_from_flat(self, best_flat: np.ndarray, best_cost: float) -> None:
         self.update_pheromone(best_flat, best_cost)

    def sync_pheromone_to_torch(self) -> None:
        phe_np = np.asarray(self._cpp.pheromone_sparse_np)
        self._pheromone_sparse.copy_(torch.from_numpy(phe_np).to(self.device))
        
    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()

    def prob_sparse_torch(self, invtemp: float = 1.0, prior: torch.Tensor = None) -> torch.Tensor:
        # Reimplementation of MMAS prob calculation for PPO / Debug
        EPS = 1e-10
        tau = self._pheromone_sparse.clamp_min(EPS)
        logit = self._cpp.alpha * torch.log(tau)
        
        h = self._h_sparse_torch.clamp_min(EPS)
        # beta is self._cpp.beta
        # Wait, PyACO_TSP exposes beta? Yes I added `get_beta`.
        logit = logit + self._cpp.beta * torch.log(h)
        
        if prior is not None:
             logit = logit + prior
             
        # This returns probabilities (unnormalized across rows in sparse graph? or normalized?)
        # MMAS uses p_ij = tau^alpha * eta^beta / sum(...).
        # This returns exp(logit). Normalization happens at selection time.
        return torch.exp(logit)

    def evaluate_log_prob_torch(self, paths: torch.Tensor, prior: torch.Tensor = None, invtemp: float = 1.0) -> torch.Tensor:
        """
        Evaluate log probability of paths (B, N).
        paths: (B, N) int64 tensor of node indices.
        prior: (N, K) tensor or None.
        Returns: (B,) log probability.
        """
        B, N = paths.shape
        device = paths.device
        
        # 1. Compute logits (N, K)
        # Logits are unnormalized log-weights.
        # logit[u, j] corresponds to edge (u, nn[u,j])
        logits = torch.log(self.prob_sparse_torch(invtemp=invtemp, prior=prior) + 1e-20) 
        # prob_sparse_torch returns exp(logit), so taking log returns logit back? 
        # prob_sparse_torch computes alpha*log(tau) + beta*log(eta) + prior.
        # Yes.
        
        # 2. Iterate steps
        # Maintain visited mask (B, N)
        visited = torch.zeros((B, N), dtype=torch.bool, device=device)
        
        log_prob_sum = torch.zeros(B, device=device)
        
        # nn indices: (N, K)
        nn = self.nn_torch # (N, K)
        
        # Initial step: Start node. Prob = 1/N? 
        # Usually ACO starts random uniform. Log prob depends if we model start choice.
        # Often ignored or treated as log(1/N).
        # We start from paths[:, 0].
        # Mark visited
        visited.scatter_(1, paths[:, 0:1], True)
        
        prev = paths[:, 0]
        
        # Pre-gather nn for all rows? No, distinct per row.
        # We process steps.
        
        range_b = torch.arange(B, device=device)
        
        for t in range(N - 1):
            curr = prev # (B,)
            next_node = paths[:, t+1] # (B,)
            
            # Identify which neighbor index 'j' corresponds to 'next_node'
            # We can lookup in nn[curr].
            # efficient way? 
            # nn is (N, K). nn[curr] is (B, K).
            # next_node is (B,).
            # match = (nn[curr] == next_node.unsqueeze(1)) -> (B, K) bool
            # idx = match.nonzero(something).
            # If not found (fallback case), probability is complicated (greedy fallback).
            # Assuming found in NN list.
            
            nn_curr = nn[curr] # (B, K)
            logits_curr = logits[curr] # (B, K)
            
            # Mask visited
            # neighbors = nn_curr. Flatten?
            # is_visited = visited.gather(1, nn_curr) # (B, K)
            # mask invalid neighbors
            is_visited = visited.gather(1, nn_curr)
            
            # Mask logits
            logits_valid = logits_curr.clone()
            logits_valid[is_visited] = float('-inf')
            
            # LogSumExp
            log_denom = torch.logsumexp(logits_valid, dim=1) # (B,)
            
            # Numerator: logit of the chosen 'next_node'
            # We need index j.
            # match mask
            is_next = (nn_curr == next_node.unsqueeze(1)) # (B, K)
            # If next_node not in nn_curr, is_next is all False.
            # In that case, we assume greedy fallback or zero prob? 
            # If MMAS C++ did greedy fallback, we should assign prob=1.0 (log=0)? 
            # Or -inf?
            # If we are training, we hope paths are within NN.
            # If fallback happened, strict log prob is -inf under "NN-only" policy.
            # But we can approximate.
            
            # To extract value efficiently:
            # sum(logits_valid * is_next) ? No, logits can be negative.
            # logits_valid[is_next] ? is_next might be sparse.
            # If multiple next (imposible), pick one.
            
            # Use max(logits_curr masked by is_next)? 
            # If is_next all false, we have a problem.
            
            # Assume unique match
            # We replace -inf with something safe for 'max'
            # But easier:
            
            has_match = is_next.any(dim=1)
            
            # For matched ones:
            log_numer = (logits_curr * is_next.float()).sum(dim=1) 
            # Note: if multiple matches (impossible for unique NN), sum might be wrong logit sum.
            # is_next is 0/1. sum gives logit value.
            
            step_logp = log_numer - log_denom
            
            # If not matched (fallback), assign logp = 0 (deterministic fallback assumption)
            # or handle gracefully.
            step_logp = torch.where(has_match, step_logp, torch.zeros_like(step_logp))
            
            log_prob_sum += step_logp
            
            # Update visited
            visited.scatter_(1, next_node.unsqueeze(1), True)
            
            prev = next_node
            
        return log_prob_sum

class MFACO_CVRP:
    """
    Unified MFACO CVRP solver wrapping C++ backend.
    
    API is designed to match MFACO_TSP for consistent usage in training code.
    """
    
    def __init__(
        self,
        coords,          
        demand,          
        capacity: float,
        n_ants: int,
        cand_list_size: int = 32,
        backup_list_size: int = 64,
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
        
        # Use _cpp for consistency with MFACO_TSP
        self._cpp = faco_opt.MFACO_CVRP(
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
        self._enable_torch_sync = enable_torch_sync
        self.alpha = alpha
        self.disable_heuristic = disable_heuristic
        
        if normalized_heuristic and not disable_heuristic:
            h = np.asarray(self._cpp.heuristic_sparse_np)
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            np.copyto(h, h_norm)
        
        # Torch buffers (matching MFACO_TSP naming)
        self._pheromone_sparse = torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(device)
        self._h_sparse_torch = torch.from_numpy(self._cpp.heuristic_sparse_np.copy()).to(device)
        self._nn_torch = torch.from_numpy(self._cpp.nn_list.copy()).to(device).long()

    # Properties (matching MFACO_TSP naming)
    @property
    def n(self) -> int: return self._cpp.n
    @property
    def m(self) -> int: return self._cpp.m
    @property
    def k(self) -> int: return self._cpp.k
    @property
    def n_ants(self) -> int: return self._cpp.n_ants

    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)
    @property
    def backup_list(self) -> np.ndarray: return np.asarray(self._cpp.backup_list)
    
    @property
    def pheromone_sparse(self) -> torch.Tensor:
        return self._pheromone_sparse
    
    @property
    def h_sparse_torch(self) -> torch.Tensor:
        """Alias for heuristic tensor (matches MFACO_TSP API)."""
        return self._h_sparse_torch
    
    @property
    def nn_torch(self) -> torch.Tensor:
        return self._nn_torch

    @property
    def source_perm(self) -> np.ndarray:
        return np.asarray(self._cpp.source_route)
    
    @property
    def source_route(self) -> np.ndarray:
        return np.asarray(self._cpp.source_route)
    
    @property
    def enable_torch_sync(self) -> bool:
        return self._enable_torch_sync

    def seed_rng(self, seed: int) -> None:
        self._cpp.seed_rng(int(seed))

    def sample(
        self,
        invtemp: float = 1.0,  # Kept for API compatibility (unused for CVRP)
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = False,
        return_decoded: bool = False,
        route_mode: int = 2,
    ):
        """
        Sample from C++ backend.
        
        Returns tuple matching MFACO_TSP:
            (costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges_count, survival)
        
        Note: For CVRP, 'touched' contains decoded routes if return_decoded=True.
        """
        if prior is not None:
            prior = _as_numpy_f32(prior)
        
        costs, routes, decoded, logps, traces, costs_raw, routes_raw, new_edges_count, survival = self._cpp.sample(
            require_prob, prior, parallel_traced, return_decoded, int(route_mode)
        )

        if isinstance(survival, np.ndarray):
            survival = torch.from_numpy(survival).to(self.device)

        # Return format matches MFACO_TSP: (costs, flats, touched, logps, traces, costs_raw, flats_raw, new_edges, survival)
        return costs, routes, decoded, logps, traces, costs_raw, routes_raw, new_edges_count, survival

    def update_pheromone(self, best_route, best_cost: float) -> None:
        """Update pheromone with best route (matches MFACO_TSP naming)."""
        p = _as_numpy_i32(best_route)
        self._cpp.update_pheromone_from_route(p, float(best_cost))
        if self._enable_torch_sync:
            self.sync_pheromone_to_torch()
    
    def _update_pheromone_from_flat(self, best_flat, best_cost: float) -> None:
        """Alias for update_pheromone (matches MFACO_TSP API)."""
        self.update_pheromone(best_flat, best_cost)

    def reset_timings(self) -> None:
        self._cpp.reset_timings()

    def get_timings(self) -> dict:
        return self._cpp.get_timings()

    def sync_pheromone_to_torch(self) -> None:
        phe_np = np.asarray(self._cpp.pheromone_sparse_np)
        self._pheromone_sparse.copy_(torch.from_numpy(phe_np).to(self.device))

    def prob_sparse_torch(self, invtemp: float = 1.0, prior: torch.Tensor = None) -> torch.Tensor:
        """Compute probability tensor (argument order matches MFACO_TSP)."""
        EPS = 1e-10
        tau = self._pheromone_sparse.clamp_min(EPS)
        logit = self.alpha * torch.log(tau)
        
        if not self.disable_heuristic:
            h = self._h_sparse_torch.clamp_min(EPS)
            if invtemp != 1.0:
                logit = logit + float(invtemp) * torch.log(h)
            else:
                logit = logit + torch.log(h)
        if prior is not None:
            logit = logit + prior
        return torch.exp(logit)

    def tau_nk_torch(self) -> torch.Tensor:
        return self._pheromone_sparse.clone()
        


class ACO_CVRP:
    """
    Standard MMAS ACO CVRP solver.
    """
    def __init__(
        self,
        coords,
        demand,
        capacity: float,
        n_ants: int,
        cand_list_size: int = 0, # Default dense 
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        p_best: float = 0.05,
        min_max: bool = True,
        elitist: bool = False,
        use_local_search: bool = False,
        device: str = "cuda",
        enable_torch_sync: bool = True,
        **kwargs
    ):
        self.device = device
        
        # Handle PyG inputs
        if not isinstance(coords, torch.Tensor) and hasattr(coords, "x"):
            coords = coords.x
        
        coords_np = _as_numpy_f32(coords)
        demand_np = _as_numpy_f32(demand)
        
        self.capacity = float(capacity)
        
        self.n = coords_np.shape[0]
        # demand buffer
        self.demand_torch = torch.from_numpy(demand_np).to(device)

        self._cpp = faco_opt.ACO_CVRP(
            coords_np,
            demand_np,
            float(capacity),
            int(n_ants),
            int(cand_list_size),
            float(decay),
            float(alpha),
            float(beta),
            float(p_best),
            bool(min_max),
            bool(elitist),
            bool(use_local_search)
        )
        
        # Torch buffers
        self._pheromone_sparse = torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(device)
        self._h_sparse_torch = torch.from_numpy(self._cpp.heuristic_sparse_np.copy()).to(device)
        self._nn_torch = torch.from_numpy(self._cpp.nn_list.copy()).to(device).long()
        self._enable_torch_sync = enable_torch_sync
        
        self.disable_heuristic = False

    # Delegate properties
    @property
    def n_ants(self) -> int: return self._cpp.n_ants
    @property
    def k(self) -> int: return self._cpp.k
    
    @property
    def best_cost(self) -> float: return self._cpp.best_cost
    @property
    def tau_min(self) -> float: return self._cpp.tau_min
    @property
    def tau_max(self) -> float: return self._cpp.tau_max
    
    @property
    def source_perm(self) -> np.ndarray: return np.asarray(self._cpp.source_perm)
    
    @property
    def best_route(self) -> np.ndarray: return self._cpp.best_route
    
    @property
    def pheromone_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.pheromone_sparse_np)
    
    @property
    def heuristic_sparse_np(self) -> np.ndarray: return np.asarray(self._cpp.heuristic_sparse_np)
    @property
    def nn_list(self) -> np.ndarray: return np.asarray(self._cpp.nn_list)

    @property 
    def h_sparse_torch(self) -> torch.Tensor: return self._h_sparse_torch
    @property
    def nn_torch(self) -> torch.Tensor: return self._nn_torch
    @property
    def pheromone_sparse(self) -> torch.Tensor: return self._pheromone_sparse

    def seed_rng(self, seed: int) -> None:
        self._cpp.seed_rng(seed)

    def run(self, n_iterations: int) -> float:
        for _ in range(n_iterations):
            # Sample (costs, routes, ...)
            ret = self.sample(require_prob=False)
            costs = ret[0]
            routes = ret[1]

            # Find best in batch
            best_idx = np.argmin(costs)
            iteration_best_cost = float(costs[best_idx])
            iteration_best_route = routes[best_idx]

            # Update pheromone (Min-Max ACO updates with iteration best or global best depending on strategy, 
            # but usually we pass iteration best to update_pheromone and C++ handles logic?
            # Actually PyACO_CVRP::update_pheromone calls solver->update_pheromone. 
            # MMAS usually updates with GLOBAL best? 
            # Looking at binding.cpp: update_pheromone takes solution, cost.
            # In mfaco_train.cpp: update_pheromone(route, cost) -> updates global best if better, then deposits pheromone.
            # So passing iteration best is correct; C++ will check if it's new global best.
            self.update_pheromone(iteration_best_route, iteration_best_cost)
        
        return self.best_cost

    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        prior: Optional[Any] = None,
        parallel_traced: bool = True,
        return_decoded: bool = False,
    ):
        if isinstance(prior, torch.Tensor):
            prior = prior.detach().to("cpu", dtype=torch.float32).numpy()
        elif prior is not None:
            prior = np.asarray(prior, dtype=np.float32)

        if prior is not None:
             arr = prior
             n, k = self.n, self.k
             if arr.size == n*k: arr = arr.reshape(n, k)
             elif arr.shape == (n, n) and k < n: 
                 # This shouldn't happen for dense ACO
                 assert False, f"Prior shape {arr.shape} is not compatible with ACO_CVRP (n={n}, k={k})"
             prior = arr

        ret = self._cpp.sample(require_prob, prior, parallel_traced)
        
        if require_prob and self._enable_torch_sync:
            self.sync_pheromone_to_torch()
            
        # ret is (costs, routes, None, logps, None, None, None, None, None)
        # For compatibility, keep same format
        return ret

    def update_pheromone(self, best_flat: np.ndarray, best_cost: float) -> None:
        self._cpp.update_pheromone(best_flat.astype(np.int32), float(best_cost))
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
        
        h = self._h_sparse_torch.clamp_min(EPS)
        logit = logit + self._cpp.beta * torch.log(h)
        
        if prior is not None:
             logit = logit + prior
             
        return torch.exp(logit)

    def evaluate_log_prob_torch(self, paths: torch.Tensor, prior: torch.Tensor = None, invtemp: float = 1.0) -> torch.Tensor:
        """
        Evaluate log probability of CVRP paths (B, L) including depots.
        paths: (B, L) int64 tensor.
        """
        B, L = paths.shape
        device = paths.device
        N = self.n
        
        # 1. Compute logits (N, K) -> (B, N, K) or (N, K) shared
        # Since nn is dense, K=N.
        logits_base = torch.log(self.prob_sparse_torch(invtemp=invtemp, prior=prior) + 1e-20) # (N, N)
        
        # 2. Iterate
        visited = torch.zeros((B, N), dtype=torch.bool, device=device) # Customers visited
        visited[:, 0] = True # Depot usually 0
        
        current_capacity = torch.full((B,), self.capacity, device=device)
        
        log_prob_sum = torch.zeros(B, device=device)
        
        prev = paths[:, 0]
        
        for t in range(L - 1):
            curr = prev
            next_node = paths[:, t+1]
            
            # Mask logits[curr] (B, N)
            lg = logits_base[curr]
            
            # Mask visited
            mask_visited = visited.clone()
            mask_visited[:, 0] = False
            
            # Capacity constraint
            demand_ok = (self.demand_torch.unsqueeze(0) <= current_capacity.unsqueeze(1))
            
            cust_cand = (~mask_visited) & demand_ok
            cust_cand[:, 0] = False # Customers only
            
            has_candidates = cust_cand.any(dim=1)
            
            # Construct allowed mask
            allowed = cust_cand.clone()
            
            # Depot allowed logic: Allowed if curr != 0
            is_at_depot = (curr == 0)
            allowed[:, 0] = (~is_at_depot)
            
            # Apply mask
            lg_masked = lg.clone()
            lg_masked[~allowed] = float('-inf')
            
            log_denom = torch.logsumexp(lg_masked, dim=1)
            
            # Numerator
            log_numer = lg.gather(1, next_node.unsqueeze(1)).squeeze(1)
            
            step_logp = log_numer - log_denom
            
            # Zero out if step invalid (done/padded)
            valid_step = allowed.any(dim=1)
            step_logp = torch.where(valid_step, step_logp, torch.zeros_like(step_logp))
            
            log_prob_sum += step_logp
            
            # Update state
            is_depot_move = (next_node == 0)
            
            visited.scatter_(1, next_node.unsqueeze(1), True)
            visited[:, 0] = False
            
            dem = self.demand_torch[next_node]
            current_capacity = torch.where(is_depot_move, 
                                           torch.tensor(self.capacity, device=device),
                                           current_capacity - dem)
            
            prev = next_node
            
        return log_prob_sum
