"""
MFACO TSP Solver - Unified interface with C++ backend and Python fallback.

This module provides  MFACO_TSP class that uses the fast C++ backend

Usage:
    from faco import MFACO_TSP, MFACOTrace
    
    # Create solver (automatically uses C++ if available)
    faco = MFACO_TSP(coords, n_ants=20, ...)
    
    # Create from snapshot for training
    faco = MFACO_TSP.from_snapshot(snapshot, config)
    
    # Sample solutions
    costs, flats, touched, logps, traces = faco.sample(require_prob=True)
    
    # Update pheromone
    faco._update_pheromone_from_flat(best_flat, best_cost)
"""

from __future__ import annotations
import warnings
from typing import Optional, Tuple, List, Any
import numpy as np
import torch

# Try to import C++ backend
try:
    from faco_tsp import MFACO_TSP as MFACO_TSP_CPP, MFACOTrace as MFACOTrace_CPP
except ImportError:
    raise ImportError(
        "C++ backend 'faco_tsp' not found. Please build the C++ extension "
        "by following the instructions in the repository."
    )


def set_faco_cpp_threads(n_threads: int) -> None:
    """Set OpenMP thread count for the C++ backend (process-global).

    If you built `faco_cpp` with OpenMP, this controls how many threads the
    *parallel* sampling path uses. The traced path (`require_prob=True`) is
    intentionally single-threaded for determinism.
    """
    faco_tsp.set_num_threads(int(n_threads))


class MFACOTrace:
    """
    Unified trace class that wraps either C++ or Python trace.
    
    For Python compatibility, provides the same interface as faco_tsp.MFACOTrace.
    """
    
    def __init__(self, trace):
        """
        Args:
            trace: Either MFACOTrace_CPP batch trace or MFACOTrace_PY
        """
        self._trace = trace
        self._is_cpp = HAS_CPP_BACKEND and isinstance(trace, MFACOTrace_CPP) if HAS_CPP_BACKEND else False
    
    @property
    def start_node(self) -> int:
        if self._is_cpp:
            # For CPP batch trace, this should be accessed per-ant
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.start_node
    
    @property
    def curr_nodes(self) -> List[int]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.curr_nodes
    
    @property
    def chosen_nodes(self) -> List[int]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.chosen_nodes
    
    @property
    def is_stochastic(self) -> List[bool]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.is_stochastic
    
    @property
    def used_uniform_fallback(self) -> List[bool]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.used_uniform_fallback

    @property
    def is_new_edge(self) -> List[bool]:
        if self._is_cpp:
            raise AttributeError("Use get_trace(ant_idx) for per-ant access from batch trace")
        return self._trace.is_new_edge



class MFACO_TSP:
    """
    Unified MFACO TSP solver with C++ backend and Python fallback.
    
    Provides the same API as faco_tsp.MFACO_TSP but uses the fast C++
    implementation when available.
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
        gbest_as_source_prob: float = 1.0,
        use_local_search: bool = True,
        extend_ls: bool = False,
        smooth_mmas: bool = False,
        enable_torch_sync: bool = True,
        device: str = "cuda",
        use_cpp: bool = True,
        disable_heuristic: bool = False,
        normalized_heuristic: bool = False,
    ):
        """
        Initialize MFACO_TSP solver.
        
        Args:
            coords: (n, 2) node coordinates
            n_ants: number of ants
            cand_list_size: candidate list size (k)
            backup_list_size: backup list size
            min_new_edges: minimum new edges before copying from source
            decay: pheromone decay rate (rho)
            alpha: pheromone exponent
            p_best: probability best for tau limits
            gbest_as_source_prob: probability to use global best as source
            use_local_search: whether to apply 2-opt local search
            extend_ls: if True, extend the local search checklist with endpoints of improving moves
            enable_torch_sync: if False, skip torch pheromone sync (faster baseline)
            device: torch device (only used by Python backend)
            use_cpp: if False, force Python backend even if C++ available
        """
        self.device = device
        self.disable_heuristic = bool(disable_heuristic)
        self.extend_ls = bool(extend_ls)
        self.smooth_mmas = bool(smooth_mmas)
        self.normalized_heuristic = bool(normalized_heuristic)

        # Some callers (e.g., notebooks) may pass a PyG `Data` object.
        # In that case we interpret `coords` as `data.x`.
        if not isinstance(coords, torch.Tensor) and hasattr(coords, "x"):
            coords = coords.x
        
        # Normalize coords and compute explicit Euclidean distances (no rounding)
        # for features and for the Python fallback backend.
        if isinstance(coords, torch.Tensor):
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            coords_t = coords.to(device=device, dtype=torch.float32)
        else:
            coords_np = np.asarray(coords, dtype=np.float32)
            coords_t = torch.from_numpy(coords_np).to(device=device, dtype=torch.float32)

        if coords_np.ndim != 2 or coords_np.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords_np.shape}")

        # Full distance matrix on CPU (float32) for fast numpy indexing in training.
        diff = coords_np[:, None, :] - coords_np[None, :, :]
        dist_np = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32), dtype=np.float32)

        self._coords_np = coords_np
        self._dist_np = dist_np

        self._cpp = MFACO_TSP_CPP(
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
        )
        
        if self.normalized_heuristic and not self.disable_heuristic:
            # Normalize heuristic to sum to 1 per row (or similar)
            # Access underlying numpy array
            h = np.asarray(self._cpp.heuristic_sparse_np) # view
            # Standard normalization: h / sum(h)
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            # Assuming h is view, updating in-place
            np.copyto(h, h_norm)

        self._py = None
        
        # For compatibility, keep torch tensors for replay_logp (use private names)
        self._coords = coords_t
        self._distances = torch.from_numpy(dist_np).to(device=device, dtype=torch.float32)
        self._pheromone_sparse = torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(device)
        self._h_sparse_torch = torch.from_numpy(self._cpp.heuristic_sparse_np.copy()).to(device)
        self._nn_torch = torch.from_numpy(self._cpp.nn_list.copy()).to(device).long()
        self._enable_torch_sync = enable_torch_sync

    def _set_source_position(self, node: int, position: int) -> None:
        """Set a single source position. Used during snapshot restore."""
        self._cpp.source_positions[node] = position
    
    # ========================================================================
    # Properties (delegated to backend)
    # ========================================================================
    
    @property
    def n(self) -> int:
        return self._cpp.n 
    
    @property
    def n_ants(self) -> int:
        return self._cpp.n_ants
    
    @property
    def k(self) -> int:
        return self._cpp.k
    
    @property
    def bl(self) -> int:
        return self._cpp.bl
    
    @property
    def min_new_edges(self) -> int:
        return self._cpp.min_new_edges
    
    @property
    def rho(self) -> float:
        return self._cpp.rho
    
    @property
    def alpha(self) -> float:
        return self._cpp.alpha
    
    @property
    def p_best(self) -> float:
        return self._cpp.p_best
    
    @property
    def use_local_search(self) -> bool:
        return self._cpp.use_local_search
    
    @property
    def source_cost(self) -> float:
        return self._cpp.source_cost
    
    @property
    def best_cost(self) -> float:
        return self._cpp.best_cost
    
    @property
    def tau_min(self) -> float:
        return self._cpp.tau_min
    
    @property
    def tau_max(self) -> float:
        return self._cpp.tau_max
    
    @property
    def source_route(self) -> np.ndarray:
        return np.asarray(self._cpp.source_route)
    @property
    def best_route(self) -> np.ndarray:
        return np.asarray(self._cpp.best_route)
    
    @property
    def pheromone_sparse_np(self) -> np.ndarray:
        return np.asarray(self._cpp.pheromone_sparse_np)
    @pheromone_sparse_np.setter
    def pheromone_sparse_np(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32)
        if self._use_cpp:
            # C++ binding exposes `set_pheromone(pheromone)`.
            self._cpp.set_pheromone(arr)
            # Keep torch cache consistent (used by prob_sparse_torch/replay).
            if hasattr(self, "_pheromone_sparse") and isinstance(self._pheromone_sparse, torch.Tensor):
                self._pheromone_sparse.copy_(torch.from_numpy(self._cpp.pheromone_sparse_np.copy()).to(self.device))
        else:
            self._py.pheromone_sparse_np = arr
    
    @property
    def heuristic_sparse_np(self) -> np.ndarray:
        return np.asarray(self._cpp.heuristic_sparse_np)
    
    @property
    def nn_list(self) -> np.ndarray:
        return np.asarray(self._cpp.nn_list) 
    
    @property
    def backup_list(self) -> np.ndarray:
        return np.asarray(self._cpp.backup_list) 
    
    @property
    def nn_pos(self) -> np.ndarray:
        return np.asarray(self._cpp.nn_pos) 
    
    @property
    def source_positions(self) -> np.ndarray:
        return np.asarray(self._cpp.source_positions)
    
    
    @property 
    def h_sparse_torch(self) -> torch.Tensor:
        """Return (n, k) heuristic sparse tensor."""
        return self._h_sparse_torch
    
    @property
    def nn_torch(self) -> torch.Tensor:
        """Return (n, k) nearest neighbor indices as torch tensor."""
        return self._nn_torch
    
    @property
    def pheromone_sparse(self) -> torch.Tensor:
        """Return (n, k) pheromone sparse tensor."""
        return self._pheromone_sparse
    
    @property
    def enable_torch_sync(self) -> bool:
        """Return whether torch sync is enabled."""
        return self._enable_torch_sync
    
    @enable_torch_sync.setter
    def enable_torch_sync(self, value: bool) -> None:
        """Set whether torch sync is enabled."""
        self._enable_torch_sync = value
    
    # ========================================================================
    def sample(
        self,
        invtemp: float = 1.0,
        require_prob: bool = False,
        prior: np.ndarray = None,
        parallel_traced: bool = True,
    ):
        """
        Sample solutions from all ants.
        
        Returns:
            costs: (n_ants,) array of costs
            flats: list of (n+1,) arrays, each with last == first
            touched_list: list of touched node arrays
            logps_nondiff: None (placeholder)
            traces: list of MFACOTrace or MFACOTrace batch if require_prob else None
        """

        # pybind expects a CPU numpy array (or None). Accept torch tensors
        # (including CUDA) and convert safely.
        if isinstance(prior, torch.Tensor):
            prior = (
                prior.detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
        elif prior is not None:
            prior = np.asarray(prior, dtype=np.float32)

        # The C++ backend requires prior to be exactly (n, k).
        # For convenience (especially in notebooks), also accept:
        # - (n, k, 1) -> squeeze
        # - (n*k,) or (n*k, 1) -> reshape row-major to (n, k)
        if prior is not None:
            arr = prior
            n = int(self.n)
            k = int(self.k)

            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]

            # Accept dense (n,n) prior matrices (e.g. from Net.reshape)
            # and project them onto the candidate list layout (n,k).
            if arr.ndim == 2 and arr.shape == (n, n):
                nn = np.asarray(self.nn_list, dtype=np.int64)
                u = np.arange(n, dtype=np.int64)[:, None]
                arr = arr[u, nn]

            if arr.ndim == 2 and arr.shape == (n * k, 1):
                arr = arr.reshape(n, k)
            elif arr.ndim == 1 and arr.size == n * k:
                arr = arr.reshape(n, k)

            # Final validation + helpful error
            if not (arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] == k):
                raise ValueError(
                    f"prior must be shape (n, k)=({n}, {k}) for C++ backend; got {arr.shape}"
                )
            prior = arr

        costs, flats, touched, logps, traces_raw, costs_raw, flats_raw = self._cpp.sample(
            invtemp, require_prob, prior, parallel_traced
        )

        # Keep the C++ batch trace object (much faster than converting to Python lists)
        traces = traces_raw
        
        # Sync pheromone to torch if needed
        if require_prob and self.enable_torch_sync:
            self.sync_pheromone_to_torch()
        
        return costs, flats, touched, logps, traces, costs_raw, flats_raw
    
    def _update_pheromone_from_flat(self, best_flat: np.ndarray, best_cost: float) -> None:
        """
        Update pheromone: evaporate + deposit on best route.
        """
        self._cpp._update_pheromone_from_flat(
            best_flat.astype(np.int32), 
            float(best_cost)
        )
        if self.enable_torch_sync:
            self.sync_pheromone_to_torch()
    
    def prob_sparse_torch(
        self,
        invtemp: float = 1.0,
        prior: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Returns unnormalized weights for candidate edges: shape (n, k).
        Matches C++ compute_probmat logic (logit space + normalization).
        """
        EPS = 1e-10
        
        # 1. Pheromone logit
        tau = self._pheromone_sparse.clamp_min(EPS)
        logit = self.alpha * torch.log(tau)
        
        # 2. Heuristic logit
        if not self.disable_heuristic:
            # Matches C++: beta * log(eta) with beta=1
            h = self._h_sparse_torch.clamp_min(EPS)
            if invtemp != 1.0:
                 logit = logit + float(invtemp) * torch.log(h)
            else:
                 logit = logit + torch.log(h)
        
        # 3. Prior logit
        if prior is not None:
             # C++: z = clamp(prior, -10, 10), then normalize
            #  z = prior.clamp(-10.0, 10.0)
            #  mean = z.mean(dim=1, keepdim=True)
            #  var = z.var(dim=1, unbiased=False, keepdim=True)
            #  std = torch.sqrt(var + 1e-6)
            #  z_norm = (z - mean) / std
             
             # Matches C++: gamma * z_norm with gamma=1
             logit = logit + prior
             
        w = torch.exp(logit)
        return w

    def sync_pheromone_to_torch(self) -> None:
        """Sync numpy pheromone to torch tensor."""
        phe_np = np.asarray(self._cpp.pheromone_sparse_np)
        phe_cpu = torch.as_tensor(phe_np)
        self.pheromone_sparse.copy_(phe_cpu.to(self.device))

