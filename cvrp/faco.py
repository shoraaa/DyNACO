import numpy as np
import torch

try:
    import faco_cvrp as faco_cpp
except ImportError as e:
    raise ImportError(
        "C++ backend 'faco_cvrp' not found. Build/compile the extension in cvrp/src first."
    ) from e



def set_faco_cpp_threads(n_threads: int) -> None:
    """Set OpenMP thread count for the C++ backend."""
    faco_cpp.set_num_threads(int(n_threads))


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


class MFACO_CVRP:
    """
    MFACO for CVRP using:
      - giant tour (perm of customers)
      - Split DP decoding for capacity routes
      - pheromone deposit on decoded depot edges + within-route edges

    Node 0 must be depot.
    demand[0] must be 0.
    """

    def __init__(
        self,
        coords,          # (n,2)
        demand,          # (n,)
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
    ):
        coords_np = _as_numpy_f32(coords)
        demand_np = _as_numpy_f32(demand)
        if demand_np.ndim != 1:
            raise ValueError("demand must be 1D")
        demand_np[0] = 0.0

        self.solver = faco_cpp.MFACO_CVRP(
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
        )
        self.device = device
        self.enable_torch_sync = enable_torch_sync
        self.alpha = alpha
        
        if normalized_heuristic and not disable_heuristic:
            # Normalize heuristic
            h = np.asarray(self.solver.heuristic_sparse_np)
            row_sums = h.sum(axis=1, keepdims=True)
            h_norm = h / (row_sums + 1e-12)
            np.copyto(h, h_norm)
        
        # Buffers
        self.disable_heuristic = disable_heuristic
        self._pheromone_sparse = torch.from_numpy(self.solver.pheromone_sparse_np.copy()).to(device)
        self._heuristic_sparse = torch.from_numpy(self.solver.heuristic_sparse_np.copy()).to(device)

    @property
    def n(self): return self.solver.n

    @property
    def m(self): return self.solver.m

    @property
    def k(self): return self.solver.k

    @property
    def n_ants(self): return self.solver.n_ants

    @property
    def pheromone_sparse(self) -> torch.Tensor:
        """Return (n, k) pheromone sparse tensor."""
        return self._pheromone_sparse

    def seed_rng(self, seed: int):
        self.solver.seed_rng(int(seed))

    def sample(self, require_prob: bool = False, prior=None, parallel_traced: bool = False, return_decoded: bool = False):
        """
        Returns:
          costs: torch.float32 [n_ants]
          perms: list[np.ndarray] each (m+1,) customers-only cycle (last==first)
          decoded_routes: list[np.ndarray] each variable length [0,...,0,...] or None
          decoded_routes: list[np.ndarray] each variable length [0,...,0,...] or None
          traces: faco_cpp.MFACOTrace or None
          new_edges_count: (n_ants,) array of new edges created
          
        Args:
          prior: (n, k) extra weights to multiply into probability. 
                 If passed, prob = (tau^alpha * eta) * prior.
        """
        if prior is not None:
            prior = _as_numpy_f32(prior)
        costs, perms, decoded, logps, traces, new_edges_count = self.solver.sample(
            require_prob, prior, parallel_traced, return_decoded
        )
        costs_t = torch.as_tensor(costs, device=self.device, dtype=torch.float32)
        return costs_t, perms, decoded, logps, traces, new_edges_count

    def update_pheromone(self, best_perm, best_cost: float):
        """
        best_perm: (m,) or (m+1,) customers-only
        best_cost: scalar float (CVRP cost)
        """
        p = _as_numpy_i32(best_perm)
        self.solver.update_pheromone_from_perm(p, float(best_cost))
        if self.enable_torch_sync:
            self.sync_pheromone_to_torch()

    def reset_timings(self):
        self.solver.reset_timings()

    def get_timings(self):
        return self.solver.get_timings()

    def sync_pheromone_to_torch(self):
        """Sync numpy pheromone to torch tensor."""
        phe_np = np.asarray(self.solver.pheromone_sparse_np)
        phe_cpu = torch.from_numpy(phe_np)
        self._pheromone_sparse.copy_(phe_cpu.to(self.device))

    def prob_sparse_torch(self, prior: torch.Tensor = None, invtemp: float = 1.0) -> torch.Tensor:
        """
        Returns unnormalized weights for candidate edges: shape (n, k).
        Matches C++ compute_probmat logic (logit space + normalization).
        """
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
            #  z = prior.clamp(-10.0, 10.0)
            #  mean = z.mean(dim=1, keepdim=True)
            #  var = z.var(dim=1, unbiased=False, keepdim=True)
            #  std = torch.sqrt(var + 1e-6)
            #  z_norm = (z - mean) / std
             
             logit = logit + prior
             
        w = torch.exp(logit)
        return w

    # Optional: expose raw views if you want (numpy views from C++)
    @property
    def pheromone_sparse_np(self):
        return self.solver.pheromone_sparse_np

    @property
    def nn_list(self):
        return self.solver.nn_list

    @property
    def heuristic_sparse_np(self):
        return self.solver.heuristic_sparse_np

    def tau_nk_torch(self) -> torch.Tensor:
        """Returns a snapshot (copy) of the current pheromone tensor (n,k)."""
        return self._pheromone_sparse.clone()

    @property
    def nn_torch(self) -> torch.Tensor:
        if not hasattr(self, '_nn_torch'):
             self._nn_torch = torch.from_numpy(self.solver.nn_list.copy()).to(self.device).long()
        return self._nn_torch