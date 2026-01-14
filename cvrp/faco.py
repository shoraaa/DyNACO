import numpy as np
import torch

try:
    import faco_cvrp as faco_cpp
except ImportError as e:
    raise ImportError(
        "C++ backend 'faco_cvrp' not found. Build/compile the extension in cvrp/src first."
    ) from e


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
        )

    @property
    def n(self): return self.solver.n

    @property
    def m(self): return self.solver.m

    @property
    def k(self): return self.solver.k

    @property
    def n_ants(self): return self.solver.n_ants

    def seed_rng(self, seed: int):
        self.solver.seed_rng(int(seed))

    def sample(self, require_prob: bool = False, residual_logits=None, parallel_traced: bool = False, return_decoded: bool = False):
        """
        Returns:
          costs: torch.float32 [n_ants]
          perms: list[np.ndarray] each (m+1,) customers-only cycle (last==first)
          decoded_routes: list[np.ndarray] each variable length [0,...,0,...] or None
          traces: faco_cpp.MFACOTrace or None
        """
        if residual_logits is not None:
            residual_logits = _as_numpy_f32(residual_logits)
        costs, perms, decoded, traces = self.solver.sample(
            require_prob, residual_logits, parallel_traced, return_decoded
        )
        costs_t = torch.from_numpy(np.asarray(costs, dtype=np.float32))
        return costs_t, perms, decoded, traces

    def update_pheromone(self, best_perm, best_cost: float):
        """
        best_perm: (m,) or (m+1,) customers-only
        best_cost: scalar float (CVRP cost)
        """
        p = _as_numpy_i32(best_perm)
        self.solver._update_pheromone_from_perm(p, float(best_cost))

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
