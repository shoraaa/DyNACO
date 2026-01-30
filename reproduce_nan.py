
import torch
import numpy as np
from train import replay_logp_from_cpp_batch_trace, log_prob_sparse_from_tau_eta_prior

class MockTraces:
    def __init__(self, n_ants, n, k):
        self.n_ants = n_ants
        # 1 step per ant
        total_steps = n_ants
        self.curr_nodes = np.zeros(total_steps, dtype=np.int64)
        self.is_stochastic = np.ones(total_steps, dtype=np.uint8) # All stochastic
        self.pick_j = np.zeros(total_steps, dtype=np.int64) # Always pick neighbor 0
        # Valid mask: all 1s (k bits)
        self.valid_mask = np.full(total_steps, (1 << k) - 1, dtype=np.uint64)
        
        self.starts = np.arange(0, total_steps + 1, 1, dtype=np.int64)

def test_numerical_stability():
    n_ants = 2
    n = 10
    k = 5
    device = "cpu"
    
    traces = MockTraces(n_ants, n, k)
    
    # Case 1: Normal values
    tau_nk = torch.ones((n, k), device=device)
    eta_nk = torch.ones((n, k), device=device)
    prior_nk = torch.zeros((n, k), device=device) # Logits around 0
    
    probs = log_prob_sparse_from_tau_eta_prior(tau_nk, eta_nk, prior_nk)
    logp, ndec = replay_logp_from_cpp_batch_trace(traces, probs)
    
    print("Case 1 (Normal):")
    print(f"Logp: {logp}")
    print(f"Has NaN: {torch.isnan(logp).any().item()}")
    
    # Case 2: Large prior values (Exploding exp)
    # exp(100) ~ 2.6e43 (Fit in float32? max float32 is ~3.4e38. So exp(90) exceeds float32 range)
    # Let's try 100.
    prior_nk = torch.full((n, k), 100.0, device=device)
    
    probs = log_prob_sparse_from_tau_eta_prior(tau_nk, eta_nk, prior_nk)
    print("\nCase 2 (Large Prior 100.0):")
    print(f"Max Log-Prob (log weights): {probs.max().item()}")
    print(f"Is Inf: {torch.isinf(probs).any().item()}")
    
    logp, ndec = replay_logp_from_cpp_batch_trace(traces, probs)
    print(f"Logp: {logp}")
    print(f"Has NaN: {torch.isnan(logp).any().item()}")

if __name__ == "__main__":
    test_numerical_stability()
