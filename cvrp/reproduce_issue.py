
import sys
import os
import torch
import numpy as np

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

from net import Net
from faco import MFACO_CVRP
from utils import gen_instance_for_mfaco, build_pyg_data

def replay_dummy(prob_sparse, n_ants):
    # Simulates returning a tensor on the same device as prob_sparse
    return torch.randn(n_ants, device=prob_sparse.device)

def reproduce():
    if not torch.cuda.is_available():
        print("CUDA is required to reproduce the device mismatch error.")
        return

    device = torch.device("cuda:0")
    print(f"Running on device: {device}")

    n_customers = 20
    n_ants = 10
    k_sparse = 10
    
    # 1. Setup data (on CPU as in train_epoch)
    coords_t, demand_t, capacity = gen_instance_for_mfaco(n_customers, device='cpu')
    coords = coords_t.detach().cpu().numpy().astype(np.float32)
    demand = demand_t.detach().cpu().numpy().astype(np.float32)
    
    # 2. Setup Model (on CUDA)
    model = Net().to(device)
    
    # 3. MFACO_CVRP init (Using default device='cpu' as in the buggy code)
    print("Initializing MFACO_CVRP with default device (cpu)...")
    aco = MFACO_CVRP(
        coords=coords,
        demand=demand,
        capacity=float(capacity),
        n_ants=n_ants,
        cand_list_size=k_sparse,
        backup_list_size=k_sparse,
        # device arg missing, defaults to 'cpu'
    )
    
    try:
        # 4. Build PyG data (on CUDA)
        pyg_data = build_pyg_data(aco, coords, demand, device, dynamic=True)
        
        # 5. Forward pass (on CUDA)
        heu_vec = model(pyg_data).view(-1)
        heu_mat = heu_vec.view(aco.n, aco.k) + 1e-10
        
        # 6. Sample (using C++, returns numpy)
        costs_np, perms, _, traces = aco.sample(require_prob=True, prior=heu_mat)
        
        # 7. Convert costs to tensor on CUDA
        costs_t = torch.as_tensor(np.asarray(costs_np, dtype=np.float32), device=device)
        
        # 8. Compute prob_sparse (THE BUG)
        # prob_sparse_torch defaults to return CPU tensor if aco is CPU
        print("Calling prob_sparse_torch...")
        prob_sparse = aco.prob_sparse_torch(prior=heu_mat)
        print(f"prob_sparse device: {prob_sparse.device}")
        
        # 9. Get logp (on same device as prob_sparse -> CPU)
        logp_per_ant = replay_dummy(prob_sparse, n_ants)
        print(f"logp_per_ant device: {logp_per_ant.device}")
        
        # 10. Loss calculation mixing CUDA and CPU
        baseline = costs_t.mean() # CUDA
        adv = (costs_t - baseline).detach() # CUDA
        print(f"adv device: {adv.device}")
        
        print("Attempting operation between adv (CUDA) and logp_per_ant (CPU)...")
        loss = (adv * logp_per_ant).mean()
        print("Loss computed successfully (Unexpected!)")
        
    except RuntimeError as e:
        print("\nSUCCESS: Caught expected RuntimeError:")
        print(e)
    except Exception as e:
        print(f"\nCaught unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    reproduce()
