
import torch
import utils
import argparse
import numpy as np


def check_dataset(name, dataset, cap_train):
    if dataset is None:
        print(f"[{name}] Not found/loaded.")
        return

    # Check first instance
    item = dataset[0]
    
    # Unpack logic from test.py
    # CVRP Tuple: (coords, demand, capacity, cost, tour)
    if isinstance(item, tuple) and len(item) >= 5:
        coords, demand, capacity, cost, tour = item[:5]
        item = (coords, demand, capacity)
    elif isinstance(item, tuple) and len(item) == 3:
         # Generated tuple
         pass
    elif isinstance(item, list) and len(item) == 3:
         item = tuple(item)

    # Convert to standard format
    # item is [coords, demand, cap] tensors/arrays
    c_val = item[0]
    d_val = item[1]
    cap_val = item[2]
    
    # Handle tensor wrappers
    if torch.is_tensor(c_val): c_val = c_val.numpy()
    if torch.is_tensor(d_val): d_val = d_val.numpy()
    if torch.is_tensor(cap_val): cap_val = float(cap_val)

    print(f"[{name}] Raw Capacity: {cap_val} | Demand Max: {d_val.max()}")

    # Apply test.py normalization logic
    if cap_val > 1.0 + 1e-6:
        print(f"[{name}] Applying normalization...")
        d_val = d_val / cap_val
        cap_val = 1.0
    
    print(f"[{name}] Norm Capacity: {cap_val} | Norm Demand Max: {d_val.max():.4f}")
    
    if abs(cap_train - cap_val) > 1e-6:
        print(f"[{name}] MISMATCH: Capacity {cap_val} != Train {cap_train}")
    
    # Check coords range
    c_min, c_max = c_val.min(), c_val.max()
    print(f"[{name}] Coords Range: {c_min:.4f} - {c_max:.4f}")
    if c_max > 1.0 + 1e-6:
         print(f"[{name}] WARNING: Coords > 1.0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_node", type=int, default=100) # Optional override
    args = parser.parse_args()
    
    n_values = [50, 100]
    if args.n_node != 100:
        n_values = [args.n_node]

    for n in n_values:
        print(f"\n{'='*20} Inspecting Data for N={n} {'='*20}")

        # 1. Generate Training Data
        print("\n[Training Data Generation]")
        torch.manual_seed(1234)
        c_train, d_train, cap_norm = utils.gen_cvrp_instance(n, device='cpu')
        
        # Calculate expected Raw Capacity based on N (same logic as utils.py)
        if n >= 50000: raw_cap = 2000
        elif n >= 10000: raw_cap = 1000
        elif n >= 5000: raw_cap = 500
        elif n >= 1000: raw_cap = 250
        else: raw_cap = 50

        print(f"[Training] Raw Capacity: {raw_cap} | Demand Max (approx raw): {d_train.max() * raw_cap:.4f}")
        print(f"[Training] Norm Capacity: {cap_norm} | Norm Demand Max: {d_train.max():.4f}")
        print(f"[Training] Coords Range: {c_train.min():.4f} - {c_train.max():.4f}")

        if cap_norm != 1.0:
            print("WARNING: Training capacity is NOT 1.0")

        # 2. Test Set
        print(f"\n[Test Set Query (N={n})]")
        try:
            test_dataset = utils.load_auto_dataset(n, problem='cvrp', data_source='test_set', device='cpu')
            check_dataset("TestSet", test_dataset, cap_norm)
        except Exception as e:
            print(f"Error checking TestSet: {e}")

        # 3. Validation Set
        print(f"\n[Validation Set Query (N={n})]")
        try:
            # Note: utils.load_auto_dataset defaults to data_source='test_set' if argument not provided
            # We explicitly ask for validation_set as train.py does
            val_dataset = utils.load_auto_dataset(n, problem='cvrp', data_source='validation_set', device='cpu')
            check_dataset("ValSet", val_dataset, cap_norm)
        except Exception as e:
            print(f"Error checking ValSet: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
