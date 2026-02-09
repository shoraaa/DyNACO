"""Read CVRPlib instance dimensions from the test file."""
from utils import load_cvrp_txt_dataset

path = "data/CVRP/data/test_set/CVRPlib_scale_larger_than1000_Li_X_XXL_n14.txt"
data_list = load_cvrp_txt_dataset(path)

print(f"\n{'Instance':<20} {'n (customers)':<15} {'n+depot':<10} {'Capacity':<10} {'BKS Cost'}")
print("-" * 75)
for coords, demand, capacity, cost, tour, name in data_list:
    n_total = coords.shape[0]       # depot + customers
    n_cust = n_total - 1
    print(f"{name:<20} {n_cust:<15} {n_total:<10} {capacity:<10.1f} {cost:.4f}")
print(f"\nTotal instances: {len(data_list)}")
