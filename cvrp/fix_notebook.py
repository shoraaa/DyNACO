
import json
import os

target_file = 'cvrp/train.ipynb'

def fix_notebook():
    with open(target_file, 'r') as f:
        nb = json.load(f)

    changed = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Fix in train_instance
            if 'def train_instance' in source:
                # Fix MFACO_CVRP init in train_instance
                if 'extend_ls=extend_ls,\n' in source and 'device=device' not in source.split('extend_ls=extend_ls,\n')[1].split(')')[0]:
                    print("Patching MFACO_CVRP init in train_instance...")
                    source = source.replace('extend_ls=extend_ls,\n', 'extend_ls=extend_ls,\n        device=device,\n')
                    changed = True

                # Fix prob_sparse_torch call
                if 'prob_sparse = aco.prob_sparse_torch(prior=heu_mat)' in source:
                    print("Patching prob_sparse_torch call...")
                    source = source.replace('prob_sparse = aco.prob_sparse_torch(prior=heu_mat)', 'prob_sparse = aco.prob_sparse_torch(prior=heu_mat, device=device)')
                    changed = True
            
            # Fix in infer_instance (likely same cell but let's be safe)
            if 'def infer_instance' in source:
                # Fix MFACO_CVRP init in infer_instance
                if 'enable_torch_sync=True,\n' in source and 'device=device' not in source.split('enable_torch_sync=True,\n')[1].split(')')[0]:
                    print("Patching MFACO_CVRP init in infer_instance...")
                    source = source.replace('enable_torch_sync=True,\n', 'enable_torch_sync=True,\n        device=device,\n')
                    changed = True
            
            # Update cell source
            # Jupyter usually expects a list of strings where each string ends with \n except maybe the last one
            # But converting to string and writing back as list of lines is safer structure-wise if we splitlines(keepends=True)
            if changed:
                cell['source'] = [line for line in source.splitlines(True)]

    if changed:
        with open(target_file, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {target_file}")
    else:
        print(f"No changes made to {target_file}. Maybe already patched?")

if __name__ == "__main__":
    fix_notebook()
