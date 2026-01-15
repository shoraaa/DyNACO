
import json
import os

target_file = 'cvrp/train.ipynb'

def instrument_notebook():
    with open(target_file, 'r') as f:
        nb = json.load(f)

    changed = False

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            if 'def train_instance' in source and 'prob_sparse =' in source:
                # Add debug prints
                lines = source.splitlines(True)
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if 'prob_sparse = aco.prob_sparse_torch' in line:
                        new_lines.append("        print(f'DEBUG: device={device}')\n")
                        new_lines.append("        print(f'DEBUG: aco.device={aco.device}')\n")
                        new_lines.append("        print(f'DEBUG: costs_t.device={costs_t.device}')\n")
                        new_lines.append("        print(f'DEBUG: prob_sparse.device={prob_sparse.device}')\n")
                    if 'logp_per_ant, _ =' in line:
                         new_lines.append("        print(f'DEBUG: logp_per_ant.device={logp_per_ant.device}')\n")
                
                cell['source'] = new_lines
                changed = True
                print("Instrumented train_instance with debug prints.")

    if changed:
        with open(target_file, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully modified {target_file}")
    else:
        print("No match found to instrument.")

if __name__ == "__main__":
    instrument_notebook()
