
import torch
import sys

def inspect(ckpt_path):
    print(f"Loading {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'config' in ckpt:
            print("Config found in checkpoint.")
            config = ckpt['config']
            print("Keys related to ablation:")
            for k, v in config.items():
                if 'ablation' in k or 'feat' in k:
                    print(f"  {k}: {v}")
        else:
            print("No 'config' key in checkpoint.")
            
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            if "emb_net.e_lin0.weight" in sd:
                shape = sd["emb_net.e_lin0.weight"].shape
                print(f"emb_net.e_lin0.weight shape: {shape}")
            else:
                print("emb_net.e_lin0.weight not found in state_dict")
                
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint>")
    else:
        inspect(sys.argv[1])
