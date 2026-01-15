import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

def parse_tsp_file(filepath):
    """
    Parse a .tsp file to extract coordinates. 
    Only supports EUC_2D or similar coordinate sections.
    """
    coords = []
    dimension = 0
    edge_weight_type = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    in_node_coord_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("DIMENSION"):
            parts = line.split(':')
            if len(parts) > 1:
                dimension = int(parts[1].strip())
            else:
                 # Try splitting by space
                 parts = line.split()
                 dimension = int(parts[-1])
                 
        if line.startswith("EDGE_WEIGHT_TYPE"):
             parts = line.split(':')
             if len(parts) > 1:
                 edge_weight_type = parts[1].strip()
             else:
                 parts = line.split()
                 edge_weight_type = parts[-1]

        if line.startswith("NODE_COORD_SECTION"):
            in_node_coord_section = True
            continue
            
        if line.startswith("EOF"):
            break
            
        if in_node_coord_section:
            # Parse coordinate line: "id x y"
            parts = line.split()
            # check if it is integer id
            try:
                # Some files might have index at parsing
                if len(parts) >= 3:
                     x = float(parts[1])
                     y = float(parts[2])
                     coords.append([x, y])
            except ValueError:
                continue

    if len(coords) != dimension and dimension > 0:
        # Some parsing mismatch, but if we have coords, trust them
        pass
        
    return np.array(coords, dtype=np.float32), edge_weight_type

def normalize_coords(coords):
    """
    Normalize coordinates to [0, 1] preserving aspect ratio.
    """
    if len(coords) == 0:
        return coords
        
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    
    # Shift to 0
    coords -= min_xy
    
    # Scale
    diff = max_xy - min_xy
    scale = diff.max()
    
    if scale > 1e-6:
        coords /= scale
        
    return coords

def main():
    parser = argparse.ArgumentParser(description="Pack TSPLIB instances to .pt")
    parser.add_argument("--tsplib_dir", type=str, default="../data/tsplib", help="Path to tsplib directory")
    parser.add_argument("--min_n", type=int, default=0, help="Min node count")
    parser.add_argument("--max_n", type=int, default=100000, help="Max node count")
    parser.add_argument("--output", type=str, default="tsplib_packed.pt", help="Output .pt file")
    
    args = parser.parse_args()
    
    files = [f for f in os.listdir(args.tsplib_dir) if f.endswith(".tsp")]
    print(f"Found {len(files)} .tsp files in {args.tsplib_dir}")
    
    packed_coords = []
    skipped = 0
    
    for fname in tqdm(files):
        path = os.path.join(args.tsplib_dir, fname)
        try:
            coords, ew_type = parse_tsp_file(path)
            
            n = len(coords)
            if n < args.min_n or n > args.max_n:
                continue
                
            # Filter mostly EUC_2D or those with coordinates
            # Some problems like ATT, GEO might have different scaling, but we treat them as 2D for this purpose
            # normalize to [0,1]
            
            if n > 0:
                coords_norm = normalize_coords(coords)
                packed_coords.append(torch.from_numpy(coords_norm))
            else:
                skipped += 1
                
        except Exception as e:
            # print(f"Error parsing {fname}: {e}")
            skipped += 1
            
    print(f"Packed {len(packed_coords)} instances. Skipped/Filtered {len(files) - len(packed_coords)}")
    
    # Save as list of tensors
    # Can also wrap in dict if preferred: {"coords": list}
    # tsp/test.py supports both list and dict["coords"]
    
    output_data = {"coords": packed_coords}
    torch.save(output_data, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
