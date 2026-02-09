#!/usr/bin/env python3
"""Debug script: Compare load_cvrp_txt_dataset across CVRPLIB vs test_set vs gen_cvrp_instance."""

import torch
import numpy as np
from pathlib import Path
from utils import load_cvrp_txt_dataset, gen_cvrp_instance

DATA_DIR = Path("data/CVRP/data/test_set")

def summarize_item(item, label):
    """Pretty-print one parsed CVRP instance."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Tuple length: {len(item)}")
    for i, v in enumerate(item):
        t = type(v).__name__
        if isinstance(v, torch.Tensor):
            print(f"  [{i}] Tensor  shape={v.shape}  dtype={v.dtype}  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")
        elif isinstance(v, np.ndarray):
            print(f"  [{i}] ndarray shape={v.shape}  dtype={v.dtype}  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")
        elif isinstance(v, (list, tuple)):
            print(f"  [{i}] {t:7s} len={len(v)}  first_5={v[:5]}")
        elif v is None:
            print(f"  [{i}] None")
        else:
            print(f"  [{i}] {t:7s} value={v}")

    # Named fields based on (coords, demand, capacity, cost, tour, name)
    coords, demand, capacity, cost = item[0], item[1], item[2], item[3]
    tour = item[4] if len(item) > 4 else None
    name = item[5] if len(item) > 5 else None

    n_total = coords.shape[0] if hasattr(coords, 'shape') else len(coords)
    n_cust = n_total - 1
    print(f"\n  --- Semantic ---")
    print(f"  Name:       {name}")
    print(f"  N_total:    {n_total}  (depot + {n_cust} customers)")
    print(f"  Capacity:   {capacity}")
    print(f"  Cost:       {cost}")
    
    if isinstance(coords, torch.Tensor):
        depot = coords[0]
        cust_coords = coords[1:]
        print(f"  Depot:      ({depot[0]:.4f}, {depot[1]:.4f})")
        print(f"  Coords range: x=[{cust_coords[:,0].min():.4f}, {cust_coords[:,0].max():.4f}]  "
              f"y=[{cust_coords[:,1].min():.4f}, {cust_coords[:,1].max():.4f}]")
    
    if demand is not None:
        if isinstance(demand, torch.Tensor):
            d = demand
        else:
            d = torch.tensor(demand)
        print(f"  Demand[0] (depot): {d[0]:.4f}")
        print(f"  Demand[1:] range:  [{d[1:].min():.4f}, {d[1:].max():.4f}]  mean={d[1:].mean():.4f}")
        print(f"  Demand[1:] sum:    {d[1:].sum():.4f}")
        
        # Check if demands look normalized or integer
        d_cust = d[1:]
        if d_cust.max() <= 1.0:
            print(f"  >>> Demands appear NORMALIZED (max <= 1.0)")
        else:
            print(f"  >>> Demands appear RAW/INTEGER (max > 1.0)")
            
        # Check if capacity is normalized
        if isinstance(capacity, float) and capacity == 1.0:
            print(f"  >>> Capacity is 1.0 -> likely NORMALIZED system")
        else:
            print(f"  >>> Capacity is {capacity} -> likely RAW/INTEGER system")
    
    if tour is not None:
        print(f"  Tour:       len={len(tour)}  first_10={tour[:10]}")
    else:
        print(f"  Tour:       None")


def main():
    # =====================================================================
    # 1) Load CVRPlib_1K.txt
    # =====================================================================
    cvrplib_path = DATA_DIR / "CVRPlib_1K.txt"
    print(f"\n{'#'*70}")
    print(f"# Loading CVRPlib: {cvrplib_path}")
    print(f"{'#'*70}")
    
    try:
        cvrplib_data = load_cvrp_txt_dataset(str(cvrplib_path))
        print(f"\nLoaded {len(cvrplib_data)} instances")
        if cvrplib_data and len(cvrplib_data) > 0:
            summarize_item(cvrplib_data[0], "CVRPlib_1K instance[0]")
            if len(cvrplib_data) > 1:
                summarize_item(cvrplib_data[1], "CVRPlib_1K instance[1]")
        else:
            print("WARNING: No instances loaded!")
    except Exception as e:
        print(f"ERROR loading CVRPlib: {e}")
        import traceback; traceback.print_exc()
        cvrplib_data = None

    # =====================================================================
    # 2) Load test_set (test_cvrp1000_hgs_n128_C250.txt)
    # =====================================================================
    test_path = DATA_DIR / "test_cvrp1000_hgs_n128_C250.txt"
    print(f"\n{'#'*70}")
    print(f"# Loading test_set: {test_path}")
    print(f"{'#'*70}")
    
    try:
        test_data = load_cvrp_txt_dataset(str(test_path))
        print(f"\nLoaded {len(test_data)} instances")
        if test_data and len(test_data) > 0:
            summarize_item(test_data[0], "test_set instance[0]")
    except Exception as e:
        print(f"ERROR loading test_set: {e}")
        import traceback; traceback.print_exc()
        test_data = None

    # =====================================================================
    # 3) Generate a fresh CVRP instance for comparison
    # =====================================================================
    print(f"\n{'#'*70}")
    print(f"# Generating fresh CVRP instance (n=1000)")
    print(f"{'#'*70}")
    
    coords, demand, capacity = gen_cvrp_instance(1000, 'cpu')
    gen_item = (coords, demand, capacity, 0.0, None, "Generated")
    summarize_item(gen_item, "gen_cvrp_instance(1000)")

    # =====================================================================
    # 4) Summary comparison
    # =====================================================================
    print(f"\n{'#'*70}")
    print(f"# COMPARISON SUMMARY")
    print(f"{'#'*70}")
    
    print(f"\n{'Feature':<30} {'CVRPlib_1K':<25} {'test_set':<25} {'gen_cvrp_instance':<25}")
    print("-" * 105)
    
    rows = []
    
    def fmt(dataset, idx, label, extractor):
        if dataset is None or len(dataset) == 0:
            return "N/A"
        try:
            return str(extractor(dataset[0]))
        except:
            return "ERR"
    
    comparisons = [
        ("tuple length", lambda d: len(d)),
        ("coords shape", lambda d: tuple(d[0].shape) if hasattr(d[0], 'shape') else "?"),
        ("coords range", lambda d: f"[{d[0].min():.2f}, {d[0].max():.2f}]"),
        ("demand shape", lambda d: tuple(d[1].shape) if d[1] is not None and hasattr(d[1], 'shape') else "None"),
        ("demand range", lambda d: f"[{d[1].min():.2f}, {d[1].max():.2f}]" if d[1] is not None else "None"),
        ("demand normalized?", lambda d: "yes" if d[1] is not None and (d[1].max() if hasattr(d[1], 'max') else max(d[1])) <= 1.0 else "NO (raw)"),
        ("capacity", lambda d: f"{d[2]}"),
        ("cost", lambda d: f"{d[3]:.4f}" if d[3] else "0"),
        ("has tour?", lambda d: d[4] is not None),
        ("name", lambda d: d[5] if len(d) > 5 else "N/A"),
    ]
    
    for label, extractor in comparisons:
        c1 = fmt(cvrplib_data, 0, label, extractor) if cvrplib_data else "LOAD_FAIL"
        c2 = fmt(test_data, 0, label, extractor) if test_data else "LOAD_FAIL"
        c3 = fmt([gen_item], 0, label, extractor)
        print(f"  {label:<28} {c1:<25} {c2:<25} {c3:<25}")

    # =====================================================================
    # 5) Check if ACO can consume each format (demand normalization issue?)
    # =====================================================================
    print(f"\n{'#'*70}")
    print(f"# DEMAND NORMALIZATION CHECK")
    print(f"{'#'*70}")
    
    if cvrplib_data and len(cvrplib_data) > 0:
        item = cvrplib_data[0]
        coords, demand, capacity = item[0], item[1], item[2]
        if demand is not None:
            d = demand if isinstance(demand, torch.Tensor) else torch.tensor(demand)
            d_cust = d[1:]
            cap = float(capacity)
            print(f"\n  CVRPlib: demand sum={d_cust.sum():.2f}, capacity={cap}")
            if d_cust.max() > 1.0:
                print(f"  >>> RAW demands (max={d_cust.max():.0f}). Need normalization: demand / capacity")
                print(f"  >>> Normalized demand range would be [{d_cust.min()/cap:.4f}, {d_cust.max()/cap:.4f}]")
                print(f"  >>> After normalization, capacity should be 1.0")
            else:
                print(f"  >>> Already normalized")
    
    if test_data and len(test_data) > 0:
        item = test_data[0]
        coords, demand, capacity = item[0], item[1], item[2]
        if demand is not None:
            d = demand if isinstance(demand, torch.Tensor) else torch.tensor(demand)
            d_cust = d[1:]
            cap = float(capacity)
            print(f"\n  test_set: demand sum={d_cust.sum():.2f}, capacity={cap}")
            if d_cust.max() > 1.0:
                print(f"  >>> RAW demands (max={d_cust.max():.0f}). Need normalization")
                print(f"  >>> Normalized demand range would be [{d_cust.min()/cap:.4f}, {d_cust.max()/cap:.4f}]")
            else:
                print(f"  >>> Already normalized")

    print(f"\n  gen_cvrp_instance: demand range=[{demand.min():.4f}, {demand.max():.4f}], capacity={capacity}")
    print(f"  >>> gen_cvrp_instance already returns NORMALIZED demands (demand/capacity) with capacity=1.0")


if __name__ == "__main__":
    main()
