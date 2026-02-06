
def get_dataset_signature(dataset, problem):
    """
    Compute a consistent hash signature for a dataset.
    Uses the first item's content as a proxy for the dataset identity.
    """
    if len(dataset) == 0:
        return "empty"
    
    first_item = dataset[0]
    hasher = hashlib.md5()
    
    # Add length as part of signature
    hasher.update(str(len(dataset)).encode('utf-8'))
    
    try:
        if problem == 'tsp':
            if torch.is_tensor(first_item):
                # TensorDataset or list of tensors
                # Quantize slightly to avoid float precision issues across machines/platforms?
                # Or just use tobytes().
                hasher.update(first_item.cpu().numpy().tobytes())
            elif isinstance(first_item, np.ndarray):
                hasher.update(first_item.tobytes())
            else:
                # Fallback
                hasher.update(str(first_item).encode('utf-8'))
        elif problem == 'cvrp':
            # Tuple: (coords, demand, capacity)
            coords, demand, cap = first_item
            if torch.is_tensor(coords): coords = coords.cpu().numpy()
            if torch.is_tensor(demand): demand = demand.cpu().numpy()
            hasher.update(coords.tobytes())
            hasher.update(demand.tobytes())
            hasher.update(str(cap).encode('utf-8'))
    except Exception as e:
        print(f"Warning: Could not hash dataset item: {e}")
        return "unknown_dataset"
        
    return hasher.hexdigest()

def get_pure_mfaco_config_hash(args):
    """
    Compute hash of configuration parameters relevant to Pure MFACO.
    Ignores model-specific args or paths that don't affect ACO logic.
    """
    # Key parameters affecting Pure MFACO behavior
    keys = [
        'problem', 'n_node', 'n_ants', 'k_sparse', 
        'seed', 'rho', 'alpha', 'beta', 'min_new_edges',
        'H', 'mini_H', 'L', # Iterations
        'parallel_traced', 'no_local_search', 'no_smooth_mmas', 
        'no_extend_ls', 'disable_heuristic', 'no_normalized_heuristic',
        # Validation specific params if they override main args
        'val_H', 'val_mini_H' 
    ]
    
    config_str = ""
    for k in sorted(keys):
        val = getattr(args, k, None)
        config_str += f"{k}={val};"
        
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

def get_pure_mfaco_cache_path(args, dataset_sig):
    config_hash = get_pure_mfaco_config_hash(args)
    # Combine config hash and dataset signature
    full_hash = hashlib.sha256(f"{config_hash}_{dataset_sig}".encode('utf-8')).hexdigest()
    
    cache_dir = Path("pretrained/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"pure_mfaco_{full_hash}.json"

def load_pure_mfaco_cache(args, dataset):
    sig = get_dataset_signature(dataset, args.problem)
    path = get_pure_mfaco_cache_path(args, sig)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_pure_mfaco_cache(args, dataset, result):
    sig = get_dataset_signature(dataset, args.problem)
    path = get_pure_mfaco_cache_path(args, sig)
    try:
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
