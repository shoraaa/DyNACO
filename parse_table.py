import re
import csv

raw_data = """CVRP1K CVRP5K CVRP10K CVRP50K CVRP100K
Method Obj. (Gap) Time Obj. (Gap) Time Obj. (Gap) Time Obj. (Gap) Time Obj. (Gap) Time
HGS 36.29 (0.00%) 2.5m 89.74 (0.00%) 2.0h 107.40 (0.00%) 5.0h 267.73 (0.00%) 8.1h 476.11 (0.00%) 24h
LKH3 37.09 (2.21%) 3.3m 93.71 (5.19%) 1.33h 118.76 (10.6%) 1.74h 399.12 (49.1%) 15.8h N/A N/A
Random Insertion 57.42 (58.2%) <1s 154.38 (72.0%) <1s 191.80 (78.6%) <1s 490.56 (83.2%) <1s 943.87 (98.3%) 2s
GLOP-G (LKH3) 39.50 (8.83%) 1.3s 98.90 (10.2%) 6.8s 116.28 (8.27%) 11.2s OOM OOM
POMO aug×8 84.89 (134%) 4.8s 393.27 (338%) 11m OOM OOM OOM
ELG aug×8 41.57 (14.56%) 1.1s 109.54 (22.06%) 30s OOM OOM OOM
LEHD RRC1,000 37.43 (3.15%) 3.4m 101.07 (12.6%) 31m 138.73 (29.2%) 41m OOM OOM
BQ bs16 38.17 (5.17%) 14s 104.40 (16.3%) 2.6m OOM OOM OOM
SIGD bs16 39.15 (7.91%) 17.3s 103.46 (15.3%) 1.91m 131.48 (22.4%) 3.97m 477.43 (78.3%) 25.9m OOM
INViT-3V greedy 42.75 (17.8%) 11.4s 109.85 (22.41%) 1.4m 141.41 (31.66%) 4.2m 402.05 (50.17%) 2.9h 688.80 (44.67%) 8.3h
LEHD greedy 38.91 (7.23%) 0.8s 105.61 (17.69%) 1.56m 146.24 (36.16%) 11.85m OOM OOM
BQ greedy 39.28 (8.23%) 1.03s 108.09 (20.48%) 8.1s 196.44 (82.9%) 1.2m OOM OOM
SIGD greedy 40.18 (10.7%) 1.2s 106.14 (18.3%) 7.9s 135.12 (25.8%) 45s 493.64 (84.4%) 4.3m OOM
Ours greedy 38.11 (5.01%) 0.2s 92.44 (3.01%) 5.49s 109.02 (1.50%) 20.62s 269.34 (0.60%) 8.06m 475.06 (-0.22%) 33.1m
Ours PRC10 37.93 (4.52%) 0.7s 93.92 (4.65%) 3.9s 112.17 (4.43%) 6.8s 285.20 (6.52%) 28s 496.24 (4.23%) 59s
Ours PRC50 37.57 (3.54%) 3.5s 92.06 (2.58%) 19.9s 108.79 (1.29%) 34s 271.77 (1.51%) 2.3m 476.71 (0.13%) 4.8m
Ours PRC100 37.49 (3.31%) 8.0s 91.58 (2.05%) 46s 108.04 (0.59%) 1.3m 268.02 (0.11%) 5.49m 471.35 (-1.00%) 11.5m
Ours PRC500 37.33 (2.88%) 44.6s 91.00 (1.41%) 4.4m 106.85 (-0.51%) 7.6m 263.56 (-1.56%) 31.1m 465.18 (-2.30%) 1.1h
Ours PRC1,000 37.28 (2.72%) 1.5m 90.81 (1.19%) 8.8m 106.69 (-0.66%) 15.2m 262.82 (-1.83%) 1.04h 463.95 (-2.55%) 2.17h"""

lines = raw_data.strip().split('\n')[2:] # Skip header lines

# Datasets and their multipliers
datasets = [
    ("CVRP1K", 128),
    ("CVRP5K", 16),
    ("CVRP10K", 16),
    ("CVRP50K", 16),
    ("CVRP100K", 16)
]

csv_header = ["Method"]
for name, _ in datasets:
    csv_header.extend([f"{name}_Obj", f"{name}_Gap(%)", f"{name}_TotalTime(s)"])

parsed_rows = []

def parse_time(time_str):
    if not time_str: return 0.0
    time_str = time_str.replace('<', '')
    if time_str.endswith('s'):
        return float(time_str[:-1])
    elif time_str.endswith('m'):
        return float(time_str[:-1]) * 60
    elif time_str.endswith('h'):
        return float(time_str[:-1]) * 3600
    else:
        # assume seconds if no unit, though table seems to always have units
        try:
            return float(time_str)
        except ValueError:
            return 0.0

# Regex for a standard data block: Obj (Gap%) Time
# Notes: Gap can have negative numbers, spaces inside parenthesis. Time can have <.
# We look for: Float, then Parenthesized Percent, then Time string.
block_pattern = re.compile(r'^([\d\.]+)\s*\(\s*([-\d\.]+)\s*%\s*\)\s*(<)?(\d+(?:\.\d+)?)([smh])')

# Regex for placeholders
# OOM appears as just OOM
# N/A N/A appears for one column?? Or N/A N/A N/A N/A for 2 cols?
# Based on analysis: 
# H-TSP: OOM OOM (Last 2 cols) -> 1 OOM token per col.
# Concorde: N/A N/A N/A N/A (Last 2 cols) -> 2 N/A tokens per col.
# DIFUSCO*: - - (2nd col) -> 2 - tokens per col.

for line in lines:
    line = line.strip()
    if not line: continue
    
    # 1. Extract Method Name
    # We assume method name ends before the first digit or OOM/- starting the data.
    # We find the first occurrence of a number pattern that starts a data block.
    
    # Heuristic: split by space, finding the index of the first token that looks like a data start.
    # Data start: matches ^\d+\.\d+$ or ^OOM$ or ^[−-]$ or ^N/A$
    # Actually, simplistic approach: Method name is everything until we match the start of a data sequence.
    
    # We scan the string for the first match of our data patterns.
    # But regex search might match inside method name (unlikely for these specific patterns).
    
    # Let's tokenize by finding the first index where a token matches our expectations.
    tokens = re.split(r'\s+', line)
    data_start_idx = -1
    
    for i, token in enumerate(tokens):
        # check if token is start of data
        if re.match(r'^\d+\.\d+$', token):
            data_start_idx = i
            break
        # Sometimes OOM or N/A or - is start
        if token == 'OOM' or token == 'N/A' or token in ['-', '−']:
            # Need to be careful. Is 'bs16' or 'augx8' a match? No.
            data_start_idx = i
            break
            
    if data_start_idx == -1:
        print(f"Skipping line (no data found): {line}")
        continue
        
    method_name = " ".join(tokens[:data_start_idx])
    
    # Replace "Ours" with "SIL"
    if method_name.startswith("Ours"):
        method_name = method_name.replace("Ours", "SIL", 1)
        
    remaining_tokens = tokens[data_start_idx:]
    
    # Now parse the 5 columns from remaining_tokens
    cols_data = []
    
    token_ptr = 0
    
    for _, multiplier in datasets:
        if token_ptr >= len(remaining_tokens):
            # Missing data
            cols_data.extend(["", "", ""])
            continue
            
        first = remaining_tokens[token_ptr]
        
        # Check standard format first (starts with number)
        if re.match(r'^\d+\.\d+$', first):
            # Expecting 3 tokens: Obj, (Gap), Time
            # Sometimes (Gap) is split into (Gap, %) due to spaces? 
            # Our tokenizer split by space. `(5.36% )` -> `(5.36%`, `)`
            
            # Let's reconstruct a small string to match regex against to be safe?
            # Or just process tokens intelligently.
            
            obj_val = float(first)
            
            # Gap
            gap_token = remaining_tokens[token_ptr+1]
            gap_val = ""
            current_ptr_advance = 2
            
            # Handle (Gap%) or (Gap %) or (Gap% )
            # If gap_token is like `(0.00%)`, simple.
            # If `(5.36%` and next is `)`, then split.
            
            full_gap_str = gap_token
            if not gap_token.endswith(')'):
                # check next tokens until closing paren
                offset = 1
                while not full_gap_str.endswith(')') and (token_ptr + 1 + offset) < len(remaining_tokens):
                    full_gap_str += remaining_tokens[token_ptr + 1 + offset]
                    offset += 1
                current_ptr_advance = 1 + offset
            
            # Extract number from full_gap_str `(1.23%)`
            gap_match = re.search(r'\(([-\d\.]+)\s*%\s*\)', full_gap_str)
            if gap_match:
                gap_val = gap_match.group(1)
            else:
                gap_val = full_gap_str # Fallback
                
            # Time
            time_token = remaining_tokens[token_ptr + current_ptr_advance]
            time_val_raw = time_token
            time_seconds = parse_time(time_val_raw)
            total_time = time_seconds * multiplier
            
            cols_data.extend([str(obj_val), gap_val, f"{total_time:.2f}"])
            token_ptr += current_ptr_advance + 1
            
        elif first == 'OOM':
            # 1 token for OOM
            cols_data.extend(["OOM", "OOM", "OOM"])
            token_ptr += 1
            
        elif first == 'N/A':
            # Check if next is also N/A (Concorde case)
            if token_ptr + 1 < len(remaining_tokens) and remaining_tokens[token_ptr+1] == 'N/A':
                token_ptr += 2
            else:
                token_ptr += 1
            cols_data.extend(["N/A", "N/A", "N/A"])
            
        elif first in ['-', '−']:
            # Check if next is also - (Difusco case)
            if token_ptr + 1 < len(remaining_tokens) and remaining_tokens[token_ptr+1] in ['-', '−']:
                token_ptr += 2
            else:
                token_ptr += 1
            cols_data.extend(["-", "-", "-"])
            
        else:
            # Unknown format
            cols_data.extend(["ERR", "ERR", "ERR"])
            token_ptr += 1

    parsed_rows.append([method_name] + cols_data)

# Write CSV
with open('result_cvrp.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(parsed_rows)

print("Done writing result_cvrp.csv")
