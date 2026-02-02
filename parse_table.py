import re
import csv

raw_data = """TSP1K TSP5K TSP10K TSP50K TSP100K
Method Obj. (Gap) Time Obj. (Gap) Time Obj. (Gap) Time Obj. (Gap) Time Obj. (Gap) Time
LKH3 23.12 (0.00%) 1.7m 50.97 (0.00%) 12m 71.78 (0.00%) 33m 159.93 (0.00%) 10h 225.99 (0.00%) 25h
Concorde 23.12 (0.00%) 1m 50.95 (-0.05%) 31m 72.00 (0.15%) 1.4h N/A N/A N/A N/A
Random Insertion 26.11 (12.9%) <1s 58.06 (13.9%) <1s 81.82 (13.9%) <1s 182.65 (14.2%) 15.4s 258.13 (14.2%) 1.7m
DIFUSCO* 23.39 (1.17%) 11.5s − − 73.62 (2.58%) 3.0m − − − −
H-TSP 24.66 (6.66%) 48s 55.16 (8.21%) 1.2m 77.75 (8.38%) 2.2m OOM OOM
GLOP 23.78 (2.85%) 10.2s 53.15 (4.26%) 1.0m 75.04 (4.39%) 1.9m 168.09 (5.10%) 1.5m 237.61 (5.14%) 3.9m
POMO aug×8 32.51 (40.6%) 4.1s 87.72 (72.1%) 8.6m OOM OOM OOM
ELG aug×8 25.738 (11.33%) 0.8s 60.19 (18.08%) 21s OOM OOM OOM
LEHD RRC1,000 23.29 (0.72%) 3.3m 54.43 (6.79%) 8.6m 80.90 (12.5%) 18.6m OOM OOM
BQ bs16 23.43 (1.37%) 13s 58.27 (10.7%) 24s OOM OOM OOM
SIGD bs16 23.36 (1.03%) 17.3s 55.77 (9.42%) 30.5m OOM OOM OOM
INViT-3V greedy 24.66 (6.66%) 9.0s 54.49 (6.90%) 1.2m 76.85 (7.07%) 3.7m 171.42 (7.18%) 1.3h 242.26 (7.20%) 5.0h
LEHD greedy 23.84 (3.11%) 0.8s 58.85 (15.46%) 1.5m 91.33 (27.24%) 11.7m OOM OOM
BQ greedy 23.65 (2.30%) 0.9s 58.27 (14.31%) 22.5s 89.73 (25.02%) 1.0m OOM OOM
SIGD greedy 23.573 (1.96%) 1.2s 57.19 (12.20%) 1.8m 93.80 (30.68%) 15.5m OOM OOM
Ours greedy 23.569 (1.95%) 0.2s 52.59 (3.17%) 5.2s 74.69 (4.05%) 20.1s 168.50 (5.36% ) 7.7m 239.84 (6.13%) 33.0m
Ours PRC10 23.396 (1.20%) 0.9s 52.36 (2.73%) 5.1s 73.99 (3.08%) 10.0s 166.69 (4.22%) 1.33m 235.38 (4.16%) 3.0m
Ours PRC50 23.279 (0.69%) 4.6s 51.92 (1.85%) 23.4s 73.41 (2.27%) 49.0s 165.01 (3.17%) 4.9m 233.13 (3.16%) 9.2m
Ours PRC100 23.254 (0.58%) 9.4s 51.82 (1.67%) 52.0s 73.29 (2.11%) 1.7m 164.59 (2.91%) 8.6m 232.55 (2.90%) 17m
Ours PRC500 23.217 (0.43%) 46s 51.70 (1.43%) 4.6m 73.12 (1.87%) 8.5m 164.09 (2.60%) 42.2m 231.75 (2.55%) 1.4h
Ours PRC1,000 23.207 (0.38%) 1.5m 51.67 (1.36%) 9.4m 73.08 (1.81%) 17.0m 163.95 (2.51%) 1.38h 231.52 (2.45%) 2.6h"""

lines = raw_data.strip().split('\n')[2:] # Skip header lines

# Datasets and their multipliers
datasets = [
    ("TSP1K", 128),
    ("TSP5K", 16),
    ("TSP10K", 16),
    ("TSP50K", 16),
    ("TSP100K", 16)
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
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(parsed_rows)

print("Done writing results.csv")
