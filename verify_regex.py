
import re

def parse_metrics(output: str):
    """Parses output from test.py to find Cost, Time, Gap."""
    # Combined regex to find "Model Cost: ..., Time: ...s" line
    # Matches: "Model Cost: 10.1234, Time: 0.5678s"
    # Group 1: Cost, Group 2: Time
    model_stats = re.search(r"Model Cost: ([-+]?\d*\.\d+|\d+).*?Time: ([-+]?\d*\.\d+|\d+)s", output)
    
    # Gap
    gap_match = re.search(r"Model Gap: ([-+]?\d*\.\d+|\d+)%", output)
    
    cost = float(model_stats.group(1)) if model_stats else None
    time_val = float(model_stats.group(2)) if model_stats else None
    gap = float(gap_match.group(1)) if gap_match else None
    
    return {"gap": gap, "time": time_val, "cost": cost}

output_sample = """
--- Results ---
Base Cost: 12.3456, Time: 0.1234s, Total Time: 1.2345s
Base Gap: 5.67%
Model Cost: 10.1234, Time: 0.5678s, Total Time: 5.6789s
Model Gap: 2.3456%
"""

res = parse_metrics(output_sample)
print(res)
assert res["cost"] == 10.1234
assert res["time"] == 0.5678
assert res["gap"] == 2.3456
print("Assertions Passed!")
