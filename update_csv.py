import csv

new_rows = [
    {
        "Method": "DyNACO H=10",
        "TSP1K_Obj": "23.2", "TSP1K_Gap(%)": "0.49", "TSP1K_TotalTime(s)": "15",
        "TSP5K_Obj": "51.58", "TSP5K_Gap(%)": "1.40", "TSP5K_TotalTime(s)": "32",
        "TSP10K_Obj": "72.97", "TSP10K_Gap(%)": "1.66", "TSP10K_TotalTime(s)": "60",
        "TSP50K_Obj": "171", "TSP50K_Gap(%)": "4.35", "TSP50K_TotalTime(s)": "180",
        "TSP100K_Obj": "246", "TSP100K_Gap(%)": "6.78", "TSP100K_TotalTime(s)": "60"
    },
    {
        "Method": "DyNACO H=20",
        "TSP1K_Obj": "-", "TSP1K_Gap(%)": "-", "TSP1K_TotalTime(s)": "-",
        "TSP5K_Obj": "51.44", "TSP5K_Gap(%)": "0.92", "TSP5K_TotalTime(s)": "58",
        "TSP10K_Obj": "-", "TSP10K_Gap(%)": "-", "TSP10K_TotalTime(s)": "-",
        "TSP50K_Obj": "-", "TSP50K_Gap(%)": "-", "TSP50K_TotalTime(s)": "-",
        "TSP100K_Obj": "-", "TSP100K_Gap(%)": "-", "TSP100K_TotalTime(s)": "-"
    },
    {
        "Method": "DyNACO H=30",
        "TSP1K_Obj": "-", "TSP1K_Gap(%)": "-", "TSP1K_TotalTime(s)": "-",
        "TSP5K_Obj": "51.38", "TSP5K_Gap(%)": "0.79", "TSP5K_TotalTime(s)": "81.6",
        "TSP10K_Obj": "-", "TSP10K_Gap(%)": "-", "TSP10K_TotalTime(s)": "-",
        "TSP50K_Obj": "-", "TSP50K_Gap(%)": "-", "TSP50K_TotalTime(s)": "-",
        "TSP100K_Obj": "-", "TSP100K_Gap(%)": "-", "TSP100K_TotalTime(s)": "-"
    }
]

rows = []
fieldnames = []

with open('results.csv', 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row['Method'].startswith('Ours'):
            row['Method'] = row['Method'].replace('Ours', 'SIL')
        rows.append(row)

# Append new rows
rows.extend(new_rows)

with open('results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Updated results.csv")
