import csv
import shutil
from csv_to_latex import parse_seconds, tsp_datasets, tsp_counts, cvrp_datasets, cvrp_counts

def process_file(filename, datasets, counts):
    backup_filename = filename + ".bak"
    shutil.copyfile(filename, backup_filename)
    print(f"Backed up {filename} to {backup_filename}")

    rows = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    updated_rows = []
    for row in rows:
        new_row = row.copy()
        for ds in datasets:
            time_key = f"{ds}_TotalTime(s)"
            if time_key in row:
                time_str = row[time_key]
                seconds = parse_seconds(time_str)
                if seconds is not None:
                    count = counts[ds]
                    avg_seconds = seconds / count
                    new_row[time_key] = f"{avg_seconds:.4f}"
        updated_rows.append(new_row)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    print(f"Updated {filename}")

def main():
    print("Converting TSP results...")
    process_file('results.csv', tsp_datasets, tsp_counts)
    
    print("\nConverting CVRP results...")
    process_file('result_cvrp.csv', cvrp_datasets, cvrp_counts)

    print("\nDone.")

if __name__ == "__main__":
    main()
