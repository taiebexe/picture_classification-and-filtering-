import json
import csv
from pathlib import Path

def fill_labels_from_json(json_path, csv_path):
    json_path = Path(json_path)
    csv_path = Path(csv_path)

    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        return
    
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    # 1. Load results.json
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    # Create a mapping from filename to label
    # results.json structure: [{"label": "GOOD", "filename": "...", ...}, ...]
    file_label_map = {}
    for item in results:
        fname = item.get('filename')
        label = item.get('label')
        if fname and label:
            file_label_map[fname] = label
    
    print(f"Loaded {len(file_label_map)} labels from {json_path}.")

    # 2. Read existing labels.csv
    rows = []
    updated_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            filename = row['filename']
            # If we have a result for this file, update it
            if filename in file_label_map:
                new_label = file_label_map[filename]
                if row['y_true'] != new_label:
                    row['y_true'] = new_label
                    updated_count += 1
            rows.append(row)

    # 3. Write back to labels.csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated_count} labels in {csv_path}.")

if __name__ == "__main__":
    fill_labels_from_json("results.json", "labels.csv")
