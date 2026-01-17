import csv
from pathlib import Path

def update_labels_csv(images_dir, csv_path):
    images_dir = Path(images_dir)
    csv_path = Path(csv_path)
    
    # 1. Load existing labels
    existing_labels = {}
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'filename' in reader.fieldnames and 'y_true' in reader.fieldnames:
                for row in reader:
                    fname = row['filename'].strip()
                    if fname:
                        existing_labels[fname] = row['y_true'].strip()
    
    print(f"Loaded {len(existing_labels)} existing labels.")

    # 2. Scan for images
    # Supports .jpg, .jpeg, .png (case insensitive)
    extensions = {".jpg", ".jpeg", ".png"}
    # Recursive search or flat? The current classify script uses rglob("*") so we should be consistent
    # classify_fotos.py: image_files = sorted([p for p in images_dir.rglob("*") ...])
    # However, existing labels.csv seems to have flat filenames. 
    # Let's check if the user has subdirectories. 
    # list_dir output showed 0 subdirectories in images. So flat search is fine.
    
    image_files = sorted([
        p for p in images_dir.rglob("*") 
        if p.is_file() and p.suffix.lower() in extensions
    ])
    
    print(f"Found {len(image_files)} images in {images_dir}.")

    # 3. Merge and Create rows
    rows = []
    # We want to store relative path if it's in a subdirectory, or just filename if flat.
    # classify_fotos.py uses `rel_path = str(img.relative_to(images_dir))`
    # The existing labels.csv has just filenames.
    
    new_count = 0
    
    for img_path in image_files:
        rel_path = img_path.name # default to simple filename as per existing csv style
        # check if we should use relative path
        if img_path.parent != images_dir:
             rel_path = str(img_path.relative_to(images_dir))
             
        current_label = existing_labels.get(rel_path, "UNKNOWN")
        if rel_path not in existing_labels:
            new_count += 1
            
        rows.append({"filename": rel_path, "y_true": current_label})
        
    # 4. Write back
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "y_true"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Updated {csv_path}. Total rows: {len(rows)}. New entries: {new_count}.")

if __name__ == "__main__":
    update_labels_csv("images", "labels.csv")
