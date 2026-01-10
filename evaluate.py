import argparse
import csv
import sys
import json
import textwrap
from pathlib import Path
from datetime import datetime

# Mathematical/Plotting dependencies
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

# Import inference logic from existing script
try:
    from classify_fotos import classify_image
except ImportError:
    print("Error: classify_fotos.py must be in the same directory.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA binary classifier (GOOD/BAD).")
    parser.add_argument("--images", required=True, type=str, help="Path to image root directory.")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels CSV. If omitted, assumes <root>/<GOOD|BAD> folder structure.")
    parser.add_argument("--out", type=str, default="reports", help="Output directory for reports.")
    return parser.parse_args()

def load_labels_csv(csv_path, images_root):
    """
    Load labels from CSV. 
    Expected format: filename,y_true
    filename is relative to images_root.
    """
    labels = []
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Labels file not found at {csv_path}")
        sys.exit(1)
        
    root = Path(images_root)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Validate headers
        if 'filename' not in reader.fieldnames or 'y_true' not in reader.fieldnames:
            print("Error: CSV must have 'filename' and 'y_true' columns.")
            sys.exit(1)
            
        for row in reader:
            fname = row['filename'].strip()
            y_true = row['y_true'].strip().upper()
            
            if y_true not in ["GOOD", "BAD"]:
                print(f"Warning: Unknown label '{y_true}' for {fname}. Skipping.")
                continue
                
            full_path = root / fname
            # We defer existence check to the main loop to aggregate errors or just skip
            labels.append((full_path, fname, y_true))
            
    return labels

def load_labels_folder(images_root):
    """
    Load labels from folder structure <root>/GOOD/ and <root>/BAD/.
    """
    labels = []
    root = Path(images_root)
    if not root.exists():
        print(f"Error: Images root not found at {images_root}")
        sys.exit(1)
        
    for label in ["GOOD", "BAD"]:
        folder = root / label
        if folder.exists():
            # Gather images, sorted for reproducibility
            files = sorted([p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            for p in files:
                # relative filename from root for CSV consistency
                # e.g. GOOD/img.jpg
                rel_name = str(p.relative_to(root))
                labels.append((p, rel_name, label))
    return labels

def get_labels_dict(images_root, csv_path=None):
    """
    Returns a dict {filename: label} for all found labels.
    """
    items = []
    if csv_path and Path(csv_path).exists():
        items.extend(load_labels_csv(csv_path, images_root))
    
    # Always check folders too? Or specific logic?
    # User said "Support both formats".
    # If we have both, one might override. Let's assume union or CSV updates folder founds.
    items_folder = load_labels_folder(images_root)
    
    # Merge: CSV takes precedence if both exist? usually CSV is explicit.
    label_map = {item[1]: item[2] for item in items_folder}
    for item in items:
        label_map[item[1]] = item[2]
        
    return label_map

def generate_confusion_matrix_pdf(y_true, y_pred, output_path):
    print(f"Generating confusion matrix to {output_path}...")
    
    # Check valid labels
    labels = ["GOOD", "BAD"]
    
    # Compute matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    ax.set_title("Confusion Matrix")
    
    # Interpretation text
    total = sum(sum(cm))
    tn, fp, fn, tp = cm.ravel()
    # Note: scikit-learn confusion matrix for labels=["GOOD", "BAD"]
    # Index 0: GOOD, Index 1: BAD
    # Row 0: True GOOD. Col 0: Pred GOOD (TP for Good), Col 1: Pred BAD (FN for Good i.e. False Bad)
    # Row 1: True BAD. Col 0: Pred GOOD (FP for Good i.e. False Good), Col 1: Pred BAD (TN for Good i.e. True Bad)
    # Wait, usually "Positive" is index 1 or specified.
    # Let's interpret explicitly based on labels=["GOOD", "BAD"]
    # C_00: True GOOD, Pred GOOD.
    # C_01: True GOOD, Pred BAD.
    # C_10: True BAD, Pred GOOD.
    # C_11: True BAD, Pred BAD.
    
    interpretation = [
        "Interpretation:",
        f"- True GOOD (correctly active): {cm[0,0]}",
        f"- False GOOD (actually BAD, missed): {cm[1,0]}",
        f"- True BAD (correctly rejected): {cm[1,1]}",
        f"- False BAD (actually GOOD, false rejection): {cm[0,1]}",
    ]
    
    # Add text to figure
    plt.figtext(0.1, -0.05, "\n".join(interpretation), fontsize=10, ha="left", style='italic', wrap=True)
    
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        
    plt.close(fig)

def generate_metrics_pdf(y_true, y_pred, output_path):
    print(f"Generating metrics report to {output_path}...")
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=["GOOD", "BAD"], output_dict=False)
    
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    title = "Classification Metrics"
    content = f"Accuracy: {acc:.4f}\n\nDetailed Report:\n{report}"
    
    ax.text(0.1, 0.9, title, fontsize=16, weight='bold')
    ax.text(0.1, 0.5, content, fontsize=12, family='monospace', va='top')
    
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
        
    plt.close(fig)

def draw_text_wrapped(ax, text, x, y, width=40, custom_fontsize=8):
    lines = textwrap.wrap(text, width=width)
    y_pos = y
    for line in lines:
        ax.text(x, y_pos, line, transform=ax.transAxes, ha="center", fontsize=custom_fontsize)
        y_pos -= 0.03 # reduced line height

def generate_errors_pdf(errors_list, output_path):
    """
    errors_list: list of dicts {filename, abs_path, y_true, y_pred, reason}
    """
    print(f"Generating error gallery to {output_path}...")
    
    # Separate types
    # False Positive: True BAD, Pred GOOD.
    fp_list = [e for e in errors_list if e['y_true'] == "BAD" and e['y_pred'] == "GOOD"]
    
    # False Negative: True GOOD, Pred BAD.
    fn_list = [e for e in errors_list if e['y_true'] == "GOOD" and e['y_pred'] == "BAD"]
    
    # Function to draw a batch
    def draw_batch(pdf, items, section_title):
        batch_size = 6
        for i in range(0, len(items), batch_size):
            batch = items[i : i+batch_size]
            fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))
            axes = axes.flatten()
            
            # Add section header on first page of section
            if i == 0:
                fig.suptitle(f"{section_title} (Total: {len(items)})", fontsize=16, weight='bold')
            
            for j, ax in enumerate(axes):
                if j < len(batch):
                    item = batch[j]
                    fname = item['filename']
                    truth = item['y_true']
                    pred = item['y_pred']
                    reason = item.get('reason', '')
                    img_path = item['abs_path']
                    
                    ax.axis("off")
                    
                    # Title
                    # Color: always Red because it's an error report? Or distinguishing?
                    # FP (Pred GOOD) but BAD -> Maybe orange?
                    # FN (Pred BAD) but GOOD -> Maybe purple?
                    # Let's simple use black text for clear info.
                    # Title: Filename
                    ax.text(0.5, 1.05, f"{fname}", transform=ax.transAxes, ha="center", weight="bold", fontsize=10)
                    ax.text(0.5, 1.0, f"True: {truth} | Pred: {pred}", transform=ax.transAxes, ha="center", fontsize=9, color="red")
                    
                    # Image
                    if img_path.exists():
                        try:
                            img = mpimg.imread(str(img_path))
                            ax.imshow(img)
                        except Exception as e:
                            ax.text(0.5, 0.5, "Error loading image", transform=ax.transAxes, ha="center")
                    else:
                        ax.text(0.5, 0.5, "Image not found", transform=ax.transAxes, ha="center")
                        
                    # Reason
                    wrapped_reason = "\n".join(textwrap.wrap(reason, width=45))
                    ax.text(0.5, -0.1, wrapped_reason, transform=ax.transAxes, ha="center", va="top", fontsize=8, wrap=True)
                    
                else:
                    ax.axis("off")
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.subplots_adjust(wspace=0.2, hspace=0.4)
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(output_path) as pdf:
        # 1. False Positives
        if fp_list:
            draw_batch(pdf, fp_list, "FALSE POSITIVES (True: BAD, Pred: GOOD)")
        else:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.text(0.5, 0.5, "No False Positives found.", ha='center', fontsize=20)
            pdf.savefig(fig)
            plt.close(fig)
            
        # 2. False Negatives
        if fn_list:
            draw_batch(pdf, fn_list, "FALSE NEGATIVES (True: GOOD, Pred: BAD)")
        else:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.text(0.5, 0.5, "No False Negatives found.", ha='center', fontsize=20)
            pdf.savefig(fig)
            plt.close(fig)

def produce_full_report(results, out_dir):
    """
    Generates all evaluation reports (CSV, Confusion Matrix, Metrics, Errors).
    results: list of dicts with keys: filename, y_true, y_pred, reason, abs_path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Predictions CSV
    csv_out = out_dir / "predictions.csv"
    print(f"Saving predictions to {csv_out}...")
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'y_true', 'y_pred', 'reason'])
        for r in results:
            writer.writerow([r['filename'], r.get('y_true', ''), r['y_pred'], r.get('reason', '')])
            
    # Filter for valid evaluation (requires known y_true)
    eval_results = [r for r in results if r.get('y_true') in ["GOOD", "BAD"]]
    
    if not eval_results:
        print("No valid labels found (y_true in GOOD/BAD). Skipping metrics reports.")
        return

    y_true_all = [r['y_true'] for r in eval_results]
    y_pred_all = [r['y_pred'] for r in eval_results]
    
    # Reports
    confusion_matrix_pdf = out_dir / "confusion_matrix.pdf"
    generate_confusion_matrix_pdf(y_true_all, y_pred_all, confusion_matrix_pdf)
    
    metrics_pdf = out_dir / "metrics.pdf"
    generate_metrics_pdf(y_true_all, y_pred_all, metrics_pdf)
    
    errors_pdf = out_dir / "errors.pdf"
    # Identify errors
    errors_list = [r for r in eval_results if r['y_true'] != r['y_pred']]
    generate_errors_pdf(errors_list, errors_pdf)
    
    print("Evaluation complete.")
    print(f"Key reports generated in {out_dir}")

def main():
    args = parse_args()
    
    images_root = Path(args.images)
    out_dir = Path(args.out)
    
    # Load Labels
    print(f"Loading labels...")
    label_map = get_labels_dict(images_root, args.labels)
        
    if not label_map:
        print("No labeled images found.")
        sys.exit(1)
        
    # Get List of files to process from the labels map implies we only process labeled files
    # args.labels might supply filenames that exist.
    # The original main traversed items from load logic.
    # Reconstruct items list from map for inference order
    items = sorted([(images_root / fname, fname, label) for fname, label in label_map.items()])
    
    print(f"Found {len(items)} labeled images to evaluate.")
    
    # Run Inference
    results = []
    
    for full_path, rel_name, y_true in items:
        if not full_path.exists():
            print(f"Error: Image not found at {full_path}")
            continue
            
        print(f"Processing {rel_name} ...")
        
        # Call existing inference
        pred_res = classify_image(full_path)
        
        y_pred = pred_res.get('label', 'GOOD')
        reason = pred_res.get('reason', '')
        
        results.append({
            'filename': rel_name,
            'y_true': y_true,
            'y_pred': y_pred,
            'reason': reason,
            'abs_path': full_path
        })
        
    if not results:
        print("No predictions generated.")
        sys.exit(1)
        
    produce_full_report(results, out_dir)

if __name__ == "__main__":
    main()
