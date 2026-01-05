import io
import base64
import json
import requests
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
MAX_IMAGE_SIZE = 1000

# Ollama server (remote RTX 5090)
OLLAMA_URL = "http://luke.nt.fh-koeln.de:11434/api/generate"
MODEL_NAME = "llava:34b"

PROMPT = """
Analyze the image.
1. IS IT AN EMPTY POT?
   - Look for: Soil, perlite (white dots), gravel, plastic cover, condensation.
   - If NO green leaves are visible -> Classify as "BAD" (Reason: "Empty pot/soil only").

2. IS IT A REAL PLANT?
   - Look for: Defined GREEN LEAVES, ROSETTE shape, or central STEM.
   - IGNORE: Circular reflections, algae, blurry green blobs.
   - If distinct leaves are visible -> Classify as "GOOD".

Output valid JSON only: { "label": "GOOD" or "BAD", "reason": "concise explanation (max 10 words)" }
NO markdown.
""".strip()


def encode_image(path: Path) -> str:
    """Return base64-encoded contents of an image file, resized to max 1024px."""
    try:
        with Image.open(path) as img:
            # Calculate new size maintaining aspect ratio
            width, height = img.size
            if max(width, height) > MAX_IMAGE_SIZE:
                scale = MAX_IMAGE_SIZE / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize with high-quality resampling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"Original size: {width}x{height} -> Resized to: {new_width}x{new_height}")
            else:
                 print(f"Original size: {width}x{height} (No resize needed)")

            # Export to JPEG in memory
            buffer = io.BytesIO()
            # Convert to RGB if necessary (e.g. for PNGs with alpha)
            if img.mode in ("RGBA", "P"): 
                img = img.convert("RGB")
                
            img.save(buffer, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
    except Exception as e:
        print(f"Warning: Image resizing failed for {path.name}: {e}. Falling back to original.")
        # Fallback to original file
        with path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def classify_image(path: Path) -> dict:
    """Send one image to LLaVA and return a dict with label, reason, filename."""
    img_b64 = encode_image(path)

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json=payload,
                auth=("user", "43tcgH!we9"),
                timeout=180 # 3 minute timeout
            )
            resp.raise_for_status()
            break
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                print(f"  Timeout/Error for {path.name} (Attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                time.sleep(2) # Brief pause before retry
            else:
                print(f"  Failed {path.name} after {max_retries} attempts: {e}")
                return {
                    "label": "ERROR", 
                    "reason": f"Request failed: {str(e)}", 
                    "filename": path.name
                }
    text = resp.json()["response"].strip()

    # --- Try to parse as JSON directly ---
    result = None
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # --- Second attempt: strip to the first {...} block (for ```json ... ``` cases) ---
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                result = None

    # --- Fallback: infer label from text, never UNKNOWN ---
    if result is None:
        upper = text.upper()
        # simple heuristics
        if "GOOD" in upper and "BAD" not in upper:
            label = "GOOD"
        elif "BAD" in upper and "GOOD" not in upper:
            label = "BAD"
        else:
            # if both appear or neither clearly, be slightly optimistic
            label = "GOOD"
        result = {
            "label": label,
            "reason": text,
        }

    # Ensure we always have label + reason keys
    if "label" not in result:
        result["label"] = "GOOD"
    if "reason" not in result:
        result["reason"] = text

    result["filename"] = path.name
    return result


def generate_pdf_report(results: list, images_dir: Path, output_path: Path):
    """Generate a PDF report with images and classification results (2x2 grid)."""
    print(f"Generating PDF report to {output_path} ...")
    
    # Process in batches of 6
    batch_size = 6
    
    with PdfPages(output_path) as pdf:
        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]
            
            # Create a figure with 3x2 subplots
            fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69)) # A4 size in inches
            # Flatten axes for easy iteration
            axes = axes.flatten()
            
            for j, ax in enumerate(axes):
                if j < len(batch):
                    item = batch[j]
                    filename = item["filename"]
                    label = item["label"]
                    reason = item["reason"]
                    
                    img_path = images_dir / filename
                    
                    # --- Content Layout ---
                    ax.axis("off")
                    
                    # 1. Title (Filename + Label)
                    color = "green" if label == "GOOD" else "red"
                    ax.text(0.5, 1.02, f"{filename}\n{label}", 
                            transform=ax.transAxes, ha="center", va="bottom", 
                            fontsize=9, color=color, weight="bold")
                    
                    # 2. Image
                    if img_path.exists():
                        try:
                            img = mpimg.imread(str(img_path))
                            ax.imshow(img)
                        except Exception as e:
                            ax.text(0.5, 0.5, f"Error loading image", transform=ax.transAxes, ha="center")
                    else:
                        ax.text(0.5, 0.5, "Image not found", transform=ax.transAxes, ha="center")

                    # 3. Reason text (below image)
                    import textwrap
                    wrapped_reason = "\n".join(textwrap.wrap(reason, width=45))
                    ax.text(0.5, -0.1, wrapped_reason, transform=ax.transAxes, 
                            ha="center", va="top", fontsize=8, wrap=True)
                            
                else:
                    # Hide unused subplots
                    ax.axis("off")
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
            # Add extra spacing for the text below the axes
            plt.subplots_adjust(wspace=0.2, hspace=0.4)
            
            pdf.savefig(fig)
            plt.close(fig)
            
    print("PDF generation complete.")


def main():
    images_dir = Path("images")
    image_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    if not image_files:
        print("No images found in ./images")
        return

    results = []
    
    results = []
    for img in image_files:
        print(f"Classifying {img.name} ...")
        res = classify_image(img)
        print(json.dumps(res, indent=2))
        results.append(res)
    
    # Save all results to a JSON file
    out_path = Path("results.json")
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Generate PDF report--
    pdf_path = Path("report.pdf")
    generate_pdf_report(results, images_dir, pdf_path)


if __name__ == "__main__":
    main()