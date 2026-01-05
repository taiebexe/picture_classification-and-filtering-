import time
import io
import base64
from pathlib import Path
from PIL import Image

# Constants
MAX_IMAGE_SIZE = 1024

def encode_image(path: Path) -> str:
    """Return base64-encoded contents of an image file, resized to max 1024px."""
    start_time = time.time()
    try:
        with Image.open(path) as img:
            # Calculate new size maintaining aspect ratio
            width, height = img.size
            print(f"Image: {path.name}, Size: {width}x{height}")
            
            if max(width, height) > MAX_IMAGE_SIZE:
                scale = MAX_IMAGE_SIZE / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize with high-quality resampling
                resize_start = time.time()
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resize_end = time.time()
                print(f"  Resize (LANCZOS): {resize_end - resize_start:.4f}s")
                print(f"  Result size: {new_width}x{new_height}")
            else:
                 print(f"  No resize needed")

            # Export to JPEG in memory
            buffer = io.BytesIO()
            # Convert to RGB if necessary (e.g. for PNGs with alpha)
            if img.mode in ("RGBA", "P"): 
                img = img.convert("RGB")
            
            save_start = time.time()
            img.save(buffer, format="JPEG", quality=85, optimize=True)
            save_end = time.time()
            original_size = path.stat().st_size
            new_size = buffer.tell()
            print(f"  Save (optimize=True): {save_end - save_start:.4f}s")
            print(f"  Size change: {original_size} -> {new_size} bytes ({(new_size/original_size)*100:.1f}%)")

            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
    except Exception as e:
        print(f"Error: {e}")
        return ""

def main():
    images_dir = Path("images")
    image_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    
    if not image_files:
        print("No images found.")
        return


    # Test with all images
    for img_path in image_files[:1]: # Run only one for server test
        print(f"--- Processing {img_path.name} ---")
        total_start = time.time()
        b64_img = encode_image(img_path)
        total_end = time.time()
        print(f"  Total local time: {total_end - total_start:.4f}s")
        
        # Test server time
        import requests
        import json
        
        OLLAMA_URL = "http://luke.nt.fh-koeln.de:11434/api/generate"
        MODEL_NAME = "llava:34b"
        PROMPT = "Describe this image in 5 words."
        
        print(f"  Sending request to {OLLAMA_URL}...")
        try:
             req_start = time.time()
             payload = {
                "model": MODEL_NAME,
                "prompt": PROMPT,
                "images": [b64_img],
                "stream": False,
                "options": {"temperature": 0.0}
            }
             resp = requests.post(OLLAMA_URL, json=payload, auth=("user", "43tcgH!we9"), timeout=120)
             resp.raise_for_status()
             req_end = time.time()
             print(f"  Result: {resp.json().get('response', '').strip()}")
             print(f"  Server response time: {req_end - req_start:.4f}s")
             
        except Exception as e:
             print(f"  Server error (Reprocessed): {e}")

        # Test server with RAW image
        print(f"  Sending RAW request to {OLLAMA_URL}...")
        try:
             with img_path.open("rb") as f:
                 raw_b64 = base64.b64encode(f.read()).decode("utf-8")
                 
             req_start = time.time()
             payload = {
                "model": MODEL_NAME,
                "prompt": PROMPT,
                "images": [raw_b64],
                "stream": False,
                "options": {"temperature": 0.0}
            }
             resp = requests.post(OLLAMA_URL, json=payload, auth=("user", "43tcgH!we9"), timeout=120)
             resp.raise_for_status()
             req_end = time.time()
             print(f"  Result: {resp.json().get('response', '').strip()}")
             print(f"  Server response time (RAW): {req_end - req_start:.4f}s")
             
        except Exception as e:
             print(f"  Server error (RAW): {e}")


if __name__ == "__main__":
    main()
