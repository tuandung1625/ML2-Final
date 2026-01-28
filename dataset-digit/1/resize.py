import os
from pathlib import Path
from PIL import Image, ImageOps

# ----- CONFIG -----
TARGET_SIZE = (128, 128)

INPUT_DIRS = [
    r"C:\Users\Admin\OneDrive\Desktop\1\test",
    r"C:\Users\Admin\OneDrive\Desktop\1\train",
    r"C:\Users\Admin\OneDrive\Desktop\1\validate",
]

OUTPUT_DIRS = [
    r"C:\Users\Admin\OneDrive\Desktop\1\test_resize",
    r"C:\Users\Admin\OneDrive\Desktop\1\train_resize",
    r"C:\Users\Admin\OneDrive\Desktop\1\train_resize",  # <-- fix below
]

# Fix output list (typo-safe)
OUTPUT_DIRS = [
    r"C:\Users\Admin\OneDrive\Desktop\1\test_resize",
    r"C:\Users\Admin\OneDrive\Desktop\1\train_resize",
    r"C:\Users\Admin\OneDrive\Desktop\1\validate_resize",
]

# Common image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Choose resize mode:
# - "stretch": force exactly 128x128 (may distort)
# - "pad": keep aspect ratio, pad to 128x128
RESIZE_MODE = "stretch"


def process_folder(input_dir: str, output_dir: str) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    count_ok = 0
    count_skip = 0
    count_err = 0

    for root, _, files in os.walk(in_path):
        root_path = Path(root)

        # Preserve subfolder structure (optional; keeps dataset structure)
        rel = root_path.relative_to(in_path)
        (out_path / rel).mkdir(parents=True, exist_ok=True)

        for fname in files:
            src = root_path / fname
            ext = src.suffix.lower()

            if ext not in IMAGE_EXTS:
                count_skip += 1
                continue

            dst = out_path / rel / fname

            try:
                with Image.open(src) as im:
                    # Respect EXIF orientation (phone photos, etc.)
                    im = ImageOps.exif_transpose(im)

                    if RESIZE_MODE == "pad":
                        # Keep aspect ratio, pad to exact size
                        im_resized = ImageOps.pad(im, TARGET_SIZE, method=Image.Resampling.LANCZOS)
                    else:
                        # Force exact size (can distort aspect ratio)
                        im_resized = im.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

                    # Save (try to keep format; fallback to PNG if needed)
                    im_resized.save(dst)
                    count_ok += 1
            except Exception as e:
                count_err += 1
                print(f"[ERROR] {src} -> {dst}: {e}")

    print(f"\nDone: {input_dir}")
    print(f"  Saved: {count_ok}")
    print(f"  Skipped (non-image): {count_skip}")
    print(f"  Errors: {count_err}")


def main():
    if len(INPUT_DIRS) != len(OUTPUT_DIRS):
        raise ValueError("INPUT_DIRS and OUTPUT_DIRS must have the same length.")

    for in_dir, out_dir in zip(INPUT_DIRS, OUTPUT_DIRS):
        process_folder(in_dir, out_dir)


if __name__ == "__main__":
    main()
