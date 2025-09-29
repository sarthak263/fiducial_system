import argparse
import os
import re
from pathlib import Path
import sys
from typing import List, Tuple

import cv2
import numpy as np


FAMILY_MAP = {
    # OpenCV's aruco module supports these AprilTag dictionaries
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

def parse_id_list(s: str) -> List[int]:
    """Parse a string like '0-5,7,10-12' into a sorted list of unique ints."""
    ids = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            if b < a:
                a, b = b, a
            ids.update(range(a, b + 1))
        else:
            ids.add(int(part))
    return sorted(ids)

def mm_to_px(mm: float, dpi: int) -> int:
    # 1 inch = 25.4 mm; pixels = inches * dpi = (mm / 25.4) * dpi
    return int(round((mm / 25.4) * dpi))

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def gen_single_tag(tag_id: int, dictionary, size: int, border_bits: int = 1, label: bool = False) -> np.ndarray:
    """Generate a single AprilTag image as a numpy array (uint8, 0..255)."""
    img = cv2.aruco.generateImageMarker(dictionary, tag_id, size, borderBits=border_bits)
    if label:
        # Add a label below the tag with the ID and family
        label_text = f"ID {tag_id}"
        # Create a small strip for the label
        strip_h = max(size // 8, 24)
        strip = np.full((strip_h, size), 255, dtype=np.uint8)
        # Put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Scale chosen to fit width reasonably
        scale = max(0.4, size / 800.0)
        thickness = max(1, int(round(scale * 2)))
        (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)
        tx = (size - tw) // 2
        ty = (strip_h + th) // 2
        cv2.putText(strip, label_text, (tx, ty), font, scale, (0,), thickness, lineType=cv2.LINE_AA)
        img = np.vstack([img, strip])
    return img

def save_png(img: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)

def make_sheet(images: List[Tuple[int, np.ndarray]], cols: int, rows: int,
               sheet_w_px: int, sheet_h_px: int,
               margin_px: int, gap_px: int) -> np.ndarray:
    """Compose a grid of images on a white sheet. images: list of (id, image)."""
    sheet = np.full((sheet_h_px, sheet_w_px), 255, dtype=np.uint8)
    # Compute cell sizes
    drawable_w = sheet_w_px - 2 * margin_px - (cols - 1) * gap_px
    drawable_h = sheet_h_px - 2 * margin_px - (rows - 1) * gap_px
    cell_w = drawable_w // cols
    cell_h = drawable_h // rows
    # Resize each tag to fit in the cell while keeping it square
    tag_size = min(cell_w, cell_h)
    x = margin_px
    y = margin_px
    i = 0
    for r in range(rows):
        x = margin_px
        for c in range(cols):
            if i >= len(images):
                break
            tag_id, img = images[i]
            resized = cv2.resize(img, (tag_size, tag_size), interpolation=cv2.INTER_NEAREST)
            h, w = resized.shape[:2]
            # Center inside the cell
            cx = x + (cell_w - w) // 2
            cy = y + (cell_h - h) // 2
            sheet[cy:cy+h, cx:cx+w] = resized
            # Optional: label at bottom of cell (small)
            i += 1
            x += cell_w + gap_px
        y += cell_h + gap_px
    return sheet

def main():
    ap = argparse.ArgumentParser(description="Generate AprilTags as PNGs (and optional printable grids).")
    ap.add_argument("--family", type=str, default="36h11",
                    help="AprilTag family: one of {16h5, 25h9, 36h10, 36h11}")
    ap.add_argument("--ids", type=str, default="0-9",
                    help="Comma-separated list/ranges of IDs (e.g. '0-9,12,20-25').")
    ap.add_argument("--size", type=int, default=512, help="Tag pixel size (square).")
    ap.add_argument("--border_bits", type=int, default=1, help="Border bits for generateImageMarker (default=1).")
    ap.add_argument("--label", action="store_true", help="Add tag ID label under each image.")
    ap.add_argument("--outdir", type=str, default="./apriltags_out", help="Output directory.")
    ap.add_argument("--prefix", type=str, default="", help="Optional filename prefix.")
    ap.add_argument("--sheet", action="store_true", help="Also render a printable grid sheet.")
    ap.add_argument("--cols", type=int, default=5, help="Columns in the sheet grid.")
    ap.add_argument("--rows", type=int, default=4, help="Rows in the sheet grid.")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for sheet sizing.")
    ap.add_argument("--sheet_width_mm", type=float, default=210.0, help="Sheet width in mm (A4 default=210).")
    ap.add_argument("--sheet_height_mm", type=float, default=297.0, help="Sheet height in mm (A4 default=297).")
    ap.add_argument("--margin_mm", type=float, default=10.0, help="Outer margin in mm.")
    ap.add_argument("--gap_mm", type=float, default=5.0, help="Gap between cells in mm.")
    args = ap.parse_args()

    family_key = args.family.strip().lower()
    if family_key not in FAMILY_MAP:
        print(f"[ERROR] Unknown family '{args.family}'. Valid: {', '.join(FAMILY_MAP.keys())}", file=sys.stderr)
        sys.exit(1)

    dictionary = cv2.aruco.getPredefinedDictionary(FAMILY_MAP[family_key])
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    ids = parse_id_list(args.ids)
    if not ids:
        print("[ERROR] No valid IDs provided.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Generating {len(ids)} tag(s) from family {args.family} into '{outdir}' ...")

    generated = []
    for tag_id in ids:
        img = gen_single_tag(tag_id, dictionary, size=args.size, border_bits=args.border_bits, label=args.label)
        fname = f"{args.prefix}apriltag_{args.family}_id{tag_id}.png"
        save_png(img, outdir / fname)
        generated.append((tag_id, img))
        print(f"  - saved {fname}")

    if args.sheet:
        # Convert mm to pixels
        sheet_w_px = mm_to_px(args.sheet_width_mm, args.dpi)
        sheet_h_px = mm_to_px(args.sheet_height_mm, args.dpi)
        margin_px = mm_to_px(args.margin_mm, args.dpi)
        gap_px = mm_to_px(args.gap_mm, args.dpi)
        sheet_img = make_sheet(generated, args.cols, args.rows, sheet_w_px, sheet_h_px, margin_px, gap_px)
        sheet_name = f"{args.prefix}apriltags_sheet_{args.family}_{args.cols}x{args.rows}_{len(generated)}ids_{args.dpi}dpi.png"
        save_png(sheet_img, outdir / sheet_name)
        print(f"[INFO] Sheet saved as {sheet_name}")

    print("[DONE]")

if __name__ == "__main__":
    main()