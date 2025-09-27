from __future__ import annotations

import argparse
from pathlib import Path
import cv2
from ..utils import logger

def resize_image(in_path: Path, out_path: Path, size: int):
    img = cv2.imread(str(in_path))
    if img is None:
        logger.warn(f"Failed to read {in_path}")
        return False
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized)
    return True

def main():
    parser = argparse.ArgumentParser(description="Resize YOLO dataset images to square size (labels unchanged).")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Root containing images/ and labels/")
    parser.add_argument("--img-size", type=int, default=640, help="Target square size (e.g., 640)")
    parser.add_argument("--out-suffix", type=str, default="_resized", help="Suffix appended to new images directory name")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma separated splits to process")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        images_dir = root / "images" / split
        if not images_dir.exists():
            logger.warn(f"Images dir missing: {images_dir}")
            continue
        out_dir = images_dir.parent / f"{split}{args.out_suffix}"
        count = 0
        for img_file in images_dir.rglob("*"):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            out_file = out_dir / img_file.relative_to(images_dir)
            if resize_image(img_file, out_file, args.img_size):
                count += 1
        logger.info(f"Split {split}: resized {count} images -> {out_dir}")
    logger.info("Done.")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
