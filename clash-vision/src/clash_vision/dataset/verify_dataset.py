from __future__ import annotations

import argparse
import yaml
from pathlib import Path
from collections import Counter
from typing import List
from ..utils import logger

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_label_file(path: Path) -> List[tuple[int, float, float, float, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                logger.warn(f"Bad line format in {path.name}: {line.strip()}")
                continue
            try:
                c = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                rows.append((c, x, y, w, h))
            except ValueError:
                logger.warn(f"Non-numeric value in {path.name}: {line.strip()}")
    return rows

def validate_box(box: tuple[int, float, float, float, float]) -> bool:
    _, x, y, w, h = box
    return all(0.0 <= v <= 1.0 for v in (x, y, w, h)) and w > 0 and h > 0

def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 dataset structure and labels.")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Root folder containing data.yaml")
    parser.add_argument("--show-missing", action="store_true", help="List missing files individually")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        logger.error(f"data.yaml not found at {data_yaml}")
        return 1

    cfg = load_yaml(data_yaml)
    names = cfg.get("names") or cfg.get("classes")
    if not names:
        logger.error("'names' field missing in data.yaml")
        return 1
    logger.info(f"Loaded {len(names)} classes: {names}")

    splits = {"train": cfg.get("train"), "val": cfg.get("val"), "test": cfg.get("test")}
    for split, value in splits.items():
        if value is None:
            logger.warn(f"Split '{split}' missing in data.yaml")

    image_exts = {".jpg", ".jpeg", ".png"}
    class_counts = Counter()
    issues = 0
    for split, split_path in splits.items():
        if not split_path:
            continue
        # Allow relative inside dataset root
        images_dir = (root / split_path) if not split_path.startswith(str(root)) else Path(split_path)
        labels_dir = images_dir.parent.parent / "labels" / split  # typical Roboflow export structure
        if not images_dir.exists():
            logger.error(f"Images dir missing for {split}: {images_dir}")
            issues += 1
            continue
        if not labels_dir.exists():
            logger.error(f"Labels dir missing for {split}: {labels_dir}")
            issues += 1
            continue
        img_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts]
        logger.info(f"Split {split}: {len(img_files)} images")
        missing_label = []
        for img in img_files:
            rel_name = img.stem + ".txt"
            label_file = labels_dir / rel_name
            if not label_file.exists():
                missing_label.append(img)
                continue
            rows = read_label_file(label_file)
            seen_boxes = set()
            for row in rows:
                cls, *coords = row
                if cls < 0 or cls >= len(names):
                    logger.warn(f"Class index out of range ({cls}) in {label_file}")
                    issues += 1
                if not validate_box(row):
                    logger.warn(f"Invalid box values in {label_file}: {row}")
                    issues += 1
                key = tuple(row)
                if key in seen_boxes:
                    logger.warn(f"Duplicate box in {label_file}: {row}")
                else:
                    seen_boxes.add(key)
                class_counts[cls] += 1
        if missing_label:
            msg = f"{len(missing_label)} images missing labels in split {split}"
            if args.show_missing:
                logger.warn(msg + ":" )
                for p in missing_label[:25]:  # limit spam
                    logger.warn(f"  - {p}")
            else:
                logger.warn(msg + " (re-run with --show-missing to list)")

    if class_counts:
        logger.banner("Class Distribution")
        name_counts = {names[c]: n for c, n in sorted(class_counts.items())}
        logger.table_dict("Counts", name_counts)
        max_c = max(class_counts.values())
        min_c = min(class_counts.values())
        if max_c / max(1, min_c) > 3:
            logger.warn("Significant class imbalance detected (max/min > 3x)")

    if issues == 0:
        logger.info("Dataset validation passed with no critical issues.")
    else:
        logger.warn(f"Dataset validation completed with {issues} issues.")

    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
