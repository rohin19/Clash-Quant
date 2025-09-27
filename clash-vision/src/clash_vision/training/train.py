from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO
from ..utils import logger

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 model for Clash Royale MVP")
    p.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (e.g., yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--img", type=int, default=640, help="Image size")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="", help="CUDA device id or 'cpu'")
    p.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    p.add_argument("--name", type=str, default="clash_mvp", help="Run name")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--resume", action="store_true", help="Resume if run exists")
    return p.parse_args()

def main():
    args = parse_args()
    logger.banner("YOLOv8 Training")
    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model)

    if not Path(args.data).exists():
        logger.error(f"data.yaml not found at {args.data}")
        return 1

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        verbose=True,
    )
    logger.info("Training complete")
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        logger.info(f"Best weights: {best}")
    else:
        logger.warn("Best weights file not found (training may have failed?)")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
