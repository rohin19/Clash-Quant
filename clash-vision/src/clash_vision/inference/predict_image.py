from __future__ import annotations

import argparse
from pathlib import Path
import json
from ultralytics import YOLO
import cv2
from ..utils import logger

def main():
    parser = argparse.ArgumentParser(description="Single image inference with YOLOv8")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="Path to image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display annotated window")
    parser.add_argument("--save", action="store_true", help="Save annotated image next to source")
    parser.add_argument("--json", action="store_true", help="Print JSON results to stdout")
    args = parser.parse_args()

    model = YOLO(args.weights)
    logger.info(f"Running inference on {args.source}")
    results = model(args.source, conf=args.conf)
    res = results[0]
    names = res.names
    output = []
    for box, cls, conf in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.cls.cpu().tolist(), res.boxes.conf.cpu().tolist()):
        x1, y1, x2, y2 = box
        output.append({
            "class_id": int(cls),
            "class_name": names[int(cls)],
            "confidence": float(conf),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]
        })

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        logger.info(f"Detections: {output}")

    if args.show or args.save:
        img = cv2.imread(args.source)
        for det in output:
            x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        if args.save:
            out_path = Path(args.source).with_suffix(".pred.jpg")
            cv2.imwrite(str(out_path), img)
            logger.info(f"Saved annotated image to {out_path}")
        if args.show:
            cv2.imshow("pred", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
