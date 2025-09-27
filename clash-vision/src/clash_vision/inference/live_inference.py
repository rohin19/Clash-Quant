from __future__ import annotations

import argparse
import time
from ultralytics import YOLO
import numpy as np
import cv2
from mss import mss
from ..utils import logger

def main():
    parser = argparse.ArgumentParser(description="Live desktop capture inference (MSS + OpenCV)")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (MSS indexing)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=800, help="Resize display width (maintain aspect)")
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT","TOP","WIDTH","HEIGHT"), help="Optional region override")
    args = parser.parse_args()

    model = YOLO(args.weights)
    sct = mss()
    monitor = sct.monitors[args.monitor]
    if args.region:
        left, top, w, h = args.region
        capture_region = {"left": left, "top": top, "width": w, "height": h}
    else:
        # default: full monitor
        capture_region = {"left": monitor["left"], "top": monitor["top"], "width": monitor["width"], "height": monitor["height"]}

    logger.info(f"Capture region: {capture_region}")
    delay = 1.0 / args.fps
    while True:
        t0 = time.time()
        frame = np.array(sct.grab(capture_region))[:, :, :3]  # BGRA -> BGR slice
        results = model.predict(frame, conf=args.conf, verbose=False)
        res = results[0]
        annotated = res.plot()  # Ultralytics renders boxes
        h, w = annotated.shape[:2]
        new_w = args.width
        new_h = int(h * new_w / w)
        annotated = cv2.resize(annotated, (new_w, new_h))
        cv2.imshow("Clash Live", annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
        elapsed = time.time() - t0
        sleep_for = delay - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
