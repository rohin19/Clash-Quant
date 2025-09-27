from __future__ import annotations
"""Screen Preview & Region Selector

Usage (PowerShell):
  python -m clash_vision.inference.screen_preview --monitor 1

Keys:
  r  -> enter ROI selection (drag a box) and store region
  s  -> save current frame to 'capture_sample.jpg'
  c  -> clear stored region
  q/ESC -> quit

After selecting ROI, a file 'capture_region.json' is written with the coordinates.
You will use those numbers with live_inference.py via --region LEFT TOP WIDTH HEIGHT.

Notes:
- scrcpy or your mirroring tool should already have the game visible on screen.
- If using Windows scaling, coordinates are still consistent for MSS.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from mss import mss
from ..utils import logger

def draw_region(frame: np.ndarray, region: dict[str,int] | None):
    if not region:
        return frame
    x1 = region['left']
    y1 = region['top']
    x2 = x1 + region['width']
    y2 = y1 + region['height']
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    cv2.putText(frame, f"REGION: {x1},{y1},{region['width']}x{region['height']}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2,cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Screen preview & ROI selection")
    parser.add_argument('--monitor', type=int, default=1, help='MSS monitor index')
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--max-width', type=int, default=1200, help='Resize display width')
    args = parser.parse_args()

    sct = mss()
    monitors = sct.monitors
    if args.monitor >= len(monitors):
        logger.error(f"Monitor index {args.monitor} out of range (have {len(monitors)-1} monitors excluding index 0 placeholder)")
        return 1
    mon = monitors[args.monitor]
    capture = {'left': mon['left'], 'top': mon['top'], 'width': mon['width'], 'height': mon['height']}
    logger.info(f"Capturing monitor {args.monitor}: {capture}")

    region: dict[str,int] | None = None
    region_file = Path('capture_region.json')
    delay = 1.0 / args.fps

    last_fps_time = time.time()
    frame_count = 0
    current_fps = 0.0

    while True:
        t0 = time.time()
        raw = np.array(sct.grab(capture))  # BGRA
        frame = raw[:, :, :3]
        frame_count += 1
        # FPS calc
        if frame_count % 10 == 0:
            now = time.time()
            current_fps = 10 / (now - last_fps_time)
            last_fps_time = now
        # Draw region overlay
        draw_region(frame, region)
        cv2.putText(frame, f"FPS:{current_fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,0),2,cv2.LINE_AA)
        # Resize for display
        h, w = frame.shape[:2]
        if w > args.max_width:
            new_h = int(h * args.max_width / w)
            frame_disp = cv2.resize(frame, (args.max_width, new_h))
        else:
            frame_disp = frame
        cv2.imshow('Screen Preview', frame_disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('s'):
            cv2.imwrite('capture_sample.jpg', frame)
            logger.info('Saved capture_sample.jpg')
        elif key == ord('r'):
            # Select ROI on the displayed (possibly resized) frame -> map back to original coordinates
            temp = frame.copy()
            disp = temp
            if w > args.max_width:
                disp = cv2.resize(temp, (args.max_width, int(h * args.max_width / w)))
            clone = disp.copy()
            r = cv2.selectROI('Screen Preview', clone, fromCenter=False, showCrosshair=True)
            if r and r[2] > 0 and r[3] > 0:
                # Map back
                scale = w / disp.shape[1]
                x, y, rw, rh = r
                x = int(x * scale)
                y = int(y * scale)
                rw = int(rw * scale)
                rh = int(rh * scale)
                region = {'left': capture['left'] + x, 'top': capture['top'] + y, 'width': rw, 'height': rh}
                region_file.write_text(json.dumps(region, indent=2), encoding='utf-8')
                logger.info(f"Saved region to {region_file}: {region}")
            else:
                logger.warn('ROI selection cancelled or invalid')
        elif key == ord('c'):
            region = None
            if region_file.exists():
                region_file.unlink()
            logger.info('Cleared region & deleted capture_region.json if existed')
        # maintain fps
        elapsed = time.time() - t0
        if delay - elapsed > 0:
            time.sleep(delay - elapsed)

    cv2.destroyAllWindows()
    logger.info('Exited screen preview.')
    return 0

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
