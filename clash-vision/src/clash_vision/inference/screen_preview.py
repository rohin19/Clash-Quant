from __future__ import annotations
"""Screen Preview & Region Selector

Usage:
  python -m clash_vision.inference.screen_preview --monitor 1

iPhone Setup with QuickTime Player:
  1. Connect iPhone to Mac via USB/USB-C cable
  2. Open QuickTime Player
  3. File > New Movie Recording
  4. Click dropdown arrow next to record button
  5. Select your iPhone under "Camera"
  6. Position QuickTime window showing your iPhone screen
  7. Run this script and select the QuickTime window area

Keys:
  r  -> Manual ROI selection (drag a box) and store region
  a  -> Auto-detect iPhone screen and save region (for QuickTime Player)
  s  -> Save current frame to 'captureq_sample.jpg'
  c  -> Clear stored region
  q/ESC -> quit

After selecting ROI, a file 'capture_region.json' is written with the coordinates.
You will use those numbers with live_inference.py via --region LEFT TOP WIDTH HEIGHT.

Notes:
- For iPhone capture: Use QuickTime Player's "New Movie Recording" with iPhone as camera
- The script will automatically position preview window to avoid mirror effects
- Select only the iPhone screen area within QuickTime Player window
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


def detect_iphone_aspect_ratio(frame: np.ndarray) -> dict | None:
    """Try to detect iPhone-like regions within the captured frame"""
    h, w = frame.shape[:2]
    
    # iPhone aspect ratios (approximate)
    iphone_ratios = [
        (19.5, 9),    # iPhone X and newer (2.17:1)
        (16, 9),      # iPhone 6/7/8 Plus (1.78:1)
        (1.78, 1),    # Landscape versions
        (2.17, 1),    # Landscape versions
    ]
    
    # Look for rectangular regions that match iPhone aspect ratios
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get bounding rectangle
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Skip small regions
        if cw < 200 or ch < 300:
            continue
            
        # Check aspect ratio
        ratio = cw / ch
        for target_w, target_h in iphone_ratios:
            target_ratio = target_w / target_h
            if abs(ratio - target_ratio) < 0.1:  # Close enough match
                return {
                    'left': x,
                    'top': y,
                    'width': cw,
                    'height': ch,
                    'detected_ratio': f"{target_w}:{target_h}"
                }
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Screen preview & ROI selection for iPhone via QuickTime")
    parser.add_argument('--monitor', type=int, default=1, help='MSS monitor index (list printed at start)')
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--max-width', type=int, default=1200, help='Resize display width')
    parser.add_argument('--freeze-select', action='store_true', help='Freeze current frame before ROI selection to avoid mirror effect')
    parser.add_argument('--auto-detect-iphone', action='store_true', help='Try to automatically detect iPhone screen region')
    args = parser.parse_args()

    sct = mss()
    monitors = sct.monitors
    logger.info('Available monitors:')
    for idx, m in enumerate(monitors):
        if idx == 0:
            continue  # mss placeholder
        logger.info(f"  {idx}: left={m['left']} top={m['top']} width={m['width']} height={m['height']}")
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
        # Ensure frame is contiguous for OpenCV
        frame = np.ascontiguousarray(frame)
        
        # Auto-detect iPhone region if enabled and no region set
        if args.auto_detect_iphone and region is None:
            detected = detect_iphone_aspect_ratio(frame)
            if detected:
                # Convert to absolute screen coordinates
                detected['left'] += capture['left']
                detected['top'] += capture['top']
                cv2.putText(frame, f"iPhone detected: {detected['detected_ratio']}", (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
                # Draw detection box in cyan
                x1, y1 = detected['left'] - capture['left'], detected['top'] - capture['top']
                x2, y2 = x1 + detected['width'], y1 + detected['height']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
        
        # Draw region overlay
        draw_region(frame, region)
        cv2.putText(frame, f"FPS:{current_fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,0),2,cv2.LINE_AA)
        
        # Add iPhone setup instructions
        if region is None:
            cv2.putText(frame, "QuickTime: File > New Movie Recording > Select iPhone", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press 'r' to select iPhone screen area", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # Resize for display
        h, w = frame.shape[:2]
        if w > args.max_width:
            new_h = int(h * args.max_width / w)
            frame_disp = cv2.resize(frame, (args.max_width, new_h))
        else:
            frame_disp = frame
        # Position window to avoid infinite mirror effect
        cv2.namedWindow('Screen Preview', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Screen Preview', 50, 50)  # Move to top-left corner
        cv2.imshow('Screen Preview', frame_disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('s'):
            cv2.imwrite('capture_sample.jpg', frame)
            logger.info('Saved capture_sample.jpg')
        elif key == ord('a'):  # Auto-detect and apply iPhone region
            detected = detect_iphone_aspect_ratio(frame)
            if detected:
                # Convert to absolute screen coordinates
                detected['left'] += capture['left']
                detected['top'] += capture['top']
                region = detected
                region_file.write_text(json.dumps(region, indent=2), encoding='utf-8')
                logger.info(f"Auto-detected and saved iPhone region: {region}")
            else:
                logger.warn('No iPhone-like region detected. Make sure QuickTime Player is showing your iPhone.')
        elif key == ord('r'):
            # Hide the preview window to avoid mirror effect during ROI selection
            cv2.destroyWindow('Screen Preview')
            # Capture a fresh frame without the preview window
            time.sleep(0.1)  # Brief pause to let window close
            raw_roi = np.array(sct.grab(capture))  # BGRA
            temp = raw_roi[:, :, :3]
            temp = np.ascontiguousarray(temp)
            disp = temp
            if w > args.max_width:
                disp = cv2.resize(temp, (args.max_width, int(h * args.max_width / w)))
            # Use a distinct window name for ROI to reduce self-capture confusion
            roi_window = 'ROI_SELECT'
            cv2.namedWindow(roi_window, cv2.WINDOW_NORMAL)
            cv2.moveWindow(roi_window, 100, 100)  # Position away from typical game area
            cv2.imshow(roi_window, disp)
            r = cv2.selectROI(roi_window, disp, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(roi_window)
            if r and r[2] > 0 and r[3] > 0:
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
