from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from mss import mss

from ..utils import logger


def letterbox(img: np.ndarray, out_w: int, out_h: int, color=(0, 0, 0)) -> tuple[np.ndarray, float, int, int]:
    """Resize with unchanged aspect ratio using padding (black bars).

    Returns: (canvas, scale, pad_x, pad_y)
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image in letterbox")
    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), color, dtype=np.uint8)
    x = (out_w - new_w) // 2
    y = (out_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas, scale, x, y


def cover(img: np.ndarray, out_w: int, out_h: int) -> tuple[np.ndarray, float, int, int]:
    """Resize to completely fill output, cropping overflow (center-crop). Returns (frame, scale, crop_x, crop_y).

    This preserves aspect ratio but removes black bars by cropping the excess.
    crop_x, crop_y are the top-left offsets in the scaled image used for the crop.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image in cover")
    scale = max(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # center-crop to out size
    x = max(0, (new_w - out_w) // 2)
    y = max(0, (new_h - out_h) // 2)
    cropped = scaled[y : y + out_h, x : x + out_w]
    # Safety: if rounding made it off by one
    if cropped.shape[0] != out_h or cropped.shape[1] != out_w:
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return cropped, scale, x, y


def compute_letterbox_padding(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[float, int, int]:
    """Compute scale and padding (pad_x, pad_y) for letterbox without actually resizing."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid source dimensions")
    scale = min(out_w / src_w, out_h / src_h)
    draw_w = int(round(src_w * scale))
    draw_h = int(round(src_h * scale))
    pad_x = (out_w - draw_w) // 2
    pad_y = (out_h - draw_h) // 2
    return scale, pad_x, pad_y


def load_capture_region(monitor: dict, cli_region: list[int] | None, region_file: Path) -> dict:
    if cli_region:
        left, top, w, h = cli_region
        return {"left": int(left), "top": int(top), "width": int(w), "height": int(h)}
    if region_file.exists():
        try:
            data = json.loads(region_file.read_text())
            required = {"left", "top", "width", "height"}
            if not required.issubset(data):
                raise ValueError("capture_region.json missing required keys")
            return {k: int(data[k]) for k in ("left", "top", "width", "height")}
        except Exception as e:
            logger.error(f"Failed to load region file {region_file}: {e}")
    # Fallback to full monitor
    logger.warn("No region specified; falling back to full monitor")
    return {"left": monitor["left"], "top": monitor["top"], "width": monitor["width"], "height": monitor["height"]}


def post_frame(api_url: str, frame_bgr: np.ndarray, send_format: str, metadata: dict) -> tuple[bool, int | None, str | None]:
    try:
        if send_format == "jpeg":
            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return False, None, "JPEG encode failed"
            files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            data = {"metadata": json.dumps(metadata)}
            r = requests.post(api_url, files=files, data=data, timeout=5)
        else:  # base64-json
            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return False, None, "JPEG encode failed"
            payload = {
                "frame": base64.b64encode(buf).decode("utf-8"),
                "metadata": metadata,
            }
            r = requests.post(api_url, json=payload, timeout=5)
        return (200 <= r.status_code < 300), r.status_code, None
    except Exception as e:
        return False, None, str(e)


def main() -> int:
    p = argparse.ArgumentParser(description="Capture iPhone via QuickTime region, letterbox to rectangle, and POST to model API")
    p.add_argument("--api-url", type=str, required=True, help="Model API endpoint to POST frames to")
    p.add_argument("--monitor", type=int, default=1, help="MSS monitor index where QuickTime window is displayed")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--out-size", type=int, nargs=2, metavar=("W", "H"), default=(1280, 720), help="Output rectangle size (W H)")
    p.add_argument("--fit-mode", choices=["letterbox", "cover", "stretch"], default="letterbox", help="How to fit source into output rectangle")
    p.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"), help="Override capture region")
    p.add_argument("--region-file", type=str, default="capture_region.json", help="Path to JSON region file saved by screen_preview")
    p.add_argument("--send-format", choices=["jpeg", "base64-json"], default="jpeg", help="How to send frames to API")
    p.add_argument("--dry-run", action="store_true", help="Do not call API; preview only")
    p.add_argument("--preview", action="store_true", help="Show a preview window of the letterboxed output")
    p.add_argument("--show-fps", action="store_true", help="Overlay FPS on output frames")
    p.add_argument("--debug-info", action="store_true", help="Overlay letterbox debug info (scale, pad, content box)")
    p.add_argument("--suggest-sizes", action="store_true", help="Print recommended output sizes and expected padding for letterbox and exit")
    args = p.parse_args()

    sct = mss()
    monitors = sct.monitors
    if args.monitor >= len(monitors):
        logger.error(f"Monitor index {args.monitor} out of range (have {len(monitors)-1} monitors excluding index 0 placeholder)")
        return 1
    mon = monitors[args.monitor]

    capture_region = load_capture_region(mon, args.region, Path(args.region_file))
    logger.info(f"Using capture region: {capture_region}")

    out_w, out_h = args.out_size

    # Optional: suggest output sizes based on current capture aspect
    if args.suggest_sizes:
        # Get a quick frame to know source size
        raw = np.array(sct.grab(capture_region))
        src_h, src_w = raw.shape[0], raw.shape[1]
        logger.banner("Letterbox size suggestions")
        candidates = [
            (640, 640), (800, 800), (1024, 1024),
            (1280, 720), (1920, 1080), (960, 540),
            (1280, 960), (1920, 1440)
        ]
        for ow, oh in candidates:
            try:
                scale, px, py = compute_letterbox_padding(src_w, src_h, ow, oh)
                logger.info(f"{ow}x{oh} -> scale={scale:.3f} pad=({px},{py}) content={int(src_w*scale)}x{int(src_h*scale)}")
            except Exception as e:
                logger.warn(f"{ow}x{oh} -> error: {e}")
        logger.info("Pick the size with desired aspect/black bars. Then re-run without --suggest-sizes.")
        return 0
    delay = 1.0 / args.fps

    last_fps_time = time.time()
    frame_count = 0
    current_fps = 0.0

    if args.preview:
        cv2.namedWindow("Letterboxed Output", cv2.WINDOW_NORMAL)
        # Place preview in top-left to reduce mirror effect; user can move it to another display if available
        cv2.moveWindow("Letterboxed Output", 50, 50)

    try:
        while True:
            t0 = time.time()
            # Capture frame from QuickTime window region
            raw = np.array(sct.grab(capture_region))  # BGRA
            frame = raw[:, :, :3]
            frame = np.ascontiguousarray(frame)

            # Fit source to target rectangle according to mode
            if args.fit_mode == "letterbox":
                out_frame, scale, pad_x, pad_y = letterbox(frame, out_w, out_h, color=(0, 0, 0))
                if args.debug_info:
                    src_h, src_w = frame.shape[:2]
                    draw_w = max(1, int(round(src_w * scale)))
                    draw_h = max(1, int(round(src_h * scale)))
                    top_left = (pad_x, pad_y)
                    bottom_right = (pad_x + draw_w, pad_y + draw_h)
                    cv2.rectangle(out_frame, top_left, bottom_right, (0, 255, 255), 2)
                    cv2.putText(out_frame, f"MODE:letterbox OUT:{out_w}x{out_h} SCALE:{scale:.3f} PAD:{pad_x},{pad_y}", (10, out_h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            elif args.fit_mode == "cover":
                out_frame, scale, crop_x, crop_y = cover(frame, out_w, out_h)
                if args.debug_info:
                    # Draw a center crosshair to visualize cropping reference
                    cv2.drawMarker(out_frame, (out_w // 2, out_h // 2), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    cv2.putText(out_frame, f"MODE:cover OUT:{out_w}x{out_h} SCALE:{scale:.3f} CROP:{crop_x},{crop_y}", (10, out_h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            else:  # stretch (no aspect preservation)
                out_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                scale = -1.0
                if args.debug_info:
                    cv2.putText(out_frame, f"MODE:stretch OUT:{out_w}x{out_h}", (10, out_h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

            # FPS overlay
            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                current_fps = 10 / (now - last_fps_time)
                last_fps_time = now
            if args.show_fps:
                cv2.putText(out_frame, f"FPS:{current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Optional preview
            if args.preview:
                cv2.imshow("Letterboxed Output", out_frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("s"):
                    cv2.imwrite("letterbox_sample.jpg", out_frame)
                    logger.info("Saved letterbox_sample.jpg")

            # Send to API
            if not args.dry_run:
                meta = {
                    "ts": time.time(),
                    "orig_shape": list(frame.shape),
                    "out_size": [out_w, out_h],
                    "scale": scale,
                    "pad": [pad_x, pad_y],
                    "region": capture_region,
                }
                ok, status, err = post_frame(args.api_url, out_frame, args.send_format, meta)
                if not ok:
                    if err:
                        logger.warn(f"API post failed: {err}")
                    else:
                        logger.warn(f"API returned status {status}")

            # Maintain FPS
            elapsed = time.time() - t0
            sleep_for = delay - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        if args.preview:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())