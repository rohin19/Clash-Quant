from __future__ import annotations

import argparse
import json
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import numpy as np
import cv2
from mss import mss
from ..utils import logger

class AsyncInferenceManager:
    """Manages asynchronous API calls to Roboflow with result caching and throttling."""
    
    def __init__(self, client, model_id, api_interval=5.0, max_queue_size=2):
        self.client = client
        self.model_id = model_id
        self.api_interval = api_interval  # seconds between API calls
        self.max_queue_size = max_queue_size
        
        # Queues for managing async inference
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue()
        
        # Latest results for display
        self.latest_results = None
        self.results_lock = threading.Lock()
        
        # Worker thread
        self.worker_thread = None
        self.should_stop = False
        
        # Throttling
        self.last_api_call = 0
        self.frames_skipped = 0
        
        # Statistics
        self.frames_sent = 0
        self.frames_processed = 0
        self.api_errors = 0
        
    def start_worker(self):
        """Start the background worker thread for API calls."""
        self.should_stop = False
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Started async inference worker")
        
    def stop_worker(self):
        """Stop the background worker thread."""
        self.should_stop = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            
    def submit_frame(self, frame):
        """Submit a frame for inference (non-blocking) with throttling."""
        current_time = time.time()
        
        # Check if enough time has passed since last API call
        if current_time - self.last_api_call < self.api_interval:
            self.frames_skipped += 1
            return False  # Frame skipped due to throttling
            
        # Clear queue if full (keep only the most recent frame)
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
                
        try:
            # Convert BGR to RGB for Roboflow API (it expects RGB numpy arrays)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_queue.put((current_time, frame_rgb), block=False)
            self.frames_sent += 1
            return True  # Frame submitted
        except Exception as e:
            logger.warning(f"Failed to submit frame: {e}")
            return False
            
    def get_latest_results(self):
        """Get the most recent inference results."""
        with self.results_lock:
            return self.latest_results
            
    def _worker_loop(self):
        """Background worker loop for processing API calls."""
        while not self.should_stop:
            try:
                # Get frame from queue (blocking with timeout)
                timestamp, frame_rgb = self.frame_queue.get(timeout=0.1)
                
                # Update last API call timestamp
                self.last_api_call = timestamp
                
                # Make API call with numpy array (RGB format)
                try:
                    result = self.client.infer(frame_rgb, model_id=self.model_id)
                    
                    # Update latest results
                    with self.results_lock:
                        self.latest_results = {
                            'predictions': result.get('predictions', []),
                            'timestamp': timestamp,
                            'error': None
                        }
                    
                    self.frames_processed += 1
                    logger.info(f"API call successful. Processed frame at {timestamp:.2f}")
                    
                except Exception as e:
                    logger.error(f"API inference failed: {e}")
                    self.api_errors += 1
                    
                    # Update with error state
                    with self.results_lock:
                        self.latest_results = {
                            'predictions': [],
                            'timestamp': timestamp,
                            'error': str(e)
                        }
                        
                self.frame_queue.task_done()
                
            except Empty:
                # Timeout - continue loop
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                
    def get_stats(self):
        """Get inference statistics."""
        return {
            'frames_sent': self.frames_sent,
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'api_errors': self.api_errors,
            'queue_size': self.frame_queue.qsize(),
            'success_rate': self.frames_processed / max(1, self.frames_sent),
            'api_interval': self.api_interval,
            'next_api_call': max(0, self.api_interval - (time.time() - self.last_api_call))
        }

def draw_detections(frame, predictions):
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()
    
    if predictions:
        for prediction in predictions:
            # Extract bounding box coordinates
            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            w = int(prediction['width'])
            h = int(prediction['height'])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{prediction['class']}: {prediction['confidence']:.2f}"
            cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return annotated

def draw_stats_overlay(frame, stats, fps_actual):
    """Draw performance statistics on frame."""
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White text
    thickness = 1
    
    # Background rectangle for better text visibility
    cv2.rectangle(frame, (10, 5), (450, 160), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 5), (450, 160), (255, 255, 255), 1)
    
    # Display stats
    texts = [
        f"Display FPS: {fps_actual:.1f}",
        f"API Interval: {stats['api_interval']:.1f}s",
        f"Next API Call: {stats['next_api_call']:.1f}s",
        f"Frames Sent: {stats['frames_sent']}",
        f"Frames Skipped: {stats['frames_skipped']}",
        f"Processed: {stats['frames_processed']}",
        f"Success Rate: {stats['success_rate']:.1%}"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (15, y_offset + i * 18), font, font_scale, color, thickness)

def main():
    parser = argparse.ArgumentParser(description="Live desktop capture inference (Roboflow API + OpenCV)")
    parser.add_argument("--model_id", type=str, default="clashquant-nvnzk/3", help="Roboflow model ID")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (MSS indexing)")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=800, help="Resize display width (maintain aspect)")
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT","TOP","WIDTH","HEIGHT"), help="Optional region override")
    parser.add_argument("--api_interval", type=float, default=5.0, help="Seconds between API calls (default: 5.0)")
    args = parser.parse_args()
    
    # Easy to change variable for API call frequency
    API_CALL_INTERVAL = args.api_interval  # seconds between API calls

    # Initialize Roboflow client
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="nfRaKWeyDCMxK2jw6k5a"
    )
    
    # Define custom configuration for confidence and other parameters
    custom_config = InferenceConfiguration(
        confidence_threshold=args.conf,  # Use confidence from command line args
        iou_threshold=0.5,               # IoU threshold for non-maximum suppression
        max_detections=100               # Maximum number of detections
    )
    
    sct = mss()
    monitor = sct.monitors[args.monitor]
    
    # Check for region in order of priority: command line -> JSON file -> full monitor
    if args.region:
        left, top, w, h = args.region
        capture_region = {"left": left, "top": top, "width": w, "height": h}
        logger.info("Using region from command line arguments")
    else:
        # Try to load from capture_region.json (created by screen_preview.py)
        region_file = Path('capture_region.json')
        if region_file.exists():
            try:
                with open(region_file, 'r') as f:
                    capture_region = json.load(f)
                logger.info("Using region from capture_region.json (iPhone QuickTime capture)")
            except Exception as e:
                logger.error(f"Failed to load capture_region.json: {e}")
                capture_region = {"left": monitor["left"], "top": monitor["top"], "width": monitor["width"], "height": monitor["height"]}
                logger.info("Falling back to full monitor")
        else:
            # default: full monitor
            capture_region = {"left": monitor["left"], "top": monitor["top"], "width": monitor["width"], "height": monitor["height"]}
            logger.info("No region file found, using full monitor. Use screen_preview.py to select iPhone region first.")

    logger.info(f"Capture region: {capture_region}")
    delay = 1.0 / args.fps
    
    # Initialize async inference manager with throttling
    with CLIENT.use_configuration(custom_config):
        inference_manager = AsyncInferenceManager(
            CLIENT, 
            args.model_id, 
            api_interval=API_CALL_INTERVAL,
            max_queue_size=2
        )
        inference_manager.start_worker()
        
        # FPS tracking
        fps_counter = 0
        fps_start_time = time.time()
        fps_actual = 0.0
        
        logger.info(f"Started live inference with {API_CALL_INTERVAL}s API interval")
        logger.info("Press ESC to stop")
        
        try:
            while True:
                t0 = time.time()
                frame = np.array(sct.grab(capture_region))[:, :, :3]  # BGRA -> BGR slice
                
                # Submit frame for async inference (with throttling)
                frame_submitted = inference_manager.submit_frame(frame)
                
                # Get latest results (if any)
                results = inference_manager.get_latest_results()
                
                # Draw detections from latest results
                if results:
                    if results['error']:
                        # Show API error
                        annotated = frame.copy()
                        cv2.putText(annotated, f"API Error: {results['error'][:50]}", 
                                  (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # Draw detections
                        annotated = draw_detections(frame, results['predictions'])
                        # Show detection count
                        detection_count = len(results['predictions'])
                        cv2.putText(annotated, f"Detections: {detection_count}", 
                                  (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # No results yet
                    annotated = frame.copy()
                    cv2.putText(annotated, "Waiting for first API response...", 
                              (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show frame submission status
                status_color = (0, 255, 0) if frame_submitted else (128, 128, 128)
                status_text = "Frame sent to API" if frame_submitted else "Frame throttled"
                cv2.putText(annotated, status_text, (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter >= 30:  # Update FPS every 30 frames
                    elapsed_fps = time.time() - fps_start_time
                    fps_actual = fps_counter / elapsed_fps
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Draw statistics overlay
                stats = inference_manager.get_stats()
                draw_stats_overlay(annotated, stats, fps_actual)
                
                # Resize for display
                h, w = annotated.shape[:2]
                new_w = args.width
                new_h = int(h * new_w / w)
                annotated = cv2.resize(annotated, (new_w, new_h))
                
                cv2.imshow("Clash Live", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
                    
                # Maintain target FPS
                elapsed = time.time() - t0
                sleep_for = delay - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                    
        finally:
            # Cleanup
            inference_manager.stop_worker()
            
    cv2.destroyAllWindows()
    
    # Print final statistics
    final_stats = inference_manager.get_stats()
    logger.info(f"Final Statistics:")
    logger.info(f"  API call interval: {final_stats['api_interval']:.1f}s")
    logger.info(f"  Frames sent to API: {final_stats['frames_sent']}")
    logger.info(f"  Frames skipped (throttled): {final_stats['frames_skipped']}")
    logger.info(f"  Frames processed: {final_stats['frames_processed']}")
    logger.info(f"  API errors: {final_stats['api_errors']}")
    logger.info(f"  Success rate: {final_stats['success_rate']:.1%}")
    
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
