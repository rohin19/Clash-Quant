from __future__ import annotations

import argparse
import json
import time
import threading
import tempfile
import os
import sys
from queue import Queue, Empty
from pathlib import Path
from roboflow import Roboflow
import numpy as np
import cv2
from mss import mss
from ..utils import logger

# Import GameState from the parent directory
sys.path.append(str(Path(__file__).parent.parent.parent))
from gamestate import GameState

class AsyncInferenceManager:
    """Manages asynchronous API calls to Roboflow with aggressive queue clearing and game state tracking."""
    
    def __init__(self, model, api_interval=5.0, confidence=40, debug=False, enable_gamestate=True):
        self.model = model  # Roboflow model object
        self.api_interval = api_interval  # seconds between API calls
        self.confidence = confidence
        self.debug = debug
        self.enable_gamestate = enable_gamestate
        
        # Queues for managing async inference (small queue)
        self.frame_queue = Queue(maxsize=1)  # Only keep 1 frame max
        
        # Latest results for display
        self.latest_results = None
        self.results_lock = threading.Lock()
        
        # Worker thread
        self.worker_thread = None
        self.should_stop = False
        
        # Throttling
        self.last_api_call = 0
        
        # Debug counter for saving frames
        self.debug_frame_counter = 0
        
        # GameState integration
        self.game_state = GameState() if enable_gamestate else None
        self.gamestate_lock = threading.Lock()
        
        # Detection statistics
        self.detection_history = {}  # timestamp -> detection data
        self.max_history = 100
        
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
        """Submit a frame for inference with aggressive queue clearing."""
        current_time = time.time()
        
        # Check if enough time has passed since last API call
        if current_time - self.last_api_call < self.api_interval:
            return False  # Frame skipped due to throttling
            
        # AGGRESSIVE: Empty entire queue first to prevent backup
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
                
        try:
            # Convert BGR to RGB for Roboflow API (it expects RGB numpy arrays)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_queue.put((current_time, frame_rgb), block=False)
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
                
                # Make API call - convert numpy array to image file path or base64
                try:
                    # DEBUG: Print image info before sending
                    if self.debug:
                        logger.info(f"Sending image: shape={frame_rgb.shape}, dtype={frame_rgb.dtype}, "
                                   f"min={frame_rgb.min()}, max={frame_rgb.max()}")
                    
                    # Convert numpy array to image file path (Roboflow prefers file paths)
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        # Convert RGB back to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(tmp_file.name, frame_bgr)
                        temp_path = tmp_file.name
                    
                    # Send file path to Roboflow API
                    result = self.model.predict(temp_path, confidence=self.confidence, overlap=30).json()
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    predictions = result.get('predictions', [])
                    
                    # DEBUG: Save frame if debugging enabled
                    if self.debug and self.debug_frame_counter < 5:  # Save first 5 frames
                        debug_dir = Path("debug_frames")
                        debug_dir.mkdir(exist_ok=True)
                        
                        # Convert RGB back to BGR for saving
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        frame_path = debug_dir / f"frame_{self.debug_frame_counter:03d}_{timestamp:.2f}.jpg"
                        cv2.imwrite(str(frame_path), frame_bgr)
                        
                        # Save JSON result too
                        json_path = debug_dir / f"result_{self.debug_frame_counter:03d}_{timestamp:.2f}.json"
                        with open(json_path, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        self.debug_frame_counter += 1
                        logger.info(f"DEBUG: Saved frame {frame_path} and result {json_path}")
                    
                    # Process detections with GameState (filter by confidence >= 0.5)
                    if self.enable_gamestate and self.game_state:
                        high_conf_predictions = [pred for pred in predictions if pred.get('confidence', 0) >= 0.5]
                        
                        if high_conf_predictions:
                            # Convert to GameState format
                            gamestate_detections = []
                            for pred in high_conf_predictions:
                                # Convert center x,y,w,h to x1,y1,x2,y2 bbox format
                                center_x, center_y = pred['x'], pred['y']
                                w, h = pred['width'], pred['height']
                                x1 = center_x - w/2
                                y1 = center_y - h/2
                                x2 = center_x + w/2
                                y2 = center_y + h/2
                                
                                gamestate_detections.append({
                                    'card': pred['class'],  # GameState expects 'card' key
                                    'bbox': [x1, y1, x2, y2],  # GameState expects x1,y1,x2,y2 format
                                    'confidence': pred['confidence']
                                })
                            
                            # Feed to GameState
                            with self.gamestate_lock:
                                self.game_state.ingest_detections(gamestate_detections, frame_ts=timestamp)
                            
                            if self.debug or len(high_conf_predictions) > 0:
                                logger.info(f"GameState updated with {len(high_conf_predictions)} high-confidence detections")
                    
                    # Store detection history
                    self.detection_history[timestamp] = {
                        'predictions': predictions,
                        'count': len(predictions),
                        'high_conf_count': len([p for p in predictions if p.get('confidence', 0) >= 0.5])
                    }
                    
                    # Maintain history size
                    if len(self.detection_history) > self.max_history:
                        oldest_ts = min(self.detection_history.keys())
                        del self.detection_history[oldest_ts]
                    
                    # DEBUG: Print detection details
                    if self.debug or len(predictions) > 0:  # Always print if detections found
                        logger.info(f"API Success - Frame: {timestamp:.2f}, Detections: {len(predictions)}")
                        for i, pred in enumerate(predictions):
                            conf_marker = "🎯" if pred['confidence'] >= 0.5 else "  "
                            logger.info(f"  {conf_marker}Detection {i+1}: class='{pred['class']}', conf={pred['confidence']:.3f}, "
                                      f"bbox=({pred['x']:.1f},{pred['y']:.1f},{pred['width']:.1f},{pred['height']:.1f})")
                    
                    # Update latest results
                    with self.results_lock:
                        self.latest_results = {
                            'predictions': predictions,
                            'timestamp': timestamp,
                            'error': None
                        }
                    
                except Exception as e:
                    logger.error(f"API inference failed: {e}")
                    
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
                
    def get_game_state(self):
        """Get current game state (thread-safe)."""
        if not self.enable_gamestate or not self.game_state:
            return None
        with self.gamestate_lock:
            return self.game_state.get_state()
    
    def reset_game_state(self):
        """Reset the game state."""
        if self.enable_gamestate and self.game_state:
            with self.gamestate_lock:
                self.game_state.reset()
                logger.info("Game state reset")
    
    def get_detection_summary(self):
        """Get detection summary statistics."""
        if not self.detection_history:
            return {'total_frames': 0, 'total_detections': 0, 'high_conf_detections': 0}
        
        total_frames = len(self.detection_history)
        total_detections = sum(d['count'] for d in self.detection_history.values())
        high_conf_detections = sum(d['high_conf_count'] for d in self.detection_history.values())
        
        return {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'high_conf_detections': high_conf_detections,
            'high_conf_rate': high_conf_detections / max(1, total_detections)
        }
    
    def get_stats(self):
        """Get simple inference statistics."""
        return {
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
    """Draw simple performance statistics on frame."""
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White text
    thickness = 1
    
    # Background rectangle for better text visibility
    cv2.rectangle(frame, (10, 5), (320, 75), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 5), (320, 75), (255, 255, 255), 1)
    
    # Display simple stats
    texts = [
        f"Display FPS: {fps_actual:.1f}",
        f"API Interval: {stats['api_interval']:.1f}s",
        f"Next API Call: {stats['next_api_call']:.1f}s"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (15, y_offset + i * 18), font, font_scale, color, thickness)

def draw_gamestate_overlay(frame, game_state_data):
    """Draw game state information on frame."""
    if not game_state_data:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 255)  # Cyan color
    thickness = 1
    
    # Position on right side of frame
    frame_height, frame_width = frame.shape[:2]
    x_start = frame_width - 250
    y_start = 30
    
    # Background rectangle
    cv2.rectangle(frame, (x_start - 10, y_start - 20), (frame_width - 10, y_start + 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (x_start - 10, y_start - 20), (frame_width - 10, y_start + 120), (0, 255, 255), 1)
    
    # Game state info
    texts = [
        f"GAME STATE",
        f"⚡ Elixir: {game_state_data.get('elixir_opponent', 0):.1f}",
        f"Last Played: {game_state_data.get('last_played', 'None')}",
        f"Hand: {', '.join([card or '?' for card in game_state_data.get('current_hand', [])[:2]])}...",  # Show first 2 cards
        f"Plays: {len(game_state_data.get('play_history', []))}",
        f"Match: {game_state_data.get('match_time', 0):.1f}s"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(frame, text, (x_start, y_start + i * 18), font, font_scale, color, thickness)

def main():
    parser = argparse.ArgumentParser(description="Live desktop capture inference (Roboflow API + OpenCV)")
    parser.add_argument("--model_id", type=str, default="clashquant-nvnzk/3", help="Roboflow model ID")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (MSS indexing)")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=800, help="Resize display width (maintain aspect)")
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT","TOP","WIDTH","HEIGHT"), help="Optional region override")
    parser.add_argument("--api_interval", type=float, default=5.0, help="Seconds between API calls (default: 5.0)")
    parser.add_argument("--debug", action="store_true", help="Enable debugging (save frames, print detections)")
    parser.add_argument("--gamestate", action="store_true", help="Enable game state tracking (default: enabled)")
    parser.add_argument("--no-gamestate", action="store_true", help="Disable game state tracking")
    args = parser.parse_args()
    
    # GameState setting (enabled by default, can be disabled)
    enable_gamestate = args.gamestate or not args.no_gamestate
    
    # Easy to change variable for API call frequency
    API_CALL_INTERVAL = args.api_interval  # seconds between API calls

    # Initialize Roboflow client
    rf = Roboflow(api_key="nfRaKWeyDCMxK2jw6k5a")
    
    # Get the project and model
    project_name, version = args.model_id.split('/')
    project = rf.workspace().project(project_name)
    model = project.version(version).model
    
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
    
    # Initialize async inference manager with aggressive queue clearing and GameState
    inference_manager = AsyncInferenceManager(
        model, 
        api_interval=API_CALL_INTERVAL,
        confidence=int(args.conf * 100),  # Convert 0.1 to 10 for Roboflow API
        debug=args.debug,
        enable_gamestate=enable_gamestate
    )
    inference_manager.start_worker()
    
    # FPS tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps_actual = 0.0
    
    logger.info(f"Started live inference with {API_CALL_INTERVAL}s API interval")
    logger.info(f"Confidence threshold: {args.conf} ({int(args.conf * 100)} for API)")
    logger.info(f"GameState tracking: {'ENABLED' if enable_gamestate else 'DISABLED'}")
    if enable_gamestate:
        logger.info("High-confidence detections (≥0.5) will trigger game state updates")
    if args.debug:
        logger.info("DEBUG MODE: Will save first 5 frames and print all detection details")
    logger.info("Press ESC to stop, R to reset game state")
    
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
            
            # Draw GameState overlay if enabled
            if enable_gamestate:
                game_state_data = inference_manager.get_game_state()
                if game_state_data:
                    draw_gamestate_overlay(annotated, game_state_data)
            
            # Resize for display
            h, w = annotated.shape[:2]
            new_w = args.width
            new_h = int(h * new_w / w)
            annotated = cv2.resize(annotated, (new_w, new_h))
            
            cv2.imshow("Clash Live", annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):  # R to reset game state
                if enable_gamestate:
                    inference_manager.reset_game_state()
                    logger.info("🔄 Game state reset by user")
                
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
    detection_summary = inference_manager.get_detection_summary()
    
    logger.info(f"Final Statistics:")
    logger.info(f"  API call interval: {final_stats['api_interval']:.1f}s")
    logger.info(f"  Total frames processed: {detection_summary['total_frames']}")
    logger.info(f"  Total detections: {detection_summary['total_detections']}")
    logger.info(f"  High-confidence detections: {detection_summary['high_conf_detections']}")
    logger.info(f"  High-confidence rate: {detection_summary['high_conf_rate']:.1%}")
    
    # Print final game state
    if enable_gamestate:
        final_game_state = inference_manager.get_game_state()
        if final_game_state:
            logger.info(f"Final Game State:")
            logger.info(f"  Opponent elixir: {final_game_state.get('elixir_opponent', 0):.1f}")
            logger.info(f"  Total card plays: {len(final_game_state.get('play_history', []))}")
            logger.info(f"  Last played: {final_game_state.get('last_played', 'None')}")
            logger.info(f"  Match duration: {final_game_state.get('match_time', 0):.1f}s")
            current_hand = final_game_state.get('current_hand', [])
            known_cards = [card for card in current_hand if card is not None]
            logger.info(f"  Known cards in hand: {', '.join(known_cards) if known_cards else 'None'}")
    
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
