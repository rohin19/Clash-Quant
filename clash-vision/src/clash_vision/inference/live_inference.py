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

# Flask integration
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Import GameState from the parent directory
sys.path.append(str(Path(__file__).parent.parent.parent))
from gamestate import GameState

# Card assets mapping - normalize card names to asset filenames
CARD_ASSETS = {
    'giant': 'giant.png',
    'knight': 'knight.png', 
    'bomber': 'bomber.png',
    'hog rider': 'hogrider.png',
    'dart goblin': 'dartgoblin.png',
    'mini pekka': 'minipekka.png',
    'baby dragon': 'babydragon.png',
    'valkyrie': 'valkyrie.png'
}

# Cache for loaded card images
_card_image_cache = {}

def load_card_image(card_name, size=(45, 45)):
    """Load and cache card image from assets."""
    if not card_name:
        return None
        
    # Normalize card name
    card_key = card_name.lower().strip()
    cache_key = f"{card_key}_{size[0]}x{size[1]}"
    
    # Return cached image if available
    if cache_key in _card_image_cache:
        return _card_image_cache[cache_key]
    
    # Try to find the card asset
    asset_filename = CARD_ASSETS.get(card_key)
    if not asset_filename:
        return None
    
    # Build asset path
    assets_dir = Path(__file__).parent.parent.parent.parent.parent / "frontend" / "src" / "assets"
    asset_path = assets_dir / asset_filename
    
    if not asset_path.exists():
        return None
    
    try:
        # Load and resize image
        img = cv2.imread(str(asset_path))
        if img is not None:
            img_resized = cv2.resize(img, size)
            _card_image_cache[cache_key] = img_resized
            return img_resized
    except Exception as e:
        logger.warning(f"Failed to load card image {asset_path}: {e}")
    
    return None

class GameStateServer:
    """Flask server to expose GameState via REST API."""
    
    def __init__(self, shared_gamestate, port=8080):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for frontend
        self.shared_gamestate = shared_gamestate
        self.port = port
        self.server_thread = None
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/api/gamestate', methods=['GET'])
        def get_gamestate():
            """GET /api/gamestate - Returns current game state."""
            try:
                state = self.shared_gamestate.get_game_state()
                return jsonify({
                    "success": True,
                    "data": state or {}
                }), 200
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/api/gamestate/detections', methods=['GET'])
        def get_detections():
            """GET /api/gamestate/detections - Returns only current detections."""
            try:
                state = self.shared_gamestate.get_game_state()
                visible_cards = state.get('visible_cards', []) if state else []
                return jsonify({
                    "success": True,
                    "detections": [card['card'] for card in visible_cards],
                    "count": len(visible_cards)
                }), 200
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/api/gamestate/reset', methods=['POST'])
        def reset_gamestate():
            """POST /api/gamestate/reset - Reset game state."""
            try:
                self.shared_gamestate.reset_game_state()
                return jsonify({"success": True, "message": "Game state reset"}), 200
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/api/predictions', methods=['GET'])
        def get_predictions():
            """GET /api/predictions - Get AI card predictions."""
            try:
                state = self.shared_gamestate.get_game_state()
                if state:
                    return jsonify({
                        "success": True,
                        "predictions": state.get('next_prediction', []),
                        "confidence": state.get('prediction_confidence', 0.0),
                        "elixir": state.get('elixir_opponent', 0)
                    }), 200
                else:
                    return jsonify({
                        "success": True,
                        "predictions": [],
                        "confidence": 0.0,
                        "elixir": 0
                    }), 200
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """GET /api/health - Health check."""
            return jsonify({"status": "healthy", "port": self.port}), 200

        # Serve React frontend if available
        frontend_dist = Path(__file__).parent.parent.parent.parent.parent / "frontend" / "dist"
        if frontend_dist.exists():
            @self.app.route('/')
            def serve_frontend():
                return send_from_directory(str(frontend_dist), 'index.html')
            
            @self.app.route('/<path:path>')
            def serve_static(path):
                if (frontend_dist / path).exists():
                    return send_from_directory(str(frontend_dist), path)
                return send_from_directory(str(frontend_dist), 'index.html')

    def start_server(self):
        """Start Flask server in background thread."""
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        logger.info(f"🌐 Web server started on http://localhost:{self.port}")
        logger.info(f"📊 API endpoints: /api/gamestate, /api/health")

class AsyncInferenceManager:
    """Manages asynchronous API calls to Roboflow with aggressive queue clearing and game state tracking."""
    
    def __init__(self, model, api_interval=5.0, confidence=40, debug=False, enable_gamestate=True, shared_gamestate=None):
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
        
        # GameState integration - use shared instance if provided
        if shared_gamestate:
            self.game_state = shared_gamestate
            self.gamestate_lock = threading.Lock()  # Still need lock for thread safety
        else:
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
        #Submit a frame for inference, clearing the queue if it
        current_time = time.time()
        
        # Check if enough time has passed since last API call
        if current_time - self.last_api_call < self.api_interval:
            return False
            
        # AGGRESSIVE: Empty entire queue first to prevent backup
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
                
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_queue.put((current_time, frame_rgb), block=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to submit frame: {e}")
            return False
            
    def get_latest_results(self):
        with self.results_lock:
            return self.latest_results
            
    def _worker_loop(self):
        while not self.should_stop:
            try:
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

def apply_nms(predictions, iou_threshold=0.3):
    """Apply Non-Maximum Suppression to remove overlapping detections."""
    if not predictions:
        return predictions
    
    # Convert predictions to format needed for NMS
    boxes = []
    scores = []
    class_ids = []
    
    for pred in predictions:
        # Convert center format to corner format
        x_center, y_center = pred['x'], pred['y']
        w, h = pred['width'], pred['height']
        x1 = x_center - w/2
        y1 = y_center - h/2
        x2 = x_center + w/2
        y2 = y_center + h/2
        
        boxes.append([x1, y1, x2, y2])
        scores.append(pred['confidence'])
        class_ids.append(pred['class'])
    
    # Convert to numpy arrays
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Apply NMS using OpenCV
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, iou_threshold)
    
    # Filter predictions based on NMS results
    if len(indices) > 0:
        indices = indices.flatten()
        filtered_predictions = [predictions[i] for i in indices]
        return filtered_predictions
    else:
        return []

def draw_detections(frame, predictions, nms_threshold=0.3):
    """Draw aesthetic bounding boxes and labels on frame with NMS to prevent overlaps."""
    annotated = frame.copy()
    
    # Apply Non-Maximum Suppression to prevent overlapping detections
    filtered_predictions = apply_nms(predictions, iou_threshold=nms_threshold)
    
    if filtered_predictions:
        for prediction in filtered_predictions:
            # Extract bounding box coordinates
            x = int(prediction['x'] - prediction['width'] / 2)
            y = int(prediction['y'] - prediction['height'] / 2)
            w = int(prediction['width'])
            h = int(prediction['height'])
            
            confidence = prediction['confidence']
            
            # Color coding based on confidence
            if confidence >= 0.5:
                color = (128, 0, 128)  # Purple for high confidence (game state triggers)
                thickness = 2
            else:
                color = (100, 100, 100)  # Gray for lower confidence
                thickness = 1
            
            # Draw bounding box with rounded corners effect
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label with background for better readability
            label = f"{prediction['class']}: {confidence:.2f}"
            font_scale = 0.4
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            
            # Label background
            cv2.rectangle(annotated, (x, y - text_size[1] - 8), (x + text_size[0] + 6, y), (0, 0, 0), -1)
            cv2.rectangle(annotated, (x, y - text_size[1] - 8), (x + text_size[0] + 6, y), color, 1)
            
            # Label text
            cv2.putText(annotated, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            
    return annotated

def draw_stats_overlay(frame, stats, fps_actual, nms_filtered=0):
    """Draw compact performance statistics in top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)  # White text
    thickness = 1
    
    # Expand box if NMS filtering is active
    box_height = 70 if nms_filtered > 0 else 55
    
    # Compact stats in top-left corner
    cv2.rectangle(frame, (5, 5), (200, box_height), (0, 0, 0, 180), -1)  # Semi-transparent background
    cv2.putText(frame, f"FPS: {fps_actual:.1f}", (10, 20), font, font_scale, color, thickness)
    cv2.putText(frame, f"API: {stats['api_interval']:.1f}s", (10, 35), font, font_scale, color, thickness)
    cv2.putText(frame, f"Next: {stats['next_api_call']:.1f}s", (10, 50), font, font_scale, color, thickness)
    
    # Show NMS filtering info if active
    if nms_filtered > 0:
        cv2.putText(frame, f"NMS: -{nms_filtered} overlaps", (10, 65), font, 0.35, (100, 255, 100), thickness)

def draw_gamestate_overlay(frame, game_state_data):
    """Draw aesthetic game state HUD inspired by Clash Quant design."""
    if not game_state_data:
        return
    
    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Colors matching the aesthetic
    purple = (128, 0, 128)  # Purple accent
    dark_bg = (40, 40, 40)  # Dark background
    white = (255, 255, 255)
    light_gray = (200, 200, 200)
    
    # === TOP RIGHT: Opponent Elixir Bar ===
    elixir_x = frame_width - 200
    elixir_y = 20
    elixir_val = game_state_data.get('elixir_opponent', 0)
    max_elixir = 10.0
    
    # Elixir label
    cv2.putText(frame, "Opponent Elixir", (elixir_x, elixir_y - 5), font, 0.5, white, 1)
    
    # Elixir bar background
    bar_width = 120
    bar_height = 20
    cv2.rectangle(frame, (elixir_x, elixir_y), (elixir_x + bar_width, elixir_y + bar_height), dark_bg, -1)
    cv2.rectangle(frame, (elixir_x, elixir_y), (elixir_x + bar_width, elixir_y + bar_height), light_gray, 1)
    
    # Elixir fill
    fill_width = int((elixir_val / max_elixir) * bar_width)
    if fill_width > 0:
        cv2.rectangle(frame, (elixir_x + 1, elixir_y + 1), (elixir_x + fill_width, elixir_y + bar_height - 1), purple, -1)
    
    # Elixir text
    cv2.putText(frame, f"{elixir_val:.1f} / {max_elixir}", (elixir_x + 25, elixir_y + 35), font, 0.5, white, 1)
    
    # === TOP RIGHT: Opponent Deck ===
    deck_x = frame_width - 380  # Start further left to fit 8 cards
    deck_y = elixir_y + 100  # Move down to avoid overlapping detection stats
    deck = game_state_data.get('deck', [])
    
    cv2.putText(frame, "Opponent Deck", (deck_x, deck_y - 10), font, 0.5, white, 1)
    
    # Draw opponent deck cards (8 cards total)
    small_card_size = 35
    cards_per_row = 4
    for i, card in enumerate(deck[:8]):  # Show all 8 deck cards
        row = i // cards_per_row
        col = i % cards_per_row
        x = deck_x + col * (small_card_size + 3)
        y = deck_y + row * (small_card_size + 3)
        
        if card:
            # Try to load card image
            card_img = load_card_image(card, (small_card_size, small_card_size))
            
            if card_img is not None:
                # Display card image
                try:
                    frame[y:y+small_card_size, x:x+small_card_size] = card_img
                    # Add thin border
                    cv2.rectangle(frame, (x, y), (x + small_card_size, y + small_card_size), white, 1)
                    
                    # Highlight current hand cards (first 4) with colored border
                    if i < 4:
                        cv2.rectangle(frame, (x, y), (x + small_card_size, y + small_card_size), purple, 2)
                except Exception:
                    # Fallback to colored box
                    color = purple if i < 4 else (60, 60, 60)  # Purple for hand, gray for queue
                    cv2.rectangle(frame, (x, y), (x + small_card_size, y + small_card_size), color, -1)
                    cv2.rectangle(frame, (x, y), (x + small_card_size, y + small_card_size), white, 1)
            else:
                # Fallback to colored box with abbreviated name
                color = purple if i < 4 else (60, 60, 60)  # Purple for hand, gray for queue
                cv2.rectangle(frame, (x, y), (x + small_card_size, y + small_card_size), color, -1)
                cv2.rectangle(frame, (x, y), (x + small_card_size, y + small_card_size), white, 1)
                if len(card) > 0:
                    card_short = card[:2].upper()
                    cv2.putText(frame, card_short, (x + 2, y + small_card_size//2 + 3), font, 0.25, white, 1)
        else:
            # Unknown card
            cv2.rectangle(frame, (x, y), (x + small_card_size, y + small_card_size), dark_bg, -1)
            cv2.rectangle(frame, (x, y), (x + small_card_size, y + small_card_size), light_gray, 1)
            cv2.putText(frame, "?", (x + small_card_size//2 - 3, y + small_card_size//2 + 3), font, 0.3, light_gray, 1)
    
    # === BOTTOM LEFT: Current Hand ===
    hand_x = 20
    hand_y = frame_height - 130  # Move up to make room for predictions below
    current_hand = game_state_data.get('current_hand', [])
    
    cv2.putText(frame, "Current Hand", (hand_x, hand_y - 10), font, 0.5, white, 1)
    
    # Draw hand cards with images
    card_size = 45
    for i, card in enumerate(current_hand[:4]):  # Show first 4 cards
        x = hand_x + i * (card_size + 5)
        y = hand_y
        
        if card:
            # Try to load card image
            card_img = load_card_image(card, (card_size, card_size))
            
            if card_img is not None:
                # Display card image
                try:
                    frame[y:y+card_size, x:x+card_size] = card_img
                    # Add border around card image
                    cv2.rectangle(frame, (x, y), (x + card_size, y + card_size), white, 2)
                except Exception:
                    # Fallback to text if image placement fails
                    cv2.rectangle(frame, (x, y), (x + card_size, y + card_size), purple, -1)
                    cv2.rectangle(frame, (x, y), (x + card_size, y + card_size), white, 1)
                    card_short = card[:3] if card else "?"
                    cv2.putText(frame, card_short, (x + 3, y + card_size//2 + 5), font, 0.4, white, 1)
            else:
                # Fallback to colored box with text if no image available
                cv2.rectangle(frame, (x, y), (x + card_size, y + card_size), purple, -1)
                cv2.rectangle(frame, (x, y), (x + card_size, y + card_size), white, 1)
                card_short = card[:3] if card else "?"
                cv2.putText(frame, card_short, (x + 3, y + card_size//2 + 5), font, 0.4, white, 1)
        else:
            # Unknown card slot
            cv2.rectangle(frame, (x, y), (x + card_size, y + card_size), dark_bg, -1)
            cv2.rectangle(frame, (x, y), (x + card_size, y + card_size), light_gray, 1)
            cv2.putText(frame, "?", (x + card_size//2 - 5, y + card_size//2 + 5), font, 0.5, light_gray, 1)
    

    
    # === BOTTOM LEFT: AI Predictions (Below Current Hand) ===
    pred_x = hand_x
    pred_y = hand_y + 60  # Position below current hand
    predictions = game_state_data.get('next_prediction', [])
    pred_confidence = game_state_data.get('prediction_confidence', 0.0)
    
    if predictions and pred_confidence > 0.3:  # Only show if confidence is reasonable
        # Background for predictions
        cv2.rectangle(frame, (pred_x - 5, pred_y - 20), (pred_x + 250, pred_y + 25), dark_bg, -1)
        cv2.rectangle(frame, (pred_x - 5, pred_y - 20), (pred_x + 250, pred_y + 25), (0, 255, 0), 1)
        
        cv2.putText(frame, f"AI Predictions ({pred_confidence:.0%}):", (pred_x, pred_y - 5), font, 0.4, (0, 255, 0), 1)
        
        # Show top 3 predictions in a horizontal layout
        for i, pred_card in enumerate(predictions[:3]):
            x_offset = pred_x + i * 80
            y_offset = pred_y + 15
            cv2.putText(frame, f"{i+1}. {pred_card[:6]}", (x_offset, y_offset), font, 0.35, white, 1)
    
    # === BOTTOM RIGHT: Game Info ===
    info_x = frame_width - 200
    info_y = frame_height - 50
    
    # Background for info section
    cv2.rectangle(frame, (info_x - 10, info_y - 15), (frame_width - 10, frame_height - 10), dark_bg, -1)
    cv2.rectangle(frame, (info_x - 10, info_y - 15), (frame_width - 10, frame_height - 10), light_gray, 1)
    
    # Game info text
    last_played = game_state_data.get('last_played') or 'None'
    last_played_display = last_played[:10] if last_played != 'None' else 'None'
    
    info_texts = [
        f"Last: {last_played_display}",  # Safely truncate long names
        f"Plays: {len(game_state_data.get('play_history', []))} | Time: {game_state_data.get('match_time', 0):.0f}s"
    ]
    
    for i, text in enumerate(info_texts):
        cv2.putText(frame, text, (info_x, info_y + i * 15), font, 0.35, white, 1)
    
    # === TOP RIGHT: Detection Stats (Below Elixir) ===
    visible_cards = game_state_data.get('visible_cards', [])
    if visible_cards:
        stats_x = frame_width - 190
        stats_y = 120  # Position below elixir bar
        cv2.rectangle(frame, (stats_x - 5, stats_y - 20), (stats_x + 180, stats_y + 20), dark_bg, -1)
        cv2.rectangle(frame, (stats_x - 5, stats_y - 20), (stats_x + 180, stats_y + 20), purple, 1)
        cv2.putText(frame, f"Visible: {len(visible_cards)} cards", (stats_x, stats_y - 5), font, 0.45, white, 1)
        cv2.putText(frame, f"NMS: Active (IoU=0.3)", (stats_x, stats_y + 10), font, 0.35, (100, 255, 100), 1)

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
    parser.add_argument("--server-port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--no-server", action="store_true", help="Disable web server")
    parser.add_argument("--nms-threshold", type=float, default=0.3, help="NMS IoU threshold to remove overlapping detections (default: 0.3)")
    args = parser.parse_args()
    
    # GameState setting (enabled by default, can be disabled)
    enable_gamestate = args.gamestate or not args.no_gamestate
    
    # Easy to change variable for API call frequency
    API_CALL_INTERVAL = args.api_interval  # seconds between API calls

    # Initialize Roboflow client
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("API_KEY")
    rf = Roboflow(api_key=api_key)
    
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
    
    # Create shared GameState instance
    shared_game_state = GameState() if enable_gamestate else None
    
    # Initialize web server if enabled
    web_server = None
    if not args.no_server and enable_gamestate and shared_game_state:
        # Create a wrapper to access game state through inference manager interface
        class GameStateWrapper:
            def __init__(self, gamestate_instance):
                self.gamestate = gamestate_instance
            
            def get_game_state(self):
                return self.gamestate.get_state() if self.gamestate else None
            
            def reset_game_state(self):
                if self.gamestate:
                    self.gamestate.reset()
        
        game_wrapper = GameStateWrapper(shared_game_state)
        web_server = GameStateServer(game_wrapper, port=args.server_port)
        web_server.start_server()
    
    # Initialize async inference manager with shared GameState
    inference_manager = AsyncInferenceManager(
        model, 
        api_interval=API_CALL_INTERVAL,
        confidence=int(args.conf * 100),  # Convert 0.1 to 10 for Roboflow API
        debug=args.debug,
        enable_gamestate=enable_gamestate,
        shared_gamestate=shared_game_state
    )
    inference_manager.start_worker()
    
    # FPS tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps_actual = 0.0
    
    logger.info(f"🎯 Started live inference with {API_CALL_INTERVAL}s API interval")
    logger.info(f"📊 Confidence threshold: {args.conf} ({int(args.conf * 100)} for API)")
    logger.info(f"🎮 GameState tracking: {'ENABLED' if enable_gamestate else 'DISABLED'}")
    logger.info(f"🔍 NMS filtering: IoU threshold = {args.nms_threshold} (prevents overlapping detections)")
    if not args.no_server and web_server:
        logger.info(f"🌐 Web server: http://localhost:{args.server_port}")
        logger.info("📱 Frontend available for remote monitoring")
    elif args.no_server:
        logger.info("🚫 Web server disabled")
    if enable_gamestate:
        logger.info("⚡ High-confidence detections (≥0.5) trigger game state updates")
        logger.info("🤖 AI card prediction engine active with visual overlay")
    if args.debug:
        logger.info("🔍 DEBUG MODE: Will save first 5 frames and print all detection details")
    logger.info("⌨️  Press ESC to stop, R to reset game state")
    
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
                    # Show API error in center
                    annotated = frame.copy()
                    cv2.putText(annotated, "API Error", 
                              (frame.shape[1]//2 - 50, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Draw detections with NMS filtering
                    raw_predictions = results['predictions']
                    annotated = draw_detections(frame, raw_predictions, args.nms_threshold)
                    
                    # Store NMS stats in results for display
                    if raw_predictions:
                        filtered_predictions = apply_nms(raw_predictions, iou_threshold=args.nms_threshold)
                        results['nms_filtered'] = len(raw_predictions) - len(filtered_predictions)
                    else:
                        results['nms_filtered'] = 0
            else:
                # No results yet - show in center
                annotated = frame.copy()
                cv2.putText(annotated, "Initializing AI Vision...", 
                          (frame.shape[1]//2 - 100, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter >= 30:  # Update FPS every 30 frames
                elapsed_fps = time.time() - fps_start_time
                fps_actual = fps_counter / elapsed_fps
                fps_counter = 0
                fps_start_time = time.time()
            
            # Draw statistics overlay
            stats = inference_manager.get_stats()
            nms_filtered_count = results.get('nms_filtered', 0) if results else 0
            draw_stats_overlay(annotated, stats, fps_actual, nms_filtered_count)
            
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
