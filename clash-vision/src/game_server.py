from flask import Flask, jsonify, send_from_directory, request, Response
from flask_cors import CORS
import time
from datetime import datetime
from gamestate import GameState
import os
import threading
import json

# Serve Vite build from Flask
app = Flask(__name__, static_folder='../../frontend/dist', static_url_path='')
CORS(app)  # Enable CORS for frontend communication

game_state = GameState()
_lock = threading.Lock()

# Background tick (elixir/time progression even when no detections posted)
_tick_event = threading.Event()

def _tick_loop():  # pragma: no cover
    while not _tick_event.is_set():
        with _lock:
            # Empty list still advances time logic
            game_state.ingest_detections([])
        _tick_event.wait(0.25)  # 4 Hz

_tick_thread = threading.Thread(target=_tick_loop, daemon=True)
_tick_thread.start()

def _normalize_incoming(detections):
    """Accept mixed detection formats (roboflow center-style or already normalized).
    Returns list formatted for GameState.ingest_detections.
    Supported shapes per item:
      {"card"|"class": name, "bbox": [x1,y1,x2,y2], "confidence": float}
      {"class": name, "x": cx, "y": cy, "width": w, "height": h, "confidence": f}
    """
    normalized = []
    for d in detections or []:
        if not isinstance(d, dict):
            continue
        name = d.get('card') or d.get('class') or d.get('name')
        if not name:
            continue
        conf = d.get('confidence')
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        # Already bbox form
        if 'bbox' in d and isinstance(d['bbox'], (list, tuple)) and len(d['bbox']) == 4:
            bbox = d['bbox']
        # center format
        elif all(k in d for k in ('x','y','width','height')):
            try:
                x = float(d['x']); y = float(d['y'])
                w = float(d['width']); h = float(d['height'])
            except (TypeError, ValueError):
                continue
            x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
            bbox = [x1, y1, x2, y2]
        else:
            continue
        normalized.append({
            'card': name,
            'bbox': bbox,
            'confidence': conf
        })
    return normalized

@app.route('/api/gamestate', methods=['GET'])
def get_gamestate():
    with _lock:
        state = game_state.get_state()
    return jsonify({"success": True, "data": state})

@app.route('/api/gamestate/detections', methods=['GET'])
def get_detections():
    with _lock:
        state = game_state.get_state()
    return jsonify({
        "success": True,
        "detections": state["visible_cards"],
        "count": len(state["visible_cards"]),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time()
    })

# Serve the React app
@app.route('/')
def serve_index():
    """Serve the main React app."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    """Serve static files from React build."""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # For client-side routing, serve index.html
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/gamestate/update', methods=['POST'])
def update_gamestate():
    data = request.get_json(force=True, silent=True) or {}
    raw_detections = data.get('detections', [])
    timestamp = data.get('timestamp')  # optional float
    norm = _normalize_incoming(raw_detections)
    with _lock:
        game_state.ingest_detections(norm, frame_ts=timestamp)
        state = game_state.get_state()
    return jsonify({
        "success": True,
        "ingested": len(norm),
        "provided": len(raw_detections),
        "deck": state['deck'],
        "elixir": state['elixir_opponent']
    })

@app.route('/api/gamestate/reset', methods=['POST'])
def reset_gamestate():
    with _lock:
        game_state.reset()
        state = game_state.get_state()
    return jsonify({"success": True, "state": state})

if __name__ == '__main__':
    print("Starting Clash Quant Game Server (Integrated GameState)")
    print("Endpoints:")
    print("  GET  /api/gamestate")
    print("  GET  /api/gamestate/detections")
    print("  POST /api/gamestate/update")
    print("  POST /api/gamestate/reset")
    print("  GET  /api/health")
    print("Background tick active (0.25s)")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
