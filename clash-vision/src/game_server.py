from flask import Flask, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json
import time
from datetime import datetime
from gamestate import GameState
import os

# Serve Vite build from Flask
app = Flask(__name__, static_folder='../../frontend/dist', static_url_path='')
CORS(app)  # Enable CORS for frontend communication

# Global game state - just use your existing class directly
game_state = GameState()

@app.route('/api/gamestate', methods=['GET'])
def get_gamestate():
    """
    GET /api/gamestate
    Returns the current game state with detection results.
    """
    try:
        state = game_state.get_state()
        return jsonify({
            "success": True,
            "data": state
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/gamestate/detections', methods=['GET'])
def get_detections():
    """
    GET /api/gamestate/detections  
    Returns only the current detections array.
    """
    try:
        state = game_state.get_state()
        return jsonify({
            "success": True,
            "detections": state["visible_cards"],
            "count": len(state["visible_cards"]),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    GET /api/health
    Health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time()
    }), 200

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
    """
    POST /api/gamestate/update
    Update the game state (for inference script to call).
    Expected format: {
        "detections": [
            {"card": "Giant", "bbox": [x1, y1, x2, y2], "confidence": 0.92}
        ]
    }
    """
    try:
        from flask import request
        data = request.get_json()
        
        detections = data.get('detections', [])
        
        # Update game state directly
        game_state.update(detections)
        
        return jsonify({
            "success": True,
            "message": "Game state updated",
            "cards_detected": len(detections),
            "elixir": game_state.elixir_opponent
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/gamestate/reset', methods=['POST'])
def reset_gamestate():
    """
    POST /api/gamestate/reset
    Reset the game state for a new match.
    """
    try:
        game_state.reset()
        return jsonify({
            "success": True,
            "message": "Game state reset for new match"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Clash Quant Full Stack Server...")
    print("API Endpoints:")
    print("   GET  /api/gamestate           - Get full game state (elixir, deck, cards)")
    print("   GET  /api/gamestate/detections - Get only detections")
    print("   GET  /api/health              - Health check")
    print("   POST /api/gamestate/update    - Update game state with detections")
    print("   POST /api/gamestate/reset     - Reset for new match")
    print("Frontend & API on http://localhost:5000")
    print("Using GameState class with elixir tracking and deck discovery!")
    print("Serving React app from /frontend/dist")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
