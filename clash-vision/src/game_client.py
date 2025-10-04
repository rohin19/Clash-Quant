"""
Simple client to send game state updates to the server.
This can be integrated with your inference script.
"""
import requests
import json
from datetime import datetime

class GameStateClient:
    """Client for posting detection batches to the game server.

    Accepts mixed detection formats and converts to server schema.
    """

    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url.rstrip('/')

    def _normalize(self, detections):
        out = []
        for d in detections or []:
            if not isinstance(d, dict):
                continue
            name = d.get('card') or d.get('class') or d.get('name')
            if not name:
                continue
            conf = d.get('confidence', 0.0)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if 'bbox' in d and isinstance(d['bbox'], (list, tuple)) and len(d['bbox']) == 4:
                bbox = d['bbox']
            elif all(k in d for k in ('x','y','width','height')):
                try:
                    x = float(d['x']); y = float(d['y'])
                    w = float(d['width']); h = float(d['height'])
                except (TypeError, ValueError):
                    continue
                bbox = [x - w/2, y - h/2, x + w/2, y + h/2]
            else:
                continue
            out.append({'card': name, 'bbox': bbox, 'confidence': conf})
        return out

    def update_gamestate(self, detections, timestamp: float | None = None):
        try:
            norm = self._normalize(detections)
            payload = {
                'detections': norm,
                'timestamp': timestamp
            }
            response = requests.post(
                f"{self.server_url}/api/gamestate/update",
                json=payload,
                timeout=1.0
            )
            return response.ok
        except requests.exceptions.RequestException as e:
            print(f"Server update failed: {e}")
            return False

    def get_gamestate(self):
        try:
            response = requests.get(f"{self.server_url}/api/gamestate", timeout=1.0)
            if response.ok:
                return response.json()
        except requests.exceptions.RequestException:
            return None
        return None

if __name__ == "__main__":
    # Example usage
    client = GameStateClient()
    
    # Test update
    sample_detections = [
        {"class": "archer", "confidence": 0.85, "x": 100, "y": 200},
        {"class": "cannon", "confidence": 0.92, "x": 300, "y": 150}
    ]
    
    success = client.update_gamestate(sample_detections, fps=15.2)
    print(f"Update successful: {success}")
    
    # Test get
    state = client.get_gamestate()
    if state:
        print("Current game state:", json.dumps(state, indent=2))