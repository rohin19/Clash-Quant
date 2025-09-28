"""
Simple client to send game state updates to the server.
This can be integrated with your inference script.
"""
import requests
import json
from datetime import datetime

class GameStateClient:
    """Client to send game state updates to the Flask server."""
    
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url.rstrip('/')
        
    def update_gamestate(self, detections, fps=0.0, api_status="active"):
        """Send detection results to the server."""
        try:
            payload = {
                "detections": detections,
                "fps": fps,
                "api_status": api_status,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{self.server_url}/api/gamestate/update",
                json=payload,
                timeout=1.0  # Quick timeout to avoid blocking inference
            )
            
            if response.status_code == 200:
                return True
            else:
                print(f"Failed to update server: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Server update failed: {e}")
            return False
            
    def get_gamestate(self):
        """Get current game state from server."""
        try:
            response = requests.get(f"{self.server_url}/api/gamestate", timeout=1.0)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except requests.exceptions.RequestException:
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