// API configuration
const API_BASE_URL = import.meta.env.PROD 
  ? 'http://207.23.170.7:5000'  // Your local server IP for production
  : '/api';  // Proxy for development

export const api = {
  // Get full game state
  getGameState: () => 
    fetch(`${API_BASE_URL}/api/gamestate`).then(res => res.json()),
  
  // Get only detections
  getDetections: () => 
    fetch(`${API_BASE_URL}/api/gamestate/detections`).then(res => res.json()),
  
  // Update game state
  updateGameState: (detections) =>
    fetch(`${API_BASE_URL}/api/gamestate/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detections })
    }).then(res => res.json()),
  
  // Reset game state
  resetGameState: () =>
    fetch(`${API_BASE_URL}/api/gamestate/reset`, {
      method: 'POST'
    }).then(res => res.json()),
  
  // Health check
  healthCheck: () =>
    fetch(`${API_BASE_URL}/api/health`).then(res => res.json())
};

export default api;