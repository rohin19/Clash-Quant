// API configuration
const API_BASE_URL = import.meta.env.PROD 
  ? 'http://207.23.170.7:5000/api'  // Your local server IP for production
  : '/api';  // Proxy for development

const defaultGameState = {
  visible_cards: [],
  elixir_opponent: 0,
  next_prediction: "",
  deck: [],
  current_hand: []
};

export const api = {
  // Get full game state
  getGameState: async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/gamestate`);
      if (!res.ok) throw new Error("API error");
      return await res.json();
    } catch (err) {
      // Return default empty state if API fails
      return defaultGameState;
    }
  },
  
  // Get only detections
  getDetections: () => 
    fetch(`${API_BASE_URL}/gamestate/detections`).then(res => res.json()),
  
  // Update game state
  updateGameState: (detections) =>
    fetch(`${API_BASE_URL}/gamestate/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detections })
    }).then(res => res.json()),
  
  // Reset game state
  resetGameState: () =>
    fetch(`${API_BASE_URL}/gamestate/reset`, {
      method: 'POST'
    }).then(res => res.json()),
  
  // Health check
  healthCheck: () =>
    fetch(`${API_BASE_URL}/health`).then(res => res.json())
};

export default api;