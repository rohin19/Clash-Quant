# Clash Vision - Real-time Clash Royale Model Detection + Analysis

AI-powered real-time analysis system for Clash Royale gameplay using computer vision and machine learning.

## Features

- **Live Card Detection**: Real-time identification of cards using Roboflow API
- **Game State Tracking**: Opponent elixir monitoring and deck discovery
- **AI Prediction Engine**: Machine learning-based next card prediction
- **Web Interface**: React frontend with live data visualization
- **Non-Maximum Suppression**: Prevents duplicate/overlapping detections
- **Match Analytics**: Historical data and performance statistics

## Quick Start

### Prerequisites
- Python 3.8+
- Roboflow API key 
- Iphone Screen Capture Software
- pip

### Installation
```bash
cd clash-vision
pip install -r requirements.txt
```
### Activate venv
```bash
cd clash-vision
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

### Setup Screen Capture
```bash
# 0. Make sure venv activated
# 1. Connect iPhone and open QuickTime Player (or OBS)
# 2. File > New Movie Recording > Select iPhone
# 3. Run region selector to define capture area
cd \Clash-Quant\clash-vision\src
python -m clash_vision.inference.screen_preview --monitor 1
# Press 'r' to select iPhone screen area
# Drag out box to identify screen area
# Press 's' then 'esc' to save
# Press 'Ctrl+C' to exit 
```

### Run Live Analysis
```bash
# Basic game state tracking and model detection
# 0. Make sure venv activated
cd Clash-Quant\clash-vision.venv\Scripts\activate
python -m clash_vision.inference.live_inference --conf 0.05 --api_interval 0.5 --fps 30

# With web server for frontend access
python -m clash_vision.inference.live_inference --conf 0.05 --server-port 8080

# High performance mode
python -m clash_vision.inference.live_inference --conf 0.03 --api_interval 0.5 --fps 30
```

### Frontend Setup - (In Trial)
```bash
cd ../frontend
npm install
npm run dev  # Development server on http://localhost:3000
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--conf` | 0.1 | Detection confidence threshold |
| `--api_interval` | 5.0 | Seconds between API calls |
| `--fps` | 15.0 | Display framerate |
| `--nms-threshold` | 0.3 | NMS IoU threshold for overlap removal |
| `--server-port` | 8080 | Web server port |
| `--no-server` | False | Disable web server |
| `--debug` | False | Enable debug mode with frame saving |

## API Endpoints

- `GET /api/gamestate` - Current game state and statistics
- `GET /api/gamestate/detections` - Active card detections
- `POST /api/gamestate/reset` - Reset match state
- `GET /api/health` - Health check

## Game State Features

- **Elixir Tracking**: Real-time opponent elixir calculation
- **Deck Discovery**: Automatic opponent deck identification
- **Card Cycling**: Hand rotation and queue prediction
- **Play History**: Chronological card usage tracking
- **Prediction Engine**: Multi-factor next card forecasting

## Controls

- **ESC**: Exit application
- **R**: Reset game state
- **S**: Save current frame (in screen preview mode)

## Troubleshooting

**No detections**: Check confidence threshold and API key
**High CPU usage**: Increase API interval or reduce FPS
**Capture issues**: Ensure QuickTime window is visible and region is selected
**Web server errors**: Check port availability and CORS settings

## Extra Notes

- Use `--api_interval 0.5` for competitive analysis
- Set `--nms-threshold 0.2` for stricter duplicate removal
- Enable `--debug` mode to analyze detection quality
- Monitor NMS filtering statistics in overlay
- Detection Model only trained on 8 cards so far
- Detection Model updates very slowly as of right now
