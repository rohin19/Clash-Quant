# Clash Vision - Real-time Clash Royale Analysis

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
- QuickTime Player (for iPhone screen capture)

### Installation
```bash
cd clash-vision
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Setup Screen Capture
```bash
# 1. Connect iPhone and open QuickTime Player
# 2. File > New Movie Recording > Select iPhone
# 3. Run region selector to define capture area
python -m clash_vision.inference.screen_preview --monitor 1
# Press 'r' to select iPhone screen area, saves to capture_region.json
```

### Run Live Analysis
```bash
# Basic inference with game state tracking
python -m clash_vision.inference.live_inference --conf 0.05 --api_interval 1.0

# With web server for frontend access
python -m clash_vision.inference.live_inference --conf 0.05 --server-port 8080

# High performance mode
python -m clash_vision.inference.live_inference --conf 0.03 --api_interval 0.5 --fps 30
```

### Frontend Setup
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

## File Structure

```
clash-vision/
├── src/clash_vision/
│   ├── inference/
│   │   ├── live_inference.py     # Main application
│   │   ├── screen_preview.py     # Region selection tool
│   │   └── predict_image.py      # Single image inference
│   ├── training/
│   │   └── train.py              # Model training utilities
│   ├── dataset/
│   │   ├── verify_dataset.py     # Dataset validation
│   │   └── resize_images.py      # Image preprocessing
│   └── utils/
│       └── logger.py             # Logging utilities
├── gamestate.py                  # Game state management
└── requirements.txt              # Python dependencies
```

## Development

### Model Training
```bash
python -m clash_vision.training.train --data dataset/data.yaml --epochs 100
```

### Dataset Validation
```bash
python -m clash_vision.dataset.verify_dataset --dataset-root dataset/
```

### Testing
```bash
# Single image inference
python -m clash_vision.inference.predict_image --weights models/best.pt --source test_image.jpg

# Batch processing
python -m clash_vision.inference.predict_image --weights models/best.pt --source test_folder/
```

## Troubleshooting

**No detections**: Check confidence threshold and API key
**High CPU usage**: Increase API interval or reduce FPS
**Capture issues**: Ensure QuickTime window is visible and region is selected
**Web server errors**: Check port availability and CORS settings

## Performance Optimization

- Use `--api_interval 0.5` for competitive analysis
- Set `--nms-threshold 0.2` for stricter duplicate removal
- Enable `--debug` mode to analyze detection quality
- Monitor NMS filtering statistics in overlay

## License

Internal use only.

