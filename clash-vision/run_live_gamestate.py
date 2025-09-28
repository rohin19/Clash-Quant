#!/usr/bin/env python3
"""
Clash Quant Live Inference with Game State Tracking
===================================================

This script runs the live inference system with integrated game state tracking.
High-confidence detections (≥0.5) automatically update elixir counts and game state.

Usage Examples:
  # Fast inference with game state
  python run_live_gamestate.py --fast
  
  # Ultra fast for competitive analysis
  python run_live_gamestate.py --ultra-fast
  
  # Debug mode with frame saving
  python run_live_gamestate.py --debug
  
  # Custom settings
  python run_live_gamestate.py --conf 0.03 --api-interval 0.5 --fps 60

Controls:
  ESC - Exit
  R   - Reset game state
"""

import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run Clash Quant live inference with game state tracking")
    
    # Preset modes
    parser.add_argument("--fast", action="store_true", help="Fast mode: 0.5s API, 30fps, 0.05 conf")
    parser.add_argument("--ultra-fast", action="store_true", help="Ultra fast: 0.2s API, 60fps, 0.03 conf")
    parser.add_argument("--debug", action="store_true", help="Debug mode with frame saving")
    
    # Individual parameters
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold (default: 0.05)")
    parser.add_argument("--api-interval", type=float, default=1.0, help="API call interval in seconds (default: 1.0)")
    parser.add_argument("--fps", type=float, default=30, help="Display FPS (default: 30)")
    parser.add_argument("--width", type=int, default=800, help="Display width (default: 800)")
    parser.add_argument("--no-gamestate", action="store_true", help="Disable game state tracking")
    
    args = parser.parse_args()
    
    # Build command
    cmd = [
        sys.executable, "-m", "clash_vision.inference.live_inference"
    ]
    
    # Apply preset modes
    if args.fast:
        cmd.extend(["--conf", "0.05", "--api_interval", "0.5", "--fps", "30"])
        print("🚀 Fast Mode: 0.5s API calls, 30fps, 0.05 confidence")
    elif args.ultra_fast:
        cmd.extend(["--conf", "0.03", "--api_interval", "0.2", "--fps", "60"])
        print("⚡ Ultra Fast Mode: 0.2s API calls, 60fps, 0.03 confidence")
    else:
        # Use individual parameters
        cmd.extend([
            "--conf", str(args.conf),
            "--api_interval", str(args.api_interval),
            "--fps", str(args.fps)
        ])
    
    # Add other parameters
    cmd.extend(["--width", str(args.width)])
    
    if args.debug:
        cmd.append("--debug")
        print("🔍 Debug mode enabled - will save first 5 frames")
    
    if args.no_gamestate:
        cmd.append("--no-gamestate")
        print("🚫 Game state tracking disabled")
    else:
        print("🎯 Game state tracking enabled (≥0.5 confidence triggers state updates)")
    
    print(f"\n🎮 Starting Clash Quant Live Inference...")
    print(f"   Confidence: {args.conf}")
    print(f"   API Interval: {args.api_interval}s")
    print(f"   Display FPS: {args.fps}")
    print(f"\n⌨️  Controls: ESC=Exit, R=Reset Game State")
    print(f"🚀 Running command: {' '.join(cmd)}\n")
    
    # Change to src directory
    src_dir = Path(__file__).parent / "src"
    
    try:
        # Run the command
        subprocess.run(cmd, cwd=src_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running live inference: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n👋 Interrupted by user")
        return 0
    
    return 0

if __name__ == "__main__":
    exit(main())