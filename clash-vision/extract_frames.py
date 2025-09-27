#!/usr/bin/env python3


import cv2
import os
import sys
from pathlib import Path


def extract_frames(video_path, output_dir, interval_ms=200):
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video properties:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Extracting frames every {interval_ms}ms...")
    
    # Calculate frame interval
    frame_interval = int(fps * interval_ms / 1000)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame at specified intervals
        if frame_count % frame_interval == 0:
            # Generate filename with timestamp
            timestamp_ms = int(frame_count / fps * 1000)
            filename = f"frame_{timestamp_ms:06d}ms.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:  # Progress update every 10 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Saved {saved_count} frames")
        
        frame_count += 1
    
    # Release video capture
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Frames saved to: {output_dir}")
    
    return True


def main():
    
    # Define paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    video_path = data_dir / "raw" / "ronVsHugo.mp4"
    frames_dir = data_dir / "frames"
    
    video_path_str = str(video_path)
    frames_dir_str = str(frames_dir)
    
    print("Clash Royale Frame Extractor")
    print("=" * 40)
    print(f"Video: {video_path_str}")
    print(f"Output: {frames_dir_str}")
    print()
    
    # Extract frames every 200ms
    success = extract_frames(video_path_str, frames_dir_str, interval_ms=1000)
    
    if success:
        print("\nFrame extraction completed successfully!")
    else:
        print("\nFrame extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
