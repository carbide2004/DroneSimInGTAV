import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_action(entry):
    """Extract action from messages"""
    msgs = entry.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            s = str(m.get("content", "")).strip()
            if s:
                return s
    return "No action"


def _get_pose(entry):
    """Extract pose information from messages"""
    msgs = entry.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            content = m.get("content", "")
            if "Current Pose:" in content:
                # Extract pose line
                lines = content.split('\n')
                for line in lines:
                    if line.strip().startswith("Current Pose:"):
                        return line.strip()
    return "Pose: Unknown"


def _wrap_text(text, max_width, font):
    """Wrap text to fit within max_width"""
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        if bbox[2] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def _create_info_panel(action, pose, awareness, panel_width=600, panel_height=400):
    """Create an information panel with action, pose, and awareness"""
    # Create a white background
    panel = Image.new('RGB', (panel_width, panel_height), color='white')
    draw = ImageDraw.Draw(panel)
    
    try:
        # Try to load a font, fallback to default if not available
        font_large = ImageFont.truetype("arial.ttf", 16)
        font_medium = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    y_offset = 10
    margin = 10
    
    # Draw Action
    draw.text((margin, y_offset), "Action:", fill='black', font=font_large)
    y_offset += 25
    draw.text((margin, y_offset), action, fill='blue', font=font_medium)
    y_offset += 35
    
    # Draw Pose
    draw.text((margin, y_offset), "Pose:", fill='black', font=font_large)
    y_offset += 25
    draw.text((margin, y_offset), pose, fill='green', font=font_small)
    y_offset += 35
    
    # Draw Awareness
    draw.text((margin, y_offset), "Awareness:", fill='black', font=font_large)
    y_offset += 25
    
    # Split awareness into lines and wrap them
    awareness_lines = awareness.split('\n')
    for line in awareness_lines:
        if line.strip():
            wrapped_lines = _wrap_text(line.strip(), panel_width - 2 * margin, font_small)
            for wrapped_line in wrapped_lines:
                if y_offset < panel_height - 20:
                    draw.text((margin, y_offset), wrapped_line, fill='red', font=font_small)
                    y_offset += 18
    
    return panel


def _load_and_resize_image(image_path, target_size=(640, 480)):
    """Load and resize image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Create a placeholder image
        img = Image.new('RGB', target_size, color='gray')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Image not found:\n{image_path}", fill='white')
        return img


def _combine_images(rgb_img, info_panel):
    """Combine RGB image and info panel horizontally"""
    total_width = rgb_img.width + info_panel.width
    total_height = max(rgb_img.height, info_panel.height)
    
    combined = Image.new('RGB', (total_width, total_height), color='black')
    combined.paste(rgb_img, (0, 0))
    combined.paste(info_panel, (rgb_img.width, 0))
    
    return combined


def main():
    parser = argparse.ArgumentParser(description="Visualize train_data_all_with_awareness.json")
    parser.add_argument(
        "--input_json",
        default=str(_repo_root() / "dataset" / "train_data_all_with_awareness.json"),
        help="Input JSON file with awareness data"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second (playback speed)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start from this entry index"
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=-1,
        help="Maximum number of entries to show (-1 for all)"
    )
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=[640, 480],
        help="RGB image display size (width height)"
    )
    parser.add_argument(
        "--panel_size",
        nargs=2,
        type=int,
        default=[600, 480],
        help="Info panel size (width height)"
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Save each frame as an image file"
    )
    parser.add_argument(
        "--output_dir",
        default=str(_repo_root() / "visualize" / "output_frames"),
        help="Directory to save frames (if --save_frames is used)"
    )
    
    args = parser.parse_args()
    
    # Read data
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading data from: {input_path}")
    data = _read_json(input_path)
    
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list")
    
    # Determine range
    start_idx = max(0, args.start_index)
    if args.max_entries > 0:
        end_idx = min(len(data), start_idx + args.max_entries)
    else:
        end_idx = len(data)
    
    print(f"Showing entries {start_idx} to {end_idx-1} ({end_idx-start_idx} total)")
    print(f"Playback speed: {args.fps} FPS")
    print("Press 'q' to quit, 'p' to pause/resume, 'n' for next frame, 'b' for previous frame")
    
    # Setup output directory if saving frames
    if args.save_frames:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {output_dir}")
    
    # Initialize variables
    root = _repo_root()
    current_idx = start_idx
    paused = False
    frame_delay = 1.0 / args.fps
    
    cv2.namedWindow('Awareness Viewer', cv2.WINDOW_NORMAL)
    
    while current_idx < end_idx:
        entry = data[current_idx]
        
        # Extract information
        action = _get_action(entry)
        pose = _get_pose(entry)
        awareness = entry.get("awareness", "No awareness data")
        
        # Get image paths
        images = entry.get("images", [])
        if not images:
            print(f"No images found for entry {current_idx}")
            current_idx += 1
            continue
        
        rgb_path = root / "dataset" / images[0]
        
        # Load and process images
        rgb_img = _load_and_resize_image(rgb_path, tuple(args.image_size))
        info_panel = _create_info_panel(action, pose, awareness, 
                                      args.panel_size[0], args.panel_size[1])
        
        # Combine images
        combined_img = _combine_images(rgb_img, info_panel)
        
        # Convert to OpenCV format for display
        combined_cv = cv2.cvtColor(np.array(combined_img), cv2.COLOR_RGB2BGR)
        
        # Add frame info
        frame_info = f"Frame {current_idx}/{end_idx-1} | FPS: {args.fps:.1f}"
        cv2.putText(combined_cv, frame_info, (10, combined_cv.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Awareness Viewer', combined_cv)
        
        # Save frame if requested
        if args.save_frames:
            frame_filename = output_dir / f"frame_{current_idx:06d}.png"
            combined_img.save(frame_filename)
        
        # Handle keyboard input
        if paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('n'):
            current_idx += 1
        elif key == ord('b'):
            current_idx = max(start_idx, current_idx - 1)
        else:
            if not paused:
                current_idx += 1
    
    cv2.destroyAllWindows()
    print("Visualization completed!")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())