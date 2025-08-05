#!/usr/bin/env python3
"""
Improved Smart Frame Extractor for Video Annotation

This script extracts visually distinct frames from videos, optimized for
annotation workflows with better scene detection and distribution features.
"""

import cv2
import os
import json
import re
from pathlib import Path
import argparse
from loguru import logger
import numpy as np
from typing import List, Tuple, Dict


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Replace multiple spaces with single underscore
    filename = re.sub(r'\s+', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Ensure it's not empty
    if not filename:
        filename = "unnamed"
    return filename


def calculate_frame_difference(frame1, frame2, method='mse'):
    """
    Calculates difference between two frames using various methods.

    Args:
        frame1, frame2: Input frames
        method: 'mse', 'ssim', or 'histogram'
    """
    if frame1 is None or frame2 is None:
        return float('inf')

    # Resize for faster comparison
    h, w = 240, 320
    gray1 = cv2.cvtColor(cv2.resize(frame1, (w, h)), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cv2.resize(frame2, (w, h)), cv2.COLOR_BGR2GRAY)

    if method == 'mse':
        # Mean Squared Error
        mse = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)
        return mse
    elif method == 'histogram':
        # Histogram comparison
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    else:
        # Default to MSE
        mse = np.mean((gray1.astype("float") - gray2.astype("float")) ** 2)
        return mse


def detect_scene_changes(video_path: Path, sample_rate: int = 30) -> List[Tuple[int, float]]:
    """
    Pre-analyze video to detect major scene changes.

    Returns:
        List of (frame_number, difference_score) for significant changes
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    scene_changes = []
    prev_frame = None
    frame_num = 0

    logger.info(f"Analyzing scene changes in {video_path.name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every nth frame for efficiency
        if frame_num % sample_rate == 0:
            if prev_frame is not None:
                diff = calculate_frame_difference(frame, prev_frame)
                if diff > 150:  # Significant change threshold
                    scene_changes.append((frame_num, diff))
            prev_frame = frame.copy()

        frame_num += 1

    cap.release()
    logger.info(f"Found {len(scene_changes)} potential scene changes")
    return scene_changes


def extract_frames_smart(video_path: Path, output_dir: Path, config: dict) -> Dict:
    """
    Extracts frames with improved scene detection and distribution.

    Returns:
        Dictionary with extraction statistics
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return {"success": False, "frames_extracted": 0, "error": "Could not open video"}

    # Get video properties - using numerical constants for compatibility
    fps = cap.get(5)  # cv2.CAP_PROP_FPS
    total_frames = int(cap.get(7))  # cv2.CAP_PROP_FRAME_COUNT
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video: {video_path.name}")
    logger.info(f"Properties: FPS={fps:.2f}, Frames={total_frames}, Duration={duration:.1f}s")

    if total_frames == 0:
        cap.release()
        return {"success": False, "frames_extracted": 0, "error": "Empty or corrupted video"}

    if fps <= 0:
        fps = 30  # Default fallback

    # Sanitize video name for filenames
    clean_video_name = sanitize_filename(video_path.stem)

    # Strategy: Extract frames at regular intervals AND at scene changes
    target_frames = min(config['max_frames_per_video'], total_frames // 10)

    # Method 1: Regular interval extraction (ensures coverage)
    interval_frames = max(1, total_frames // (target_frames // 2))

    # Method 2: Scene change detection
    scene_changes = detect_scene_changes(video_path, sample_rate=max(1, int(fps)))

    # Combine both methods
    selected_frames = set()

    # Add regular intervals
    for i in range(0, total_frames, interval_frames):
        if len(selected_frames) < target_frames:
            selected_frames.add(i)

    # Add scene changes (top scoring ones)
    scene_changes.sort(key=lambda x: x[1], reverse=True)
    for frame_num, score in scene_changes[:target_frames // 3]:
        if len(selected_frames) < target_frames:
            selected_frames.add(frame_num)

    # Sort frame numbers for sequential processing
    selected_frames = sorted(list(selected_frames))

    logger.info(f"Selected {len(selected_frames)} frames for extraction")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract selected frames - using sequential reading instead of seeking
    extracted_frames = []
    cap.set(1, 0)  # Reset to beginning (cv2.CAP_PROP_POS_FRAME = 1)

    current_frame = 0
    saved_count = 0
    selected_idx = 0

    while selected_idx < len(selected_frames):
        ret, frame = cap.read()
        if not ret:
            break

        target_frame = selected_frames[selected_idx]

        if current_frame == target_frame:
            timestamp = target_frame / fps
            frame_filename = f"{clean_video_name}_frame_{saved_count + 1:04d}_t{timestamp:.1f}s.jpg"
            save_path = output_dir / frame_filename

            # Ensure the directory exists before writing
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save frame with error handling
            try:
                success = cv2.imwrite(str(save_path), frame)
                if success and save_path.exists():
                    file_size = os.path.getsize(save_path)
                    extracted_frames.append({
                        "filename": frame_filename,
                        "frame_number": target_frame,
                        "timestamp": timestamp,
                        "file_size": file_size
                    })
                    saved_count += 1
                    logger.info(f"Saved: {frame_filename} (t={timestamp:.1f}s)")
                else:
                    logger.error(f"Failed to write frame: {frame_filename} - cv2.imwrite returned {success}")
            except Exception as e:
                logger.error(f"Exception writing frame {frame_filename}: {e}")

            selected_idx += 1
        elif current_frame > target_frame:
            # Skip this target frame as we've passed it
            selected_idx += 1
            continue

        current_frame += 1

    cap.release()

    # Save metadata
    metadata = {
        "video_file": video_path.name,
        "clean_video_name": clean_video_name,
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": duration,
        "frames_extracted": saved_count,
        "extraction_config": config,
        "extracted_frames": extracted_frames
    }

    metadata_filename = f"{clean_video_name}_metadata.json"
    metadata_path = output_dir / metadata_filename

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {metadata_filename}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")

    return {
        "success": True,
        "frames_extracted": saved_count,
        "metadata": metadata
    }


def distribute_for_annotation(output_dir: Path, annotator_count: int = 2):
    """
    Distribute extracted frames among annotators for balanced workload.
    """
    logger.info("Creating annotation distribution...")

    # Find all extracted frames
    frame_files = []
    for video_dir in output_dir.iterdir():
        if video_dir.is_dir():
            frames = list(video_dir.glob("*.jpg"))
            if frames:
                frame_files.extend([(f, video_dir.name) for f in frames])

    if not frame_files:
        logger.warning("No frames found for distribution")
        return

    # Create annotator directories
    for i in range(annotator_count):
        annotator_dir = output_dir / f"annotator_{i + 1}"
        annotator_dir.mkdir(exist_ok=True)

        # Create CSV template for annotations
        csv_template = annotator_dir / "annotations_template.csv"
        with open(csv_template, 'w') as f:
            f.write("video_name,frame_filename,timestamp,annotation_label,notes\n")

    # Distribute frames (round-robin)
    logger.info(f"Distributing {len(frame_files)} frames among {annotator_count} annotators")

    distribution = {f"annotator_{i + 1}": [] for i in range(annotator_count)}

    for idx, (frame_path, video_name) in enumerate(frame_files):
        annotator_id = f"annotator_{(idx % annotator_count) + 1}"
        annotator_dir = output_dir / annotator_id

        # Copy frame to annotator directory
        dest_path = annotator_dir / f"{video_name}_{frame_path.name}"
        import shutil
        shutil.copy2(frame_path, dest_path)

        distribution[annotator_id].append({
            "video": video_name,
            "frame": frame_path.name,
            "path": str(dest_path.relative_to(output_dir))
        })

    # Save distribution info
    for annotator_id, frames in distribution.items():
        dist_file = output_dir / annotator_id / "assigned_frames.json"
        with open(dist_file, 'w') as f:
            json.dump({
                "annotator": annotator_id,
                "total_frames": len(frames),
                "frames": frames
            }, f, indent=2)

        logger.info(f"{annotator_id}: {len(frames)} frames assigned")


def main():
    parser = argparse.ArgumentParser(
        description="Smart frame extraction for video annotation with improved scene detection"
    )
    parser.add_argument("video_dir", type=str, help="Directory containing video files")
    parser.add_argument("output_dir", type=str, help="Output directory for extracted frames")
    parser.add_argument("--max_frames", type=int, default=50, help="Max frames per video (default: 50)")
    parser.add_argument("--annotators", type=int, default=2, help="Number of annotators (default: 2)")
    parser.add_argument("--change_threshold", type=float, default=100, help="Scene change sensitivity")
    parser.add_argument("--distribute", action="store_true", help="Distribute frames among annotators")

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if not video_dir.is_dir():
        logger.error(f"Video directory not found: {video_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = {
        "change_threshold": args.change_threshold,
        "max_frames_per_video": args.max_frames,
        "min_seconds_between_captures": 1.0,
        "max_seconds_between_captures": 10.0
    }

    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv']
    video_files = [f for f in video_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        logger.warning(f"No video files found in {video_dir}")
        return

    logger.info(f"Found {len(video_files)} videos to process")
    logger.info(f"Max frames per video: {args.max_frames}")

    # Process each video
    results = {}
    for video_file in video_files:
        # Sanitize video name for directory creation
        clean_name = sanitize_filename(video_file.stem)
        video_output_dir = output_dir / clean_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"Processing: {video_file.name}")
        logger.info(f"Output dir: {video_output_dir}")
        logger.info("=" * 60)

        result = extract_frames_smart(video_file, video_output_dir, config)
        results[video_file.name] = result

        if result["success"]:
            logger.success(f"✓ Extracted {result['frames_extracted']} frames from {video_file.name}")
        else:
            logger.error(f"✗ Failed to process {video_file.name}: {result.get('error', 'Unknown error')}")

    # Save overall summary
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Distribute frames for annotation if requested
    if args.distribute:
        distribute_for_annotation(output_dir, args.annotators)

    # Print summary
    total_frames = sum(r.get("frames_extracted", 0) for r in results.values())
    successful_videos = sum(1 for r in results.values() if r.get("success", False))

    logger.success("=" * 60)
    logger.success("EXTRACTION COMPLETE!")
    logger.success(f"Videos processed: {successful_videos}/{len(video_files)}")
    logger.success(f"Total frames extracted: {total_frames}")
    if args.distribute:
        logger.success(f"Frames distributed among {args.annotators} annotators")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()