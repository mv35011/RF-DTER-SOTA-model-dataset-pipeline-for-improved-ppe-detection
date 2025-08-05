#!/usr/bin/env python3
"""
Converts a dataset from a custom format (ID, polygon, name, etc.)
to the standard YOLO format (class_id, x_center, y_center, width, height).
FIXED VERSION: Resolves class mapping inconsistencies
"""
import cv2
import yaml
from pathlib import Path
import argparse
from loguru import logger
import shutil

# FIXED: This dictionary maps the class NAMES found in your custom files
# to the final class IDs (0-7) your model needs.
# welding-helmet is mapped to welding-glass, but we only include one in the final names
TARGET_CLASS_MAP = {
    "no-safety-glove": 0,
    "no-safety-helmet": 1,
    "no-safety-shoes": 2,
    "no-welding-glass": 3,
    "safety-glove": 4,
    "safety-helmet": 5,
    "safety-shoes": 6,
    "welding-glass": 7,
    "welding-helmet": 7  # Map 'welding-helmet' to the same class as 'welding-glass'
}

# FIXED: Define the final 8 class names explicitly (no duplicates)
FINAL_CLASS_NAMES = [
    "no-safety-glove",  # 0
    "no-safety-helmet",  # 1
    "no-safety-shoes",  # 2
    "no-welding-glass",  # 3
    "safety-glove",  # 4
    "safety-helmet",  # 5
    "safety-shoes",  # 6
    "welding-glass"  # 7 (welding-helmet maps to this)
]


def convert_dataset(input_dir: str, output_dir: str):
    """
    Converts a single dataset to standard YOLO format.
    """
    source_path = Path(input_dir)
    output_path = Path(output_dir)

    logger.info(f"Converting dataset: {source_path.name}")
    logger.info(f"Outputting to: {output_path}")

    # We only care about the original training data
    source_images = source_path / "train" / "images"
    source_labels = source_path / "train" / "labelTxt"

    if not source_images.exists() or not source_labels.exists():
        logger.error(f"Could not find 'train/images' or 'train/labelTxt' in {source_path}")
        return

    # Create output structure
    (output_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "train" / "labels").mkdir(parents=True, exist_ok=True)

    converted_files = 0
    class_usage_stats = {}

    for label_file in source_labels.glob("*.txt"):
        # Find corresponding image file
        img_file = next(source_images.glob(f"{label_file.stem}.*"), None)
        if not img_file:
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"Could not read image: {img_file}")
            continue
        img_h, img_w, _ = img.shape

        new_annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Expected format: [id, x1, y1, x2, y2, x3, y3, x4, y4, class_name, flag]
                if len(parts) < 10:
                    continue

                class_name = parts[-2]
                if class_name in TARGET_CLASS_MAP:
                    new_class_id = TARGET_CLASS_MAP[class_name]

                    # FIXED: Track class usage for debugging
                    if class_name not in class_usage_stats:
                        class_usage_stats[class_name] = 0
                    class_usage_stats[class_name] += 1

                    # Extract the 8 coordinate points
                    try:
                        coords = [float(p) for p in parts[1:9]]
                        x_coords = [coords[i] for i in range(0, 8, 2)]
                        y_coords = [coords[i] for i in range(1, 8, 2)]
                    except ValueError:
                        logger.warning(f"Skipping line with invalid coordinates in {label_file.name}")
                        continue

                    # Create a standard bounding box
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)

                    # Convert to YOLO format (normalized center x, center y, width, height)
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    center_x = x_min + box_w / 2
                    center_y = y_min + box_h / 2

                    yolo_x = center_x / img_w
                    yolo_y = center_y / img_h
                    yolo_w = box_w / img_w
                    yolo_h = box_h / img_h

                    new_annotations.append(f"{new_class_id} {yolo_x} {yolo_y} {yolo_w} {yolo_h}")

        if new_annotations:
            # Copy the image to the new clean directory
            shutil.copy(img_file, output_path / "train" / "images")
            # Write the new, clean label file
            with open(output_path / "train" / "labels" / label_file.name, 'w') as f:
                f.write("\n".join(new_annotations))
            converted_files += 1

    # FIXED: Create the correct data.yaml for the new clean dataset
    final_yaml_data = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 8,  # Only 8 actual classes
        'names': FINAL_CLASS_NAMES  # Use the explicit 8-class list
    }
    with open(output_path / "data.yaml", 'w') as f:
        yaml.dump(final_yaml_data, f, default_flow_style=False, sort_keys=False)

    # FIXED: Print class mapping statistics
    logger.info("Class mapping statistics:")
    for original_class, count in class_usage_stats.items():
        mapped_id = TARGET_CLASS_MAP[original_class]
        mapped_name = FINAL_CLASS_NAMES[mapped_id]
        if original_class == mapped_name:
            logger.info(f"  {original_class} -> ID {mapped_id} ({count} instances)")
        else:
            logger.info(f"  {original_class} -> ID {mapped_id} ({mapped_name}) ({count} instances)")

    logger.success(f"Successfully converted {converted_files} files for {source_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom format to standard YOLO.")
    parser.add_argument("input_dir", type=str, help="Path to the source dataset directory.")
    parser.add_argument("output_dir", type=str, help="Path to the new directory for the clean dataset.")
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_dir)