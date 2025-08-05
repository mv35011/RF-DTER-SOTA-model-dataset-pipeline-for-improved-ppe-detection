#!/usr/bin/env python3
"""
Generates a small test set with a specific number of samples for each class,
and also creates visualized images with bounding boxes drawn on them.
"""

import yaml
from pathlib import Path
from collections import defaultdict
import argparse
import random
import shutil
from loguru import logger
import cv2
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 255), (151, 157, 255),
    (56, 56, 255), (157, 151, 255), (112, 255, 255), (255, 255, 112)
]


def generate_samples(yaml_path: str, output_dir: str, num_samples: int):
    """
    Creates a sample test set from a larger dataset by searching all splits.
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.is_file():
        logger.error(f"YAML file not found: {yaml_file}")
        return
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
        class_names = data.get('names', [])
        dataset_root = Path(data.get('path', yaml_file.parent))

    logger.info(f"Source dataset: {dataset_root}")
    logger.info(f"Looking for {num_samples} samples for each of the {len(class_names)} classes across all splits.")
    class_to_images = defaultdict(list)
    logger.info("Indexing images from all splits... this may take a moment.")
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                classes_in_file = set()
                for line in f:
                    try:
                        class_id = int(float(line.strip().split()[0]))
                        classes_in_file.add(class_id)
                    except (ValueError, IndexError):
                        continue
            for cid in classes_in_file:
                class_to_images[cid].append(label_file)
    labels_to_copy = set()
    for class_id, class_name in enumerate(class_names):
        available_labels = class_to_images.get(class_id, [])
        if not available_labels:
            logger.warning(f"No images found for class: '{class_name}'")
            continue
        random.shuffle(available_labels)
        num_to_select = min(num_samples, len(available_labels))
        selected_for_class = available_labels[:num_to_select]
        labels_to_copy.update(selected_for_class)
        logger.info(f"Selected {len(selected_for_class)} samples for '{class_name}'")

    if not labels_to_copy:
        logger.error("No images were selected. Please check the dataset.")
        return
    output_path = Path(output_dir)
    output_images_path = output_path / "images"
    output_labels_path = output_path / "labels"
    output_viz_path = output_path / "visualized_images"
    output_images_path.mkdir(parents=True, exist_ok=True)
    output_labels_path.mkdir(parents=True, exist_ok=True)
    output_viz_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nCopying {len(labels_to_copy)} unique images and labels to '{output_path}'...")
    for label_path in labels_to_copy:
        image_stem = label_path.stem
        split_name = label_path.parent.parent.name
        source_images_dir = dataset_root / split_name / "images"
        source_img_file = next(source_images_dir.glob(f"{image_stem}.*"), None)

        if source_img_file and label_path.exists():
            shutil.copy(source_img_file, output_images_path)
            shutil.copy(label_path, output_labels_path)
            img = cv2.imread(str(source_img_file))
            h, w, _ = img.shape

            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        class_id = int(float(parts[0]))
                        x_center, y_center, width, height = map(float, parts[1:5])
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        color = COLORS[class_id % len(COLORS)]
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = class_names[class_id]
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except (ValueError, IndexError):
                        continue
            cv2.imwrite(str(output_viz_path / source_img_file.name), img)

    logger.info(f"✅ Sample generation complete! Visualized images are in '{output_viz_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a test set with N samples per class.")
    parser.add_argument("yaml_file", type=str, help="Path to the dataset's .yaml file.")
    parser.add_argument("output_dir", type=str, help="Name of the new directory for the samples.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of images to select for each class.")
    args = parser.parse_args()

    generate_samples(args.yaml_file, args.output_dir, args.num_samples)