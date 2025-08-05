#!/usr/bin/env python3
"""
Analyzes the class distribution of a final YOLO dataset.

Reads the dataset's .yaml file to get class names and paths,
then manually counts all annotations in the train, valid, and test sets.
"""

import yaml
from pathlib import Path
from collections import defaultdict
import argparse
from loguru import logger


def analyze_distribution(yaml_path: str):
    """
    Counts annotations and prints the class distribution for a given dataset.

    Args:
        yaml_path (str): The path to the dataset's .yaml configuration file.
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.is_file():
        logger.error(f"YAML file not found: {yaml_file}")
        return

    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data.get('names', [])
            dataset_root = Path(data.get('path', yaml_file.parent))
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        return

    if not class_names:
        logger.error("Could not find 'names' in the YAML file.")
        return

    logger.info(f"Analyzing dataset at: {dataset_root}")
    logger.info(f"Found {len(class_names)} classes: {', '.join(class_names)}")

    class_counts = defaultdict(int)
    total_annotations = 0
    for split in ['train', 'valid', 'test']:
        if split in data:
            labels_dir = dataset_root / data[split].replace('images', 'labels')
            if not labels_dir.exists():
                logger.warning(f"Labels directory for '{split}' split not found at: {labels_dir}")
                continue

            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(float(parts[0]))
                                if 0 <= class_id < len(class_names):
                                    class_counts[class_id] += 1
                                    total_annotations += 1
                                else:
                                    logger.warning(f"Invalid class ID '{class_id}' in {label_file.name}")
                            except ValueError:
                                logger.warning(
                                    f"Could not parse class ID from line '{line.strip()}' in {label_file.name}")
    logger.info("\n" + "=" * 50)
    logger.info("📊 DATASET CLASS DISTRIBUTION REPORT")
    logger.info("=" * 50)
    logger.info(f"Total annotations found: {total_annotations}\n")
    for class_id, class_name in enumerate(class_names):
        count = class_counts.get(class_id, 0)
        percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
        logger.info(f"- {class_name:<20}: {count:>6} annotations ({percentage:>5.1f}%)")

    logger.info("\n" + "=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check class distribution of a YOLO dataset.")
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the dataset's .yaml file (e.g., 'focused_enhanced_dataset/focused_dataset.yaml')."
    )
    args = parser.parse_args()

    analyze_distribution(args.yaml_file)