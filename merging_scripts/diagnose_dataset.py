#!/usr/bin/env python3
"""
Debug script to verify class mappings and identify label flipping issues
"""
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import random


def load_yaml_config(yaml_path):
    """Load YAML configuration file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def check_label_files(dataset_path, class_names, sample_size=10):
    """Check a sample of label files to verify class IDs"""
    labels_path = dataset_path / 'train' / 'labels'
    if not labels_path.exists():
        print(f"Labels path not found: {labels_path}")
        return

    label_files = list(labels_path.glob("*.txt"))
    if not label_files:
        print(f"No label files found in {labels_path}")
        return

    # Sample random files
    sample_files = random.sample(label_files, min(sample_size, len(label_files)))

    class_id_counts = Counter()

    print(f"\nAnalyzing {len(sample_files)} label files from {dataset_path.name}:")
    print(f"Expected classes: {class_names}")
    print(f"Class mapping:")
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")

    for label_file in sample_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_id = int(float(parts[0]))
                        class_id_counts[class_id] += 1
                    except ValueError:
                        print(f"Invalid class ID in {label_file.name}: {parts[0]}")

    print(f"\nClass ID distribution in sample:")
    for class_id in sorted(class_id_counts.keys()):
        if class_id < len(class_names):
            class_name = class_names[class_id]
            print(f"  Class {class_id} ({class_name}): {class_id_counts[class_id]} instances")
        else:
            print(f"  Class {class_id} (OUT OF BOUNDS): {class_id_counts[class_id]} instances")

    # Check for out-of-bounds class IDs
    max_valid_id = len(class_names) - 1
    invalid_ids = [cid for cid in class_id_counts.keys() if cid > max_valid_id]
    if invalid_ids:
        print(f"⚠️  WARNING: Found invalid class IDs: {invalid_ids}")
        print(f"   Max valid ID should be: {max_valid_id}")

    return class_id_counts


def compare_datasets():
    """Compare original vs converted datasets"""

    # Define paths - UPDATE THESE TO YOUR ACTUAL PATHS
    datasets = {
        'OBJECT_DETECTION_converted': {
            'path': Path(
                "D:/Compressed/PPE detection files/mywork/converted_datasets/OBJECT DETECTION.v2i.yolov5-obb_converted"),
            'yaml_path': Path(
                "D:/Compressed/PPE detection files/mywork/converted_datasets/OBJECT DETECTION.v2i.yolov5-obb_converted/data.yaml")
        },
        'project2.0_converted': {
            'path': Path(
                "D:/Compressed/PPE detection files/mywork/converted_datasets/project2.0.v2i.yolov5-obb_converted"),
            'yaml_path': Path(
                "D:/Compressed/PPE detection files/mywork/converted_datasets/project2.0.v2i.yolov5-obb_converted/data.yaml")
        },
        'SAIL_PROJECT_converted': {
            'path': Path(
                "D:/Compressed/PPE detection files/mywork/converted_datasets/SAIL PROJECT.v2i.yolov5-obb_converted"),
            'yaml_path': Path(
                "D:/Compressed/PPE detection files/mywork/converted_datasets/SAIL PROJECT.v2i.yolov5-obb_converted/data.yaml")
        }
    }

    print("=" * 60)
    print("DATASET CLASS MAPPING VERIFICATION")
    print("=" * 60)

    for dataset_name, info in datasets.items():
        print(f"\n{'=' * 40}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 40}")

        # Load YAML config
        if info['yaml_path'].exists():
            config = load_yaml_config(info['yaml_path'])
            class_names = config.get('names', [])
            print(f"Classes from YAML ({len(class_names)}): {class_names}")
        else:
            print(f"YAML not found: {info['yaml_path']}")
            continue

        # Check label files
        if info['path'].exists():
            class_counts = check_label_files(info['path'], class_names)
        else:
            print(f"Dataset path not found: {info['path']}")

    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS:")
    print(f"{'=' * 60}")
    print("1. Ensure all datasets use the SAME 9-class list in test_merge.py")
    print("2. Check for any out-of-bounds class IDs (should be 0-8)")
    print("3. Verify that converted datasets have consistent class mappings")
    print("4. If issues persist, re-run the conversion process")


def main():
    """Main function"""
    print("Starting class mapping verification...")
    compare_datasets()

    print(f"\n{'=' * 60}")
    print("NEXT STEPS:")
    print(f"{'=' * 60}")
    print("1. Run this script to identify any remaining issues")
    print("2. Use the fixed test_merge.py with consistent 9-class lists")
    print("3. If problems persist, check the original conversion process")
    print("4. Consider re-converting the problematic dataset")


if __name__ == "__main__":
    main()