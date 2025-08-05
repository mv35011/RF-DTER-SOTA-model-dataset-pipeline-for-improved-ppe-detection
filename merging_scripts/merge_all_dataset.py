#!/usr/bin/env python3
"""
A robust script to merge three disparate YOLO datasets into a single,
consistent 10-class dataset for final model training.
"""
import shutil
from pathlib import Path
import yaml
from loguru import logger
from collections import defaultdict

# --- 1. DEFINE THE FINAL 10-CLASS STRUCTURE ---
# This is our "source of truth" based on your master dataset.
FINAL_CLASS_NAMES = [
    'boots', 'gloves', 'goggles', 'helmet',
    'no-boots', 'no-gloves', 'no-goggles', 'no-helmet',
    'no-vest', 'vest'
]

# This master dictionary maps all possible input class names to the final names.
FINAL_CLASS_MAP = {
    # Perfect Matches from master list
    'boots': 'boots', 'gloves': 'gloves', 'goggles': 'goggles', 'helmet': 'helmet',
    'no-boots': 'no-boots', 'no-gloves': 'no-gloves', 'no-goggles': 'no-goggles',
    'no-helmet': 'no-helmet', 'no-vest': 'no-vest', 'vest': 'vest',

    # Variations from other datasets
    'safety-shoes': 'boots',
    'no-safety-shoes': 'no-boots',
    'safety-glove': 'gloves',
    'no-safety-glove': 'no-gloves',
    'safety-helmet': 'helmet',
    'no-safety-helmet': 'no-helmet',

    # --- THIS IS THE FIX ---
    # Classes to explicitly ignore from all source datasets
    'person': None,
    'welding-glass': None,
    'no-welding-glass': None,
}

# Create the final mapping from name to ID
FINAL_NAME_TO_ID = {name: i for i, name in enumerate(FINAL_CLASS_NAMES)}


def harmonize_and_merge(source_datasets: dict, output_dir: str):
    """
    Reads multiple source datasets, maps their classes to a final unified
    format, and copies them into a single output directory.
    """
    output_path = Path(output_dir)
    logger.info(f"Starting unified merge. Output will be in: {output_path}")

    if output_path.exists():
        logger.warning(f"Output directory '{output_path}' already exists. Deleting it to start fresh.")
        shutil.rmtree(output_path)

    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Process each source dataset
    for name, info in source_datasets.items():
        source_path = Path(info['path'])
        source_classes = info['classes']
        logger.info(f"Processing: {name}...")

        for split in ['train', 'valid', 'test']:
            images_dir = source_path / split / 'images'
            labels_dir = source_path / split / 'labels'

            if not labels_dir.exists(): continue

            for label_file in labels_dir.glob('*.txt'):
                # Find corresponding image file, checking multiple extensions
                img_file = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_img_file = images_dir / f"{label_file.stem}{ext}"
                    if potential_img_file.exists():
                        img_file = potential_img_file
                        break
                if not img_file:
                    continue

                with open(label_file, 'r') as f:
                    lines = f.readlines()

                new_annotations = []
                for line in lines:
                    try:
                        parts = line.strip().split()
                        if not parts: continue
                        source_class_id = int(parts[0])

                        if source_class_id < len(source_classes):
                            source_class_name = source_classes[source_class_id]

                            if source_class_name in FINAL_CLASS_MAP:
                                final_class_name = FINAL_CLASS_MAP[source_class_name]
                                if final_class_name is None:  # Skip ignored classes
                                    continue

                                final_class_id = FINAL_NAME_TO_ID[final_class_name]
                                new_line = f"{final_class_id} {' '.join(parts[1:])}"
                                new_annotations.append(new_line)
                    except (ValueError, IndexError):
                        continue

                if new_annotations:
                    # Prepend dataset name to avoid file name collisions
                    new_file_name = f"{name}_{img_file.name}"
                    shutil.copy(img_file, output_path / split / 'images' / new_file_name)
                    with open(output_path / split / 'labels' / f"{name}_{label_file.stem}.txt", 'w') as f:
                        f.write('\n'.join(new_annotations))

    # Create the final data.yaml
    final_yaml_data = {
        'path': str(output_path.absolute()),
        'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
        'nc': len(FINAL_CLASS_NAMES),
        'names': FINAL_CLASS_NAMES
    }
    with open(output_path / "dataset.yaml", 'w') as f:
        yaml.dump(final_yaml_data, f, default_flow_style=False, sort_keys=False)

    logger.success(f"🚀 Merge complete! Your final dataset is ready in '{output_dir}'.")


if __name__ == "__main__":
    # --- DEFINE YOUR THREE SOURCE DATASETS HERE ---

    datasets_to_merge = {
        "sirs_and_annotated": {
            "path": Path("D:/Compressed/PPE detection files/mywork/clean_and_merge/sirs_and_annotated"),
            "classes": ['boots', 'gloves', 'goggles', 'helmet', 'no-boots', 'no-gloves', 'no-goggles', 'no-helmet',
                        'no-vest', 'vest']
        },
        "construction_safety": {
            "path": Path("D:/Compressed/PPE detection files/mywork/clean_and_merge/construction safety.v2-release.yolov9"),
            "classes": ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']
        },
        "final_merged_dataset": {
            "path": Path("D:/Compressed/PPE detection files/mywork/clean_and_merge/final_merged_dataset"),
            "classes": ['no-safety-glove', 'no-safety-helmet', 'no-safety-shoes', 'no-welding-glass', 'safety-glove',
                        'safety-helmet', 'safety-shoes', 'welding-glass']
        }
    }

    harmonize_and_merge(datasets_to_merge, "final_10_class_dataset")
