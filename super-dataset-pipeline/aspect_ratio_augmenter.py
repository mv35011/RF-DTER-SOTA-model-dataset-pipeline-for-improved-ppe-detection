#!/usr/bin/env python3
"""
A specialized augmentation script that creates new versions of images
with varied aspect ratios to help the model detect seated or crouching people.
"""

import os
import cv2
import yaml
from pathlib import Path
import albumentations as A
from tqdm import tqdm
from loguru import logger
import argparse
import re


class YOLOAspectRatioAugmenter:
    def __init__(self, base_path, output_path):
        """
        Initialize the augmentation pipeline for aspect ratio changes.

        Args:
            base_path: The source dataset directory (e.g., 'final_unified_dataset').
            output_path: Directory for the new, aspect-ratio-augmented dataset.
        """
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.class_names = self.load_class_names()
        if not self.class_names:
            raise ValueError("Could not load class names from source dataset's YAML.")
        self.create_output_structure()
        self.setup_augmentations()

    def sanitize_filename(self, name: str) -> str:
        """Removes characters that are forbidden in filenames."""
        return re.sub(r"[^a-zA-Z0-9_.-]", "", name)

    def load_class_names(self):
        """Loads the class names from the source dataset's data.yaml."""
        yaml_path = self.base_path / 'dataset.yaml'
        if not yaml_path.exists():
            logger.error(f"dataset.yaml not found in {self.base_path}")
            return None
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('names', [])

    def create_output_structure(self):
        """Create the output directory structure."""
        for split in ['train', 'valid', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    def setup_augmentations(self):
        """Setup the specialized augmentation pipelines."""
        self.augmentation_sets = {
            'aspect_ratio_wide': A.Compose([
                A.Resize(height=480, width=854, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

            'aspect_ratio_tall': A.Compose([
                A.Resize(height=854, width=480, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

            'sitting_focused': A.Compose([
                A.Resize(height=480, width=640, p=1.0),
                A.RandomCrop(height=400, width=580, p=0.7),
                A.Resize(height=640, width=640, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.4),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        }

    def read_yolo_labels(self, label_path):
        """Read and validate YOLO format labels and class IDs."""
        bboxes = []
        class_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(float(parts[0]))
                            x_center, y_center, width, height = [float(p) for p in parts[1:5]]
                            if not (0 < width <= 1 and 0 < height <= 1):
                                logger.warning(
                                    f"Invalid width/height in {label_path.name} on line {line_num}. Skipping: {line.strip()}")
                                continue
                            x_min = x_center - width / 2
                            x_max = x_center + width / 2
                            y_min = y_center - height / 2
                            y_max = y_center + height / 2

                            if not (x_min >= 0 and x_max <= 1 and y_min >= 0 and y_max <= 1):
                                logger.warning(
                                    f"Bbox outside bounds in {label_path.name} on line {line_num}. Skipping: {line.strip()}")
                                continue

                            class_labels.append(class_id)
                            bboxes.append([x_center, y_center, width, height])
                        except ValueError:
                            logger.warning(
                                f"Could not parse line in {label_path.name} on line {line_num}. Skipping: {line.strip()}")
                            continue
        return bboxes, class_labels

    def write_yolo_labels(self, label_path, bboxes, class_labels):
        """Write YOLO format labels."""
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def run_pipeline(self):
        """Run the complete augmentation pipeline."""
        logger.info("Starting Aspect Ratio Augmentation Pipeline")
        logger.info(f"Source dataset: {self.base_path}")
        logger.info(f"Output path: {self.output_path}")
        for split in ['train', 'valid', 'test']:
            images_dir = self.base_path / split / 'images'
            labels_dir = self.base_path / split / 'labels'

            if not images_dir.exists():
                logger.warning(f"Directory not found, skipping: {images_dir}")
                continue

            image_files = [f for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

            for image_path in tqdm(image_files, desc=f"Augmenting {split} split"):
                label_path = labels_dir / f"{image_path.stem}.txt"

                image = cv2.imread(str(image_path))
                if image is None: continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                bboxes, class_labels = self.read_yolo_labels(label_path)
                for aug_name, aug_transform in self.augmentation_sets.items():
                    try:
                        if not bboxes:
                            continue

                        augmented = aug_transform(image=image, bboxes=bboxes, class_labels=class_labels)
                        aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                        sanitized_stem = self.sanitize_filename(image_path.stem)
                        output_image_name = f"{sanitized_stem}_{aug_name}{image_path.suffix}"
                        output_image_path = self.output_path / split / 'images' / output_image_name
                        cv2.imwrite(str(output_image_path), aug_image)
                        output_label_path = self.output_path / split / 'labels' / f"{sanitized_stem}_{aug_name}.txt"
                        self.write_yolo_labels(output_label_path, augmented['bboxes'], augmented['class_labels'])
                    except Exception as e:
                        logger.error(f"Failed to augment {image_path.name} with {aug_name}. Error: {e}")
        self.create_dataset_yaml()
        logger.success(f"Augmentation complete. New dataset saved to: {self.output_path}")

    def create_dataset_yaml(self):
        """Create the dataset.yaml file for the new augmented dataset."""
        yaml_content = {
            'path': str(self.output_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Created new dataset YAML at: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply aspect ratio augmentations to a YOLO dataset.')
    parser.add_argument('--source_dataset', required=True,
                        help='Path to the source dataset directory (e.g., final_unified_dataset).')
    parser.add_argument('--output_dir', required=True, help='Path to save the new augmented dataset.')
    args = parser.parse_args()
    pipeline = YOLOAspectRatioAugmenter(args.source_dataset, args.output_dir)
    pipeline.run_pipeline()
