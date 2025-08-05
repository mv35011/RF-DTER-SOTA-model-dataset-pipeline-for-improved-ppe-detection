#!/usr/bin/env python3
"""
Fixed Dataset Harmonizer for PPE Detection
Handles annotation files with both numeric IDs and class names
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from loguru import logger
from collections import defaultdict
import requests
from tqdm import tqdm


class PPEDatasetHarmonizer:
    def __init__(self, output_dir: str = "harmonized_dataset"):
        """Initialize dataset harmonizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Your current model's classes
        self.target_classes = {
            0: "no-safety-glove",
            1: "no-safety-helmet",
            2: "no-safety-shoes",
            3: "no-welding-glass",
            4: "safety-glove",
            5: "safety-helmet",
            6: "safety-shoes",
            7: "welding-glass"
        }

        # Common PPE class variations found in different datasets
        self.class_mappings = {
            # Helmet variations
            "helmet": "safety-helmet",
            "hard-hat": "safety-helmet",
            "hardhat": "safety-helmet",
            "safety_helmet": "safety-helmet",
            "safety-helmet": "safety-helmet",
            "construction_helmet": "safety-helmet",
            "welding-helmet": "welding-glass",  # Map welding helmet to welding glass
            "no_helmet": "no-safety-helmet",
            "no-helmet": "no-safety-helmet",
            "no_hardhat": "no-safety-helmet",
            "without_helmet": "no-safety-helmet",
            "person": None,  # Skip if only person class

            # Glove variations
            "gloves": "safety-glove",
            "glove": "safety-glove",
            "safety_gloves": "safety-glove",
            "safety-gloves": "safety-glove",
            "safety-glove": "safety-glove",
            "work_gloves": "safety-glove",
            "no_gloves": "no-safety-glove",
            "no-gloves": "no-safety-glove",
            "no_glove": "no-safety-glove",
            "no-safety-glove": "no-safety-glove",
            "without_gloves": "no-safety-glove",

            # Shoe variations
            "safety_shoes": "safety-shoes",
            "safety-shoes": "safety-shoes",
            "safety_boots": "safety-shoes",
            "boots": "safety-shoes",
            "safety-boots": "safety-shoes",
            "shoes": "safety-shoes",
            "protective_footwear": "safety-shoes",
            "no_shoes": "no-safety-shoes",
            "no-safety-shoes": "no-safety-shoes",
            "no_boots": "no-safety-shoes",
            "without_shoes": "no-safety-shoes",

            # Glasses/Goggles variations
            "glasses": "welding-glass",
            "goggles": "welding-glass",
            "safety_glasses": "welding-glass",
            "safety-goggles": "welding-glass",
            "eye_protection": "welding-glass",
            "welding-glass": "welding-glass",
            "no_glasses": "no-welding-glass",
            "no-goggles": "no-welding-glass",
            "no_goggles": "no-welding-glass",
            "no-welding-glass": "no-welding-glass",
            "without_glasses": "no-welding-glass",

            # Additional mappings
            "mask": None,  # Skip mask if not in your model
            "no_mask": None,  # Skip mask if not in your model

            # Vest variations (map to closest class or skip)
            "vest": None,  # Skip if not in your model
            "safety_vest": None,
            "reflective_vest": None,
        }

        # Reverse mapping for target classes
        self.target_to_index = {v: k for k, v in self.target_classes.items()}

    def discover_datasets(self) -> List[Dict]:
        """Discover available PPE datasets"""
        datasets = [
            {
                "name": "Roboflow Hard Hat Workers Dataset",
                "url": "https://universe.roboflow.com/roboflow-universe-projects/hard-hat-workers-voc",
                "format": "yolov5",
                "classes": ["helmet", "no_helmet", "person"],
                "size": "5k+ images",
                "focus": "helmet detection"
            },
            {
                "name": "Construction Site Safety",
                "url": "https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety",
                "format": "yolov8",
                "classes": ["hardhat", "no-hardhat", "person", "safety-vest"],
                "size": "3k images",
                "focus": "helmet and vest"
            },
            {
                "name": "PPE Detection Dataset",
                "url": "https://universe.roboflow.com/objectdetection-instinct/ppe-detection-cctv",
                "format": "yolov5",
                "classes": ["gloves", "no_gloves", "helmet", "no_helmet", "boots", "no_boots"],
                "size": "2k images",
                "focus": "multiple PPE types"
            },
            {
                "name": "Safety Equipment Detection",
                "url": "https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection",
                "format": "pascal_voc",
                "classes": ["helmet", "head", "person"],
                "size": "5k images",
                "focus": "helmet detection"
            },
            {
                "name": "PPE Violation Dataset",
                "url": "https://universe.roboflow.com/ppe-violation/ppe-violation-detection",
                "format": "yolov5",
                "classes": ["helmet", "no-helmet", "vest", "no-vest", "gloves", "no-gloves"],
                "size": "1.5k images",
                "focus": "violation detection"
            }
        ]

        logger.info(f"Found {len(datasets)} potential datasets for transfer learning")
        return datasets

    def download_roboflow_dataset(self, dataset_key: str, api_key: str, format: str = "yolov9"):
        """Download dataset from Roboflow"""
        try:
            from roboflow import Roboflow

            rf = Roboflow(api_key=api_key)
            workspace, project, version = dataset_key.split("/")

            project = rf.workspace(workspace).project(project)
            dataset = project.version(int(version))

            # Download in YOLOv9 format (or closest available)
            download_path = self.output_dir / "raw_downloads" / dataset_key.replace("/", "_")

            if format == "yolov9":
                # Try YOLOv8 format first (closest to v9)
                try:
                    dataset.download("yolov8", location=str(download_path))
                except:
                    # Fall back to YOLOv5
                    dataset.download("yolov5", location=str(download_path))
            else:
                dataset.download(format, location=str(download_path))

            return download_path

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return None

    def convert_annotation_format(self, source_format: str, target_format: str,
                                  annotation_path: str, image_shape: Tuple[int, int]) -> List[List[float]]:
        """Convert between annotation formats"""

        if source_format == "pascal_voc":
            return self._convert_voc_to_yolo(annotation_path, image_shape)
        elif source_format == "coco":
            return self._convert_coco_to_yolo(annotation_path, image_shape)
        elif source_format in ["yolov5", "yolov8", "yolov9"]:
            # Already in YOLO format, just read
            return self._read_yolo_annotation(annotation_path)
        else:
            logger.warning(f"Unknown format: {source_format}")
            return []

    def _read_yolo_annotation(self, annotation_path: str) -> List[List[float]]:
        """Read YOLO format annotation - handles both numeric IDs and class names"""
        annotations = []

        try:
            with open(annotation_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        logger.warning(f"Invalid annotation in {annotation_path} line {line_num}: {line}")
                        continue

                    try:
                        # Try to convert first element to float (class ID)
                        class_id = float(parts[0])
                        # Convert coordinates
                        coords = [float(x) for x in parts[1:5]]
                        annotations.append([class_id] + coords)

                    except ValueError:
                        # First element is a class name, not numeric ID
                        class_name = parts[0]
                        logger.info(f"Found class name '{class_name}' in {annotation_path}")

                        # Map class name to target class
                        mapped_class = self.map_class_name(class_name)
                        if mapped_class and mapped_class in self.target_to_index:
                            class_id = self.target_to_index[mapped_class]
                            try:
                                coords = [float(x) for x in parts[1:5]]
                                annotations.append([float(class_id)] + coords)
                            except ValueError as coord_error:
                                logger.error(f"Invalid coordinates in {annotation_path} line {line_num}: {coord_error}")
                                continue
                        else:
                            logger.warning(f"Skipping unmapped class '{class_name}' in {annotation_path}")

        except Exception as e:
            logger.error(f"Error reading annotation file {annotation_path}: {e}")

        return annotations

    def _convert_voc_to_yolo(self, xml_path: str, image_shape: Tuple[int, int]) -> List[List[float]]:
        """Convert Pascal VOC to YOLO format"""
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        root = tree.getroot()

        height, width = image_shape
        annotations = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower()
            bbox = obj.find('bndbox')

            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to YOLO format (normalized center coordinates)
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # Map class
            mapped_class = self.map_class_name(class_name)
            if mapped_class and mapped_class in self.target_to_index:
                class_id = self.target_to_index[mapped_class]
                annotations.append([class_id, x_center, y_center, w, h])

        return annotations

    def map_class_name(self, original_class: str) -> Optional[str]:
        """Map external dataset class to your model's classes"""
        original_lower = original_class.lower().strip()

        # Direct mapping
        if original_lower in self.class_mappings:
            return self.class_mappings[original_lower]

        # Check if it contains key terms
        for key, target in self.class_mappings.items():
            if key in original_lower:
                return target

        logger.warning(f"Unknown class: {original_class}")
        return None

    def harmonize_dataset(self, dataset_path: Path, dataset_info: Dict) -> Dict[str, int]:
        """Harmonize a dataset to match your model's classes with corrected logic."""
        logger.info(f"Harmonizing dataset: {dataset_info['name']}")

        stats = defaultdict(int)

        # Create output structure
        for split in ['train', 'valid', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Process each split
        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                continue

            images_path = split_path / 'images'
            labels_path = split_path / 'labels'

            if not images_path.exists():
                continue

            # Process each image
            image_count_in_split = 0
            for img_file in images_path.glob("*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                label_file = labels_path / f"{img_file.stem}.txt"
                if not label_file.exists():
                    continue

                original_annotations = self._read_yolo_annotation(str(label_file))

                # This list will hold the newly mapped annotations for the current file
                mapped_annotations = []

                for ann in original_annotations:
                    if len(ann) < 5:
                        continue

                    orig_class_id = int(ann[0])

                    # --- THIS IS THE CORRECTED LOGIC ---
                    # Always look up the original name and map it.
                    if orig_class_id < len(dataset_info['classes']):
                        orig_class_name = dataset_info['classes'][orig_class_id]
                        mapped_class = self.map_class_name(orig_class_name)

                        if mapped_class and mapped_class in self.target_to_index:
                            new_class_id = self.target_to_index[mapped_class]
                            mapped_ann = [new_class_id] + ann[1:]
                            mapped_annotations.append(mapped_ann)
                            stats[mapped_class] += 1
                    else:
                        logger.warning(
                            f"Class ID {orig_class_id} out of bounds for dataset {dataset_info['name']}. Skipping.")

                # Save the image and the new label file ONLY if it has valid, mapped annotations
                if mapped_annotations:
                    # Copy image
                    dst_img = self.output_dir / split / 'images' / img_file.name
                    shutil.copy(str(img_file), str(dst_img))

                    # Save the new, correctly mapped labels
                    dst_label = self.output_dir / split / 'labels' / f"{img_file.stem}.txt"
                    with open(dst_label, 'w') as f:
                        for mapped_ann in mapped_annotations:
                            f.write(' '.join(map(str, mapped_ann)) + '\n')

                    image_count_in_split += 1

            stats['total_images'] += image_count_in_split

        return dict(stats)

    def create_merged_yaml(self, dataset_stats: List[Dict[str, int]]):
        """Create YAML file for merged dataset"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.target_classes),
            'names': list(self.target_classes.values()),

            # Add statistics
            'dataset_info': {
                'total_images': sum(s.get('total_images', 0) for s in dataset_stats),
                'class_distribution': {}
            }
        }

        # Aggregate class distribution
        for class_name in self.target_classes.values():
            total = sum(s.get(class_name, 0) for s in dataset_stats)
            yaml_content['dataset_info']['class_distribution'][class_name] = total

        # Save YAML
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        logger.info(f"Created dataset YAML: {yaml_path}")
        return yaml_path


def main():
    """Main demonstration"""
    harmonizer = PPEDatasetHarmonizer()

    # Discover available datasets
    logger.info("Discovering PPE datasets...")
    datasets = harmonizer.discover_datasets()

    logger.info("\nAvailable datasets for transfer learning:")
    for i, ds in enumerate(datasets):
        logger.info(f"\n{i + 1}. {ds['name']}")
        logger.info(f"   URL: {ds['url']}")
        logger.info(f"   Classes: {ds['classes']}")
        logger.info(f"   Format: {ds['format']}")
        logger.info(f"   Size: {ds['size']}")
        logger.info(f"   Focus: {ds['focus']}")

    # Show class mapping examples
    logger.info("\nClass mapping examples:")
    test_classes = ["helmet", "no_helmet", "boots", "no-boots", "safety_gloves", "person"]
    for cls in test_classes:
        mapped = harmonizer.map_class_name(cls)
        logger.info(f"  '{cls}' -> '{mapped}'")


if __name__ == "__main__":
    main()