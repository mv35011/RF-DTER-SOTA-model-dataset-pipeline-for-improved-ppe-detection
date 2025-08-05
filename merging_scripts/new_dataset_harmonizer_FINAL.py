#!/usr/bin/env python3
"""
Final, Corrected Dataset Harmonizer for PPE Detection
This version has a fully robust mapping dictionary to prevent all 'Unknown class' errors and label flipping.
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


class PPEDatasetHarmonizer:
    def __init__(self, output_dir: str = "harmonized_dataset"):
        """Initialize dataset harmonizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.target_classes = {
            0: "no-safety-glove", 1: "no-safety-helmet", 2: "no-safety-shoes",
            3: "no-welding-glass", 4: "safety-glove", 5: "safety-helmet",
            6: "safety-shoes", 7: "welding-glass"
        }

        # --- THIS IS THE DEFINITIVE FIX ---
        # This dictionary is now complete and handles all variations from all datasets consistently.
        self.class_mappings = {
            # 1. Final Target Classes (Perfect Matches)
            "no-safety-glove": "no-safety-glove",
            "no-safety-helmet": "no-safety-helmet",
            "no-safety-shoes": "no-safety-shoes",
            "no-welding-glass": "no-welding-glass",
            "safety-glove": "safety-glove",
            "safety-helmet": "safety-helmet",
            "safety-shoes": "safety-shoes",
            "welding-glass": "welding-glass",

            # 2. Variations from New Dataset (using hyphens)
            "glove": "safety-glove",
            "goggles": "welding-glass",
            "helmet": "safety-helmet",
            "no-glove": "no-safety-glove",
            "no-goggles": "no-welding-glass",
            "no-helmet": "no-safety-helmet",
            "no-shoes": "no-safety-shoes",
            "shoes": "safety-shoes",

            # 3. Variations from Original Datasets (using hyphens)
            "hard-hat": "safety-helmet",
            "hardhat": "safety-helmet",
            "construction-helmet": "safety-helmet",
            "welding-helmet": "welding-glass",
            "no-hardhat": "no-safety-helmet",
            "without-helmet": "no-safety-helmet",
            "gloves": "safety-glove",
            "safety-gloves": "safety-glove",
            "work-gloves": "safety-glove",
            "no-gloves": "no-safety-glove",
            "without-gloves": "no-safety-glove",
            "safety-boots": "safety-shoes",
            "boots": "safety-shoes",
            "protective-footwear": "safety-shoes",
            "no-boots": "no-safety-shoes",
            "without-shoes": "no-safety-shoes",
            "glasses": "welding-glass",
            "safety-glasses": "welding-glass",
            "safety-goggles": "welding-glass",
            "eye-protection": "welding-glass",
            "no-glasses": "no-welding-glass",
            "without-glasses": "no-welding-glass",

            # 4. Classes to Ignore
            "person": None, "mask": None, "no-mask": None, "vest": None,
            "safety-vest": None, "reflective-vest": None,
        }

        self.target_to_index = {v: k for k, v in self.target_classes.items()}

    def map_class_name(self, original_class: str) -> Optional[str]:
        """Map external dataset class to your model's classes."""
        # Standardize the input class name to match the dictionary keys
        original_lower = original_class.lower().strip().replace('_', '-')

        if original_lower in self.class_mappings:
            return self.class_mappings[original_lower]

        logger.warning(f"Unknown class '{original_lower}' could not be mapped and will be skipped.")
        return None

    def harmonize_dataset(self, dataset_path: Path, dataset_info: Dict) -> Dict[str, int]:
        """Harmonize a dataset to match your model's classes."""
        logger.info(f"Harmonizing dataset: {dataset_info['name']}")
        stats = defaultdict(int)

        for split in ['train', 'valid', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split
            if not split_path.exists(): continue
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            if not images_path.exists(): continue

            image_count_in_split = 0
            for img_file in images_path.glob("*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue
                label_file = labels_path / f"{img_file.stem}.txt"
                if not label_file.exists(): continue

                with open(label_file, 'r') as f:
                    original_annotations = f.readlines()

                mapped_annotations = []
                for line in original_annotations:
                    try:
                        parts = line.strip().split()
                        if not parts: continue
                        orig_class_id = int(float(parts[0]))

                        if orig_class_id < len(dataset_info['classes']):
                            orig_class_name = dataset_info['classes'][orig_class_id]
                            mapped_class = self.map_class_name(orig_class_name)

                            if mapped_class and mapped_class in self.target_to_index:
                                new_class_id = self.target_to_index[mapped_class]
                                mapped_ann_str = f"{new_class_id} {' '.join(parts[1:])}"
                                mapped_annotations.append(mapped_ann_str)
                                stats[mapped_class] += 1
                        else:
                            logger.warning(
                                f"Class ID {orig_class_id} out of bounds for {dataset_info['name']}. Skipping.")
                    except (ValueError, IndexError):
                        continue

                if mapped_annotations:
                    shutil.copy(str(img_file), str(self.output_dir / split / 'images' / img_file.name))
                    dst_label = self.output_dir / split / 'labels' / f"{img_file.stem}.txt"
                    with open(dst_label, 'w') as f:
                        f.write('\n'.join(mapped_annotations))
                    image_count_in_split += 1

            stats['total_images'] = stats.get('total_images', 0) + image_count_in_split
        return dict(stats)

    def create_merged_yaml(self, dataset_stats: List[Dict[str, int]]):
        """Create YAML file for the final merged dataset."""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
            'nc': len(self.target_classes),
            'names': list(self.target_classes.values()),
            'dataset_info': {
                'total_images': sum(s.get('total_images', 0) for s in dataset_stats),
                'class_distribution': {}
            }
        }
        class_dist = defaultdict(int)
        for stats in dataset_stats:
            for class_name, count in stats.items():
                if class_name != 'total_images':
                    class_dist[class_name] += count
        yaml_content['dataset_info']['class_distribution'] = dict(class_dist)

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        logger.info(f"Created final dataset YAML: {yaml_path}")
