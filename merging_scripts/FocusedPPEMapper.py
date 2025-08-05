#!/usr/bin/env python3
"""
Focused PPE Class Mapper
Maps specific external dataset classes to target underperforming classes
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from loguru import logger
from collections import defaultdict
import yaml


class FocusedPPEMapper:
    def __init__(self, output_dir: str = "focused_enhanced_dataset"):
        """Initialize focused mapper for underperforming classes"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Your target model classes
        self.target_classes = {
            0: "no-safety-glove",
            1: "no-safety-helmet",
            2: "no-safety-shoes",  # UNDERPERFORMING
            3: "no-welding-glass",  # UNDERPERFORMING
            4: "safety-glove",
            5: "safety-helmet",
            6: "safety-shoes",
            7: "welding-glass"  # UNDERPERFORMING
        }

        # External dataset classes you have
        self.external_classes = [
            'glove', 'goggles', 'helmet', 'mask',
            'no_glove', 'no_goggles', 'no_helmet', 'no_mask',
            'no_shoes', 'shoes'
        ]

        # FOCUSED mapping for underperforming classes only
        self.focused_mapping = {
            # Target: no-safety-shoes (class 2)
            'no_shoes': 'no-safety-shoes',

            # Target: no-welding-glass (class 3)
            'no_goggles': 'no-welding-glass',

            # Target: welding-glass (class 7)
            'goggles': 'welding-glass',

            # Target: safety-shoes (optional - might help with shoes detection)
            'shoes': 'safety-shoes',

            # Skip irrelevant classes for focused approach
            'glove': None,  # Skip - not targeting glove issues
            'no_glove': None,  # Skip - not targeting glove issues
            'helmet': None,  # Skip - not targeting helmet issues
            'no_helmet': None,  # Skip - not targeting helmet issues
            'mask': None,  # Skip - not in your model
            'no_mask': None,  # Skip - not in your model
        }

        # Reverse mapping for target classes
        self.target_to_index = {v: k for k, v in self.target_classes.items()}

        logger.info("Focused mapper initialized for underperforming classes:")
        logger.info("- no-safety-shoes")
        logger.info("- no-welding-glass")
        logger.info("- welding-glass")

    def map_external_dataset(self, external_dataset_path: str,
                             original_datasets: List[str] = None,
                             external_ratio: float = 0.3) -> Dict:
        """
        Map external dataset focusing only on underperforming classes

        Args:
            external_dataset_path: Path to external dataset with the 10 classes
            original_datasets: List of paths to your original datasets (optional)
            external_ratio: Ratio of external data to add (0.3 = 30% external, 70% original)
        """
        logger.info(f"Processing external dataset: {external_dataset_path}")

        external_path = Path(external_dataset_path)
        if not external_path.exists():
            logger.error(f"External dataset not found: {external_dataset_path}")
            return {}

        # Create output structure
        for split in ['train', 'valid', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        mapping_stats = {
            'processed_images': 0,
            'mapped_annotations': defaultdict(int),
            'skipped_annotations': defaultdict(int),
            'class_distribution': defaultdict(int)
        }

        # Process each split
        for split in ['train', 'valid', 'test']:
            split_stats = self._process_split(external_path, split, external_ratio)

            # Aggregate stats
            mapping_stats['processed_images'] += split_stats['processed_images']

            for class_name, count in split_stats['mapped_annotations'].items():
                mapping_stats['mapped_annotations'][class_name] += count

            for class_name, count in split_stats['skipped_annotations'].items():
                mapping_stats['skipped_annotations'][class_name] += count

            for class_name, count in split_stats['class_distribution'].items():
                mapping_stats['class_distribution'][class_name] += count

        # Copy original datasets if provided
        if original_datasets:
            logger.info("Copying original datasets...")
            orig_stats = self._copy_original_datasets(original_datasets)

            # Add original stats
            mapping_stats['original_images'] = orig_stats['total_images']
            for class_name, count in orig_stats['class_distribution'].items():
                mapping_stats['class_distribution'][class_name] += count

        self._log_mapping_results(mapping_stats)
        return mapping_stats

    def _process_split(self, dataset_path: Path, split: str, sample_ratio: float) -> Dict:
        """Process a single split of the external dataset"""

        # Look for images and labels
        images_path = dataset_path / split / 'images'
        labels_path = dataset_path / split / 'labels'

        # Alternative structure
        if not images_path.exists():
            images_path = dataset_path / 'images' / split
            labels_path = dataset_path / 'labels' / split

        # Another alternative
        if not images_path.exists():
            images_path = dataset_path / 'images'
            labels_path = dataset_path / 'labels'

        if not images_path.exists():
            logger.warning(f"No images found for split: {split}")
            return {'processed_images': 0, 'mapped_annotations': {}, 'skipped_annotations': {},
                    'class_distribution': {}}

        split_stats = {
            'processed_images': 0,
            'mapped_annotations': defaultdict(int),
            'skipped_annotations': defaultdict(int),
            'class_distribution': defaultdict(int)
        }

        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))

        # Sample images if needed
        if sample_ratio < 1.0:
            import random
            random.shuffle(image_files)
            n_keep = int(len(image_files) * sample_ratio)
            image_files = image_files[:n_keep]
            logger.info(f"Sampling {n_keep}/{len(image_files)} images for {split}")

        # Process each image
        for img_file in image_files:
            # Find corresponding label file
            label_file = labels_path / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue

            # Read original annotations
            original_annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        original_annotations.append([float(x) for x in parts])

            if not original_annotations:
                continue

            # Map annotations to target classes
            mapped_annotations = []
            has_target_classes = False

            for ann in original_annotations:
                class_id = int(ann[0])

                # Map external class to your target class
                if class_id < len(self.external_classes):
                    external_class_name = self.external_classes[class_id]

                    if external_class_name in self.focused_mapping:
                        target_class_name = self.focused_mapping[external_class_name]

                        if target_class_name:  # Not None (skip)
                            target_class_id = self.target_to_index[target_class_name]
                            mapped_ann = [target_class_id] + ann[1:]
                            mapped_annotations.append(mapped_ann)

                            split_stats['mapped_annotations'][external_class_name] += 1
                            split_stats['class_distribution'][target_class_name] += 1
                            has_target_classes = True
                        else:
                            split_stats['skipped_annotations'][external_class_name] += 1
                    else:
                        split_stats['skipped_annotations'][f'unknown_{class_id}'] += 1

            # Only save if we have annotations for target classes
            if has_target_classes and mapped_annotations:
                # Copy image
                dst_img = self.output_dir / split / 'images' / f"ext_{img_file.name}"
                shutil.copy(str(img_file), str(dst_img))

                # Save mapped labels
                dst_label = self.output_dir / split / 'labels' / f"ext_{img_file.stem}.txt"
                with open(dst_label, 'w') as f:
                    for ann in mapped_annotations:
                        f.write(' '.join(map(str, ann)) + '\n')

                split_stats['processed_images'] += 1

        logger.info(f"Split {split}: processed {split_stats['processed_images']} images")
        return split_stats

    def _copy_original_datasets(self, original_datasets: List[str]) -> Dict:
        """Copy original datasets to maintain existing knowledge"""
        logger.info("Copying original datasets...")

        orig_stats = {
            'total_images': 0,
            'class_distribution': defaultdict(int)
        }

        for dataset_path in original_datasets:
            path = Path(dataset_path)
            if not path.exists():
                logger.warning(f"Original dataset not found: {dataset_path}")
                continue

            dataset_stats = self._copy_single_original_dataset(path)
            orig_stats['total_images'] += dataset_stats['images']

            for class_name, count in dataset_stats['classes'].items():
                orig_stats['class_distribution'][class_name] += count

        return orig_stats

    def _copy_single_original_dataset(self, dataset_path: Path) -> Dict:
        """Copy a single original dataset"""
        stats = {'images': 0, 'classes': defaultdict(int)}

        for split in ['train', 'valid', 'test']:
            images_path = dataset_path / split / 'images'
            labels_path = dataset_path / split / 'labels'

            if not images_path.exists():
                continue

            for img_file in images_path.glob("*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                label_file = labels_path / f"{img_file.stem}.txt"
                if not label_file.exists():
                    continue

                # Copy image
                dst_img = self.output_dir / split / 'images' / f"orig_{img_file.name}"
                shutil.copy(str(img_file), str(dst_img))

                # Copy label and count classes
                dst_label = self.output_dir / split / 'labels' / f"orig_{img_file.stem}.txt"
                shutil.copy(str(label_file), str(dst_label))

                # Count classes
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(float(parts[0]))
                            if class_id in self.target_classes:
                                class_name = self.target_classes[class_id]
                                stats['classes'][class_name] += 1

                stats['images'] += 1

        return stats

    def _log_mapping_results(self, stats: Dict):
        """Log detailed mapping results"""
        logger.info("\n" + "=" * 50)
        logger.info("FOCUSED MAPPING RESULTS")
        logger.info("=" * 50)

        logger.info(f"Total processed images: {stats['processed_images']}")
        if 'original_images' in stats:
            logger.info(f"Original images: {stats['original_images']}")

        logger.info("\nSuccessfully mapped classes:")
        for ext_class, count in stats['mapped_annotations'].items():
            target_class = self.focused_mapping.get(ext_class, 'Unknown')
            logger.info(f"  {ext_class} → {target_class}: {count} annotations")

        logger.info("\nSkipped classes:")
        for ext_class, count in stats['skipped_annotations'].items():
            logger.info(f"  {ext_class}: {count} annotations (not needed)")

        logger.info("\nFinal class distribution:")
        for class_name, count in stats['class_distribution'].items():
            logger.info(f"  {class_name}: {count} annotations")

        # Check improvement for target classes
        logger.info("\nImprovements for underperforming classes:")
        target_classes = ['no-safety-shoes', 'no-welding-glass', 'welding-glass']
        for class_name in target_classes:
            count = stats['class_distribution'].get(class_name, 0)
            status = "✓ IMPROVED" if count > 0 else "⚠ NO IMPROVEMENT"
            logger.info(f"  {class_name}: {count} annotations {status}")

    def create_training_config(self, stats: Dict) -> Path:
        """Create YAML config for training"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.target_classes),
            'names': list(self.target_classes.values()),

            # Focus on underperforming classes
            'training_notes': {
                'focus': 'Improving underperforming classes',
                'target_classes': ['no-safety-shoes', 'no-welding-glass', 'welding-glass'],
                'mapping_used': dict(self.focused_mapping),
                'external_additions': {
                    k: v for k, v in stats['class_distribution'].items()
                    if k in ['no-safety-shoes', 'no-welding-glass', 'welding-glass']
                }
            },

            'dataset_stats': {
                'total_images': stats.get('processed_images', 0) + stats.get('original_images', 0),
                'external_images': stats.get('processed_images', 0),
                'original_images': stats.get('original_images', 0),
                'class_distribution': dict(stats['class_distribution'])
            }
        }

        config_path = self.output_dir / 'focused_dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"\nCreated training config: {config_path}")
        return config_path


def main():
    """Example usage"""
    mapper = FocusedPPEMapper()
    parent = Path(__file__).parent.parent
    annotations = parent / "Annotations Corrected" / "Annotations Corrected" / "YoloV5"

    # Your external dataset path (update this)
    external_dataset_path = parent / "PPEs.v7-raw-images_ommittedsuitclasses.yolov9"

    # Your original datasets (update these paths)
    original_datasets = [
        annotations / "OBJECT DETECTION.v2i.yolov5-obb",
        annotations / "project2.0.v2i.yolov5-obb",
        annotations / "SAIL PROJECT.v2i.yolov5-obb"
    ]

    logger.info("Starting focused mapping for underperforming classes...")
    logger.info("Target improvements:")
    logger.info("- no-safety-shoes (from 'no_shoes')")
    logger.info("- no-welding-glass (from 'no_goggles')")
    logger.info("- welding-glass (from 'goggles')")

    # Process datasets
    stats = mapper.map_external_dataset(
        external_dataset_path=external_dataset_path,
        original_datasets=original_datasets,
        external_ratio=0.2  # Add 20% external data focused on weak classes
    )

    # Create training config
    config_path = mapper.create_training_config(stats)

    logger.info(f"\n✅ Ready for training!")
    logger.info(f"Dataset path: {mapper.output_dir}")
    logger.info(f"Config file: {config_path}")
    logger.info(f"\nNext: Train your YOLOv9 model with this enhanced dataset")


if __name__ == "__main__":
    main()