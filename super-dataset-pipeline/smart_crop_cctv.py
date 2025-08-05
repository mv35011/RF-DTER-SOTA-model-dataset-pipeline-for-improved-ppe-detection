#!/usr/bin/env python3
"""
Advanced Smart Cropper for PPE Detection with support for different camera profiles.
"""
import os
import cv2
import yaml
import numpy as np
from pathlib import Path
import shutil
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass
from loguru import logger


@dataclass
class CropConfig:
    """Configuration for different crop types"""
    name: str
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    description: str


class PPESmartCropper:
    def __init__(self, dataset_yaml_path: str, output_base_path: str, profile: str = 'front_camera'):
        """
        Initialize the smart cropper with a specific profile.

        Args:
            dataset_yaml_path: Path to your dataset.yaml file.
            output_base_path: Base path where cropped datasets will be saved.
            profile (str): The cropping profile to use ('front_camera' or 'cctv').
        """
        self.dataset_yaml_path = dataset_yaml_path
        self.output_base_path = Path(output_base_path)
        self.dataset_config = self.load_dataset_config()
        self.profile = profile

        self.min_object_size = 0.01

        # Load the correct configurations based on the selected profile
        self.load_profile_configs()
        logger.info(f"Initialized cropper with '{self.profile}' profile.")

    def load_profile_configs(self):
        """Loads the appropriate crop configurations and categories for the selected profile."""
        if self.profile == 'cctv':
            self.ppe_categories = {
                'cctv_head_focus': ['helmet', 'no-helmet', 'goggles', 'no-goggles'],
                'cctv_lower_body': ['boots', 'no-boots'],
                'cctv_full_body_vertical': ['helmet', 'no-helmet', 'vest', 'no-vest', 'boots', 'no-boots'],
            }
            self.crop_configs = {
                'cctv_head_focus': CropConfig(
                    name='cctv_head_focus',
                    x_start=0.0, y_start=0.0, x_end=1.0, y_end=0.5,
                    description='Tighter top-down crop for helmets/goggles in CCTV footage'
                ),
                'cctv_lower_body': CropConfig(
                    name='cctv_lower_body',
                    x_start=0.0, y_start=0.4, x_end=1.0, y_end=1.0,
                    description='Focus on the lower part of the frame for boots in CCTV footage'
                ),
                'cctv_full_body_vertical': CropConfig(
                    name='cctv_full_body_vertical',
                    x_start=0.25, y_start=0.0, x_end=0.75, y_end=1.0,
                    description='A vertical slice to capture a full person who may be distant'
                ),
            }
        else:  # Default to 'front_camera'
            self.ppe_categories = {
                'upper_body': ['helmet', 'no-helmet', 'goggles', 'no-goggles', 'vest', 'no-vest'],
                'lower_body': ['boots', 'no-boots', 'vest', 'no-vest'],
                'torso_focus': ['vest', 'no-vest', 'gloves', 'no-gloves'],
            }
            self.crop_configs = {
                'upper_body': CropConfig(
                    name='upper_body',
                    x_start=0.0, y_start=0.0, x_end=1.0, y_end=0.7,
                    description='Focus on helmet, goggles, upper vest - good for seated detection'
                ),
                'lower_body': CropConfig(
                    name='lower_body',
                    x_start=0.0, y_start=0.3, x_end=1.0, y_end=1.0,
                    description='Focus on boots, lower vest - handles desk occlusion'
                ),
                'torso_focus': CropConfig(
                    name='torso_focus',
                    x_start=0.1, y_start=0.2, x_end=0.9, y_end=0.8,
                    description='Central torso focus - vest specialist'
                ),
            }

    def load_dataset_config(self) -> Dict:
        with open(self.dataset_yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def get_class_id_to_name(self) -> Dict[int, str]:
        return {idx: name for idx, name in enumerate(self.dataset_config['names'])}

    def load_annotations(self, annotation_path: str) -> List[List[float]]:
        annotations = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            parts = list(map(float, line.split()))
                            annotations.append(parts)
                        except ValueError:
                            logger.warning(f"Could not parse line in {annotation_path}: {line}")
        return annotations

    def save_annotations(self, annotations: List[List[float]], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(' '.join(map(str, ann)) + '\n')

    def crop_image_and_adjust_annotations(self, image: np.ndarray, annotations: List[List[float]],
                                          crop_config: CropConfig) -> Tuple[np.ndarray, List[List[float]]]:
        h, w = image.shape[:2]
        x1, y1 = int(crop_config.x_start * w), int(crop_config.y_start * h)
        x2, y2 = int(crop_config.x_end * w), int(crop_config.y_end * h)

        cropped_image = image[y1:y2, x1:x2]
        crop_h, crop_w = cropped_image.shape[:2]
        if crop_h == 0 or crop_w == 0:
            return None, []

        adjusted_annotations = []
        for ann in annotations:
            class_id, center_x, center_y, bbox_w, bbox_h = ann[:5]

            orig_bbox_x1 = (center_x - bbox_w / 2) * w
            orig_bbox_y1 = (center_y - bbox_h / 2) * h
            orig_bbox_x2 = (center_x + bbox_w / 2) * w
            orig_bbox_y2 = (center_y + bbox_h / 2) * h

            if (orig_bbox_x2 > x1 and orig_bbox_x1 < x2 and orig_bbox_y2 > y1 and orig_bbox_y1 < y2):
                clipped_x1 = max(orig_bbox_x1, x1)
                clipped_y1 = max(orig_bbox_y1, y1)
                clipped_x2 = min(orig_bbox_x2, x2)
                clipped_y2 = min(orig_bbox_y2, y2)

                clipped_w = clipped_x2 - clipped_x1
                clipped_h = clipped_y2 - clipped_y1

                if (clipped_w * clipped_h) / (crop_w * crop_h) > self.min_object_size:
                    new_center_x = (clipped_x1 - x1 + clipped_w / 2) / crop_w
                    new_center_y = (clipped_y1 - y1 + clipped_h / 2) / crop_h
                    new_bbox_w = clipped_w / crop_w
                    new_bbox_h = clipped_h / crop_h

                    adjusted_annotations.append([class_id, new_center_x, new_center_y, new_bbox_w, new_bbox_h])

        return cropped_image, adjusted_annotations

    def should_apply_crop(self, annotations: List[List[float]], crop_type: str) -> bool:
        if not annotations: return False
        class_id_to_name = self.get_class_id_to_name()
        present_classes = set()
        for ann in annotations:
            class_id = int(ann[0])
            if class_id in class_id_to_name:
                class_name = class_id_to_name[class_id].replace('safety-', '').replace('no-', '')
                present_classes.add(class_name)
        relevant_classes = set(self.ppe_categories.get(crop_type, []))
        return bool(present_classes.intersection(relevant_classes))

    def process_dataset_split(self, split_name: str):
        print(f"\nProcessing {split_name} split...")
        dataset_root = Path(self.dataset_config['path'])
        images_path = dataset_root / f"{split_name}/images"
        labels_path = dataset_root / f"{split_name}/labels"

        if not images_path.exists():
            print(f"Warning: {images_path} does not exist, skipping {split_name}")
            return

        for crop_type, crop_config in self.crop_configs.items():
            print(f"  Applying {crop_type} crop: {crop_config.description}")
            output_img_dir = self.output_base_path / crop_type / f"{split_name}/images"
            output_label_dir = self.output_base_path / crop_type / f"{split_name}/labels"
            output_img_dir.mkdir(parents=True, exist_ok=True)
            output_label_dir.mkdir(parents=True, exist_ok=True)

            processed_count, skipped_count = 0, 0
            for img_file in images_path.glob("*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']: continue
                image = cv2.imread(str(img_file))
                if image is None: continue

                label_file = labels_path / f"{img_file.stem}.txt"
                annotations = self.load_annotations(str(label_file))

                if not self.should_apply_crop(annotations, crop_type):
                    skipped_count += 1
                    continue

                cropped_image, adjusted_annotations = self.crop_image_and_adjust_annotations(image, annotations,
                                                                                             crop_config)

                if cropped_image is None or not adjusted_annotations:
                    skipped_count += 1
                    continue

                output_img_path = output_img_dir / f"{crop_type}_{img_file.name}"
                output_label_path = output_label_dir / f"{crop_type}_{img_file.stem}.txt"
                cv2.imwrite(str(output_img_path), cropped_image)
                self.save_annotations(adjusted_annotations, str(output_label_path))
                processed_count += 1

            print(f"    {crop_type}: {processed_count} images processed, {skipped_count} skipped")

    def create_cropped_dataset_configs(self):
        print("\nCreating dataset configuration files...")
        for crop_type in self.crop_configs.keys():
            new_config = self.dataset_config.copy()
            new_config['path'] = str((self.output_base_path / crop_type).absolute())
            config_path = self.output_base_path / crop_type / "dataset.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            print(f"Created dataset config: {config_path}")

    def run_smart_cropping(self, splits: Optional[List[str]] = None):
        print("=" * 60)
        print(f"PPE Smart Cropping Pipeline (Profile: {self.profile})")
        print("=" * 60)

        # --- THIS IS THE FIX ---
        # Auto-detect available splits instead of using a fixed list.
        if splits is None:
            dataset_root = Path(self.dataset_config['path'])
            available_splits = []
            for potential_split in ['train', 'valid', 'val', 'test']:
                if (dataset_root / potential_split).exists():
                    available_splits.append(potential_split)
            splits = available_splits
            logger.info(f"Auto-detected splits to process: {splits}")

        for split in splits:
            self.process_dataset_split(split)

        self.create_cropped_dataset_configs()
        print("\n" + "=" * 60)
        print("Smart cropping completed successfully!")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Smart Cropping for PPE Detection with Profiles')
    parser.add_argument('--dataset_yaml', required=True, help='Path to original dataset.yaml')
    parser.add_argument('--output_path', required=True, help='Output base path for cropped datasets')
    parser.add_argument('--profile', choices=['front_camera', 'cctv'], default='front_camera',
                        help='Cropping profile to use')

    args = parser.parse_args()
    cropper = PPESmartCropper(args.dataset_yaml, args.output_path, args.profile)
    cropper.run_smart_cropping()


if __name__ == "__main__":
    main()
