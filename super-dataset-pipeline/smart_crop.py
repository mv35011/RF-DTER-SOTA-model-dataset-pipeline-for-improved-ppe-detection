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
    def __init__(self, dataset_yaml_path: str, output_base_path: str):
        """
        Initialize the smart cropper for PPE detection

        Args:
            dataset_yaml_path: Path to your dataset.yaml file
            output_base_path: Base path where cropped datasets will be saved
        """
        self.dataset_yaml_path = dataset_yaml_path
        self.output_base_path = Path(output_base_path)
        self.dataset_config = self.load_dataset_config()
        self.ppe_categories = {
            'upper_body': ['helmet', 'no-helmet', 'goggles', 'no-goggles', 'vest', 'no-vest'],
            'lower_body': ['boots', 'no-boots', 'vest', 'no-vest'],
            'torso_focus': ['vest', 'no-vest', 'gloves', 'no-gloves'],
            'side_profile': ['helmet', 'no-helmet', 'vest', 'no-vest', 'boots', 'no-boots']
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
            'side_profile': CropConfig(
                name='side_profile',
                x_start=0.2, y_start=0.0, x_end=0.8, y_end=1.0,
                description='Side view focus - CCTV angle specialist'
            )
        }
        self.min_object_size = 0.01

    def load_dataset_config(self) -> Dict:
        """Load the YOLO dataset configuration"""
        with open(self.dataset_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get_class_id_to_name(self) -> Dict[int, str]:
        """Create mapping from class IDs to names"""
        return {idx: name for idx, name in enumerate(self.dataset_config['names'])}

    def load_annotations(self, annotation_path: str) -> List[List[float]]:
        """Load YOLO format annotations from file"""
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
        """Save YOLO format annotations to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(' '.join(map(str, ann)) + '\n')

    def crop_image_and_adjust_annotations(
            self,
            image: np.ndarray,
            annotations: List[List[float]],
            crop_config: CropConfig
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Crop image and adjust annotations accordingly

        Returns:
            Tuple of (cropped_image, adjusted_annotations)
        """
        h, w = image.shape[:2]
        x1 = int(crop_config.x_start * w)
        y1 = int(crop_config.y_start * h)
        x2 = int(crop_config.x_end * w)
        y2 = int(crop_config.y_end * h)
        cropped_image = image[y1:y2, x1:x2]
        crop_h, crop_w = cropped_image.shape[:2]
        if crop_h == 0 or crop_w == 0:
            return None, []
        adjusted_annotations = []

        for ann in annotations:
            class_id, center_x, center_y, bbox_w, bbox_h = ann[:5]
            orig_center_x = center_x * w
            orig_center_y = center_y * h
            orig_bbox_w = bbox_w * w
            orig_bbox_h = bbox_h * h
            bbox_x1 = orig_center_x - orig_bbox_w / 2
            bbox_y1 = orig_center_y - orig_bbox_h / 2
            bbox_x2 = orig_center_x + orig_bbox_w / 2
            bbox_y2 = orig_center_y + orig_bbox_h / 2
            if (bbox_x2 > x1 and bbox_x1 < x2 and
                    bbox_y2 > y1 and bbox_y1 < y2):
                clipped_x1 = max(bbox_x1, x1)
                clipped_y1 = max(bbox_y1, y1)
                clipped_x2 = min(bbox_x2, x2)
                clipped_y2 = min(bbox_y2, y2)
                clipped_w = clipped_x2 - clipped_x1
                clipped_h = clipped_y2 - clipped_y1
                if (clipped_w * clipped_h) / (crop_w * crop_h) > self.min_object_size:
                    new_center_x = ((clipped_x1 + clipped_x2) / 2 - x1) / crop_w
                    new_center_y = ((clipped_y1 + clipped_y2) / 2 - y1) / crop_h
                    new_bbox_w = clipped_w / crop_w
                    new_bbox_h = clipped_h / crop_h
                    new_center_x = max(0, min(1, new_center_x))
                    new_center_y = max(0, min(1, new_center_y))
                    new_bbox_w = max(0, min(1, new_bbox_w))
                    new_bbox_h = max(0, min(1, new_bbox_h))

                    adjusted_annotations.append([
                        class_id, new_center_x, new_center_y, new_bbox_w, new_bbox_h
                    ])

        return cropped_image, adjusted_annotations

    def should_apply_crop(self, annotations: List[List[float]], crop_type: str) -> bool:
        """
        Determine if a crop should be applied based on present PPE classes
        """
        if not annotations:
            return False

        class_id_to_name = self.get_class_id_to_name()
        present_classes = set()

        for ann in annotations:
            class_id = int(ann[0])
            if class_id in class_id_to_name:
                class_name = class_id_to_name[class_id].replace('safety-', '').replace('no-', '')
                present_classes.add(class_name)
        relevant_classes = set(self.ppe_categories.get(crop_type, []))
        return bool(present_classes.intersection(relevant_classes))

    def process_dataset_split(self, split_name: str, crop_types: List[str]):
        """
        Process a dataset split (train/val/test) with smart cropping
        """
        print(f"\nProcessing {split_name} split...")
        dataset_root = Path(self.dataset_config['path'])
        images_path = dataset_root / f"{split_name}/images"
        labels_path = dataset_root / f"{split_name}/labels"

        if not images_path.exists():
            print(f"Warning: {images_path} does not exist, skipping {split_name}")
            return
        for crop_type in crop_types:
            if crop_type not in self.crop_configs:
                print(f"Warning: Unknown crop type '{crop_type}', skipping")
                continue

            crop_config = self.crop_configs[crop_type]
            print(f"  Applying {crop_type} crop: {crop_config.description}")
            output_img_dir = self.output_base_path / crop_type / f"{split_name}/images"
            output_label_dir = self.output_base_path / crop_type / f"{split_name}/labels"
            output_img_dir.mkdir(parents=True, exist_ok=True)
            output_label_dir.mkdir(parents=True, exist_ok=True)
            processed_count = 0
            skipped_count = 0

            for img_file in images_path.glob("*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue

                image = cv2.imread(str(img_file))
                if image is None:
                    continue

                label_file = labels_path / f"{img_file.stem}.txt"
                annotations = self.load_annotations(str(label_file))
                if not self.should_apply_crop(annotations, crop_type):
                    skipped_count += 1
                    continue
                cropped_image, adjusted_annotations = self.crop_image_and_adjust_annotations(
                    image, annotations, crop_config
                )

                if cropped_image is None or not adjusted_annotations:
                    skipped_count += 1
                    continue
                output_img_path = output_img_dir / f"{crop_type}_{img_file.name}"
                output_label_path = output_label_dir / f"{crop_type}_{img_file.stem}.txt"

                cv2.imwrite(str(output_img_path), cropped_image)
                self.save_annotations(adjusted_annotations, str(output_label_path))

                processed_count += 1

            print(f"    {crop_type}: {processed_count} images processed, {skipped_count} skipped")

    def create_cropped_dataset_configs(self, crop_types: List[str]):
        """Create dataset.yaml files for each crop type"""
        for crop_type in crop_types:
            if crop_type not in self.crop_configs:
                continue

            crop_config = self.crop_configs[crop_type]
            new_config = self.dataset_config.copy()
            new_config['path'] = str((self.output_base_path / crop_type).absolute())
            config_path = self.output_base_path / crop_type / "dataset.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

            print(f"Created dataset config: {config_path}")

    def run_smart_cropping(self, crop_types: Optional[List[str]] = None, splits: Optional[List[str]] = None):
        """
        Run the complete smart cropping pipeline
        """
        if crop_types is None:
            crop_types = list(self.crop_configs.keys())

        if splits is None:
            splits = ['train', 'val', 'test']

        print("=" * 60)
        print("PPE Smart Cropping Pipeline")
        print("=" * 60)
        print(f"Original dataset: {self.dataset_yaml_path}")
        print(f"Output base path: {self.output_base_path}")
        print(f"Classes: {self.dataset_config['names']}")
        print(f"Crop types: {crop_types}")
        print(f"Splits to process: {splits}")
        for split in splits:
            self.process_dataset_split(split, crop_types)
        print("\nCreating dataset configuration files...")
        self.create_cropped_dataset_configs(crop_types)

        print("\n" + "=" * 60)
        print("Smart cropping completed successfully!")
        print("=" * 60)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Smart Cropping for PPE Detection YOLO Dataset')
    parser.add_argument('--dataset_yaml', required=True, help='Path to original dataset.yaml')
    parser.add_argument('--output_path', required=True, help='Output base path for cropped datasets')
    parser.add_argument('--crop_types', nargs='+',
                        choices=['upper_body', 'lower_body', 'torso_focus', 'side_profile'],
                        default=['upper_body', 'lower_body', 'torso_focus'],
                        help='Crop types to apply')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'],
                        default=['train', 'val', 'test'], help='Dataset splits to process')

    args = parser.parse_args()
    cropper = PPESmartCropper(args.dataset_yaml, args.output_path)
    cropper.run_smart_cropping(args.crop_types, args.splits)


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        print("Running with direct paths defined in the script.")
        dataset_yaml_path = r"D:\Compressed\PPE detection files\mywork\super-dataset-pipeline\datasets\2_merged_dataset\dataset.yaml"
        output_base_path = r"D:\Compressed\PPE detection files\mywork\super-dataset-pipeline\datasets\cropped_datasets"

        cropper = PPESmartCropper(dataset_yaml_path, output_base_path)
        cropper.run_smart_cropping(
            crop_types=['upper_body', 'lower_body', 'torso_focus'],
            splits=['train', 'val', 'test']
        )
    else:
        main()
