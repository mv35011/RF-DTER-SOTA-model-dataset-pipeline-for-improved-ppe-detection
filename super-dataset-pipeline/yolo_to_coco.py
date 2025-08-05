import os
import json
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
from loguru import logger
import yaml  # <-- THIS IS THE FIX


def convert_yolo_to_coco(yolo_dataset_path: str, output_dir: str):
    """
    Converts a YOLO-formatted dataset to COCO JSON format.

    Args:
        yolo_dataset_path (str): Path to the root of the YOLO dataset.
        output_dir (str): Directory to save the COCO annotation files.
    """
    root_path = Path(yolo_dataset_path)
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Load class names from the data.yaml file
    try:
        with open(root_path / "dataset.yaml", 'r') as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml['names']
    except Exception as e:
        logger.error(f"Could not read dataset.yaml. Error: {e}")
        return

    categories = [{"id": i, "name": name, "supercategory": "ppe"} for i, name in enumerate(class_names)]

    for split in ["train", "valid", "test"]:
        image_dir = root_path / split / "images"
        label_dir = root_path / split / "labels"

        if not image_dir.exists() or not label_dir.exists():
            logger.warning(f"Skipping '{split}' split as it was not found.")
            continue

        coco_output = {
            "info": {},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }

        image_id = 0
        annotation_id = 0

        image_files = sorted(list(image_dir.glob("*")))

        logger.info(f"Processing {split} split...")
        for img_file in tqdm(image_files):
            try:
                # Get image dimensions
                image = cv2.imread(str(img_file))
                height, width, _ = image.shape
            except Exception:
                logger.warning(f"Could not read image {img_file}, skipping.")
                continue

            coco_output["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": img_file.name,
            })

            label_file = label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        class_id, x_center, y_center, bbox_w, bbox_h = map(float, parts[:5])

                        # Convert from YOLO (normalized center) to COCO (absolute top-left)
                        x_min = (x_center - bbox_w / 2) * width
                        y_min = (y_center - bbox_h / 2) * height
                        coco_w = bbox_w * width
                        coco_h = bbox_h * height

                        coco_output["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "bbox": [x_min, y_min, coco_w, coco_h],
                            "area": coco_w * coco_h,
                            "iscrowd": 0,
                        })
                        annotation_id += 1
            image_id += 1

        output_json_path = output_path / f"{split}_annotations.json"
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f, indent=4)

        logger.success(f"Successfully created COCO annotations for '{split}' split at: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO format for DINO training.")
    parser.add_argument("--yolo_path", required=True,
                        help="Path to the root of your final YOLO dataset (e.g., 'super_dataset').")
    parser.add_argument("--output_dir", required=True, help="Path to save the output COCO JSON annotation files.")

    # You will need to install PyYAML for this script
    # pip install pyyaml

    args = parser.parse_args()
    convert_yolo_to_coco(args.yolo_path, args.output_dir)
