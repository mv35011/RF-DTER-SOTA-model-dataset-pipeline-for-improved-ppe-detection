#!/usr/bin/env python3
"""
Final, robust script to merge all clean, converted, and split datasets into one
final dataset ready for training. This version reads the data.yaml from each
clean source to prevent any mapping errors.
"""
from pathlib import Path
import yaml
from loguru import logger
# Make sure your harmonizer script is named new_dataset_harmonizer.py
from new_dataset_harmonizer_FINAL import PPEDatasetHarmonizer


def get_classes_from_yaml(dataset_path: Path) -> list:
    """Helper function to read the class list from a data.yaml file."""
    yaml_file = dataset_path / "data.yaml"
    if not yaml_file.exists():
        logger.error(f"Could not find data.yaml in {dataset_path}")
        return []
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])


def main():
    """Main function to execute the final merge."""

    # --- 1. DEFINE YOUR PATHS ---
    # This points to the folder containing your CLEAN, SPLIT datasets
    clean_data_root = Path("./converted_datasets")
    new_dataset_root = Path("D:\Compressed\PPE detection files")  # Assuming new dataset is in the current folder

    # --- 2. INITIALIZE THE TOOL ---
    # The final output will be in 'final_merged_dataset'
    harmonizer = PPEDatasetHarmonizer("final_merged_dataset")
    all_stats = []

    # --- 3. PROCESS THE CLEAN, SPLIT DATASETS ---

    # --- Dataset 1: OBJECT DETECTION (Clean) ---
    ds1_path = clean_data_root / "OBJECT DETECTION.v2i.yolov5-obb_converted"
    if ds1_path.exists():
        ds1_classes = get_classes_from_yaml(ds1_path)
        if ds1_classes:
            logger.info(f"Processing {ds1_path.name}...")
            stats1 = harmonizer.harmonize_dataset(
                dataset_path=ds1_path,
                dataset_info={'name': 'CleanObjectDetection', 'format': 'yolov5', 'classes': ds1_classes}
            )
            all_stats.append(stats1)

    # --- Dataset 2: project2.0 (Clean) ---
    ds2_path = clean_data_root / "project2.0.v2i.yolov5-obb_converted"
    if ds2_path.exists():
        ds2_classes = get_classes_from_yaml(ds2_path)
        if ds2_classes:
            logger.info(f"Processing {ds2_path.name}...")
            stats2 = harmonizer.harmonize_dataset(
                dataset_path=ds2_path,
                dataset_info={'name': 'CleanProject2', 'format': 'yolov5', 'classes': ds2_classes}
            )
            all_stats.append(stats2)

    # --- Dataset 3: SAIL PROJECT (Clean) ---
    ds3_path = clean_data_root / "SAIL PROJECT.v2i.yolov5-obb_converted"
    if ds3_path.exists():
        ds3_classes = get_classes_from_yaml(ds3_path)
        if ds3_classes:
            logger.info(f"Processing {ds3_path.name}...")
            stats3 = harmonizer.harmonize_dataset(
                dataset_path=ds3_path,
                dataset_info={'name': 'CleanSailProject', 'format': 'yolov5', 'classes': ds3_classes}
            )
            all_stats.append(stats3)

    # --- Dataset 4: New Downloaded Dataset ---
    new_dataset_folder_name = "PPEs.v7-raw-images_ommittedsuitclasses.yolov9"
    new_dataset_path = new_dataset_root / new_dataset_folder_name
    if new_dataset_path.exists():
        new_dataset_class_list = get_classes_from_yaml(new_dataset_path)
        if new_dataset_class_list:
            logger.info(f"Processing {new_dataset_path.name}...")
            stats4 = harmonizer.harmonize_dataset(
                dataset_path=new_dataset_path,
                dataset_info={'name': 'NewExternalDataset', 'format': 'yolov9', 'classes': new_dataset_class_list}
            )
            all_stats.append(stats4)

    # --- 4. FINALIZE ---
    if all_stats:
        harmonizer.create_merged_yaml(all_stats)
        logger.success("\n🚀 Final merge complete! Your 'final_merged_dataset' is ready for training.")
    else:
        logger.error("No datasets were processed. Please check your paths.")


if __name__ == "__main__":
    main()
