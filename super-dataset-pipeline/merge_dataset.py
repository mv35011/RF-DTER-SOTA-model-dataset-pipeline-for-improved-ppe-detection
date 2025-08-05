#!/usr/bin/env python3
"""
Merges the original, full-frame dataset with multiple cropped datasets
to create a final "super dataset" for training.
"""
import shutil
from pathlib import Path
import yaml
from loguru import logger
import argparse
import os


def merge_datasets(source_dirs: list, output_dir: str):
    """
    Merges multiple source dataset directories into a single output directory.

    Args:
        source_dirs (list): A list of paths to the source dataset folders.
        output_dir (str): The name for the final merged dataset folder.
    """
    output_path = Path(output_dir)
    logger.info(f"Starting merge process to create 'super dataset'.")
    logger.info(f"Output will be in: {output_path}")
    if output_path.exists():
        logger.warning(f"Output directory '{output_path}' already exists. Deleting it.")
        shutil.rmtree(output_path)
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    total_images_copied = 0
    total_labels_copied = 0
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"Source directory not found, skipping: {source_path}")
            continue

        logger.info(f"Processing source: {source_path.name}")
        for split in ['train', 'valid', 'test']:
            source_images_path = source_path / split / 'images'
            source_labels_path = source_path / split / 'labels'
            if source_images_path.exists():
                for f in source_images_path.glob('*'):
                    shutil.copy(f, output_path / split / 'images' / f.name)
                    total_images_copied += 1
            if source_labels_path.exists():
                for f in source_labels_path.glob('*.txt'):
                    shutil.copy(f, output_path / split / 'labels' / f.name)
                    total_labels_copied += 1
    reference_yaml_path = Path(source_dirs[0]) / "dataset.yaml"
    if reference_yaml_path.exists():
        with open(reference_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        data['path'] = str(output_path.absolute())

        final_yaml_path = output_path / "dataset.yaml"
        with open(final_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.success(f"Created final dataset.yaml at {final_yaml_path}")
    else:
        logger.error(f"Could not find reference dataset.yaml at {reference_yaml_path}")

    logger.success("\n" + "=" * 60)
    logger.success(f"🚀 Super dataset merge complete!")
    logger.success(f"Total images copied: {total_images_copied}")
    logger.success(f"Total labels copied: {total_labels_copied}")
    logger.success(f"Your final dataset is ready in '{output_path}'.")


def main():
    """Main function to run the merge process."""
    parser = argparse.ArgumentParser(description='Merge original and cropped datasets.')
    parser.add_argument('--original_dataset', required=True, help='Path to the original, full-frame dataset.')
    parser.add_argument('--cropped_base_path', required=True, help='Base path containing the cropped dataset folders.')
    parser.add_argument('--output_dir', default='super_dataset', help='Name for the final output directory.')

    args = parser.parse_args()
    cropped_base = Path(args.cropped_base_path)
    cropped_dirs = [d for d in cropped_base.iterdir() if d.is_dir()]
    datasets_to_merge = [args.original_dataset] + [str(d) for d in cropped_dirs]

    logger.info("The following datasets will be merged:")
    for d in datasets_to_merge:
        logger.info(f"- {d}")

    merge_datasets(datasets_to_merge, args.output_dir)


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        logger.info("Running with direct paths defined in the script.")

        original_dataset_path = r"D:\Compressed\PPE detection files\mywork\super-dataset-pipeline\datasets\2_merged_dataset"
        cropped_datasets_base_path = r"D:\Compressed\PPE detection files\mywork\super-dataset-pipeline\datasets\cropped_datasets"
        output_directory_name = "super_dataset"
        cropped_base = Path(cropped_datasets_base_path)
        cropped_dirs = [str(d) for d in cropped_base.iterdir() if d.is_dir()]
        all_datasets = [original_dataset_path] + cropped_dirs

        logger.info("The following datasets will be merged:")
        for d in all_datasets:
            logger.info(f"- {d}")

        merge_datasets(all_datasets, output_directory_name)
    else:
        main()
