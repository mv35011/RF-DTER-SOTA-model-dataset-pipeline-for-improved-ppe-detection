#!/usr/bin/env python3
"""
Test script to merge the three original datasets with ONE new dataset.
This uses the robust PPEDatasetHarmonizer to ensure all class names
and IDs are mapped correctly.
"""
from pathlib import Path
from new_dataset_harmonizer import PPEDatasetHarmonizer

# --- 1. DEFINE YOUR PATHS ---
# Define the main root paths using pathlib.Path
original_root = Path("D:/Compressed/PPE detection files/Annotations Corrected/Annotations Corrected/YoloV5")
new_dataset_root = Path("D:/Compressed/PPE detection files/mywork")

# --- 2. INITIALIZE THE TOOL ---
# The output will be in a new 'merged_test_dataset' folder
harmonizer = PPEDatasetHarmonizer("merged_test_dataset")
all_stats = []

# --- 3. PROCESS THE DATASETS ---

# --- Original Dataset 1: OBJECT DETECTION ---
print("Processing Original Dataset 1: OBJECT DETECTION")
stats1 = harmonizer.harmonize_dataset(
    dataset_path=original_root / "OBJECT DETECTION.v2i.yolov5-obb",
    dataset_info={
        'name': 'OriginalObjectDetection',
        'format': 'yolov5',
        'classes': ['no-safety-glove', 'no-safety-helmet', 'no-safety-shoes', 'no-welding-glass', 'safety-glove', 'safety-helmet', 'safety-shoes', 'welding-glass']
    }
)
all_stats.append(stats1)

# --- Original Dataset 2: project2.0 ---
print("\nProcessing Original Dataset 2: project2.0")
stats2 = harmonizer.harmonize_dataset(
    dataset_path=original_root / "project2.0.v2i.yolov5-obb",
    dataset_info={
        'name': 'OriginalProject2',
        'format': 'yolov5',
        'classes': ['no-safety-helmet', 'no-safety-shoes', 'safety-glove', 'safety-helmet', 'safety-shoes', 'welding-helmet']
    }
)
all_stats.append(stats2)

# --- Original Dataset 3: SAIL PROJECT ---
print("\nProcessing Original Dataset 3: SAIL PROJECT")
stats3 = harmonizer.harmonize_dataset(
    dataset_path=original_root / "SAIL PROJECT.v2i.yolov5-obb",
    dataset_info={
        'name': 'OriginalSailProject',
        'format': 'yolov5',
        'classes': ['no-safety-glove', 'no-safety-helmet', 'no-safety-shoes', 'no-welding-glass', 'safety-glove', 'safety-helmet', 'safety-shoes', 'welding-glass', 'welding-helmet']
    }
)
all_stats.append(stats3)

# --- New Downloaded Dataset 1 (FOR TESTING) ---
# !!! IMPORTANT: You must update the two values below !!!
new_dataset_folder_name = "PPEs.v7-raw-images_ommittedsuitclasses.yolov9" #<-- CONFIRM THIS FOLDER NAME
new_dataset_class_list = ['glove', 'goggles', 'helmet', 'mask', 'no_glove', 'no_goggles', 'no_helmet', 'no_mask', 'no_shoes', 'shoes']


print(f"\nProcessing New Downloaded Dataset: {new_dataset_folder_name}")
stats4 = harmonizer.harmonize_dataset(
    dataset_path=Path(__file__).parent.parent / new_dataset_folder_name,
    dataset_info={
        'name': 'NewDataset1',
        'format': 'yolov9', # Or yolov8/yolov9
        'classes': new_dataset_class_list
    }
)
all_stats.append(stats4)


# --- 4. FINALIZE ---
# Create the final dataset.yaml file for the merged data
harmonizer.create_merged_yaml(all_stats)
print("\n🚀 Test merge complete! Your 'merged_test_dataset' is ready.")