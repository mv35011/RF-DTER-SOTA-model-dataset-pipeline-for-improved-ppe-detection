# PPE Detection Dataset Creation Pipeline

This repository contains a comprehensive pipeline for creating a robust PPE (Personal Protective Equipment) detection dataset with advanced preprocessing steps designed to minimize standing, sitting, and lighting biases.

## Overview

The pipeline creates a "super-dataset" by merging multiple datasets, applying smart cropping techniques, performing aspect ratio augmentations, and incorporating custom-annotated data to improve model generalization and reduce environmental biases.

## Features

- **Bias Reduction**: Advanced preprocessing to minimize standing/sitting position and lighting condition biases
- **Smart Cropping**: Intelligent cropping to focus on PPE items while maintaining context
- **Dataset Harmonization**: Unified class mapping across multiple source datasets
- **Custom Data Addition**: Integration of 210 manually annotated frames from front camera footage
- **Aspect Ratio Augmentation**: Enhances model robustness to different image dimensions
- **Comprehensive Class Coverage**: Supports 10 master classes for complete PPE detection

## Prerequisites

```bash
pip install -r requirements.txt
```

## Pipeline Overview

The dataset creation follows this sequence:

```
Raw Datasets → Format Conversion → Dataset Merging → Custom Frame Addition → 
Smart Cropping → Dataset Merge → Aspect Ratio Augmentation → Final Super-Dataset
```

## Step-by-Step Instructions

### Step 1: Convert Invalid YOLO Datasets

Navigate to the merge scripts directory and convert datasets with invalid class IDs:

```bash
cd merge_scripts
python convert_to_yolo.py "path/to/input_dir" "path/to/output_dir"
```

This step adds the required intermediate classes to `dataset.yaml`:

```yaml
names:
- no-safety-glove
- no-safety-helmet
- no-safety-shoes
- no-welding-glass
- safety-glove
- safety-helmet
- safety-shoes
- welding-glass
- welding-helmet
```

### Step 2: Merge Datasets

Using the modified dataset harmonizer, merge the converted dataset with downloaded datasets:

1. Update the paths in the script
2. Run the merge process:

```bash
python test_merge.py
```

This creates a unified dataset ready for advanced preprocessing.

### Step 3: Custom Frame Addition

Enhance the dataset with manually annotated front camera footage:

1. Extract frames strategically using the smart frame extractor:
```bash
python smart_frame_extractor.py
```

2. **Manual Annotation**: 210 frames were carefully selected and annotated from front camera video footage to add diverse perspectives and scenarios

**Benefits of Custom Data:**
- Adds domain-specific scenarios not present in public datasets
- Provides front camera perspective variations
- Increases dataset diversity and real-world applicability

### Step 4: Smart Cropping (Bias Reduction)

Apply intelligent cropping to reduce positional and contextual biases:

1. Configure paths in `smart_crop.py`:
```python
dataset_yaml_path = r"path/to/merged_dataset/dataset.yaml"
output_base_path = r"path/to/cropped_datasets"
```

2. Execute smart cropping:
```bash
python smart_crop.py
```

**Smart Cropping Benefits:**
- Creates three distinct body region crops: torso, upper body, and lower body
- Focuses on relevant PPE items for each body region
- Reduces background distractions and positional context
- Enables region-specific PPE detection training

### Step 5: Merge Original and Cropped Datasets

Combine the original dataset with the smart-cropped variations:

1. Update paths in `merge_dataset.py`:
```python
original_dataset_path = r"path/to/merged_dataset"
cropped_datasets_base_path = r"path/to/cropped_datasets"
output_directory_name = "super_dataset"
```

2. Run the merge:
```bash
python merge_dataset.py
```

### Step 6: Aspect Ratio Augmentation

Apply aspect ratio transformations to further reduce bias and improve generalization:

```bash
python aspect_ratio_augmenter.py \
    --source_dataset "path/to/super_dataset" \
    --output_dir "path/to/completed_works"
```

**Aspect Ratio Benefits:**
- Reduces bias toward specific image dimensions
- Improves model performance on various camera setups
- Enhances robustness to different viewing angles

## Dataset Visualization

Generate test samples to visualize your dataset:

```bash
python generate_test_samples.py "final_merged_dataset/dataset.yaml" "my_test_samples"
```

This creates visual samples from your dataset for quality assurance and presentation purposes.

## Final Dataset Structure

The completed dataset (`completed_works`) contains 10 master classes:

```yaml
nc: 10
names:
- boots
- gloves
- goggles
- helmet
- no-boots
- no-gloves
- no-goggles
- no-helmet
- no-vest
- vest
```

## Bias Reduction Techniques

### 1. Smart Cropping
- **Purpose**: Eliminates background context that may correlate with worker positions
- **Method**: Creates focused crops of three body regions:
  - **Torso Region**: Captures helmet, vest, and upper body PPE
  - **Upper Body**: Focuses on gloves, safety equipment on arms/hands
  - **Lower Body**: Isolates boots and lower body safety gear
- **Benefit**: Reduces standing/sitting position bias by creating region-specific training data

### 2. Aspect Ratio Augmentation
- **Purpose**: Prevents model from learning camera-specific biases
- **Method**: Systematic aspect ratio transformations
- **Benefit**: Improves generalization across different camera setups and viewing angles

### 3. Dataset Harmonization
- **Purpose**: Ensures consistent labeling across source datasets
- **Method**: Unified class mapping and validation
- **Benefit**: Reduces dataset-specific biases

### 4. Custom Data Integration
- **Purpose**: Adds domain-specific scenarios and perspectives
- **Method**: Smart frame extraction and manual annotation of 210 front camera frames
- **Benefit**: Enhances real-world applicability and scenario diversity

## Best Practices

1. **Path Configuration**: Always use absolute paths and verify they exist before running scripts
2. **Data Validation**: Check dataset integrity after each step
3. **Backup**: Keep backups of intermediate datasets for debugging
4. **Memory Management**: Monitor system resources during processing of large datasets

## Troubleshooting

### Common Issues

1. **Path Errors**: Ensure all paths use the correct format for your operating system
2. **Memory Issues**: For large datasets, consider processing in batches
3. **Class Mapping**: Verify class names match exactly between steps

### Validation

After each step, validate:
- Image count preservation (except during cropping, which increases count)
- Annotation format consistency
- Class distribution balance

## Output

The final super-dataset in the `completed_works` directory is ready for PPE detection model training with significantly reduced environmental and positional biases.

## Resources

### Documentation
- **[Comprehensive Project Report](https://drive.google.com/file/d/1ScwPcU7UsnDsEeZ33i4I6AtyLWQPQhoK/view?usp=sharing)**: Detailed analysis and methodology documentation

### Pre-built Datasets
- **[YOLO Format Dataset](https://drive.google.com/file/d/1pm18XQAyfayr4-d1Cy9ECakrF-IObZqK/view?usp=sharing)**: Ready for YOLO-based object detection models
- **[COCO Format Dataset](https://drive.google.com/file/d/1dQkG__0FmF2zBZ44n3hhPfBGTI_g4PVk/view?usp=sharing)**: Compatible with transformer models (RF-DETR, D-FINE, etc.)

## Contributing

When adding new preprocessing steps, ensure they contribute to bias reduction and maintain dataset integrity. Document any new bias reduction techniques and their effectiveness.

