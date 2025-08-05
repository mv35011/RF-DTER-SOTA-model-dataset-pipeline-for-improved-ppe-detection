# PPE Detection Inference

Simple video inference script for PPE detection using trained RF-DETR model.

## Model Checkpoint

**[Download Trained Model](https://drive.google.com/file/d/1Jd9TJaclKBvQM9S5303SPGA5fcEMca0d/view?usp=sharing)**

## Usage

1. **Download the model checkpoint** from the link above
2. **Update paths** and change the confedence threshold in the test script:
   ```python
   # Update these paths in your test script
   CHECKPOINT_PATH = "/kaggle/input/rfdter_ppe_pretrained/pytorch/default/1/checkpoint_best_total.pth"
    INPUT_VIDEO_PATH = "/kaggle/input/violation-nonviolation/resized_870.mp4"  # Make sure to include the .mp4 extension
    OUTPUT_VIDEO_PATH = "/kaggle/working/cctv_0.25_2.mp4" # Output as an .mp4 file

# 2. Model and dataset configuration
CONFIDENCE_THRESHOLD = 0.25  # Adjust as needed
   ```


## Output

The script will generate detection videos with:
- Bounding boxes around detected PPE items
- Class labels and confidence scores
- Real-time processing feedback

## Requirements

```bash
pip install torch torchvision
pip install rfdetr supervision opencv-python numpy
```

## Classes Detected

- boots, gloves, goggles, helmet, vest
- no-boots, no-gloves, no-goggles, no-helmet, no-vest

**Note**: Make sure to use the correct model checkpoint for best performance.