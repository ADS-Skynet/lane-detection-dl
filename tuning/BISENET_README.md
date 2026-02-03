# BiSeNet Training for fine-tune3

BiSeNet (Bilateral Segmentation Network) scripts adapted for the fine-tune3 directory structure. These scripts use the BiSeNet model from the parent `BiSeNet/` directory while working with your local dataset.

## Quick Start

### Option 1: Automated Workflow

```bash
# Run everything with one command
./run_bisenet.sh
```

### Option 2: Step-by-Step

#### Step 1: Convert Dataset to Segmentation Masks

```bash
# Binary segmentation (lane vs background)
python convert_to_bisenet.py \
    --input_dir ./dataset/augmented \
    --output_dir ./dataset/bisenet_data \
    --binary \
    --thickness 10

# OR Multi-class segmentation (4 lanes + background)
python convert_to_bisenet.py \
    --input_dir ./dataset/augmented \
    --output_dir ./dataset/bisenet_multiclass \
    --thickness 10
```

This will create:
```
dataset/bisenet_data/
├── images/          # Copied from original
├── masks/           # Generated segmentation masks
└── visualizations/  # Sample overlays for inspection
```

#### Step 2: Train BiSeNet Model

```bash
# Binary segmentation
python train_bisenet.py \
    --train_dir ./dataset/bisenet_data \
    --n_classes 2 \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.005 \
    --image_size 512 1024 \
    --output_dir ./outputs/bisenet_binary \
    --use_ohem \
    --use_aux_loss

# Multi-class segmentation (5 classes)
python train_bisenet.py \
    --train_dir ./dataset/bisenet_multiclass \
    --n_classes 5 \
    --epochs 100 \
    --batch_size 8 \
    --output_dir ./outputs/bisenet_multiclass
```

**Training Parameters:**
- `--train_dir`: Dataset directory with images/ and masks/
- `--val_dir`: Optional validation directory
- `--n_classes`: 2 for binary, 5 for multi-class (4 lanes + background)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (reduce if OOM)
- `--lr`: Learning rate (default: 0.005)
- `--image_size`: Model input size (H W)
- `--use_ohem`: Use Online Hard Example Mining loss
- `--use_aux_loss`: Use auxiliary losses (recommended)

#### Step 3: Evaluate Model

```bash
python diagnose_bisenet.py \
    --checkpoint ./outputs/bisenet_binary/checkpoints/best_model.pth \
    --data_dir ./dataset/bisenet_data \
    --output_dir ./diagnostics/bisenet_binary \
    --n_classes 2 \
    --image_size 512 1024 \
    --visualize \
    --num_viz 30
```

**Outputs:**
- `metrics.json`: Accuracy, mIoU, per-class IoU, precision, recall
- `confusion_matrix.png`: Confusion matrix visualization
- `visualizations/`: Prediction overlays

#### Step 4: Run Inference

```bash
# Single image
python inference_bisenet.py \
    --checkpoint ./outputs/bisenet_binary/checkpoints/best_model.pth \
    --input /path/to/image.jpg \
    --output ./results/output.png \
    --n_classes 2

# Directory of images
python inference_bisenet.py \
    --checkpoint ./outputs/bisenet_binary/checkpoints/best_model.pth \
    --input /path/to/images/ \
    --output ./results/ \
    --n_classes 2 \
    --mode directory

# Video
python inference_bisenet.py \
    --checkpoint ./outputs/bisenet_binary/checkpoints/best_model.pth \
    --input /path/to/video.mp4 \
    --output ./results/output.mp4 \
    --n_classes 2 \
    --mode video
```

## Files Overview

### New BiSeNet Scripts (fine-tune3/)
- `convert_to_bisenet.py`: Convert JSON annotations to segmentation masks
- `model_bisenet.py`: BiSeNet model wrapper (uses ../BiSeNet/lib/)
- `train_bisenet.py`: Training script for BiSeNet
- `diagnose_bisenet.py`: Evaluation and diagnostics
- `inference_bisenet.py`: Run inference on images/videos
- `run_bisenet.sh`: Automated workflow script
- `BISENET_README.md`: This file

### Existing Files (fine-tune3/)
- `train.py`, `train_improved.py`, etc.: Your existing UFLD training scripts
- `model.py`: UFLD model architecture
- `diagnose.py`: Existing diagnostics

## Comparison: UFLD vs BiSeNet

### UFLD (existing model.py)
- **Type**: Lane detection with row-wise classification
- **Output**: Grid-based lane coordinates
- **Pros**: Fast, lightweight, designed for lane detection
- **Use Case**: Real-time lane detection applications

### BiSeNet (new scripts)
- **Type**: Semantic segmentation
- **Output**: Pixel-wise lane segmentation masks
- **Pros**: More accurate, better for complex scenes, flexible
- **Use Case**: High-accuracy lane segmentation, research

## Directory Structure

```
fine-tune3/
├── dataset/
│   ├── augmented/           # Original dataset
│   │   ├── images/
│   │   └── annotations/
│   └── bisenet_data/        # Converted for BiSeNet
│       ├── images/
│       ├── masks/
│       └── visualizations/
├── outputs/
│   └── bisenet_binary/      # Training outputs
│       ├── checkpoints/
│       │   ├── best_model.pth
│       │   └── final_model.pth
│       └── config.json
├── diagnostics/             # Evaluation results
│   └── bisenet_binary/
│       ├── metrics.json
│       ├── confusion_matrix.png
│       └── visualizations/
│
# BiSeNet scripts
├── convert_to_bisenet.py
├── model_bisenet.py
├── train_bisenet.py
├── diagnose_bisenet.py
├── inference_bisenet.py
├── run_bisenet.sh
├── BISENET_README.md
│
# Existing UFLD scripts
├── train.py
├── model.py
├── diagnose.py
└── ...
```

## Tips

### For Better Performance
1. **Data Quality**: Inspect visualizations after conversion
2. **Thickness**: Adjust based on image resolution (8-15px typical)
3. **Image Size**: Larger = better accuracy but slower
   - Small: 384x768 (faster)
   - Medium: 512x1024 (balanced)
   - Large: 768x1536 (accurate)
4. **Learning Rate**: Start with 0.005, reduce if unstable
5. **OHEM Loss**: Helps with class imbalance (lane pixels << background)
6. **Augmentations**: Built-in augmentations improve robustness

### Troubleshooting

**Out of Memory:**
```bash
python train_bisenet.py ... --batch_size 4 --image_size 384 768
```

**Poor Performance:**
1. Check data quality in visualizations/
2. Verify class balance in metrics.json
3. Increase lane thickness in conversion
4. Train longer (150-200 epochs)
5. Use OHEM loss: `--use_ohem`

**Import Errors:**
```bash
# Make sure BiSeNet directory exists
ls -la ../BiSeNet/lib/

# Test model import
python model_bisenet.py
```

## Example Workflow

```bash
# 1. Convert dataset
python convert_to_bisenet.py \
    --input_dir ./dataset/augmented \
    --output_dir ./dataset/bisenet_data \
    --binary \
    --thickness 12

# 2. Check visualizations
ls dataset/bisenet_data/visualizations/

# 3. Train model
python train_bisenet.py \
    --train_dir ./dataset/bisenet_data \
    --n_classes 2 \
    --epochs 100 \
    --batch_size 8 \
    --output_dir ./outputs/experiment1 \
    --use_ohem

# 4. Monitor training
# Watch: outputs/experiment1/

# 5. Evaluate
python diagnose_bisenet.py \
    --checkpoint ./outputs/experiment1/checkpoints/best_model.pth \
    --data_dir ./dataset/bisenet_data \
    --output_dir ./diagnostics/experiment1

# 6. Review results
cat diagnostics/experiment1/metrics.json
ls diagnostics/experiment1/visualizations/

# 7. Run inference
python inference_bisenet.py \
    --checkpoint ./outputs/experiment1/checkpoints/best_model.pth \
    --input ./test_images/ \
    --output ./results/ \
    --mode directory
```

## Dependencies

```bash
pip install torch torchvision opencv-python numpy albumentations tqdm matplotlib
```

## Model Architecture

BiSeNetV2 uses:
- **Detail Branch**: Preserves spatial information
- **Semantic Branch**: Captures high-level context
- **Bilateral Guided Aggregation**: Fuses both branches
- **Auxiliary Heads**: Deep supervision during training

Total Parameters: ~3.4M (lightweight for semantic segmentation)

## Performance Metrics

The diagnostic script reports:
- **Accuracy**: Overall pixel-wise accuracy
- **mIoU**: Mean Intersection over Union (primary metric)
- **Per-Class IoU**: IoU for each lane class
- **Precision/Recall**: Per-class precision and recall
- **Confusion Matrix**: Visualization of classification errors

## Citation

If using BiSeNet in your work:
```
@inproceedings{yu2021bisenet,
  title={Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation},
  author={Yu, Changqian and Gao, Changxin and Wang, Jingbo and Yu, Gang and Shen, Chunhua and Sang, Nong},
  booktitle={IJCV},
  year={2021}
}
```
