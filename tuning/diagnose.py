#!/usr/bin/env python3
"""
Universal diagnostic tool for any lane detection model
Supports: ResNet (train.py), ResNet+Dropout (train_improved.py), MobileNet, DETR, etc.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys

# Import all model types (with error handling for missing models)
try:
    from model import ParsingNet as ResNetParsingNet
    from train_improved import ParsingNetWithDropout
    from train_mobilenet import SimpleLaneNet, LaneDataset
    from train_simple_grid import SimpleGridNet
except ImportError as e:
    print(f"Warning: Some model imports failed: {e}")
    print("Some model types may not be available.")

# Import DETR model
try:
    from rfdetr import RFDETRBase
    import supervision as sv
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    print("Note: rfdetr package not available. DETR inference will be limited to inspection only.")


def load_detr_model(checkpoint_path, device):
    """Load DETR model from checkpoint"""
    if not DETR_AVAILABLE:
        print("Error: rfdetr package not available. Please install it first:")
        print("  pip install rfdetr supervision")
        return None, None

    print(f"Loading DETR model from: {checkpoint_path}")

    try:
        # RFDETRBase is a high-level wrapper that handles the model internally
        # It loads from a checkpoint file path directly
        model = RFDETRBase(checkpoint=checkpoint_path)
        print("✓ DETR model loaded successfully")

        # Load checkpoint for metadata
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return model, checkpoint
    except Exception as e:
        print(f"⚠️  Error loading DETR model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_detr_predictions(model, image_dir, num_samples=10, threshold=0.5, save_path='diagnostics_detr.png'):
    """Visualize DETR predictions on sample images"""
    from PIL import Image

    print(f"\n{'='*60}")
    print("DETR PREDICTION VISUALIZATION")
    print(f"{'='*60}")

    # Get image files
    image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    if len(image_files) == 0:
        print(f"⚠️  No images found in {image_dir}")
        return

    # Sample random images
    np.random.seed(42)
    sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)

    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, num_samples * 4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    total_detections = 0
    detection_counts = []
    confidences = []

    for i, img_path in enumerate(sample_files):
        # Load image
        image = Image.open(img_path).convert('RGB')

        # Run inference
        detections = model.predict(image, threshold=threshold)

        # Count detections
        num_dets = len(detections) if hasattr(detections, '__len__') else (detections.xyxy.shape[0] if hasattr(detections, 'xyxy') else 0)
        total_detections += num_dets
        detection_counts.append(num_dets)

        # Get confidences if available
        if hasattr(detections, 'confidence'):
            confidences.extend(detections.confidence.tolist())

        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Sample {i+1}: Original\n{img_path.name}', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Annotated image
        annotated_image = np.array(image.copy())
        if num_dets > 0:
            # Annotate with boxes and labels
            annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
            annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)

        axes[i, 1].imshow(annotated_image)
        avg_conf = np.mean(detections.confidence) if hasattr(detections, 'confidence') and len(detections.confidence) > 0 else 0
        axes[i, 1].set_title(f'Predictions: {num_dets} detections\nAvg Conf: {avg_conf:.3f}',
                            fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")

    # Print statistics
    print(f"\nDetection Statistics:")
    print(f"  Total images: {len(sample_files)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections per image: {total_detections / len(sample_files):.1f}")
    if confidences:
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Median confidence: {np.median(confidences):.3f}")
        print(f"  High confidence (>{threshold}): {sum(1 for c in confidences if c > threshold)/len(confidences)*100:.1f}%")


def inspect_detr_checkpoint(checkpoint_path, device, run_inference=False, image_dir=None, num_samples=10):
    """Inspect DETR checkpoint and optionally run inference"""
    print(f"\n{'='*60}")
    print(f"DETR CHECKPOINT INSPECTION")
    print(f"{'='*60}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract args if available
    if 'args' in checkpoint:
        args = checkpoint['args']
        print(f"\nModel Configuration:")
        print(f"  Encoder: {getattr(args, 'encoder', 'N/A')}")
        print(f"  Decoder layers: {getattr(args, 'dec_layers', 'N/A')}")
        print(f"  Hidden dim: {getattr(args, 'hidden_dim', 'N/A')}")
        print(f"  Num queries: {getattr(args, 'num_queries', 'N/A')}")
        print(f"  Num classes: {getattr(args, 'num_classes', 'N/A')}")
        print(f"  Class names: {getattr(args, 'class_names', 'N/A')}")
        print(f"  Segmentation head: {getattr(args, 'segmentation_head', 'N/A')}")
        print(f"  Resolution: {getattr(args, 'resolution', 'N/A')}")
        print(f"  Pretrained weights: {getattr(args, 'pretrain_weights', 'N/A')}")

    # Get model state dict
    state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', {}))

    if isinstance(state_dict, dict):
        print(f"\nModel Structure:")
        print(f"  Total parameters: {len(state_dict)} layers")

        # Categorize layers
        encoder_layers = [k for k in state_dict.keys() if 'encoder' in k.lower()]
        decoder_layers = [k for k in state_dict.keys() if 'decoder' in k.lower()]
        transformer_layers = [k for k in state_dict.keys() if 'transformer' in k.lower()]
        head_layers = [k for k in state_dict.keys() if 'class_embed' in k or 'bbox_embed' in k or 'mask' in k]

        print(f"  Encoder layers: {len(encoder_layers)}")
        print(f"  Decoder/Transformer layers: {len(decoder_layers) + len(transformer_layers)}")
        print(f"  Head layers: {len(head_layers)}")

        # Sample keys
        print(f"\nSample keys:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            tensor_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {key}: {tensor_shape}")

    # Run inference if requested and available
    if run_inference and DETR_AVAILABLE and image_dir:
        print(f"\n{'='*60}")
        print("RUNNING INFERENCE")
        print(f"{'='*60}")

        model_loaded, _ = load_detr_model(checkpoint_path, device)
        if model_loaded:
            visualize_detr_predictions(model_loaded, image_dir, num_samples=num_samples)
        else:
            print("⚠️  Could not load model for inference")
    elif run_inference and not DETR_AVAILABLE:
        print(f"\n{'='*60}")
        print(f"INFERENCE NOT AVAILABLE")
        print(f"{'='*60}")
        print("To run inference, install required packages:")
        print("  pip install rfdetr supervision")
    elif run_inference and not image_dir:
        print(f"\n⚠️  Image directory not specified. Use --data to provide image directory.")

    if not run_inference:
        print(f"\n{'='*60}")
        print(f"NEXT STEPS:")
        print(f"{'='*60}")
        print("To run full diagnostics on this DETR model:")
        print("1. Install dependencies: pip install rfdetr supervision")
        print("2. Run with --run-inference flag and --data <image_dir>")
        print(f"\nExample:")
        print(f"  python diagnose.py --checkpoint {checkpoint_path} --model detr --run-inference --data training_data/images")
        print(f"{'='*60}\n")

    return checkpoint


def auto_detect_model_type(checkpoint_path, device):
    """Automatically detect model type AND parameters from checkpoint"""

    print(f"Auto-detecting model type from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if this is a DETR model
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'encoder'):
        # This looks like a DETR/Roboflow model
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', {}))
        if isinstance(state_dict, dict):
            sample_keys = list(state_dict.keys())[:5]
            if any('transformer' in k.lower() or 'decoder.layers' in k for k in sample_keys):
                print(f"  Detected: DETR/Transformer-based model")
                print(f"  Sample keys: {sample_keys}")
                # Note: This is called from load_model_and_checkpoint, so we just return
                # The main function handles DETR models separately
                return 'detr', 97, 68, 6, 512  # Return dummy values for compatibility

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Checkpoint might be the state_dict itself
        state_dict = checkpoint

    # Print some keys for debugging
    sample_keys = list(state_dict.keys())[:5]
    print(f"  Sample keys: {sample_keys}")

    # Remove 'model.' prefix if present for consistent checking
    normalized_dict = {}
    for k, v in state_dict.items():
        key = k[6:] if k.startswith('model.') else k
        normalized_dict[key] = v

    # Detect number of lanes from output layer size
    num_lanes = 6  # default
    num_gridding = 97  # default
    num_cls_row = 68  # default
    fc_hidden_dim = 512  # default

    # Find the final output layer
    final_layer_keys = ['cls.5.weight', 'cls.3.weight', 'cls.2.weight']
    for key in final_layer_keys:
        if key in normalized_dict:
            output_size = normalized_dict[key].shape[0]
            print(f"  Output layer '{key}' size: {output_size}")

            # Try to deduce parameters: output_size = num_gridding × num_cls_row × num_lanes
            for nl in [3, 4, 5, 6]:
                for ng in [50, 97, 100]:
                    for ncr in [30, 56, 68]:
                        if ng * ncr * nl == output_size:
                            num_lanes = nl
                            num_gridding = ng
                            num_cls_row = ncr
                            print(f"  → Detected config: {num_gridding} cols × {num_cls_row} rows × {num_lanes} lanes")
                            break
            break

    # Detect hidden dimension for MobileNet models
    if 'cls.2.weight' in normalized_dict:
        hidden_dim_shape = normalized_dict['cls.2.weight'].shape
        fc_hidden_dim = hidden_dim_shape[0]  # Output features of first FC layer
        print(f"  Hidden dim: {fc_hidden_dim}")

    # Strategy 1: Check classifier layer dimensions (most reliable)
    # MobileNet: cls.2.weight has shape [512, 1280] - 1280 is MobileNetV2 output
    # ResNet: cls.2.weight has shape [num_outputs, 2048] - 2048 is ResNet18 FC output
    # ResNet+Dropout: cls.3.weight exists (has extra dropout layer)

    if 'cls.2.weight' in normalized_dict:
        cls_weight_shape = normalized_dict['cls.2.weight'].shape
        input_features = cls_weight_shape[1] if len(cls_weight_shape) > 1 else cls_weight_shape[0]

        print(f"  Classifier input features: {input_features}")

        if input_features == 1280:
            detected = 'mobilenet'
            print(f"  → Detected: MobileNetV2 (classifier input = 1280)")
        elif input_features == 2048:
            # Check for dropout layer to distinguish resnet vs resnet_dropout
            if 'cls.3.weight' in normalized_dict:
                detected = 'resnet_dropout'
                print(f"  → Detected: ResNet18 with Dropout (has cls.3 layer)")
            else:
                detected = 'resnet'
                print(f"  → Detected: ResNet18 without Dropout")
        else:
            # Fallback: unknown architecture
            print(f"  ⚠️  Unknown classifier dimension: {input_features}")
            detected = 'resnet'  # Default fallback
            print(f"  → Defaulting to: ResNet18")
    else:
        # Strategy 2: Fallback to key pattern matching
        has_dropout = 'cls.3.weight' in normalized_dict
        has_mobilenet = any('features' in k for k in normalized_dict.keys())

        if has_mobilenet:
            detected = 'mobilenet'
            print(f"  → Detected: MobileNetV2 (has 'features' layers)")
        elif has_dropout:
            detected = 'resnet_dropout'
            print(f"  → Detected: ResNet18 with Dropout")
        else:
            detected = 'resnet'
            print(f"  → Detected: ResNet18 (default)")

    has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
    if has_model_prefix:
        print(f"  → Note: Checkpoint has 'model.' prefix (pretrained format)")

    return detected, num_gridding, num_cls_row, num_lanes, fc_hidden_dim


def load_model_and_checkpoint(model_type, checkpoint_path, device, auto_detect=False):
    """Load the appropriate model architecture and weights"""

    print(f"\n{'='*60}")
    print(f"LOADING MODEL")
    print(f"{'='*60}")

    # Load checkpoint first to check
    if not Path(checkpoint_path).exists():
        print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for ckpt in Path('.').glob('*.pth'):
            print(f"  - {ckpt}")
        sys.exit(1)

    # Auto-detect model type and parameters if requested or if loading fails
    if auto_detect or model_type == 'auto':
        model_type, num_gridding, num_cls_row, num_lanes, fc_hidden_dim = auto_detect_model_type(checkpoint_path, device)
    else:
        # Use defaults if not auto-detecting
        num_gridding = 97
        num_cls_row = 68
        num_lanes = 6
        fc_hidden_dim = 512

    print(f"Model type: {model_type}")
    print(f"Model config: {num_gridding} × {num_cls_row} × {num_lanes} lanes")

    # Create model based on type with detected parameters
    if model_type == 'resnet':
        print("Using ResNet18 (from train.py)")
        model = ResNetParsingNet(
            num_gridding=num_gridding,
            num_cls_row=num_cls_row,
            num_lanes=num_lanes,
            fc_hidden_dim=2048
        ).to(device)

    elif model_type == 'resnet_dropout':
        print("Using ResNet18 with Dropout (from train_improved.py)")
        model = ParsingNetWithDropout(
            num_gridding=num_gridding,
            num_cls_row=num_cls_row,
            num_lanes=num_lanes,
            fc_hidden_dim=2048,
            dropout_p=0.5
        ).to(device)

    elif model_type == 'mobilenet':
        # Choose the right MobileNet variant based on hidden dim
        if fc_hidden_dim == 256:
            print(f"Using SimpleGridNet (reduced complexity, hidden_dim={fc_hidden_dim})")
            model = SimpleGridNet(
                num_gridding=num_gridding,
                num_cls_row=num_cls_row,
                num_lanes=num_lanes
            ).to(device)
        else:
            print(f"Using SimpleLaneNet (standard, hidden_dim={fc_hidden_dim})")
            model = SimpleLaneNet(
                num_gridding=num_gridding,
                num_cls_row=num_cls_row,
                num_lanes=num_lanes
            ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state_dict from different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle 'model.' prefix in pretrained checkpoints
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'model.' prefix if present
        if k.startswith('model.'):
            new_key = k[6:]  # Remove 'model.'
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✓ Model loaded successfully")
        if 'loss' in checkpoint:
            print(f"  Checkpoint val_loss: {checkpoint.get('loss', checkpoint.get('val_loss', 'N/A')):.4f}")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
    except Exception as e:
        print(f"⚠️  Error loading model state: {e}")
        if not auto_detect:
            print("\nTrying auto-detection...")
            return load_model_and_checkpoint('auto', checkpoint_path, device, auto_detect=True)
        else:
            print("⚠️  Could not load model even with auto-detection")
            print("\nTroubleshooting tips:")
            print("1. Check if checkpoint is compatible with the model")
            print("2. Try specifying --model manually (resnet, resnet_dropout, or mobilenet)")
            print("3. Verify checkpoint file is not corrupted")
            sys.exit(1)

    model.eval()
    return model


def visualize_predictions(model, dataset, device, num_samples=10, save_path='diagnostics.png'):
    """Visualize GT vs predictions"""

    model.eval()

    # Get model grid dimensions
    num_gridding = model.num_gridding
    num_cls_row = model.num_cls_row
    num_lanes = model.num_lanes

    print(f"  Using grid: {num_gridding} × {num_cls_row} × {num_lanes} lanes")

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, num_samples * 4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
    ]

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img_tensor, label = dataset[i]

            # Denormalize image for visualization
            img = img_tensor.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)

            # Get prediction
            img_input = img_tensor.unsqueeze(0).to(device)
            output = model(img_input)
            output = output.permute(0, 2, 3, 1)  # [B, num_cls_row, num_lanes, num_gridding]

            # Convert to probabilities
            pred = torch.softmax(output[0], dim=-1)  # [num_cls_row, num_lanes, num_gridding]
            pred_indices = torch.argmax(pred, dim=-1)  # [num_cls_row, num_lanes]
            pred_conf = torch.max(pred, dim=-1)[0]  # [num_cls_row, num_lanes]

            h, w = img.shape[:2]

            # 1. Original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Sample {i+1}: Original Image', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')

            # 2. Ground truth overlay
            img_gt = img.copy()
            gt_lane_counts = [0] * num_lanes

            for lane_idx in range(min(label.shape[1], num_lanes)):  # num_lanes
                lane_points = []
                for row_idx in range(label.shape[0]):  # num_cls_row
                    col_idx = label[row_idx, lane_idx].item()
                    if col_idx != -1:
                        # Convert grid to pixel coordinates
                        x = int(col_idx / num_gridding * w)
                        y = int(row_idx / num_cls_row * h)
                        lane_points.append((x, y))
                        gt_lane_counts[lane_idx] += 1

                # Draw lane points and lines
                if len(lane_points) > 0:
                    color_bgr = colors[lane_idx % len(colors)]
                    # Draw points
                    for pt in lane_points:
                        cv2.circle(img_gt, pt, 3, color_bgr, -1)
                    # Draw lines
                    if len(lane_points) > 1:
                        for j in range(len(lane_points) - 1):
                            cv2.line(img_gt, lane_points[j], lane_points[j+1],
                                   color_bgr, 2)

            gt_text = f'GT Lanes: {sum(1 for c in gt_lane_counts if c > 0)}'
            axes[i, 1].imshow(img_gt)
            axes[i, 1].set_title(f'{gt_text} | Total Points: {sum(gt_lane_counts)}',
                                fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')

            # 3. Prediction overlay with confidence threshold
            img_pred = img.copy()
            pred_lane_counts = [0] * num_lanes
            avg_confidences = []

            for lane_idx in range(pred_indices.shape[1]):  # num_lanes
                lane_points = []
                lane_confidences = []

                for row_idx in range(pred_indices.shape[0]):  # num_cls_row
                    col_idx = pred_indices[row_idx, lane_idx].item()
                    confidence = pred_conf[row_idx, lane_idx].item()

                    # Confidence threshold
                    if confidence > 0.2:  # Lower threshold to see what model predicts
                        x = int(col_idx / num_gridding * w)
                        y = int(row_idx / num_cls_row * h)
                        lane_points.append((x, y))
                        lane_confidences.append(confidence)
                        pred_lane_counts[lane_idx] += 1

                # Draw predictions
                if len(lane_points) > 0:
                    color_bgr = colors[lane_idx % len(colors)]
                    avg_conf = np.mean(lane_confidences)
                    avg_confidences.append(avg_conf)

                    # Draw points with transparency based on confidence
                    for pt, conf in zip(lane_points, lane_confidences):
                        # Brighter = higher confidence
                        alpha = min(1.0, conf * 1.5)
                        color_with_alpha = tuple(c * alpha for c in color_bgr)
                        cv2.circle(img_pred, pt, 3, color_with_alpha, -1)

                    # Draw lines
                    if len(lane_points) > 1:
                        for j in range(len(lane_points) - 1):
                            cv2.line(img_pred, lane_points[j], lane_points[j+1],
                                   color_bgr, 2)

            pred_text = f'Pred Lanes: {sum(1 for c in pred_lane_counts if c > 0)}'
            avg_conf_text = f'Avg Conf: {np.mean(avg_confidences):.3f}' if avg_confidences else 'No predictions'
            axes[i, 2].imshow(img_pred)
            axes[i, 2].set_title(f'{pred_text} | {avg_conf_text}',
                                fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")


def analyze_label_statistics(dataset):
    """Analyze ground truth label statistics"""

    print(f"\n{'='*60}")
    print("GROUND TRUTH LABEL STATISTICS")
    print(f"{'='*60}")

    total_labels = 0
    labels_per_lane = [0] * 6
    rows_with_labels = [0] * 68
    images_with_lanes = 0

    for i in range(len(dataset)):
        _, label = dataset[i]
        has_labels = False

        for row_idx in range(label.shape[0]):
            for lane_idx in range(label.shape[1]):
                if label[row_idx, lane_idx] != -1:
                    total_labels += 1
                    labels_per_lane[lane_idx] += 1
                    rows_with_labels[row_idx] += 1
                    has_labels = True

        if has_labels:
            images_with_lanes += 1

    print(f"Total images: {len(dataset)}")
    print(f"Images with labels: {images_with_lanes}")
    print(f"Total labeled points: {total_labels}")
    if len(dataset) > 0:
        print(f"Avg labels per image: {total_labels / len(dataset):.1f}")
    else:
        print(f"Avg labels per image: N/A (no images)")

    print(f"\nLabels per lane:")
    for i, count in enumerate(labels_per_lane):
        if count > 0:
            print(f"  Lane {i}: {count:5d} points ({count/len(dataset):6.1f} per image)")

    print(f"\nRow coverage:")
    rows_used = sum(1 for x in rows_with_labels if x > 0)
    print(f"  Rows with labels: {rows_used}/68 ({rows_used/68*100:.1f}%)")
    print(f"  Most labeled row: {np.argmax(rows_with_labels)} with {max(rows_with_labels)} labels")
    print(f"  Least labeled row range: {np.argmin(rows_with_labels)}")


def analyze_prediction_accuracy(model, dataset, device, num_samples=None):
    """Analyze prediction accuracy and confidence"""

    if num_samples is None:
        num_samples = len(dataset)

    model.eval()

    # Get model dimensions
    num_lanes = model.num_lanes
    num_cls_row = model.num_cls_row
    num_gridding = model.num_gridding

    all_confidences = []
    correct_predictions = 0
    total_predictions = 0

    # Per-lane statistics
    lane_correct = [0] * num_lanes
    lane_total = [0] * num_lanes
    lane_confidences = [[] for _ in range(num_lanes)]

    print(f"\n{'='*60}")
    print("PREDICTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Model config: {num_gridding} × {num_cls_row} × {num_lanes} lanes")

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            img_tensor, label = dataset[i]

            img_input = img_tensor.unsqueeze(0).to(device)
            output = model(img_input)
            output = output.permute(0, 2, 3, 1)

            # Get predictions
            pred = torch.softmax(output[0], dim=-1)
            pred_indices = torch.argmax(pred, dim=-1)
            max_conf = torch.max(pred, dim=-1)[0]

            # Compare with ground truth - only for rows and lanes the model was trained on
            for row_idx in range(min(label.shape[0], num_cls_row)):  # Only check rows model has
                for lane_idx in range(min(label.shape[1], num_lanes)):  # Only check lanes model was trained on
                    gt_col = label[row_idx, lane_idx].item()

                    if gt_col != -1:  # Has ground truth
                        pred_col = pred_indices[row_idx, lane_idx].item()
                        confidence = max_conf[row_idx, lane_idx].item()

                        all_confidences.append(confidence)
                        lane_confidences[lane_idx].append(confidence)

                        # Check if prediction is close to ground truth (within 2 grid cells)
                        if abs(pred_col - gt_col) <= 2:
                            correct_predictions += 1
                            lane_correct[lane_idx] += 1

                        total_predictions += 1
                        lane_total[lane_idx] += 1

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        print(f"Overall Accuracy (±2 grid cells): {accuracy:.2f}%")
        print(f"Total predictions evaluated: {total_predictions}")

        print(f"\nConfidence Statistics:")
        print(f"  Mean confidence: {np.mean(all_confidences):.3f}")
        print(f"  Median confidence: {np.median(all_confidences):.3f}")
        print(f"  Std confidence: {np.std(all_confidences):.3f}")
        print(f"  High confidence (>0.5): {sum(1 for x in all_confidences if x > 0.5)/len(all_confidences)*100:.1f}%")
        print(f"  Medium confidence (0.3-0.5): {sum(1 for x in all_confidences if 0.3 <= x <= 0.5)/len(all_confidences)*100:.1f}%")
        print(f"  Low confidence (<0.3): {sum(1 for x in all_confidences if x < 0.3)/len(all_confidences)*100:.1f}%")

        print(f"\nPer-Lane Accuracy:")
        for i in range(num_lanes):
            if lane_total[i] > 0:
                lane_acc = lane_correct[i] / lane_total[i] * 100
                lane_conf = np.mean(lane_confidences[i]) if lane_confidences[i] else 0
                print(f"  Lane {i}: {lane_acc:5.1f}% accuracy | "
                      f"Avg conf: {lane_conf:.3f} | "
                      f"Samples: {lane_total[i]}")
    else:
        print("⚠️  No ground truth labels found to evaluate!")


def main():
    parser = argparse.ArgumentParser(description='Diagnose lane detection model')
    parser.add_argument('--model', type=str,
                       choices=['auto', 'resnet', 'resnet_dropout', 'mobilenet', 'detr'],
                       default='auto',
                       help='Model architecture type (use "auto" for automatic detection)')
    parser.add_argument('--checkpoint', type=str,
                       default='finetuned_best.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--data', type=str,
                       default='training_data',
                       help='Data directory (training_data or training_data_augmented)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default=None,
                       help='Output visualization filename')
    parser.add_argument('--inspect-only', action='store_true',
                       help='Only inspect checkpoint without running inference (useful for DETR models)')
    parser.add_argument('--run-inference', action='store_true',
                       help='Run inference on DETR models (requires rfdetr package)')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check if checkpoint exists first
    if not Path(args.checkpoint).exists():
        print(f"\n⚠️  Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        for ckpt in Path('.').glob('*.pth') + Path('.').glob('*.pt'):
            print(f"  - {ckpt}")
        sys.exit(1)

    # Quick check if this is a DETR model before loading dataset
    if args.model == 'auto':
        checkpoint_peek = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'args' in checkpoint_peek and hasattr(checkpoint_peek['args'], 'encoder'):
            state_dict = checkpoint_peek.get('model', checkpoint_peek.get('model_state_dict', {}))
            if isinstance(state_dict, dict):
                sample_keys = list(state_dict.keys())[:5]
                if any('transformer' in k.lower() or 'decoder.layers' in k for k in sample_keys):
                    # This is a DETR model
                    image_dir = f'{args.data}/images' if args.run_inference else None
                    inspect_detr_checkpoint(args.checkpoint, device,
                                          run_inference=args.run_inference,
                                          image_dir=image_dir,
                                          num_samples=args.samples)
                    return

    # If inspect-only mode or DETR model, just inspect and exit
    if args.inspect_only or args.model == 'detr':
        image_dir = f'{args.data}/images' if args.run_inference else None
        inspect_detr_checkpoint(args.checkpoint, device,
                              run_inference=args.run_inference,
                              image_dir=image_dir,
                              num_samples=args.samples)
        return

    # Load dataset (only for grid-based models)
    print(f"\nLoading dataset from: {args.data}")
    image_dir = f'{args.data}/images'
    annotation_dir = f'{args.data}/annotations'

    try:
        dataset = LaneDataset(image_dir, annotation_dir, augment=False)
    except NameError:
        print("Error: LaneDataset not available. Make sure required imports are available.")
        sys.exit(1)

    # Use validation split (same seed as training)
    val_size = int(len(dataset) * 0.2)
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    val_indices = indices[-val_size:]

    val_dataset = LaneDataset(image_dir, annotation_dir, augment=False)
    val_dataset.annotations = [dataset.annotations[i] for i in val_indices]

    print(f"Validation samples: {len(val_dataset)}")

    # Analyze labels
    analyze_label_statistics(val_dataset)

    # Load model (with auto-detection if needed)
    model = load_model_and_checkpoint(
        args.model, args.checkpoint, device,
        auto_detect=(args.model == 'auto')
    )

    # Set output filename based on detected/specified model type if not specified
    if args.output is None:
        # Get actual model type (may have been auto-detected)
        model_name = type(model).__name__.lower()
        args.output = f'diagnostics_{model_name}.png'

    # Analyze predictions
    analyze_prediction_accuracy(model, val_dataset, device)

    # Visualize
    print(f"\nGenerating visualizations ({args.samples} samples)...")
    visualize_predictions(model, val_dataset, device,
                         num_samples=args.samples,
                         save_path=args.output)

    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Visualization saved to: {args.output}")
    print("\nRecommendations based on results:")
    print("1. Check if predictions show any lane structure")
    print("2. If accuracy < 30%, consider simpler model or more data")
    print("3. If confidence < 0.3, model is very uncertain")
    print("4. Compare GT vs Pred to spot labeling errors")


if __name__ == '__main__':
    main()
