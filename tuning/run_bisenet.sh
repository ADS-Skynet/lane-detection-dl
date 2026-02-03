#!/bin/bash

# BiSeNet Training - Quick Start Script for fine-tune3
# Automates: dataset conversion, training, and evaluation

set -e  # Exit on error

# Configuration
DATASET_DIR="./dataset/augmented"
OUTPUT_BASE="./outputs"
DATASET_OUTPUT="./dataset/bisenet_data"
N_CLASSES=2  # 2 for binary, 5 for multi-class
BINARY_MODE=true  # true for binary, false for multi-class
EPOCHS=100
BATCH_SIZE=8
IMAGE_H=512
IMAGE_W=1024
THICKNESS=10

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}BiSeNet Lane Detection - fine-tune3${NC}"
echo -e "${BLUE}======================================${NC}"

# Step 1: Convert dataset
echo -e "\n${GREEN}[Step 1/3] Converting dataset to segmentation masks...${NC}"

if [ "$BINARY_MODE" = true ]; then
    python convert_to_bisenet.py \
        --input_dir "$DATASET_DIR" \
        --output_dir "$DATASET_OUTPUT" \
        --binary \
        --thickness $THICKNESS
    echo -e "${GREEN}✓ Binary segmentation masks created${NC}"
else
    python convert_to_bisenet.py \
        --input_dir "$DATASET_DIR" \
        --output_dir "$DATASET_OUTPUT" \
        --thickness $THICKNESS
    N_CLASSES=5
    echo -e "${GREEN}✓ Multi-class segmentation masks created${NC}"
fi

echo -e "${YELLOW}Check visualizations: $DATASET_OUTPUT/visualizations/${NC}"
read -p "Press Enter to continue with training, or Ctrl+C to stop..."

# Step 2: Train model
echo -e "\n${GREEN}[Step 2/3] Training BiSeNet model...${NC}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="bisenet_${TIMESTAMP}"
OUTPUT_DIR="$OUTPUT_BASE/$EXPERIMENT_NAME"

python train_bisenet.py \
    --train_dir "$DATASET_OUTPUT" \
    --n_classes $N_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 0.005 \
    --image_size $IMAGE_H $IMAGE_W \
    --output_dir "$OUTPUT_DIR" \
    --use_ohem \
    --use_aux_loss

echo -e "${GREEN}✓ Training complete${NC}"
echo -e "${YELLOW}Model saved to: $OUTPUT_DIR/checkpoints/${NC}"

# Step 3: Evaluate model
echo -e "\n${GREEN}[Step 3/3] Evaluating model...${NC}"

CHECKPOINT="$OUTPUT_DIR/checkpoints/best_model.pth"
DIAG_DIR="./diagnostics/$EXPERIMENT_NAME"

if [ -f "$CHECKPOINT" ]; then
    python diagnose_bisenet.py \
        --checkpoint "$CHECKPOINT" \
        --data_dir "$DATASET_OUTPUT" \
        --output_dir "$DIAG_DIR" \
        --n_classes $N_CLASSES \
        --image_size $IMAGE_H $IMAGE_W \
        --visualize \
        --num_viz 20

    echo -e "${GREEN}✓ Evaluation complete${NC}"
    echo -e "${YELLOW}Results saved to: $DIAG_DIR${NC}"

    # Display metrics
    if [ -f "$DIAG_DIR/metrics.json" ]; then
        echo -e "\n${BLUE}Metrics Summary:${NC}"
        python -c "import json; metrics = json.load(open('$DIAG_DIR/metrics.json')); print(f\"Accuracy: {metrics['accuracy']:.4f}\"); print(f\"mIoU: {metrics['miou']:.4f}\")"
    fi
else
    echo -e "${YELLOW}⚠ Best model checkpoint not found, skipping evaluation${NC}"
fi

echo -e "\n${BLUE}======================================${NC}"
echo -e "${GREEN}✓ All steps completed!${NC}"
echo -e "${BLUE}======================================${NC}"

echo -e "\n${YELLOW}Summary:${NC}"
echo -e "  Dataset: $DATASET_OUTPUT"
echo -e "  Model: $OUTPUT_DIR"
echo -e "  Diagnostics: $DIAG_DIR"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  - Review visualizations: $DIAG_DIR/visualizations/"
echo -e "  - Check metrics: $DIAG_DIR/metrics.json"
echo -e "  - Run inference:"
echo -e "    python inference_bisenet.py \\"
echo -e "      --checkpoint $CHECKPOINT \\"
echo -e "      --input /path/to/images \\"
echo -e "      --output ./results \\"
echo -e "      --n_classes $N_CLASSES"
