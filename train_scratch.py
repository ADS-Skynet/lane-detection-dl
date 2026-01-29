#!/usr/bin/env python3
"""
Fine-tune UFLD V2 model on custom track data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pathlib import Path
from model_scratch import ParsingNet
import time

class LaneDataset(Dataset):
    """Custom dataset for lane detection"""

    def __init__(self, image_dir, annotation_dir, img_w=800, img_h=320,
                 num_gridding=97, num_cls_row=68, num_lanes=6):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.img_w = img_w
        self.img_h = img_h
        self.num_gridding = num_gridding
        self.num_cls_row = num_cls_row
        self.num_lanes = num_lanes

        # Load annotations
        self.annotations = []
        for json_file in sorted(self.annotation_dir.glob('*.json')):
            try:
                # Try UTF-8 first
                with open(json_file, encoding='utf-8') as f:
                    self.annotations.append(json.load(f))
            except UnicodeDecodeError:
                # Fall back to latin-1 if UTF-8 fails
                try:
                    with open(json_file, encoding='latin-1') as f:
                        self.annotations.append(json.load(f))
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
                    continue
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in {json_file}: {e}")
                continue

        print(f"Loaded {len(self.annotations)} annotated images")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get image and ground truth"""
        ann = self.annotations[idx]

        # Load image
        img_path = self.image_dir / ann['image']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # Data augmentation (random)
        if np.random.rand() > 0.5:
            # Random brightness
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        if np.random.rand() > 0.5:
            # Random contrast
            contrast = np.random.uniform(0.8, 1.2)
            img = np.clip((img - 128) * contrast + 128, 0, 255).astype(np.uint8)

        if np.random.rand() > 0.5:
            # Horizontal flip
            img = cv2.flip(img, 1)
            # Flip lane annotations too
            lanes_flipped = []
            for lane in ann['lanes']:
                if lane:
                    flipped_lane = [(orig_w - x, y) for x, y in lane]
                    lanes_flipped.append(flipped_lane)
                else:
                    lanes_flipped.append([])
            ann = {'image': ann['image'], 'lanes': lanes_flipped}

        # Resize
        img = cv2.resize(img, (self.img_w, self.img_h))

        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # To tensor [C, H, W]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # Create ground truth label [num_gridding, num_cls_row, num_lanes]
        label = self.create_ground_truth(ann['lanes'], orig_w, orig_h)

        return img_tensor, label

    def create_ground_truth(self, lanes, orig_w, orig_h):
        """Convert lane annotations to UFLD ground truth format"""
        # Initialize with -1 (no lane)
        label = np.full((self.num_cls_row, self.num_lanes), -1, dtype=np.int64)

        for lane_idx, lane in enumerate(lanes[:self.num_lanes]):
            if not lane:
                continue

            # Convert lane points to grid coordinates
            for x, y in lane:
                # Scale to resized image
                x_scaled = x * self.img_w / orig_w
                y_scaled = y * self.img_h / orig_h

                # Convert to grid indices
                col = int(x_scaled / self.img_w * self.num_gridding)
                row = int(y_scaled / self.img_h * self.num_cls_row)

                # Clamp to valid range
                col = max(0, min(col, self.num_gridding - 1))
                row = max(0, min(row, self.num_cls_row - 1))

                # Set label (column index for this row)
                label[row, lane_idx] = col

        return torch.from_numpy(label)


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4):
    """Train the model"""

    # Loss function - CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Optimizer - Adam with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler - more gradual reduction
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_loss = float('inf')

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Reshape outputs [B, num_gridding, num_cls_row, num_lanes]
            #   to [B, num_cls_row, num_lanes, num_gridding] for loss calculation
            outputs = outputs.permute(0, 2, 3, 1)  # [B, num_cls_row, num_lanes, num_gridding]

            # Calculate loss - reshape and compute in one go
            # Flatten: [B, num_cls_row, num_lanes, num_gridding] -> [B*num_cls_row*num_lanes, num_gridding]
            batch_size = outputs.shape[0]
            outputs_flat = outputs.reshape(-1, outputs.shape[-1])  # [B*68*6, 97]
            labels_flat = labels.reshape(-1)  # [B*68*6]

            # Only compute loss on valid labels (not -1)
            valid_mask = labels_flat != -1

            if valid_mask.any():
                loss = criterion(outputs_flat[valid_mask], labels_flat[valid_mask])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    outputs = outputs.permute(0, 2, 3, 1)

                    # Calculate loss
                    outputs_flat = outputs.reshape(-1, outputs.shape[-1])
                    labels_flat = labels.reshape(-1)
                    valid_mask = labels_flat != -1

                    if valid_mask.any():
                        loss = criterion(outputs_flat[valid_mask], labels_flat[valid_mask])
                        val_loss += loss.item()


            val_loss /= len(val_loader)

            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, 'finetuned_best.pth')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s")

        # Step scheduler based on validation loss
        if val_loader:
            scheduler.step(val_loss)

        # Early stopping if validation loss increases for too long
        if val_loader and epoch > 10 and val_loss > best_loss * 1.5:
            print(f"\n⚠️  Early stopping: validation loss diverging (current: {val_loss:.4f}, best: {best_loss:.4f})")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'finetuned_final.pth')

    print("\n✓ Training complete!")
    print(f"✓ Best model saved to: finetuned_best.pth")
    print(f"✓ Final model saved to: finetuned_final.pth")


def main():
    # Configuration
    image_dir = 'training_data_augmented/images'
    annotation_dir = 'training_data_augmented/annotations'
    batch_size = 16  # Larger batches with augmentation
    epochs = 200  # More epochs needed
    learning_rate = 5e-5
    val_split = 0.15  # Keep more for training (85% train, 15% val)

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = LaneDataset(image_dir, annotation_dir)

    # Split train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2) if val_size > 0 else None

    # Create model with configurable backbone
    # Options: 'resnet18', 'resnet34', 'mobilenet_v2', 'efficientnet_b0'
    backbone_choice = 'mobilenet_v2'  # Lighter model for small dataset

    model = ParsingNet(
        num_gridding=97,
        num_cls_row=68,
        num_lanes=6,
        fc_hidden_dim=1024,  # Reduced from 2048
        backbone=backbone_choice,
        pretrained=True  # Use ImageNet pretrained weights
    ).to(device)

    print(f"✓ Using {backbone_choice} with ImageNet pretraining")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    train_model(model, train_loader, val_loader, device, epochs, learning_rate)


if __name__ == '__main__':
    main()