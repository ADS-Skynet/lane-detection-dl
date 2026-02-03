#!/usr/bin/env python3
"""
BiSeNet Training Script for fine-tune3
Train BiSeNetV2 on lane segmentation dataset
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add BiSeNet to path
bisenet_path = Path(__file__).parent.parent / 'BiSeNet' / 'lib'
sys.path.insert(0, str(bisenet_path))

from model_bisenet import create_bisenet_model
from ohem_ce_loss import OhemCELoss


class LaneSegmentationDataset(Dataset):
    """Dataset for BiSeNet lane segmentation"""

    def __init__(self, data_dir, mode='train', image_size=(512, 1024), n_classes=2):
        """
        Args:
            data_dir: Directory containing 'images' and 'masks' folders
            mode: 'train' or 'val'
            image_size: (height, width)
            n_classes: Number of classes
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.image_size = image_size
        self.n_classes = n_classes

        self.images_dir = self.data_dir / 'images'
        self.masks_dir = self.data_dir / 'masks'

        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) +
                                 list(self.images_dir.glob('*.png')))

        # Filter to only include images with corresponding masks
        self.samples = []
        for img_path in self.image_files:
            mask_path = self.masks_dir / (img_path.stem + '.png')
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        print(f"Loaded {len(self.samples)} samples for {mode} mode")

        # Setup transforms
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        """Get augmentation transforms"""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                    val_shift_limit=10, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask = torch.from_numpy(mask).long()

        return image, mask


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize model
        print(f"Initializing BiSeNet with {args.n_classes} classes")
        self.model = create_bisenet_model(n_classes=args.n_classes, pretrained=True)
        self.model.to(self.device)

        # Initialize datasets
        print(f"Loading training data from {args.train_dir}")
        train_dataset = LaneSegmentationDataset(
            data_dir=args.train_dir,
            mode='train',
            image_size=tuple(args.image_size),
            n_classes=args.n_classes
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        if args.val_dir:
            print(f"Loading validation data from {args.val_dir}")
            val_dataset = LaneSegmentationDataset(
                data_dir=args.val_dir,
                mode='val',
                image_size=tuple(args.image_size),
                n_classes=args.n_classes
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None

        # Initialize loss
        if args.use_ohem:
            print("Using OHEM Cross Entropy Loss")
            self.criterion = OhemCELoss(thresh=0.7, n_min=100000)
        else:
            print("Using standard Cross Entropy Loss")
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Initialize optimizer
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = self.model.get_params()
        self.optimizer = optim.AdamW([
            {'params': wd_params, 'weight_decay': args.weight_decay},
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': args.lr * 10, 'weight_decay': args.weight_decay},
            {'params': lr_mul_nowd_params, 'lr': args.lr * 10, 'weight_decay': 0},
        ], lr=args.lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        # Training state
        self.current_epoch = 0
        self.best_miou = 0.0
        self.global_step = 0

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.model.set_mode('train')
        epoch_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.args.epochs}')

        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits = outputs[0]

            # Calculate loss
            loss_main = self.criterion(logits, masks)

            # Add auxiliary losses
            if self.args.use_aux_loss:
                loss_aux2 = self.criterion(outputs[1], masks)
                loss_aux3 = self.criterion(outputs[2], masks)
                loss_aux4 = self.criterion(outputs[3], masks)
                loss_aux5_4 = self.criterion(outputs[4], masks)

                loss = loss_main + 0.4 * (loss_aux2 + loss_aux3 + loss_aux4 + loss_aux5_4)
            else:
                loss = loss_main

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': epoch_loss / (batch_idx + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })

        return epoch_loss / len(self.train_loader)

    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        self.model.set_mode('eval')
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        class_iou_sum = torch.zeros(self.args.n_classes).to(self.device)
        class_count = torch.zeros(self.args.n_classes).to(self.device)

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)
                logits = outputs[0]

                # Calculate loss
                loss = self.criterion(logits, masks)
                total_loss += loss.item()

                # Calculate accuracy
                preds = logits.argmax(dim=1)
                total_correct += (preds == masks).sum().item()
                total_pixels += masks.numel()

                # Calculate IoU per class
                for cls in range(self.args.n_classes):
                    pred_mask = (preds == cls)
                    true_mask = (masks == cls)

                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()

                    if union > 0:
                        class_iou_sum[cls] += intersection / union
                        class_count[cls] += 1

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_pixels

        class_iou = class_iou_sum / (class_count + 1e-8)
        miou = class_iou.mean().item()

        print(f"\nValidation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  mIoU: {miou:.4f}")

        return miou, avg_loss

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'global_step': self.global_step,
            'args': vars(self.args)
        }

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

    def train(self):
        """Main training loop"""
        print(f"\nStarting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Epochs: {self.args.epochs}\n")

        for epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}")

            # Validate
            if self.val_loader and (epoch + 1) % self.args.val_interval == 0:
                miou, val_loss = self.validate()

                if miou > self.best_miou:
                    self.best_miou = miou
                    self.save_checkpoint(is_best=True)
                    print(f"New best mIoU: {miou:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch_{epoch + 1}.pth')

            # Update learning rate
            self.scheduler.step()

        # Save final model
        self.save_checkpoint(filename='final_model.pth')
        print(f"\nTraining completed! Best mIoU: {self.best_miou:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train BiSeNet for Lane Detection')

    # Dataset
    parser.add_argument('--train_dir', type=str, default='./dataset/bisenet_data',
                       help='Training dataset directory')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Validation dataset directory')

    # Model
    parser.add_argument('--n_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 1024],
                       help='Image size (height width)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Loss
    parser.add_argument('--use_ohem', action='store_true',
                       help='Use OHEM loss')
    parser.add_argument('--use_aux_loss', action='store_true', default=True,
                       help='Use auxiliary losses')

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Run validation every N epochs')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
