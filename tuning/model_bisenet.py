"""
BiSeNetV2 Model for Lane Detection - fine-tune3 version
Uses the BiSeNet model from the parent BiSeNet directory
"""

import sys
import os
from pathlib import Path

# Add BiSeNet to path
bisenet_path = Path(__file__).parent.parent / 'BiSeNet' / 'lib'
sys.path.insert(0, str(bisenet_path))

from models.bisenetv2 import BiSeNetV2
import torch
import torch.nn as nn


class BiSeNetLaneDetector(nn.Module):
    """
    Wrapper for BiSeNetV2 for lane detection
    Provides a simplified interface compatible with fine-tune3 workflow
    """

    def __init__(self, n_classes=2, aux_mode='train', pretrained=True):
        """
        Args:
            n_classes: Number of classes (2 for binary, 5 for 4 lanes + background)
            aux_mode: 'train', 'eval', or 'pred'
            pretrained: Load pretrained backbone weights
        """
        super(BiSeNetLaneDetector, self).__init__()
        self.n_classes = n_classes
        self.aux_mode = aux_mode

        self.model = BiSeNetV2(n_classes=n_classes, aux_mode=aux_mode)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            During training: (logits, aux2, aux3, aux4, aux5_4)
            During eval: (logits,)
            During pred: predictions
        """
        return self.model(x)

    def set_mode(self, mode):
        """
        Change auxiliary mode

        Args:
            mode: 'train', 'eval', or 'pred'
        """
        self.aux_mode = mode
        self.model.aux_mode = mode

    def get_params(self):
        """
        Get parameter groups for different learning rates

        Returns:
            (wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params)
        """
        return self.model.get_params()


def create_bisenet_model(n_classes=2, pretrained=True):
    """
    Factory function to create BiSeNet model

    Args:
        n_classes: Number of segmentation classes
        pretrained: Load pretrained backbone weights

    Returns:
        BiSeNetLaneDetector model
    """
    model = BiSeNetLaneDetector(n_classes=n_classes, aux_mode='train', pretrained=pretrained)
    return model


if __name__ == '__main__':
    # Test model
    print("Testing BiSeNet model...")

    model = create_bisenet_model(n_classes=2)
    model.eval()

    # Test input
    x = torch.randn(2, 3, 512, 1024)

    print(f"Input shape: {x.shape}")

    # Test training mode
    model.set_mode('train')
    outputs = model(x)
    print(f"Training outputs: {len(outputs)} tensors")
    print(f"  Main output shape: {outputs[0].shape}")

    # Test eval mode
    model.set_mode('eval')
    outputs = model(x)
    print(f"Eval outputs: {len(outputs)} tensor(s)")
    print(f"  Output shape: {outputs[0].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nBiSeNet model test completed successfully!")
