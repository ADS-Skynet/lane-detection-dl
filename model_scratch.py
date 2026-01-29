import torch
import torch.nn as nn
import torchvision.models as models


class ParsingNet(nn.Module):
    """UFLD V2 style lane detection network with configurable backbone"""

    def __init__(self, num_gridding=97, num_cls_row=68, num_lanes=6,
                 fc_hidden_dim=2048, backbone='resnet18', pretrained=True):
        super(ParsingNet, self).__init__()

        self.num_gridding = num_gridding
        self.num_cls_row = num_cls_row
        self.num_lanes = num_lanes

        # Select backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_out_channels = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_out_channels = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_channels = 2048
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            backbone_out_channels = 1280
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_out_channels = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classification head from backbone
        if 'resnet' in backbone:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        elif 'mobilenet' in backbone:
            self.backbone = self.backbone.features
        elif 'efficientnet' in backbone:
            self.backbone = self.backbone.features

        # Adaptive pooling to consistent size
        self.pool = nn.AdaptiveAvgPool2d((num_cls_row, 1))

        # Classification head
        self.cls = nn.Sequential(
            nn.Linear(backbone_out_channels, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout for regularization
            nn.Linear(fc_hidden_dim, num_gridding * num_cls_row * num_lanes)
        )

        print(f"✓ Using {backbone} backbone")
        print(f"  Backbone output channels: {backbone_out_channels}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        # Extract features with backbone
        x = self.backbone(x)  # [B, C, H, W]

        # Pool to consistent height
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # [B, C, H*W]
        x = x.permute(0, 2, 1)  # [B, H*W, C]

        # Global average pooling
        x = x.mean(dim=1)  # [B, C]

        # Classification
        x = self.cls(x)  # [B, num_gridding * num_cls_row * num_lanes]

        # Reshape to output format
        x = x.view(B, self.num_gridding, self.num_cls_row, self.num_lanes)

        return x