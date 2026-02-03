"""
BiSeNet V2 - Bilateral Segmentation Network
Efficient semantic segmentation for real-time applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, ks=3, stride=2, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
            ConvBNReLU(64, 64, ks=3, stride=1, padding=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, ks=3, stride=2, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
            ConvBNReLU(128, 128, ks=3, stride=1, padding=1),
        )

    def forward(self, x):
        feat_s1 = self.S1(x)
        feat_s2 = self.S2(feat_s1)
        feat_s3 = self.S3(feat_s2)
        return feat_s3


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, ks=3, stride=2, padding=1)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, ks=1, stride=1, padding=0),
            ConvBNReLU(8, 16, ks=3, stride=2, padding=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, ks=3, stride=1, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class GELayer(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6, stride=1):
        super(GELayer, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, ks=3, stride=1, padding=1)

        # Different structure based on whether we need shortcut
        self.use_shortcut = (in_chan != out_chan) or (stride != 1)

        if self.use_shortcut:
            # Layers with stride or channel change use dwconv1 and dwconv2
            self.dwconv1 = nn.Sequential(
                nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=stride, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
            )
            self.dwconv2 = nn.Sequential(
                nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=stride, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        else:
            # Layers with same channels and stride=1 use single dwconv
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
            )
            self.shortcut = None

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)

        if self.use_shortcut:
            feat = self.dwconv1(feat)
            feat = self.dwconv2(feat)
            feat = self.conv2(feat)
            shortcut = self.shortcut(x)
            feat = feat + shortcut
        else:
            feat = self.dwconv(feat)
            feat = self.conv2(feat)
            feat = feat + x

        feat = self.relu(feat)
        return feat


class CEBlock(nn.Module):
    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, ks=1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class SemanticBranch(nn.Module):
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayer(16, 32, exp_ratio=6, stride=2),
            GELayer(32, 32, exp_ratio=6, stride=1),
        )
        self.S4 = nn.Sequential(
            GELayer(32, 64, exp_ratio=6, stride=2),
            GELayer(64, 64, exp_ratio=6, stride=1),
        )
        self.S5_4 = nn.Sequential(
            GELayer(64, 128, exp_ratio=6, stride=2),
            GELayer(128, 128, exp_ratio=6, stride=1),
            GELayer(128, 128, exp_ratio=6, stride=1),
            GELayer(128, 128, exp_ratio=6, stride=1),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGA(nn.Module):
    def __init__(self):
        super(BGA, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(right, size=dsize, mode='bilinear', align_corners=True)
        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, use_aux=False):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        if use_aux:
            # Aux heads have an extra conv layer
            self.conv_out = nn.Sequential(
                nn.Sequential(
                    nn.Identity(),  # Placeholder for index 0
                    ConvBNReLU(mid_chan, mid_chan // 8, ks=3, stride=1, padding=1)
                ),
                nn.Conv2d(mid_chan // 8, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        else:
            # Main head uses direct output
            self.conv_out = nn.Sequential(
                nn.Identity(),  # Placeholder for index 0
                nn.Conv2d(mid_chan, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out[1](x)
        return x


class BiSeNetV2(nn.Module):
    def __init__(self, n_classes=2, aux_mode='train'):
        super(BiSeNetV2, self).__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SemanticBranch()
        self.bga = BGA()
        self.head = SegmentHead(128, 1024, n_classes, use_aux=False)

        if self.aux_mode == 'train':
            self.aux2 = SegmentHead(16, 128, n_classes, use_aux=True)
            self.aux3 = SegmentHead(32, 128, n_classes, use_aux=True)
            self.aux4 = SegmentHead(64, 128, n_classes, use_aux=True)
            self.aux5_4 = SegmentHead(128, 128, n_classes, use_aux=True)

    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        logits = self.head(feat_head)
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)

        if self.aux_mode == 'train':
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            logits_aux2 = F.interpolate(logits_aux2, size=size, mode='bilinear', align_corners=True)
            logits_aux3 = F.interpolate(logits_aux3, size=size, mode='bilinear', align_corners=True)
            logits_aux4 = F.interpolate(logits_aux4, size=size, mode='bilinear', align_corners=True)
            logits_aux5_4 = F.interpolate(logits_aux5_4, size=size, mode='bilinear', align_corners=True)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        elif self.aux_mode == 'eval':
            return logits,
        elif self.aux_mode == 'pred':
            pred = logits.argmax(dim=1)
            return pred
        else:
            raise NotImplementedError
