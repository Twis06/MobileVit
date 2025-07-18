import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilevit import MobileViT, mobilevit_xxs, mobilevit_xs, mobilevit_s
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class SegMobileViT(nn.Module):
    def __init__(self, backbone_name='mobilevit_xxs', num_classes=1, image_size=(256, 256)):
        super().__init__()
        # 选择 backbone
        if backbone_name == 'mobilevit_xxs':
            self.backbone = mobilevit_xxs()
            backbone_out_channels = 320
        elif backbone_name == 'mobilevit_xs':
            self.backbone = mobilevit_xs()
            backbone_out_channels = 384
        elif backbone_name == 'mobilevit_s':
            self.backbone = mobilevit_s()
            backbone_out_channels = 640
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        self.image_size = image_size
        # 去掉分类头
        self.backbone.fc = nn.Identity()
        self.backbone.pool = nn.Identity()
        # 分割头：上采样到输入尺寸
        self.seg_head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_classes, 1),
        )
    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.backbone.conv1(x)
        for i in range(4):
            feat = self.backbone.mv2[i](feat)
        for i in range(3):
            feat = self.backbone.mv2[4 + i](feat)
            feat = self.backbone.mvit[i](feat)
        feat = self.backbone.conv2(feat)
        out = self.seg_head(feat)
        out = F.interpolate(out, size=self.image_size, mode='bilinear', align_corners=False)
        return out

class SegMobileViT_DeepLabV3(nn.Module):
    def __init__(self, backbone_name='mobilevit_xxs', num_classes=1, image_size=(256, 256)):
        super().__init__()
        # 选择 backbone
        if backbone_name == 'mobilevit_xxs':
            self.backbone = mobilevit_xxs()
            backbone_out_channels = 320
        elif backbone_name == 'mobilevit_xs':
            self.backbone = mobilevit_xs()
            backbone_out_channels = 384
        elif backbone_name == 'mobilevit_s':
            self.backbone = mobilevit_s()
            backbone_out_channels = 640
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        self.image_size = image_size
        # 去掉分类头
        self.backbone.fc = nn.Identity()
        self.backbone.pool = nn.Identity()
        # DeepLabV3 Head: in_channels=backbone_out_channels, out_channels=num_classes
        self.deeplab_head = DeepLabHead(backbone_out_channels, num_classes)
    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.backbone.conv1(x)
        for i in range(4):
            feat = self.backbone.mv2[i](feat)
        for i in range(3):
            feat = self.backbone.mv2[4 + i](feat)
            feat = self.backbone.mvit[i](feat)
        feat = self.backbone.conv2(feat)
        out = self.deeplab_head(feat)
        out = F.interpolate(out, size=self.image_size, mode='bilinear', align_corners=False)
        return out 