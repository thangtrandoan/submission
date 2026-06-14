from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        num_groups = 16 if out_channels % 16 == 0 else 32
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FCOSHead(nn.Module):
    def __init__(self, in_channels: int = 128, num_classes: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        self.cls_tower = nn.Sequential(
            ConvBlock(in_channels, in_channels, dropout=dropout),
            ConvBlock(in_channels, in_channels, dropout=dropout),
            ConvBlock(in_channels, in_channels, dropout=dropout),
            ConvBlock(in_channels, in_channels, dropout=dropout),
        )
        self.reg_tower = nn.Sequential(
            ConvBlock(in_channels, in_channels, dropout=dropout),
            ConvBlock(in_channels, in_channels, dropout=dropout),
            ConvBlock(in_channels, in_channels, dropout=dropout),
            ConvBlock(in_channels, in_channels, dropout=dropout),
        )
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.cnt_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self._init_weights()

    def forward(self, x: torch.Tensor, scale: Scale | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_features = self.cls_tower(x)
        reg_features = self.reg_tower(x)
        cls_logits = self.cls_head(cls_features)
        raw_reg = self.reg_head(reg_features)
        if scale is not None:
            raw_reg = scale(raw_reg)
        reg_preds = F.relu(raw_reg)
        cnt_logits = self.cnt_head(reg_features)
        return cls_logits, reg_preds, cnt_logits

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)


class TinyGridDetector(nn.Module):
    level_strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64}

    def __init__(
        self,
        num_classes: int = 5,
        pretrained_backbone: bool = True,
        use_p2: bool = False,
        use_p6: bool = False,
        use_scales: bool = False,
        channels: int = 128,
        use_bifpn: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_p2 = use_p2
        self.use_p6 = use_p6
        self.use_scales = use_scales
        self.channels = channels
        self.use_bifpn = use_bifpn
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.p5_conv = nn.Conv2d(2048, channels, kernel_size=1)
        self.p4_conv = nn.Conv2d(1024, channels, kernel_size=1)
        self.p3_conv = nn.Conv2d(512, channels, kernel_size=1)
        if self.use_p2:
            self.p2_conv = nn.Conv2d(256, channels, kernel_size=1)
            self.p2_smooth = ConvBlock(channels, channels)
            self.p3_out_smooth = ConvBlock(channels, channels)
        self.p5_smooth = ConvBlock(channels, channels)
        self.p4_smooth = ConvBlock(channels, channels)
        self.p3_smooth = ConvBlock(channels, channels)
        if self.use_bifpn:
            self.p4_out_smooth = ConvBlock(channels, channels)
            self.p5_out_smooth = ConvBlock(channels, channels)
        if self.use_p6:
            self.p6_out_smooth = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(16 if channels % 16 == 0 else 32, channels),
                nn.ReLU(inplace=True),
            )
        self.head = FCOSHead(channels, num_classes)
        if self.use_scales:
            self.scales = nn.ModuleDict({level: Scale(1.0) for level in self.level_strides})

    def _head(self, level: str, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.scales[level] if self.use_scales else None
        return self.head(feature, scale)

    def forward(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5_td = self.p5_conv(c5)
        p4_td = self.p4_smooth(self.p4_conv(c4) + F.interpolate(p5_td, size=c4.shape[-2:], mode="nearest"))
        p3_td = self.p3_smooth(self.p3_conv(c3) + F.interpolate(p4_td, size=c3.shape[-2:], mode="nearest"))
        if self.use_p2:
            p2_out = self.p2_smooth(self.p2_conv(c2) + F.interpolate(p3_td, size=c2.shape[-2:], mode="nearest"))
            p3_out = self.p3_out_smooth(p3_td + F.max_pool2d(p2_out, kernel_size=2, stride=2))
        else:
            p2_out = None
            p3_out = p3_td
        if self.use_bifpn:
            p4_out = self.p4_out_smooth(p4_td + F.max_pool2d(p3_out, kernel_size=2, stride=2))
            p5_out = self.p5_out_smooth(p5_td + F.max_pool2d(p4_out, kernel_size=2, stride=2))
        else:
            p4_out = p4_td
            p5_out = self.p5_smooth(p5_td)

        outputs = {
            "p3": self._head("p3", p3_out),
            "p4": self._head("p4", p4_out),
            "p5": self._head("p5", p5_out),
        }
        if p2_out is not None:
            outputs = {"p2": self._head("p2", p2_out), **outputs}
        if self.use_p6:
            outputs["p6"] = self._head("p6", self.p6_out_smooth(p5_out))
        return outputs
