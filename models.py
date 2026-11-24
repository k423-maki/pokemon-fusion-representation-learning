"""Architectures recovered from the two project notebooks.

Portable organization added with AI assistance on 2026-11-23.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, *, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(in_channels, out_channels, 3, stride, 1), nn.BatchNorm2d(out_channels), nn.ReLU()]


def up_block(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1), nn.BatchNorm2d(out_channels), nn.ReLU()]


class ConvVAE(nn.Module):
    def __init__(self, architecture: str = "baseline", latent_channels: int = 8,
                 num_classes: int = 170, pretrained: bool = False):
        super().__init__()
        if architecture not in {"baseline", "resnet"}:
            raise ValueError("architecture must be baseline or resnet")
        if latent_channels < 1 or num_classes < 2:
            raise ValueError("Invalid latent size or class count")
        self.architecture = architecture
        if architecture == "baseline":
            self.encoder = nn.Sequential(*conv_block(3, 64, stride=2),
                                         *conv_block(64, 128, stride=2),
                                         *conv_block(128, 256))
            self.decoder = nn.Sequential(*conv_block(256, 128), *up_block(128, 64),
                                         *up_block(64, 64), nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid())
            self.classifier = None
        else:
            from torchvision.models import ResNet18_Weights, resnet18
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                         resnet.layer1, resnet.layer2, resnet.layer3)
            self.decoder = nn.Sequential(*conv_block(256, 128), *up_block(128, 64),
                                         *up_block(64, 32), *up_block(32, 16),
                                         nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid())
            self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                            nn.Linear(latent_channels, 256), nn.ReLU(),
                                            nn.Linear(256, num_classes))
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.quant_conv(self.encoder(2 * x - 1))
        mean, logvar = h.chunk(2, dim=1)
        if self.training:
            limits = (-30.0, 20.0) if self.architecture == "baseline" else (-20.0, 10.0)
            logvar = logvar.clamp(*limits)
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            z = mean
        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z, mean, logvar = self.encode(x)
        result = (self.decode(z), z, mean, logvar)
        return result if self.classifier is None else (*result, self.classifier(z))
