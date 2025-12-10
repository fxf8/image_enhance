from dataclasses import dataclass

import torch
import py_real_esrgan.model

import image_enhance.database as idb


@dataclass
class EnhanceModelInput:
    image_sample: idb.ImageSample
    new_size: tuple[int, int]  # height, width


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        identity = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)

        return identity + out


class EMBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = py_real_esrgan.model.RealESRGAN(device="cpu", scale=4).model

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            return self.model(x.float().clamp(0, 1))


class EMHead(torch.nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()

        self.entry = torch.nn.Conv2d(3, channels, 3, padding=1)

        self.blocks = torch.nn.Sequential(
            *(ResidualBlock(channels) for _ in range(blocks))
        )

        self.exit = torch.nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor):
        original: torch.Tensor = x

        x = torch.relu(self.entry(x))
        x = self.blocks(x)
        x = self.exit(x)

        return x + original


class EnhanceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = EMBackbone()
        self.head = EMHead()

    def forward(self, model_input: EnhanceModelInput) -> torch.Tensor:
        sample_image: torch.Tensor = (
            model_input.image_sample.get_tensor().unsqueeze(0).float()
        )

        with torch.no_grad():
            x = self.backbone(sample_image)

        x = torch.nn.functional.interpolate(
            x,
            size=model_input.new_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.head(x)

        return x.squeeze(0)
