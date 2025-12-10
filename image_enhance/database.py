from typing import cast

from dataclasses import dataclass
import glob
import pathlib

import PIL.Image
import torch
import torchvision.transforms.functional

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt


@dataclass
class ImageSample:
    image_path: pathlib.Path
    image_tensor: torch.Tensor | None = None
    is_modified: bool = False

    """
    Field image_tensor has shape (3, height, width)
    Component range is [0, 1]
    """

    def get_tensor(self, cache: bool = False) -> torch.Tensor:
        if self.image_tensor is not None:
            return self.image_tensor

        image_tensor: torch.Tensor = torchvision.transforms.functional.to_tensor(
            PIL.Image.open(self.image_path).convert("RGB")
        )

        if cache:
            self.image_tensor = image_tensor

        return image_tensor

    def get_size(self, cache: bool = False) -> tuple[int, int]:
        _, height, width = self.get_tensor(cache=cache).shape

        return width, height

    def resize(self, new_size: tuple[int, int]) -> "ImageSample":
        return ImageSample(
            image_path=self.image_path,
            image_tensor=torchvision.transforms.functional.resize(
                self.get_tensor(), [*new_size]
            ),
            is_modified=True,
        )

    def gaussian_blur(self, kernel_size: int = 5, sigma: float = 1.0) -> "ImageSample":
        return ImageSample(
            image_path=self.image_path,
            image_tensor=torchvision.transforms.functional.gaussian_blur(
                self.get_tensor(), [kernel_size, kernel_size], sigma=[sigma, sigma]
            ),
            is_modified=True,
        )

    def add_noise(self, std: float = 0.05) -> "ImageSample":
        return ImageSample(
            image_path=self.image_path,
            image_tensor=self.get_tensor() + torch.randn_like(self.get_tensor()) * std,
            is_modified=True,
        )

    def cutout(self, scale: tuple[float, float] = (0.1, 0.3)) -> "ImageSample":
        eraser = torchvision.transforms.RandomErasing(scale=scale, ratio=(0.3, 3.3))

        return ImageSample(
            image_path=self.image_path,
            image_tensor=eraser(self.get_tensor().clone()),
            is_modified=True,
        )

    def dropout(self, probability: float = 0.1) -> "ImageSample":
        return ImageSample(
            image_path=self.image_path,
            image_tensor=self.get_tensor()
            * (torch.rand_like(self.get_tensor()) > probability),
            is_modified=True,
        )

    def jpeg_compress(self, quality=15) -> "ImageSample":
        img = (self.get_tensor() * 255).byte()
        encoded = torchvision.io.encode_jpeg(img, quality=quality)
        decoded = (
            cast(torch.Tensor, torchvision.io.decode_jpeg(encoded)).float() / 255.0
        )

        return ImageSample(self.image_path, decoded, True)

    def display_image(self):
        plt.imshow(self.get_tensor().permute(1, 2, 0))
        plt.show()


def import_glob(glob_pattern: str = "*") -> list[ImageSample]:
    return [
        ImageSample(image_path=pathlib.Path(image_path))
        for image_path in glob.glob(glob_pattern)
        if image_path.endswith(".jpg")
        or image_path.endswith(".png")
        or image_path.endswith(".jpeg")
    ]
