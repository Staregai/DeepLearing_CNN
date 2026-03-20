from __future__ import annotations

from io import BytesIO
import random

from PIL import Image
from torchvision import transforms


class Cutout:
    def __init__(self, size: int = 8):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        x = random.randint(0, img.width - 1)
        y = random.randint(0, img.height - 1)
        half = self.size // 2

        left = max(0, x - half)
        upper = max(0, y - half)
        right = min(img.width, x + half)
        lower = min(img.height, y + half)

        img = img.copy()
        fill = (0, 0, 0)
        for i in range(left, right):
            for j in range(upper, lower):
                img.putpixel((i, j), fill)
        return img


class CompressionArtifact:
    def __init__(self, min_quality: int = 10, max_quality: int = 50):
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.min_quality, self.max_quality)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_train_transforms(profile: str = "baseline") -> transforms.Compose:
    base = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]

    advanced = {
        "baseline": [],
        "color_jitter": [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)],
        "autoaugment": [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)],
        "cutout": [Cutout(size=8)],
        "compression": [CompressionArtifact()],
        "combo": [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            Cutout(size=8),
            CompressionArtifact(),
        ],
    }

    if profile not in advanced:
        raise ValueError(f"Unknown augmentation profile: {profile}")

    return transforms.Compose(
        base
        + advanced[profile]
        + [
            transforms.ToTensor(),
            transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
        ]
    )


def build_eval_transforms(image_size: int = 32) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
        ]
    )
