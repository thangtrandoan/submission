from __future__ import annotations

import random
from typing import Any

import torch
from PIL import Image, ImageEnhance


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
LETTERBOX_FILL = (114, 114, 114)


class DetectionTransform:
    def __init__(
        self,
        img_size: int = 416,
        train: bool = True,
        hflip_prob: float = 0.5,
        color_jitter: float = 1.0,
    ) -> None:
        self.img_size = img_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter

    def __call__(self, image: Image.Image, targets: list[dict[str, Any]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        image = image.convert("RGB")
        width, height = image.size
        targets = [{"class_id": item["class_id"], "bbox": list(item["bbox"])} for item in targets]

        if self.train and random.random() < self.hflip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for item in targets:
                xmin, ymin, xmax, ymax = item["bbox"]
                item["bbox"] = [width - xmax, ymin, width - xmin, ymax]

        if self.train and self.color_jitter > 0:
            image = self._color_jitter(image)

        image, scale, pad_x, pad_y = self._letterbox(image)
        for item in targets:
            xmin, ymin, xmax, ymax = item["bbox"]
            item["bbox"] = [
                max(0.0, min(self.img_size, xmin * scale + pad_x)),
                max(0.0, min(self.img_size, ymin * scale + pad_y)),
                max(0.0, min(self.img_size, xmax * scale + pad_x)),
                max(0.0, min(self.img_size, ymax * scale + pad_y)),
            ]

        tensor = self._to_tensor(image)
        return tensor, targets

    def _letterbox(self, image: Image.Image) -> tuple[Image.Image, float, int, int]:
        width, height = image.size
        scale = min(self.img_size / width, self.img_size / height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        resized = image.resize((resized_w, resized_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.img_size, self.img_size), LETTERBOX_FILL)
        pad_x = (self.img_size - resized_w) // 2
        pad_y = (self.img_size - resized_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        return canvas, scale, pad_x, pad_y

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.30))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.30))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.80, 1.25))
        return image

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.view(image.size[1], image.size[0], 3)
        tensor = data.permute(2, 0, 1).float().div(255.0)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD
