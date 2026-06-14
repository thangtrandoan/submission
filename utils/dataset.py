from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import DetectionTransform


class ObjectDetectionDataset(Dataset):
    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        img_size: int = 416,
        train: bool = True,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.train = train

        with self.annotation_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.class_names: list[str] = data["classes"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.images: list[dict[str, Any]] = data["images"]
        self.targets_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for ann in data["annotations"]:
            self.targets_by_image[ann["image_id"]].append(
                {
                    "class_id": self.class_to_idx[ann["class"]],
                    "bbox": [float(value) for value in ann["bbox"]],
                }
            )

        self.transform = DetectionTransform(img_size=img_size, train=train)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        info = self.images[index]
        image_id = info["id"]
        image_path = self.image_dir / image_id
        image = Image.open(image_path)
        targets = self.targets_by_image.get(image_id, [])
        image_tensor, targets = self.transform(image, targets)
        return image_tensor, targets


def collate_fn(batch: list[tuple[torch.Tensor, list[dict[str, Any]]]]) -> tuple[torch.Tensor, list[list[dict[str, Any]]]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
