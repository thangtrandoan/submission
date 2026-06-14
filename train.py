from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.detector import TinyGridDetector
from utils.dataset import ObjectDetectionDataset, collate_fn
from utils.loss import DetectionLoss, flatten_fcos_outputs, make_fcos_points
from utils.nms import nms


MULTI_SCALE_MIN = 640
MULTI_SCALE_MAX = 640
EARLY_STOPPING_PATIENCE = 6
MAP_CONF_THRESHOLD = 0.005
MAP_NMS_THRESHOLD = 0.55
MAP_MAX_DETECTIONS_PER_IMAGE = 300
MAP_PRE_NMS_TOPK = 1500
EVAL_MAP_EVERY = 1
MODEL_TYPE_BASE = "fcos_resnet50_bifpn"
MODEL_TYPE_P2 = "fcos_resnet50_bifpn_p2"
MODEL_TYPE_P6_SCALE = "fcos_resnet50_fpn_p6_scale_v4"
PREPROCESS_MODE = "letterbox"


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            key: value.detach().clone()
            for key, value in unwrap_model(model).state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        model_state = unwrap_model(model).state_dict()
        for key, value in model_state.items():
            value = value.detach()
            if torch.is_floating_point(value):
                self.shadow[key].mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                self.shadow[key].copy_(value)

    def store(self, model: torch.nn.Module) -> None:
        self.backup = {
            key: value.detach().clone()
            for key, value in unwrap_model(model).state_dict().items()
        }

    def copy_to(self, model: torch.nn.Module) -> None:
        unwrap_model(model).load_state_dict(self.shadow, strict=True)

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        unwrap_model(model).load_state_dict(self.backup, strict=True)
        self.backup = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], num_updates: int = 0) -> None:
        self.num_updates = num_updates
        for key, value in state_dict.items():
            if key in self.shadow:
                self.shadow[key].copy_(value.detach().to(self.shadow[key].device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shadow.items()}


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyGridDetector.")
    parser.add_argument("--train_data", required=True, type=Path)
    parser.add_argument("--val_data", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--val_image_dir", required=True, type=Path)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models/"))
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_noobj", type=float, default=2.0)
    parser.add_argument("--chair_loss_boost", type=float, default=1.0)
    parser.add_argument("--multi_scale_min", type=int, default=MULTI_SCALE_MIN)
    parser.add_argument("--multi_scale_max", type=int, default=MULTI_SCALE_MAX)
    parser.add_argument("--early_stopping_patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min_epochs", type=int, default=8)
    parser.add_argument("--map_conf_threshold", type=float, default=MAP_CONF_THRESHOLD)
    parser.add_argument("--map_nms_threshold", type=float, default=MAP_NMS_THRESHOLD)
    parser.add_argument(
        "--map_max_detections_per_image",
        type=int,
        default=MAP_MAX_DETECTIONS_PER_IMAGE,
    )
    parser.add_argument("--map_pre_nms_topk", type=int, default=MAP_PRE_NMS_TOPK)
    parser.add_argument("--eval_map_every", type=int, default=EVAL_MAP_EVERY)
    parser.add_argument("--scheduler", choices=("plateau", "onecycle"), default="onecycle")
    parser.add_argument("--skip_val_loss", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_channels_last", action="store_true")
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--enable_class_aware_sampler", action="store_true")
    parser.add_argument("--disable_class_aware_sampler", action="store_true")
    parser.add_argument("--enable_class_weights", action="store_true")
    parser.add_argument("--enable_p2", action="store_true")
    parser.add_argument("--disable_p2", action="store_true")
    parser.add_argument("--disable_p6", action="store_true")
    parser.add_argument("--disable_level_scales", action="store_true")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--enable_bifpn", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume_from_best", action="store_true")
    parser.add_argument("--resume_from_last", action="store_true")
    parser.add_argument("--resume_checkpoint", type=Path)
    return parser.parse_args()


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def bbox_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def class_weights_from_dataset(dataset: ObjectDetectionDataset, chair_loss_boost: float = 1.0) -> torch.Tensor:
    counts = torch.ones(len(dataset.class_names), dtype=torch.float32)
    for targets in dataset.targets_by_image.values():
        for item in targets:
            counts[item["class_id"]] += 1
    weights = counts.sum() / (counts * len(counts))
    if "chair" in dataset.class_to_idx:
        weights[dataset.class_to_idx["chair"]] *= chair_loss_boost
    return weights / weights.mean()


def image_sampling_weights(dataset: ObjectDetectionDataset) -> torch.Tensor:
    counts = torch.ones(len(dataset.class_names), dtype=torch.float32)
    for targets in dataset.targets_by_image.values():
        for item in targets:
            counts[item["class_id"]] += 1
    class_weights = torch.sqrt(counts.sum() / (counts * len(counts)))

    weights = []
    for image in dataset.images:
        targets = dataset.targets_by_image.get(image["id"], [])
        if not targets:
            weights.append(0.75)
            continue
        weights.append(float(max(class_weights[item["class_id"]] for item in targets)))
    return torch.tensor(weights, dtype=torch.double)


def run_epoch(
    model: TinyGridDetector,
    dataloader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    step_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ModelEMA | None = None,
    use_amp: bool = False,
    channels_last: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            raw = model(images)
            loss, metrics = criterion(raw, targets)
        if training:
            optimizer_stepped = True
            if scaler is not None and use_amp:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            if step_scheduler is not None and optimizer_stepped:
                step_scheduler.step()
            if ema is not None and optimizer_stepped:
                ema.update(model)

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    img_size: int,
    device: torch.device,
    conf_threshold: float,
    nms_threshold: float,
    max_detections_per_image: int,
    pre_nms_topk: int,
    iou_threshold: float = 0.5,
    use_amp: bool = False,
    channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    gt_by_class: dict[int, dict[int, list[dict[str, object]]]] = {
        class_id: {} for class_id in range(num_classes)
    }
    pred_by_class: dict[int, list[dict[str, object]]] = {class_id: [] for class_id in range(num_classes)}
    image_index = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            cls_logits, reg_preds, cnt_logits = flatten_fcos_outputs(outputs)
            points, strides, _ = make_fcos_points(outputs, device)
            probs = torch.sigmoid(cls_logits) * torch.sqrt(torch.sigmoid(cnt_logits).clamp(min=0.0)).unsqueeze(-1)

        for batch_idx, image_targets in enumerate(targets):
            current_image_index = image_index + batch_idx
            for item in image_targets:
                xmin, ymin, xmax, ymax = [float(value) / img_size for value in item["bbox"]]
                class_id = int(item["class_id"])
                gt_by_class[class_id].setdefault(current_image_index, []).append(
                    {"bbox": torch.tensor([xmin, ymin, xmax, ymax]), "matched": False}
                )

            scores, flat_indices = torch.topk(
                probs[batch_idx].reshape(-1),
                k=min(pre_nms_topk, probs.shape[1] * probs.shape[2]) if pre_nms_topk > 0 else probs.shape[1] * probs.shape[2],
            )
            keep = scores >= conf_threshold
            if not keep.any():
                continue
            scores = scores[keep]
            flat_indices = flat_indices[keep]
            point_indices = flat_indices // num_classes
            class_ids = flat_indices % num_classes
            distances = reg_preds[batch_idx, point_indices] * strides[point_indices, None] / img_size
            selected_points = points[point_indices] / img_size
            boxes = torch.stack(
                (
                    selected_points[:, 0] - distances[:, 0],
                    selected_points[:, 1] - distances[:, 1],
                    selected_points[:, 0] + distances[:, 2],
                    selected_points[:, 1] + distances[:, 3],
                ),
                dim=1,
            ).clamp(0.0, 1.0)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not valid.any():
                continue
            boxes = boxes[valid]
            scores = scores[valid]
            class_ids = class_ids[valid]

            selected_indices: list[torch.Tensor] = []
            for class_id in class_ids.unique():
                class_indices = torch.where(class_ids == class_id)[0]
                selected_indices.append(class_indices[nms(boxes[class_indices], scores[class_indices], nms_threshold)])
            if not selected_indices:
                continue
            selected = torch.cat(selected_indices)
            selected = selected[scores[selected].argsort(descending=True)][:max_detections_per_image]
            for index in selected.tolist():
                pred_by_class[int(class_ids[index].item())].append(
                    {
                        "image_id": current_image_index,
                        "class_id": int(class_ids[index].item()),
                        "confidence": float(scores[index].item()),
                        "bbox": boxes[index].detach().cpu(),
                    }
                )

        image_index += images.shape[0]

    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    for class_id in range(num_classes):
        class_gt = gt_by_class[class_id]
        num_gt = sum(len(items) for items in class_gt.values())
        class_preds = sorted(
            pred_by_class[class_id], key=lambda item: float(item["confidence"]), reverse=True
        )
        tp_flags = []
        fp_flags = []

        for pred in class_preds:
            candidates = class_gt.get(int(pred["image_id"]), [])
            best_iou = 0.0
            best_index = -1
            for index, gt in enumerate(candidates):
                if bool(gt["matched"]):
                    continue
                iou = bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index >= 0 and best_iou >= iou_threshold:
                candidates[best_index]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        cumulative_tp = []
        cumulative_fp = []
        tp_sum = 0
        fp_sum = 0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            cumulative_tp.append(tp_sum)
            cumulative_fp.append(fp_sum)

        recalls = [value / num_gt if num_gt else 0.0 for value in cumulative_tp]
        precisions = [tp / max(tp + fp, 1) for tp, fp in zip(cumulative_tp, cumulative_fp)]
        if num_gt:
            aps.append(compute_ap(recalls, precisions))

        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += num_gt

    map_50 = sum(aps) / len(aps) if aps else 0.0
    return {
        "map_50": map_50,
        "micro_precision": total_tp / max(total_tp + total_fp, 1),
        "micro_recall": total_tp / total_gt if total_gt else 0.0,
        "num_predictions": float(sum(len(items) for items in pred_by_class.values())),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    class_names: list[str],
    img_size: int,
    grid_size: int,
    epoch: int,
    best_val_loss: float,
    best_metric: float,
    model_type: str,
    use_p2: bool,
    use_p6: bool,
    use_scales: bool,
    channels: int,
    use_bifpn: bool,
    ema: ModelEMA | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "ema_updates": ema.num_updates if ema is not None else 0,
            "ema_decay": ema.decay if ema is not None else None,
            "class_names": class_names,
            "img_size": img_size,
            "grid_size": grid_size,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_metric": best_metric,
            "model_type": model_type,
            "preprocess": PREPROCESS_MODE,
            "use_p2": use_p2,
            "use_p6": use_p6,
            "use_scales": use_scales,
            "channels": channels,
            "use_bifpn": use_bifpn,
        },
        path,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_p2 = args.enable_p2 and not args.disable_p2
    use_p6 = not args.disable_p6 and not use_p2
    use_scales = not args.disable_level_scales
    use_bifpn = args.enable_bifpn
    if use_p2:
        model_type = MODEL_TYPE_P2
    elif use_p6 or use_scales or not use_bifpn or args.channels != 256:
        model_type = MODEL_TYPE_P6_SCALE
    else:
        model_type = MODEL_TYPE_BASE
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = device.type == "cuda" and not args.no_amp
    channels_last = device.type == "cuda" and not args.no_channels_last
    best_path = args.checkpoint_dir / "best.pth"
    last_path = args.checkpoint_dir / "last.pth"
    resume_path = args.resume_checkpoint
    if args.resume_from_best:
        resume_path = best_path
    if args.resume_from_last:
        resume_path = last_path
    resume_option_count = int(args.resume_checkpoint is not None) + int(args.resume_from_best) + int(args.resume_from_last)
    if resume_option_count > 1:
        raise ValueError("Use only one of --resume_from_best, --resume_from_last, or --resume_checkpoint.")
    checkpoint = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_model_type = checkpoint.get("model_type")
        if checkpoint_model_type != model_type:
            raise ValueError(
                "Resume checkpoint is not compatible with the current model. "
                f"checkpoint_model_type={checkpoint_model_type!r}, current_model_type={model_type!r}. "
                "Train from scratch or resume from a checkpoint created by the current FCOS model."
            )
        checkpoint_preprocess = checkpoint.get("preprocess", "stretch")
        if checkpoint_preprocess != PREPROCESS_MODE:
            print(
                "Warning: resume checkpoint uses different preprocessing "
                f"checkpoint_preprocess={checkpoint_preprocess!r} "
                f"current_preprocess={PREPROCESS_MODE!r}. "
                "Weights are compatible, but a fresh run is usually cleaner.",
                flush=True,
            )

    train_dataset = ObjectDetectionDataset(args.train_data, args.image_dir, img_size=args.img_size, train=True)
    val_dataset = ObjectDetectionDataset(args.val_data, args.val_image_dir, img_size=args.img_size, train=False)
    loader_options = {
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 2
    sampler = None
    use_class_aware_sampler = args.enable_class_aware_sampler and not args.disable_class_aware_sampler
    if use_class_aware_sampler:
        sampler = WeightedRandomSampler(
            weights=image_sampling_weights(train_dataset),
            num_samples=len(train_dataset),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_options,
    )
    val_batch_size = args.val_batch_size or max(1, min(args.batch_size, 16))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        **loader_options,
    )

    model = TinyGridDetector(
        num_classes=len(train_dataset.class_names),
        pretrained_backbone=True,
        use_p2=use_p2,
        use_p6=use_p6,
        use_scales=use_scales,
        channels=args.channels,
        use_bifpn=use_bifpn,
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    ema = None if args.disable_ema else ModelEMA(model, decay=args.ema_decay)
    if ema is not None and checkpoint is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(
            checkpoint["ema_state_dict"],
            num_updates=int(checkpoint.get("ema_updates", 0)),
        )
    if gpu_count > 1 and not args.single_gpu:
        model = torch.nn.DataParallel(model)
    if args.enable_class_weights or args.chair_loss_boost != 1.0:
        class_weights = class_weights_from_dataset(train_dataset, args.chair_loss_boost).to(device)
    else:
        class_weights = torch.ones(len(train_dataset.class_names), dtype=torch.float32, device=device)
    criterion = DetectionLoss(
        img_size=args.img_size,
        grid_size=args.img_size // 32,
        num_classes=len(train_dataset.class_names),
        class_weights=class_weights,
        lambda_noobj=args.lambda_noobj,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.15,
            div_factor=10,
            final_div_factor=100,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.01,
            threshold_mode="rel",
            min_lr=1e-6,
        )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_val_loss = float("inf")
    best_map = float("-inf")
    epochs_without_improvement = 0
    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_map = float(checkpoint.get("best_metric", float("-inf")))
        print(
            "Resuming training "
            f"checkpoint={resume_path} "
            f"start_epoch={start_epoch} "
            f"best_val_loss={best_val_loss:.4f} "
            f"best_mAP@0.5={best_map:.4f}",
            flush=True,
        )
    multi_scale_sizes = list(range(args.multi_scale_min, args.multi_scale_max + 1, 32))
    if not multi_scale_sizes:
        raise ValueError("Multi-scale range must include at least one size.")
    if args.eval_map_every <= 0:
        raise ValueError("--eval_map_every must be >= 1.")
    print(
        "Starting training "
        f"device={device} "
        f"epochs={args.epochs} "
        f"batch_size={args.batch_size} "
        f"val_batch_size={val_batch_size} "
        f"gpus={gpu_count} "
        f"single_gpu={args.single_gpu} "
        f"architecture={model_type} "
        f"preprocess={PREPROCESS_MODE} "
        f"use_p2={use_p2} "
        f"use_p6={use_p6} "
        f"use_scales={use_scales} "
        f"channels={args.channels} "
        f"use_bifpn={use_bifpn} "
        f"pretrained_backbone=True "
        f"multi_scale=True "
        f"eval_map=True "
        f"eval_map_every={args.eval_map_every} "
        f"scheduler={args.scheduler} "
        f"skip_val_loss={args.skip_val_loss or args.scheduler == 'onecycle'} "
        f"amp={use_amp} "
        f"channels_last={channels_last} "
        f"ema={ema is not None} "
        f"ema_decay={(ema.decay if ema is not None else 0.0):.5f} "
        f"class_aware_sampler={sampler is not None} "
        f"class_weights={args.enable_class_weights or args.chair_loss_boost != 1.0} "
        f"chair_loss_boost={args.chair_loss_boost} "
        f"map_pre_nms_topk={args.map_pre_nms_topk} "
        f"early_stopping_patience={args.early_stopping_patience} "
        f"min_epochs={args.min_epochs} "
        f"seed={args.seed} "
        f"train_images={len(train_dataset)} "
        f"val_images={len(val_dataset)} "
        f"best_checkpoint={best_path} "
        f"last_checkpoint={last_path}",
        flush=True,
    )
    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.perf_counter()
        current_img_size = random.choice(multi_scale_sizes)
        train_dataset.transform.img_size = current_img_size
        criterion.img_size = current_img_size
        criterion.grid_size = current_img_size // 32
        print(f"epoch={epoch:03d} multi_scale_img_size={current_img_size}", flush=True)

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler=scaler,
            step_scheduler=scheduler if args.scheduler == "onecycle" else None,
            ema=ema,
            use_amp=use_amp,
            channels_last=channels_last,
        )
        should_eval_map = epoch == end_epoch or epoch % args.eval_map_every == 0
        map_metrics: dict[str, float] | None = None
        val_metrics: dict[str, float] | None = None
        val_loss = float("inf")
        if should_eval_map:
            val_dataset.transform.img_size = args.img_size
            criterion.img_size = args.img_size
            criterion.grid_size = args.img_size // 32
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
            try:
                should_run_val_loss = not args.skip_val_loss and args.scheduler == "plateau"
                if should_run_val_loss:
                    with torch.no_grad():
                        val_metrics = run_epoch(
                            model,
                            val_loader,
                            criterion,
                            device,
                            use_amp=use_amp,
                            channels_last=channels_last,
                        )
                    val_loss = val_metrics["loss"]
                    scheduler.step(val_loss)
                map_metrics = evaluate_map(
                    model=model,
                    dataloader=val_loader,
                    num_classes=len(train_dataset.class_names),
                    img_size=args.img_size,
                    device=device,
                    conf_threshold=args.map_conf_threshold,
                    nms_threshold=args.map_nms_threshold,
                    max_detections_per_image=args.map_max_detections_per_image,
                    pre_nms_topk=args.map_pre_nms_topk,
                    use_amp=use_amp,
                    channels_last=channels_last,
                )

                improved = map_metrics["map_50"] > best_map
                if improved:
                    best_val_loss = val_loss
                    best_map = map_metrics["map_50"]
                    epochs_without_improvement = 0
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        scheduler,
                        train_dataset.class_names,
                        args.img_size,
                        args.img_size // 32,
                        epoch,
                        best_val_loss,
                        best_metric=best_map,
                        model_type=model_type,
                        use_p2=use_p2,
                        use_p6=use_p6,
                        use_scales=use_scales,
                        channels=args.channels,
                        use_bifpn=use_bifpn,
                        ema=ema,
                    )
                else:
                    epochs_without_improvement += 1
            finally:
                if ema is not None:
                    ema.restore(model)
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            train_dataset.class_names,
            args.img_size,
            args.img_size // 32,
            epoch,
            best_val_loss,
            best_metric=best_map,
            model_type=model_type,
            use_p2=use_p2,
            use_p6=use_p6,
            use_scales=use_scales,
            channels=args.channels,
            use_bifpn=use_bifpn,
            ema=ema,
        )

        message = (
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"best_val_loss={best_val_loss:.4f} "
        )
        if map_metrics is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            message += (
                f"val_loss={(val_metrics['loss'] if val_metrics is not None else float('nan')):.4f} "
                f"val_obj_conf={(val_metrics['obj_conf'] if val_metrics is not None else float('nan')):.4f} "
                f"val_noobj_conf={(val_metrics['noobj_conf'] if val_metrics is not None else float('nan')):.4f} "
                f"val_mAP@0.5={map_metrics['map_50']:.4f} "
                f"best_mAP@0.5={best_map:.4f} "
                f"val_precision={map_metrics['micro_precision']:.4f} "
                f"val_recall={map_metrics['micro_recall']:.4f} "
                f"val_predictions={int(map_metrics['num_predictions'])} "
                f"patience={epochs_without_improvement}/{args.early_stopping_patience} evals "
                f"lr={current_lr:.2e}"
            )
        else:
            message += f" best_mAP@0.5={best_map:.4f} map_eval=skipped lr={current_lr:.2e}"
        message += f" epoch_time={format_duration(time.perf_counter() - epoch_start_time)}"
        print(message)

        if (
            map_metrics is not None
            and epoch >= args.min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch={epoch:03d} "
                f"best_mAP@0.5={best_map:.4f} "
                f"best_checkpoint={best_path}",
                flush=True,
            )
            break

    print(f"Best checkpoint saved to {best_path}")
    print(f"Last checkpoint saved to {last_path}")


if __name__ == "__main__":
    main()
