from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


LEVEL_SPECS = {
    "p2": {"stride": 4, "min_size": 0.0, "max_size": 64.0},
    "p3": {"stride": 8, "min_size": 32.0, "max_size": 128.0},
    "p4": {"stride": 16, "min_size": 64.0, "max_size": 192.0},
    "p5": {"stride": 32, "min_size": 128.0, "max_size": 384.0},
    "p6": {"stride": 64, "min_size": 256.0, "max_size": float("inf")},
}


def level_spec_for(level: str, active_levels: set[str]) -> dict[str, float]:
    spec = dict(LEVEL_SPECS[level])
    if "p2" not in active_levels and level == "p3":
        spec["min_size"] = 0.0
        spec["max_size"] = 96.0
    if "p6" not in active_levels and level == "p5":
        spec["max_size"] = float("inf")
    return spec


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if class_weights is not None:
        positive_weights = class_weights.view(1, -1, 1, 1).to(logits.device)
        loss = loss * (1.0 + targets * (positive_weights - 1.0))
    return loss


def centerness_from_ltrb(ltrb: torch.Tensor) -> torch.Tensor:
    left, top, right, bottom = ltrb.unbind(dim=-1)
    lr = torch.minimum(left, right) / torch.maximum(left, right).clamp(min=1e-6)
    tb = torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp(min=1e-6)
    return torch.sqrt((lr * tb).clamp(min=0.0, max=1.0))


def ciou_loss_from_ltrb(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_left, pred_top, pred_right, pred_bottom = pred.unbind(dim=-1)
    tgt_left, tgt_top, tgt_right, tgt_bottom = target.unbind(dim=-1)

    inter_w = torch.minimum(pred_left, tgt_left) + torch.minimum(pred_right, tgt_right)
    inter_h = torch.minimum(pred_top, tgt_top) + torch.minimum(pred_bottom, tgt_bottom)
    inter = inter_w.clamp(min=0.0) * inter_h.clamp(min=0.0)

    pred_area = (pred_left + pred_right).clamp(min=0.0) * (pred_top + pred_bottom).clamp(min=0.0)
    target_area = (tgt_left + tgt_right).clamp(min=0.0) * (tgt_top + tgt_bottom).clamp(min=0.0)
    union = pred_area + target_area - inter
    iou = inter / union.clamp(min=1e-6)

    pred_x1 = -pred_left
    pred_y1 = -pred_top
    pred_x2 = pred_right
    pred_y2 = pred_bottom
    tgt_x1 = -tgt_left
    tgt_y1 = -tgt_top
    tgt_x2 = tgt_right
    tgt_y2 = tgt_bottom

    pred_cx = (pred_x1 + pred_x2) * 0.5
    pred_cy = (pred_y1 + pred_y2) * 0.5
    tgt_cx = (tgt_x1 + tgt_x2) * 0.5
    tgt_cy = (tgt_y1 + tgt_y2) * 0.5
    center_distance = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)

    enc_x1 = torch.minimum(pred_x1, tgt_x1)
    enc_y1 = torch.minimum(pred_y1, tgt_y1)
    enc_x2 = torch.maximum(pred_x2, tgt_x2)
    enc_y2 = torch.maximum(pred_y2, tgt_y2)
    enclosing_diagonal = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2)

    pred_w = (pred_left + pred_right).clamp(min=1e-6)
    pred_h = (pred_top + pred_bottom).clamp(min=1e-6)
    tgt_w = (tgt_left + tgt_right).clamp(min=1e-6)
    tgt_h = (tgt_top + tgt_bottom).clamp(min=1e-6)
    v = (4.0 / torch.pi**2) * (torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v).clamp(min=1e-6)

    ciou = iou - center_distance / enclosing_diagonal.clamp(min=1e-6) - alpha * v
    return 1.0 - ciou.clamp(min=-1.0, max=1.0)


def giou_loss_from_ltrb(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_left, pred_top, pred_right, pred_bottom = pred.unbind(dim=-1)
    tgt_left, tgt_top, tgt_right, tgt_bottom = target.unbind(dim=-1)

    inter_w = torch.minimum(pred_left, tgt_left) + torch.minimum(pred_right, tgt_right)
    inter_h = torch.minimum(pred_top, tgt_top) + torch.minimum(pred_bottom, tgt_bottom)
    inter = inter_w.clamp(min=0.0) * inter_h.clamp(min=0.0)

    pred_area = (pred_left + pred_right).clamp(min=0.0) * (pred_top + pred_bottom).clamp(min=0.0)
    target_area = (tgt_left + tgt_right).clamp(min=0.0) * (tgt_top + tgt_bottom).clamp(min=0.0)
    union = pred_area + target_area - inter
    iou = inter / union.clamp(min=1e-6)

    pred_x1 = -pred_left
    pred_y1 = -pred_top
    pred_x2 = pred_right
    pred_y2 = pred_bottom
    tgt_x1 = -tgt_left
    tgt_y1 = -tgt_top
    tgt_x2 = tgt_right
    tgt_y2 = tgt_bottom

    enc_x1 = torch.minimum(pred_x1, tgt_x1)
    enc_y1 = torch.minimum(pred_y1, tgt_y1)
    enc_x2 = torch.maximum(pred_x2, tgt_x2)
    enc_y2 = torch.maximum(pred_y2, tgt_y2)
    enc_area = (enc_x2 - enc_x1).clamp(min=0.0) * (enc_y2 - enc_y1).clamp(min=0.0)
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return 1.0 - giou.clamp(min=-1.0, max=1.0)


def encode_fcos_targets(
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    targets: list[list[dict[str, Any]]],
    img_size: int,
    num_classes: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    encoded: dict[str, dict[str, torch.Tensor]] = {}
    active_levels = set(outputs)
    for level, (cls_logits, _, _) in outputs.items():
        batch_size, _, height, width = cls_logits.shape
        level_spec = level_spec_for(level, active_levels)
        stride = level_spec["stride"]
        cls_target = torch.zeros((batch_size, num_classes, height, width), device=device)
        reg_target = torch.zeros((batch_size, 4, height, width), device=device)
        cnt_target = torch.zeros((batch_size, 1, height, width), device=device)
        pos_mask = torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device=device)
        area_target = torch.full((batch_size, 1, height, width), float("inf"), device=device)

        shifts_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        shifts_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        for batch_idx, image_targets in enumerate(targets):
            for item in image_targets:
                xmin, ymin, xmax, ymax = [float(value) for value in item["bbox"]]
                box_w = xmax - xmin
                box_h = ymax - ymin
                if box_w <= 0 or box_h <= 0:
                    continue
                max_side = max(box_w, box_h)
                if max_side < level_spec["min_size"] or max_side > level_spec["max_size"]:
                    continue

                left = xx - xmin
                top = yy - ymin
                right = xmax - xx
                bottom = ymax - yy
                inside_box = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)

                cx = (xmin + xmax) * 0.5
                cy = (ymin + ymax) * 0.5
                radius = 1.5 * stride
                inside_center = (
                    (xx >= max(xmin, cx - radius))
                    & (xx <= min(xmax, cx + radius))
                    & (yy >= max(ymin, cy - radius))
                    & (yy <= min(ymax, cy + radius))
                )
                candidate = inside_box & inside_center
                if not candidate.any():
                    grid_x = min(width - 1, max(0, int(cx / stride)))
                    grid_y = min(height - 1, max(0, int(cy / stride)))
                    candidate = torch.zeros((height, width), dtype=torch.bool, device=device)
                    candidate[grid_y, grid_x] = True

                area = box_w * box_h
                update = candidate & (area < area_target[batch_idx, 0])
                if not update.any():
                    continue

                class_id = int(item["class_id"])
                cls_target[batch_idx, :, update] = 0.0
                cls_target[batch_idx, class_id, update] = 1.0
                ltrb = torch.stack((left, top, right, bottom), dim=0) / stride
                reg_target[batch_idx, :, update] = ltrb[:, update]
                cnt_target[batch_idx, 0, update] = centerness_from_ltrb(ltrb.permute(1, 2, 0)[update])
                pos_mask[batch_idx, 0, update] = True
                area_target[batch_idx, 0, update] = area

        encoded[level] = {
            "cls": cls_target,
            "reg": reg_target,
            "cnt": cnt_target,
            "pos_mask": pos_mask,
        }
    return encoded


def flatten_fcos_outputs(
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cls_logits = torch.cat(
        [item[0].permute(0, 2, 3, 1).reshape(item[0].shape[0], -1, item[0].shape[1]) for item in outputs.values()],
        dim=1,
    )
    reg_preds = torch.cat(
        [item[1].permute(0, 2, 3, 1).reshape(item[1].shape[0], -1, 4) for item in outputs.values()],
        dim=1,
    )
    cnt_logits = torch.cat(
        [item[2].permute(0, 2, 3, 1).reshape(item[2].shape[0], -1) for item in outputs.values()],
        dim=1,
    )
    return cls_logits, reg_preds, cnt_logits


def make_fcos_points(
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_levels = set(outputs)
    points_by_level = []
    strides_by_level = []
    ranges_by_level = []
    for level, (cls_logits, _, _) in outputs.items():
        _, _, height, width = cls_logits.shape
        spec = level_spec_for(level, active_levels)
        stride = float(spec["stride"])
        shifts_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        shifts_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        points_by_level.append(points)
        strides_by_level.append(torch.full((points.shape[0],), stride, device=device))
        ranges_by_level.append(
            torch.tensor([spec["min_size"], spec["max_size"]], device=device, dtype=torch.float32)
            .view(1, 2)
            .expand(points.shape[0], 2)
        )
    return torch.cat(points_by_level), torch.cat(strides_by_level), torch.cat(ranges_by_level)


def encode_fcos_targets_flat(
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    targets: list[list[dict[str, Any]]],
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points, strides, reg_ranges = make_fcos_points(outputs, device)
    batch_size = len(targets)
    num_points = points.shape[0]
    labels = torch.full((batch_size, num_points), -1, dtype=torch.long, device=device)
    reg_targets = torch.zeros((batch_size, num_points, 4), dtype=torch.float32, device=device)
    cnt_targets = torch.zeros((batch_size, num_points), dtype=torch.float32, device=device)

    xs = points[:, 0]
    ys = points[:, 1]
    for batch_idx, image_targets in enumerate(targets):
        boxes = []
        gt_labels = []
        for item in image_targets:
            xmin, ymin, xmax, ymax = [float(value) for value in item["bbox"]]
            if xmax <= xmin or ymax <= ymin:
                continue
            class_id = int(item["class_id"])
            if 0 <= class_id < num_classes:
                boxes.append([xmin, ymin, xmax, ymax])
                gt_labels.append(class_id)
        if not boxes:
            continue

        box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        label_tensor = torch.tensor(gt_labels, dtype=torch.long, device=device)
        left = xs[:, None] - box_tensor[None, :, 0]
        top = ys[:, None] - box_tensor[None, :, 1]
        right = box_tensor[None, :, 2] - xs[:, None]
        bottom = box_tensor[None, :, 3] - ys[:, None]
        reg = torch.stack((left, top, right, bottom), dim=2)
        inside_box = reg.min(dim=2).values > 0

        centers = (box_tensor[:, :2] + box_tensor[:, 2:]) * 0.5
        radius = strides[:, None] * 1.5
        center_x1 = torch.maximum(box_tensor[None, :, 0], centers[None, :, 0] - radius)
        center_y1 = torch.maximum(box_tensor[None, :, 1], centers[None, :, 1] - radius)
        center_x2 = torch.minimum(box_tensor[None, :, 2], centers[None, :, 0] + radius)
        center_y2 = torch.minimum(box_tensor[None, :, 3], centers[None, :, 1] + radius)
        inside_center = (
            (xs[:, None] >= center_x1)
            & (xs[:, None] <= center_x2)
            & (ys[:, None] >= center_y1)
            & (ys[:, None] <= center_y2)
        )

        max_reg = reg.max(dim=2).values
        in_range = (max_reg >= reg_ranges[:, None, 0]) & (max_reg <= reg_ranges[:, None, 1])
        areas = (
            (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
        )[None, :].expand(num_points, len(boxes)).clone()
        areas[~(inside_box & inside_center & in_range)] = float("inf")
        min_area, min_indices = areas.min(dim=1)
        pos = torch.isfinite(min_area)
        if not pos.any():
            continue

        matched = min_indices[pos]
        labels[batch_idx, pos] = label_tensor[matched]
        reg_targets[batch_idx, pos] = reg[pos, matched] / strides[pos, None]
        cnt_targets[batch_idx, pos] = centerness_from_ltrb(reg_targets[batch_idx, pos])

    return labels, reg_targets, cnt_targets


class DetectionLoss(nn.Module):
    def __init__(
        self,
        img_size: int = 416,
        grid_size: int = 13,
        num_classes: int = 5,
        lambda_box: float = 2.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 1.0,
        lambda_cls: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        if class_weights is None:
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights.float())

    def forward(
        self,
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        targets: list[list[dict[str, Any]]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(iter(outputs.values()))[0].device
        cls_logits, reg_preds, cnt_logits = flatten_fcos_outputs(outputs)
        labels, reg_targets, cnt_targets = encode_fcos_targets_flat(outputs, targets, self.num_classes, device)

        pos_mask = labels >= 0
        num_pos = pos_mask.sum().clamp(min=1).float()
        cls_target = torch.zeros_like(cls_logits)
        pos_b, pos_i = torch.where(pos_mask)
        if pos_b.numel():
            cls_target[pos_b, pos_i, labels[pos_mask]] = 1.0

        bce = F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction="none")
        probs = torch.sigmoid(cls_logits)
        p_t = probs * cls_target + (1.0 - probs) * (1.0 - cls_target)
        alpha_t = 0.25 * cls_target + 0.75 * (1.0 - cls_target)
        cls_loss = alpha_t * (1.0 - p_t).pow(2.0) * bce
        if self.class_weights is not None:
            class_weights = self.class_weights.view(1, 1, -1).to(device)
            cls_loss = cls_loss * (1.0 + cls_target * (class_weights - 1.0))
        total_cls_loss = cls_loss.sum() / num_pos

        if pos_b.numel():
            pred_ltrb = reg_preds[pos_b, pos_i]
            target_ltrb = reg_targets[pos_b, pos_i]
            cnt_weights = cnt_targets[pos_b, pos_i].detach()
            total_box_loss = (
                giou_loss_from_ltrb(pred_ltrb, target_ltrb) * cnt_weights
            ).sum() / cnt_weights.sum().clamp(min=1e-6)
            total_cnt_loss = F.binary_cross_entropy_with_logits(
                cnt_logits[pos_b, pos_i], cnt_targets[pos_b, pos_i], reduction="sum"
            ) / num_pos
            pos_conf = torch.sigmoid(cnt_logits[pos_b, pos_i]).detach().mean()
        else:
            total_box_loss = reg_preds.sum() * 0.0
            total_cnt_loss = cnt_logits.sum() * 0.0
            pos_conf = total_box_loss.detach()

        neg_conf = probs[~pos_mask].detach().mean() if (~pos_mask).any() else probs.detach().mean()
        loss = self.lambda_cls * total_cls_loss + self.lambda_box * total_box_loss + self.lambda_obj * total_cnt_loss
        metrics = {
            "loss": float(loss.detach().cpu()),
            "box_loss": float(total_box_loss.detach().cpu()),
            "obj_loss": float(total_cnt_loss.detach().cpu()),
            "noobj_loss": float(total_cls_loss.detach().cpu()),
            "cls_loss": float(total_cls_loss.detach().cpu()),
            "obj_conf": float(pos_conf.detach().cpu()),
            "noobj_conf": float(neg_conf.detach().cpu()),
            "num_pos": float(pos_mask.sum().detach().cpu()),
            "pos_ratio": float(pos_mask.float().mean().detach().cpu()),
        }
        return loss, metrics


def decode_raw_predictions(*args: Any, **kwargs: Any) -> torch.Tensor:
    raise RuntimeError("FCOS model outputs are decoded directly in predict.py/train.py.")
