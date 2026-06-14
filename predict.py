from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from PIL import Image, ImageEnhance

from models.detector import TinyGridDetector
from utils.box_ops import box_iou
from utils.json_utils import write_json
from utils.loss import flatten_fcos_outputs, make_fcos_points
from utils.nms import nms


TTA_BRIGHTNESS_FACTORS: list[float] = [0.9, 1.1]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
LETTERBOX_FILL = (114, 114, 114)
TTA_IMAGE_SIZES: list[int] = [640, 512, 704]
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "models" / "best.pth"
DEFAULT_CHECKPOINT_URL = "https://drive.google.com/file/d/1ub2AicWIkYZNLvyCZKrQOVTwOtgrb6J2/view?usp=sharing"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyGridDetector inference.")
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint_url", default=DEFAULT_CHECKPOINT_URL)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--conf_threshold", type=float, default=0.01)
    parser.add_argument("--nms_threshold", type=float, default=0.6)
    parser.add_argument("--max_detections_per_image", type=int, default=300)
    parser.add_argument("--pre_nms_topk", type=int, default=1500)
    parser.add_argument("--preprocess", choices=("auto", "letterbox", "stretch"), default="auto")
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--tta_img_sizes", nargs="*", type=int, default=TTA_IMAGE_SIZES)
    parser.add_argument("--tta_brightness", nargs="*", type=float, default=TTA_BRIGHTNESS_FACTORS)
    parser.add_argument("--merge_method", choices=("wbf", "nms"), default="wbf")
    parser.add_argument("--wbf_iou_threshold", type=float, default=0.55)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--no_channels_last", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def google_drive_download_url(url: str) -> str:
    match = re.search(r"/file/d/([^/]+)", url)
    if match:
        file_id = match.group(1)
    else:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        file_id = query.get("id", [""])[0]
    if not file_id:
        return url
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def confirm_google_drive_download(response: object) -> tuple[str | None, bool]:
    for value in response.headers.get_all("Set-Cookie") or []:
        match = re.search(r"download_warning[^=]*=([^;]+)", value)
        if match:
            return match.group(1), False

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None, False
    preview = response.read(200_000).decode("utf-8", errors="ignore")
    patterns = [
        r"confirm=([0-9A-Za-z_-]+)",
        r'name="confirm"\s+value="([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, preview)
        if match:
            return match.group(1), False
    return None, True


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".download")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    request_url = google_drive_download_url(url)
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = opener.open(urllib.request.Request(request_url, headers=headers))
        confirm, is_unhandled_html = confirm_google_drive_download(response)
        if is_unhandled_html:
            response.close()
            raise RuntimeError("Google Drive returned an HTML page instead of the checkpoint file.")
        if confirm:
            response.close()
            parsed = urllib.parse.urlparse(request_url)
            query = urllib.parse.parse_qs(parsed.query)
            query["confirm"] = [confirm]
            request_url = urllib.parse.urlunparse(
                parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
            )
            response = opener.open(urllib.request.Request(request_url, headers=headers))

        with response, temp_path.open("wb") as file:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
        temp_path.replace(destination)
    except (OSError, urllib.error.URLError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download checkpoint from {url}: {exc}") from exc


def ensure_checkpoint(checkpoint_path: Path, checkpoint_url: str) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path
    if not checkpoint_url:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Checkpoint not found at {checkpoint_path}. Downloading from {checkpoint_url}", flush=True)
    download_file(checkpoint_url, checkpoint_path)
    print(f"Downloaded checkpoint to {checkpoint_path}", flush=True)
    return checkpoint_path


def letterbox_image(image: Image.Image, img_size: int) -> tuple[Image.Image, float, int, int]:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(img_size / width, img_size / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = image.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), LETTERBOX_FILL)
    pad_x = (img_size - resized_w) // 2
    pad_y = (img_size - resized_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def image_to_tensor(image: Image.Image, img_size: int, preprocess: str) -> tuple[torch.Tensor, float, int, int]:
    if preprocess == "letterbox":
        image, scale, pad_x, pad_y = letterbox_image(image, img_size)
    else:
        original_w, original_h = image.size
        image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        scale = img_size / max(original_w, 1)
        pad_x = 0
        pad_y = 0
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(img_size, img_size, 3)
    tensor = data.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD, scale, pad_x, pad_y


def rounded_valid_bbox(box: list[float], max_width: int | None = None, max_height: int | None = None) -> list[float] | None:
    xmin, ymin, xmax, ymax = box
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = max(0.0, xmax)
    ymax = max(0.0, ymax)
    if max_width is not None:
        xmin = min(float(max_width), xmin)
        xmax = min(float(max_width), xmax)
    if max_height is not None:
        ymin = min(float(max_height), ymin)
        ymax = min(float(max_height), ymax)

    rounded = [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)]
    if rounded[2] <= rounded[0] or rounded[3] <= rounded[1]:
        return None
    return rounded


@torch.no_grad()
def predict_image(
    model: TinyGridDetector,
    image: Image.Image,
    image_id: str,
    class_names: list[str],
    img_size: int,
    conf_threshold: float,
    nms_threshold: float,
    pre_nms_topk: int,
    device: torch.device,
    preprocess: str,
    channels_last: bool,
) -> dict[str, object]:
    original_w, original_h = image.size
    tensor, scale, pad_x, pad_y = image_to_tensor(image, img_size, preprocess)
    tensor = tensor.to(device)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    outputs = model(tensor)
    cls_logits, reg_preds, cnt_logits = flatten_fcos_outputs(outputs)
    points, strides, _ = make_fcos_points(outputs, device)
    probs = torch.sigmoid(cls_logits[0]) * torch.sqrt(torch.sigmoid(cnt_logits[0]).clamp(min=0.0)).unsqueeze(-1)
    num_classes = len(class_names)
    scores, flat_indices = torch.topk(
        probs.reshape(-1),
        k=min(pre_nms_topk, probs.shape[0] * probs.shape[1]) if pre_nms_topk > 0 else probs.shape[0] * probs.shape[1],
    )
    keep = scores >= conf_threshold
    output_boxes: list[dict[str, object]] = []
    if not keep.any():
        return {"image_id": image_id, "boxes": output_boxes}
    scores = scores[keep]
    flat_indices = flat_indices[keep]
    point_indices = flat_indices // num_classes
    class_ids = flat_indices % num_classes
    distances = reg_preds[0, point_indices] * strides[point_indices, None]
    selected_points = points[point_indices]
    boxes = torch.stack(
        (
            selected_points[:, 0] - distances[:, 0],
            selected_points[:, 1] - distances[:, 1],
            selected_points[:, 0] + distances[:, 2],
            selected_points[:, 1] + distances[:, 3],
        ),
        dim=1,
    )
    if preprocess == "letterbox":
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    else:
        resize_scale = torch.tensor(
            [original_w / img_size, original_h / img_size, original_w / img_size, original_h / img_size],
            device=device,
            dtype=torch.float32,
        )
        boxes = boxes * resize_scale
    boxes = boxes.clamp(min=0)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=original_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=original_h)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not valid.any():
        return {"image_id": image_id, "boxes": output_boxes}
    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    for class_id in class_ids.unique():
        class_mask = class_ids == class_id
        selected = nms(boxes[class_mask], scores[class_mask], nms_threshold)
        class_boxes = boxes[class_mask][selected]
        class_scores_selected = scores[class_mask][selected]
        for box, score in zip(class_boxes, class_scores_selected):
            xmin, ymin, xmax, ymax = box.tolist()
            bbox = rounded_valid_bbox([xmin, ymin, xmax, ymax], original_w, original_h)
            if bbox is None:
                continue
            output_boxes.append(
                {
                    "class": class_names[int(class_id.item())],
                    "confidence": round(float(score.item()), 6),
                    "bbox": bbox,
                }
            )

    output_boxes.sort(key=lambda item: item["confidence"], reverse=True)
    return {"image_id": image_id, "boxes": output_boxes}


def merge_boxes(
    image_id: str,
    boxes: list[dict[str, object]],
    class_names: list[str],
    nms_threshold: float,
    max_detections_per_image: int,
    device: torch.device,
) -> dict[str, object]:
    if not boxes:
        return {"image_id": image_id, "boxes": []}

    merged: list[dict[str, object]] = []
    for class_name in class_names:
        class_boxes = [
            box
            for box in boxes
            if box["class"] == class_name and rounded_valid_bbox([float(value) for value in box["bbox"]]) is not None
        ]
        if not class_boxes:
            continue
        box_tensor = torch.tensor([box["bbox"] for box in class_boxes], dtype=torch.float32, device=device)
        score_tensor = torch.tensor([box["confidence"] for box in class_boxes], dtype=torch.float32, device=device)
        selected = nms(box_tensor, score_tensor, nms_threshold)
        for index in selected.tolist():
            merged.append(class_boxes[index])

    merged.sort(key=lambda item: item["confidence"], reverse=True)
    return {"image_id": image_id, "boxes": merged[:max_detections_per_image]}


def weighted_fusion_boxes(
    image_id: str,
    boxes: list[dict[str, object]],
    class_names: list[str],
    iou_threshold: float,
    max_detections_per_image: int,
    device: torch.device,
) -> dict[str, object]:
    if not boxes:
        return {"image_id": image_id, "boxes": []}

    fused: list[dict[str, object]] = []
    for class_name in class_names:
        class_boxes = [
            box
            for box in boxes
            if box["class"] == class_name and rounded_valid_bbox([float(value) for value in box["bbox"]]) is not None
        ]
        if not class_boxes:
            continue
        box_tensor = torch.tensor([box["bbox"] for box in class_boxes], dtype=torch.float32, device=device)
        score_tensor = torch.tensor([box["confidence"] for box in class_boxes], dtype=torch.float32, device=device)
        order = score_tensor.argsort(descending=True)

        while order.numel() > 0:
            current = order[0]
            if order.numel() == 1:
                cluster_indices = current.unsqueeze(0)
                order = order[1:]
            else:
                ious = box_iou(box_tensor[current].unsqueeze(0), box_tensor[order]).squeeze(0)
                cluster_mask = ious >= iou_threshold
                cluster_indices = order[cluster_mask]
                order = order[~cluster_mask]

            cluster_boxes = box_tensor[cluster_indices]
            cluster_scores = score_tensor[cluster_indices]
            weights = cluster_scores.clamp(min=1e-6)
            fused_box = (cluster_boxes * weights[:, None]).sum(dim=0) / weights.sum()
            bbox = rounded_valid_bbox(fused_box.tolist())
            if bbox is None:
                continue
            fused.append(
                {
                    "class": class_name,
                    "confidence": round(float(cluster_scores.max().item()), 6),
                    "bbox": bbox,
                }
            )

    fused.sort(key=lambda item: item["confidence"], reverse=True)
    return {"image_id": image_id, "boxes": fused[:max_detections_per_image]}


def predict_with_tta(
    model: TinyGridDetector,
    image_path: Path,
    class_names: list[str],
    img_sizes: list[int],
    conf_threshold: float,
    nms_threshold: float,
    max_detections_per_image: int,
    pre_nms_topk: int,
    brightness_factors: list[float],
    merge_method: str,
    wbf_iou_threshold: float,
    device: torch.device,
    preprocess: str,
    use_tta: bool,
    channels_last: bool,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    all_boxes: list[dict[str, object]] = []

    variants: list[tuple[Image.Image, bool, int]] = []
    for img_size in img_sizes:
        variants.append((image, False, img_size))
        if use_tta:
            variants.append((image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), True, img_size))
            for factor in brightness_factors:
                variants.append((ImageEnhance.Brightness(image).enhance(factor), False, img_size))

    for variant, flipped, img_size in variants:
        prediction = predict_image(
            model,
            variant,
            image_path.name,
            class_names,
            img_size,
            conf_threshold,
            nms_threshold,
            pre_nms_topk,
            device,
            preprocess,
            channels_last,
        )
        for box in prediction["boxes"]:
            box = dict(box)
            if flipped:
                xmin, ymin, xmax, ymax = box["bbox"]
                bbox = rounded_valid_bbox([original_w - xmax, ymin, original_w - xmin, ymax], original_w, original_h)
                if bbox is None:
                    continue
                box["bbox"] = bbox
            all_boxes.append(box)

    if merge_method == "wbf":
        return weighted_fusion_boxes(
            image_path.name,
            all_boxes,
            class_names,
            wbf_iou_threshold,
            max_detections_per_image,
            device,
        )
    return merge_boxes(
        image_path.name,
        all_boxes,
        class_names,
        nms_threshold,
        max_detections_per_image,
        device,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    checkpoint_path = ensure_checkpoint(args.checkpoint, args.checkpoint_url)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    img_size = int(checkpoint.get("img_size", args.img_size))
    checkpoint_model_type = checkpoint.get("model_type")
    use_p2 = bool(checkpoint.get("use_p2", checkpoint.get("model_type") == "fcos_resnet50_bifpn_p2"))
    use_p6 = bool(checkpoint.get("use_p6", checkpoint_model_type in {"fcos_resnet50_bifpn_p6_scale", "fcos_resnet50_fpn_p6_scale_v4"}))
    use_scales = bool(checkpoint.get("use_scales", checkpoint_model_type in {"fcos_resnet50_bifpn_p6_scale", "fcos_resnet50_fpn_p6_scale_v4"}))
    channels = int(checkpoint.get("channels", 256 if checkpoint_model_type == "fcos_resnet50_bifpn" else 128))
    use_bifpn = bool(checkpoint.get("use_bifpn", checkpoint_model_type == "fcos_resnet50_bifpn"))
    preprocess = args.preprocess
    if preprocess == "auto":
        preprocess = str(checkpoint.get("preprocess", "stretch"))
    use_tta = not args.disable_tta
    if args.tta_img_sizes is None:
        tta_img_sizes = [img_size, 704] if use_tta and img_size == 640 else [img_size]
    else:
        tta_img_sizes = args.tta_img_sizes or [img_size]
    invalid_sizes = [size for size in tta_img_sizes if size <= 0 or size % 32 != 0]
    if invalid_sizes:
        raise ValueError(f"TTA image sizes must be positive multiples of 32: {invalid_sizes}")
    channels_last = device.type == "cuda" and not args.no_channels_last
    print(
        "Starting prediction "
        f"device={device} "
        f"image_dir={args.image_dir} "
        f"checkpoint={checkpoint_path} "
        f"use_p2={use_p2} "
        f"use_p6={use_p6} "
        f"use_scales={use_scales} "
        f"channels={channels} "
        f"use_bifpn={use_bifpn} "
        f"preprocess={preprocess} "
        f"conf_threshold={args.conf_threshold} "
        f"max_detections_per_image={args.max_detections_per_image} "
        f"pre_nms_topk={args.pre_nms_topk} "
        f"tta={use_tta} "
        f"tta_img_sizes={tta_img_sizes} "
        f"tta_brightness={args.tta_brightness} "
        f"merge_method={args.merge_method} "
        f"wbf_iou_threshold={args.wbf_iou_threshold} "
        f"channels_last={channels_last} "
        f"output={args.output}",
        flush=True,
    )

    model = TinyGridDetector(
        num_classes=len(class_names),
        pretrained_backbone=False,
        use_p2=use_p2,
        use_p6=use_p6,
        use_scales=use_scales,
        channels=channels,
        use_bifpn=use_bifpn,
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_paths = sorted(
        [
            path
            for path in args.image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    predictions = []
    for index, image_path in enumerate(image_paths, start=1):
        predictions.append(
            predict_with_tta(
                model,
                image_path,
                class_names,
                tta_img_sizes,
                args.conf_threshold,
                args.nms_threshold,
                args.max_detections_per_image,
                args.pre_nms_topk,
                args.tta_brightness,
                args.merge_method,
                args.wbf_iou_threshold,
                device,
                preprocess,
                use_tta,
                channels_last,
            )
        )
        if args.progress_every > 0 and (index == len(image_paths) or index % args.progress_every == 0):
            print(f"predicted={index}/{len(image_paths)}", flush=True)
    write_json(predictions, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
