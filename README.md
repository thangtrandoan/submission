# FCOS Object Detection

Dự án này cài đặt một one-stage object detector từ đầu cho 5 lớp:

- `person`
- `car`
- `dog`
- `cat`
- `chair`


## Cài Đặt

```bash
pip install -r requirements.txt
```

## Cấu Trúc

```text
models/
  detector.py
utils/
  box_ops.py
  dataset.py
  json_utils.py
  loss.py
  nms.py
  transforms.py
train.py
predict.py
requirements.txt
```

## Dữ Liệu

Dữ liệu theo cấu trúc:

```text
public/
  classes.json
  train/images/
  val/images/
  annotations/train.json
  annotations/val.json
  tools/evaluate_predictions.py
```

Annotation dùng bbox dạng:

```text
[xmin, ymin, xmax, ymax]
```

Tọa độ bbox là tọa độ trên ảnh gốc.

## Mô Hình

Detector anchor-free theo hướng FCOS:

- Backbone: ResNet50 pretrained.
- Neck: FPN nhẹ, mặc định có P6 stride 64.
- Head: classification tower, box regression tower, centerness head.
- Output mỗi level gồm:
  - class logits
  - box distances `[left, top, right, bottom]`
  - centerness/objectness

Checkpoint lưu thêm các tùy chọn kiến trúc:

- `use_p2`
- `use_p6`
- `use_scales`
- `channels`
- `use_bifpn`
- `preprocess`
- `model_type`

`predict.py` tự đọc metadata trong checkpoint để khởi tạo đúng kiến trúc.

## Tiền Xử Lý Và Augment

Pipeline dữ liệu có:

- Đọc JSON annotation và nhiều object trong một ảnh.
- Letterbox resize để giữ tỉ lệ ảnh.
- Normalize theo ImageNet mean/std.
- Horizontal flip.
- Color jitter.
- Multi-scale training.

## Loss

Loss gồm các thành phần:

- Focal loss cho classification.
- GIoU loss cho box regression.
- BCEWithLogits cho centerness.
- Có tùy chọn class weight/boost cho `chair`, mặc định tắt để giảm bias.

## Train

Lệnh bắt buộc theo đề bài:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Checkpoint tốt nhất được lưu tại:

```text
./models/best.pth
```

Checkpoint mới nhất được lưu tại:

```text
./models/last.pth
```

Resume:

```bash
python train.py ... --resume_from_best
python train.py ... --resume_from_last
```

Lệnh train:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --img_size 640 \
  --multi_scale_min 640 \
  --multi_scale_max 640 \
  --batch_size 16 \
  --val_batch_size 16 \
  --lr 1.5e-4 \
  --scheduler onecycle \
  --eval_map_every 1 \
  --early_stopping_patience 6
```

Mặc định train:

- `img_size=640`
- `fixed scale=640`
- `epochs=16`
- `channels=128`
- `scheduler=onecycle`
- `map_nms_threshold=0.55`
- `map_conf_threshold=0.005`
- `map_pre_nms_topk=1500`
- `map_max_detections_per_image=300`
- EMA bật mặc định, `best.pth` được chọn/evaluate bằng EMA weights
- không bật class-aware sampler và class weights mặc định


## Predict

Lệnh predict:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output ./val_predictions.json \
  --checkpoint ./models/best.pth 
```

Mặc định:

- `--checkpoint ./models/best.pth`
- Neu checkpoint chua ton tai, `predict.py` se tu tai weight ve duong dan nay.
- Link weight mac dinh: `https://github.com/thangtrandoan/submission/releases/download/latest/best.pth`
- Co the doi link tai bang `--checkpoint_url`.
- `--conf_threshold 0.01`
- `--max_detections_per_image 300`
- `--nms_threshold 0.6`
- `--wbf_iou_threshold 0.55`
- `--tta_img_sizes 640 512 704`
- `--tta_brightness 0.9 1.1`

- TTA flip ngang và điều chỉnh độ sáng được bật mặc định.
- Mặc định suy luận bằng multi-scale TTA với các kích thước: `640`, `512`, và `704`.
- Merge TTA bằng weighted box fusion (`--merge_method wbf`).

Có thể tắt TTA để chạy nhanh hơn:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output val_predictions.json \
  --disable_tta \
  --img_size 640
```

## Output

`predictions.json` là một mảng JSON:

```json
[
  {
    "image_id": "example.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

Mỗi ảnh trong `image_dir` đều có một phần tử output. Ảnh không có detection sẽ có:

```json
{"image_id": "example.jpg", "boxes": []}
```

NMS được cài đặt trong `utils/nms.py` và được chạy riêng theo từng class.

## Evaluate

```bash
python ./public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions ./val_predictions.json \
  --output ./val_score.json
```
