# Model Conversion Guide: YOLO11n → ONNX → RKNN

This document covers the full pipeline for training YOLO11n on a custom human detection dataset and converting it to an RKNN model for inference on a Rockchip RK3588 board.

---

## Overview

```
Roboflow Dataset
      │
      ▼
YOLO11n Training (.pt)       ← any machine with GPU (or CPU)
      │
      ▼
Export to ONNX (.onnx)       ← same machine, ultralytics export
      │
      ▼
RKNN Conversion (.rknn)      ← Ubuntu 22.04 x86, RKNN Toolkit2 v1.6.0
      │
      ▼
RK3588 Inference             ← Orange Pi 5, rknnlite
```

---

## Environment Requirements

| Stage | OS | Key packages |
|---|---|---|
| Training & ONNX export | Any (Linux/macOS/Windows) | `ultralytics`, `torch` |
| RKNN conversion | Ubuntu 22.04 (x86_64) | `rknn-toolkit2==1.6.0` |
| Board inference | Ubuntu on RK3588 | `rknnlite` (Rockchip official wheel) |

> **Note:** RKNN Toolkit2 is used on the **host PC** for conversion only.  
> `rknnlite` is the lightweight runtime used **on the board** for inference.  
> Do not use Docker for the RKNN conversion step — it is significantly slower.

---

## Step 1: Prepare Dataset (Roboflow)

1. Go to [Roboflow](https://roboflow.com) and find or create a human detection dataset.
2. Export in **YOLOv8 format** (compatible with YOLO11n).
3. Download and extract — you should get:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## Step 2: Train YOLO11n

Install Ultralytics:

```bash
pip install ultralytics
```

Train the model:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load pretrained YOLO11n weights

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="kiomind_person"
)
```

The best weights are saved at:

```
runs/detect/kiomind_person/weights/best.pt
```

---

## Step 3: Export to ONNX

```python
from ultralytics import YOLO

model = YOLO("runs/detect/kiomind_person/weights/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    opset=12,       # RKNN Toolkit2 works best with opset 11 or 12
    simplify=True
)
```

Output: `best.onnx`

---

## Step 4: Convert ONNX → RKNN (Ubuntu 22.04, host PC)

### Install RKNN Toolkit2

```bash
pip install rknn-toolkit2==1.6.0
```

Or install from the [official Rockchip GitHub release](https://github.com/rockchip-linux/rknn-toolkit2/releases/tag/v1.6.0).

### Conversion Script

```python
from rknn.api import RKNN

ONNX_MODEL = "best.onnx"
RKNN_MODEL = "yolo11n-rk3588.rknn"

rknn = RKNN(verbose=False)

# Configure for RK3588 target
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3588"
)

# Load ONNX model
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print("Load ONNX model failed!")
    exit(ret)

# Build (INT8 quantization)
# If you have a calibration dataset, pass dataset= parameter for accurate INT8
ret = rknn.build(do_quantization=True, dataset="calibration.txt")
if ret != 0:
    print("Build failed!")
    exit(ret)

# Export RKNN model
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print("Export failed!")
    exit(ret)

print(f"Exported: {RKNN_MODEL}")
rknn.release()
```

> **Quantization note:** `do_quantization=True` uses INT8 by default, which is recommended for RK3588 NPU performance.  
> `calibration.txt` is a text file listing paths to ~100–200 representative images (one path per line). If omitted, quantization accuracy may degrade.

### calibration.txt format

```
/path/to/dataset/train/images/img001.jpg
/path/to/dataset/train/images/img002.jpg
...
```

---

## Step 5: Deploy to RK3588 Board

Copy the converted model to the board:

```bash
scp yolo11n-rk3588.rknn user@<board-ip>:/home/KioMind/kiomind/yolo11n_rknn_model/
```

The model is loaded in `object_detection.py` via `rknnlite`:

```python
from rknnlite.api import RKNNLite

rknn = RKNNLite()
rknn.load_rknn("yolo11n_rknn_model/yolo11n-rk3588.rknn")
rknn.init_runtime()
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Slow conversion in Docker | Docker overhead on RKNN | Use native Ubuntu 22.04 install |
| `opset` error during load | Unsupported ONNX opset | Re-export with `opset=11` or `opset=12` |
| Low detection accuracy after INT8 | Missing calibration dataset | Provide a `calibration.txt` with representative images |
| `init_runtime()` fails on board | rknnlite version mismatch | Match rknnlite version to the RKNN Toolkit2 version used for conversion (v1.6.0) |
