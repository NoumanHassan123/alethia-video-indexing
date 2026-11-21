"""Preparation step: run once, WITH internet, before the timed run.
Downloads the official yolo11l.pt next to run.py and checks that the GPU works."""
import os
from pathlib import Path

here = Path(__file__).resolve().parent
os.chdir(here)

import numpy as np
import torch
from ultralytics import YOLO

model = YOLO("yolo11l.pt")                      # downloads the official weights into this folder
assert (here / "yolo11l.pt").exists(), "yolo11l.pt was not downloaded"
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=640, quantize=16, verbose=False)   # warm-up check
print("prepare.py: ready")
