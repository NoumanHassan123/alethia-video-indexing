# Video indexing with YOLO11l

For every second of every video in a folder of FineVideo Parquet files, `run.py` records which of five
objects (person, car, dog, laptop, cell phone) appear and how many people are visible, and writes one
JSON line per second.

## Requirements

- Ubuntu 22.04, Python 3.11, an NVIDIA GPU with a driver that supports CUDA 12
- `torch` / `torchvision` are installed from the PyTorch CUDA 12.6 index (first line of `requirements.txt`)

## 1. System packages

OpenCV (used by Ultralytics) needs these libraries on a minimal Ubuntu install:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## 2. Python packages

```bash
pip install -r requirements.txt
```

## 3. Preparation command (run once, needs internet, not timed)

```bash
python prepare.py
```

This downloads the official `yolo11l.pt` into the same folder as `run.py` and runs a short GPU check.
It prints `prepare.py: ready` when done.

## 4. Run (no internet needed)

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

Defaults are the submitted settings. Optional flags (not needed for grading):

| flag | default | meaning |
|---|---|---|
| `--sample` | 2 | base frames per second sent to YOLO (`0` = every frame) |
| `--extra` | 6 | frames per second used for seconds whose base frames disagree (`0` = off) |
| `--batch` | 32 | YOLO batch size |
| `--fp32` | off | full precision instead of FP16 |

## Output

One line per second of each video, from second 0 up to the second holding the last frame:

```json
{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}
```

A video that cannot be processed gets one line with an `error` field. If it fails partway, the seconds
already processed are kept and one error line is added. A damaged Parquet file never stops the run.

## Model settings

YOLO11l (`yolo11l.pt`), `imgsz=640`, `conf=0.4`, class-aware NMS with `iou=0.7`, BGR frames, letterbox
resize kept from Ultralytics, FP16. `iou=0.7` is passed explicitly because in Ultralytics 8.4 the
predict default is 0.5. Frames are rotated using the video's rotation tag before the model sees them.
Each YOLO batch holds frames from a single video, so all frames in a batch have the same size and get
the same letterbox as single-frame inference.

## Repository layout

| path | contents |
|---|---|
| `run.py` | the script |
| `prepare.py` | preparation command |
| `requirements.txt` | pinned packages (tested in a fresh Python 3.11 venv) |
| `results.jsonl.gz` | results for the first 50 FineVideo Parquet files |
| `HARDWARE.md` | where and how long the full run took |
| `WRITEUP.md` | answers to the write-up questions |
| `ai_chats/` | AI chat history |
| `tools/` | development tools, not needed to run: `check.py` (scores a results file against a reference with the three grading metrics), `dump_frames.py` (every-frame detections), `simulate.py` (compares sampling strategies offline), `full_run.py` (resumable driver used for the 50-file run on Colab) |
