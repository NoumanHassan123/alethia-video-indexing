#!/usr/bin/env python3
"""Video indexing: for each second of each video, which of 5 objects appear and how many people.

Pipeline
  decoder thread : Parquet row -> mp4 bytes -> PyAV decode -> choose frames -> batches on a queue
  main thread    : YOLO11l on batches -> per-second results -> JSONL lines

Adaptive sampling
  Every second gets `--sample` base frames (default 2). If the base frames of a second disagree
  (different people count or different objects), that second also gets the frames at `--extra`
  per second (default 6). The brief's rules (>= half the frames, floor of the median) are then
  applied to every frame that went through YOLO in that second."""
import argparse, io, json, math, os, queue, statistics, sys, threading, time, traceback
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
os.environ.setdefault("YOLO_OFFLINE", "1")   # the timed run has no internet: never try to go online
from ultralytics import YOLO

# COCO class id -> output name (names must match the brief exactly)
CLASSES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
NAME_ORDER = ["person", "car", "dog", "laptop", "cell phone"]
DONE = object()   # marks the end of the decoder's work


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="folder of .parquet files")
    p.add_argument("--output", required=True, help="output .jsonl file")
    p.add_argument("--weights", default=str(Path(__file__).resolve().parent / "yolo11l.pt"),
                   help="weights file (prepare.py downloads it next to run.py)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--sample", type=int, default=2, help="base YOLO frames per second (0 = every frame)")
    p.add_argument("--extra", type=int, default=6, help="frames per second for seconds whose base frames disagree (0 = off)")
    p.add_argument("--fp32", action="store_true", help="full precision (used to build a reference)")
    p.add_argument("--rows", default=None, help="testing only: comma-separated row numbers")
    return p.parse_args()


def rotate(img, rotation):
    """Rotate by the video's rotation tag. PyAV gives degrees counter-clockwise; np.rot90 also turns counter-clockwise."""
    k = int(round(rotation / 90.0)) % 4
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


# ----------------------------- decoder thread -----------------------------

def decode_video(key, data, args, q):
    """Decode one video; put ('frames', key, imgs, secs, is_base) batches on the queue, then ('end', key, error)."""
    q.put(("start", key))
    if not data:
        q.put(("end", key, "empty mp4 bytes"))
        return
    imgs, secs, flags, err = [], [], [], None
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = stream.time_base
            rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(30)
            first_t = last_base = last_extra = None
            for idx, frame in enumerate(container.decode(stream)):
                # timestamp in seconds (exact fraction) from the frame's pts
                if frame.pts is not None and tb is not None:
                    t = frame.pts * tb
                else:
                    t = (first_t or 0) + Fraction(idx) / rate
                if first_t is None:
                    first_t = t
                rel = t - first_t
                sec = max(0, math.floor(rel))

                is_base = True
                if args.sample:
                    part = rel - sec
                    b = (sec, math.floor(part * args.sample))     # first frame in each 1/sample of the second
                    is_base = b != last_base
                    if is_base:
                        last_base = b
                    is_extra = False
                    if args.extra:
                        x = (sec, math.floor(part * args.extra))  # first frame in each 1/extra of the second
                        is_extra = x != last_extra
                        if is_extra:
                            last_extra = x
                    if not (is_base or is_extra):
                        continue          # skip: no BGR conversion, no YOLO for this frame

                img = frame.to_ndarray(format="bgr24")          # BGR, as the brief requires
                rot = getattr(frame, "rotation", 0) or 0
                if rot:
                    img = rotate(img, rot)
                imgs.append(img)
                secs.append(sec)
                flags.append(is_base)
                if len(imgs) >= args.batch:
                    q.put(("frames", key, imgs, secs, flags))
                    imgs, secs, flags = [], [], []
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:200]
    if imgs:
        q.put(("frames", key, imgs, secs, flags))   # frames decoded before an error are still used
    q.put(("end", key, err))


def decoder(files, only_rows, args, q):
    """Walk every Parquet file and row; never crash, always finish with DONE."""
    try:
        for f in files:
            name = f.name
            try:
                pf = pq.ParquetFile(f)
                num_rows = pf.metadata.num_rows
            except Exception as e:
                q.put(("error", f"{name}:0", f"could not open parquet file: {type(e).__name__}"))
                continue
            row = 0
            try:
                for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                    col = batch.column("mp4")
                    for j in range(batch.num_rows):
                        r = row
                        if only_rows is None or r in only_rows:
                            decode_video(f"{name}:{r}", col[j].as_py(), args, q)
                        row = r + 1
            except Exception as e:
                # the Parquet file broke partway: every remaining row gets an error line
                for r in range(row, num_rows):
                    if only_rows is None or r in only_rows:
                        q.put(("error", f"{name}:{r}", f"could not read row: {type(e).__name__}"))
    except Exception:
        traceback.print_exc()
    finally:
        q.put(DONE)


# ----------------------------- main thread (GPU) -----------------------------

class Video:
    """State for the video being processed in the main thread."""
    def __init__(self):
        self.stats = {}                    # sec -> [n_frames, [people per frame], {name: frames present}]
        self.base_imgs, self.base_secs = [], []
        self.base_res = {}                 # sec -> set of (people, objects) seen in its base frames
        self.waiting = {}                  # sec -> extra frames kept until we know if the second is uncertain
        self.x_imgs, self.x_secs = [], []  # extra frames that will go to YOLO
        self.err = None


def predict(model, imgs, args):
    """YOLO on one batch (all frames from the same video, so the same size). Returns [(people, objects)]."""
    results = model.predict(
        imgs, imgsz=640, conf=0.4, iou=0.7, agnostic_nms=False,
        classes=list(CLASSES), quantize=None if args.fp32 else 16, verbose=False,
    )
    out = []
    for r in results:
        cls = r.boxes.cls.int().tolist()
        out.append((cls.count(0), frozenset(CLASSES[c] for c in cls)))
    return out


def add(v, sec, people, objs):
    rec = v.stats.setdefault(sec, [0, [], dict.fromkeys(NAME_ORDER, 0)])
    rec[0] += 1
    rec[1].append(people)
    for n in objs:
        rec[2][n] += 1


def step(model, v, args, final=False):
    """Run YOLO on the waiting base frames; send the extra frames of uncertain seconds to YOLO too."""
    last = None
    if v.base_imgs:
        for sec, (p, o) in zip(v.base_secs, predict(model, v.base_imgs, args)):
            add(v, sec, p, o)
            v.base_res.setdefault(sec, set()).add((p, o))
        last = v.base_secs[-1]
        v.base_imgs, v.base_secs = [], []
    for sec in sorted(v.waiting):
        if not final and (last is None or sec >= last):
            break                              # this second may still get more base frames
        imgs = v.waiting.pop(sec)
        if len(v.base_res.get(sec, ())) > 1:   # base frames disagree -> look at more frames
            v.x_imgs += imgs
            v.x_secs += [sec] * len(imgs)
    while len(v.x_imgs) >= args.batch or (final and v.x_imgs):
        imgs, secs = v.x_imgs[:args.batch], v.x_secs[:args.batch]
        del v.x_imgs[:args.batch], v.x_secs[:args.batch]
        for sec, (p, o) in zip(secs, predict(model, imgs, args)):
            add(v, sec, p, o)


def second_lines(key, stats):
    """One output line per second, from second 0 up to the second holding the last frame."""
    if not stats:
        return []
    lines = []
    for s in range(max(stats) + 1):
        rec = stats.get(s)
        if rec is None:                                   # no frames in this second
            objs, people = [], 0
        else:
            n, people_counts, present = rec
            objs = [name for name in NAME_ORDER if present[name] * 2 >= n]   # at least half the frames
            people = math.floor(statistics.median(people_counts))            # median, halfway -> round down
        lines.append({"video": key, "second": s, "objects": objs, "people": people})
    return lines


def main():
    args = parse_args()
    t_start = time.time()
    model = YOLO(args.weights)
    files = sorted(Path(args.input).glob("*.parquet"))
    only_rows = {int(x) for x in args.rows.split(",")} if args.rows else None

    q = queue.Queue(maxsize=6)          # small queue: decoder stays a few batches ahead, RAM stays low
    threading.Thread(target=decoder, args=(files, only_rows, args, q), daemon=True).start()

    n_videos = n_errors = 0
    v = Video()
    with open(args.output, "w") as out:
        def write(obj):
            out.write(json.dumps(obj) + "\n")

        while True:
            item = q.get()
            if item is DONE:
                break
            kind = item[0]
            if kind == "start":
                v = Video()
            elif kind == "frames":
                if v.err:
                    continue
                try:
                    for img, sec, is_base in zip(item[2], item[3], item[4]):
                        if is_base:
                            v.base_imgs.append(img)
                            v.base_secs.append(sec)
                        else:
                            v.waiting.setdefault(sec, []).append(img)
                    if len(v.base_imgs) >= args.batch:
                        step(model, v, args)
                except Exception as e:
                    v.err = f"model failed: {type(e).__name__}: {e}"[:200]
            elif kind == "end":
                key = item[1]
                if not v.err:
                    try:
                        step(model, v, args, final=True)
                    except Exception as e:
                        v.err = f"model failed: {type(e).__name__}: {e}"[:200]
                err = item[2] or v.err
                for line in second_lines(key, v.stats):
                    write(line)
                if err or not v.stats:
                    write({"video": key, "error": err or "no frames decoded"})
                    n_errors += 1
                n_videos += 1
                out.flush()
            elif kind == "error":
                write({"video": item[1], "error": item[2]})
                n_errors += 1

    print(f"Done: {n_videos} videos, {n_errors} errors, {time.time() - t_start:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
