"""Run YOLO on EVERY frame and save per-frame class counts, so sampling strategies can be simulated offline."""
import argparse, io, json, sys
from pathlib import Path
import av, pyarrow.parquet as pq
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from run import CLASSES, rotate

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()

model = YOLO("yolo11l.pt")
ids = list(CLASSES)                      # [0, 2, 16, 63, 67]
with open(a.output, "w") as fo:
    for f in sorted(Path(a.input).glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for r, b in enumerate(pf.iter_batches(batch_size=1, columns=["mp4"])):
            key = f"{f.name}:{r}"
            imgs, pts, counts = [], [], []

            def flush():
                if not imgs:
                    return
                res = model.predict(imgs, imgsz=640, conf=0.4, iou=0.7, agnostic_nms=False,
                                    classes=ids, quantize=16, verbose=False)
                for x in res:
                    c = x.boxes.cls.int().tolist()
                    counts.append([c.count(i) for i in ids])
                imgs.clear()

            try:
                with av.open(io.BytesIO(b.column("mp4")[0].as_py())) as cont:
                    s = cont.streams.video[0]
                    s.thread_type = "AUTO"
                    tb = s.time_base
                    for fr in cont.decode(s):
                        pts.append(fr.pts)
                        img = fr.to_ndarray(format="bgr24")
                        rot = getattr(fr, "rotation", 0) or 0
                        if rot:
                            img = rotate(img, rot)
                        imgs.append(img)
                        if len(imgs) >= a.batch:
                            flush()
                    flush()
                fo.write(json.dumps({"video": key, "tb": [tb.numerator, tb.denominator],
                                     "pts": pts, "counts": counts}) + "\n")
                print(key, len(pts), "frames", flush=True)
            except Exception as e:
                print(key, "ERROR", e, flush=True)
