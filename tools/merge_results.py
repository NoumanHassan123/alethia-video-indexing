"""Join the 50 per-file results into results.jsonl.gz, check them, and summarise the full-run log.
Usage: python merge_results.py <results folder> <output .jsonl.gz> [--partial]"""
import gzip, json, re, sys
from datetime import datetime, timedelta
from pathlib import Path

res_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
partial = "--partial" in sys.argv
names = [f"train-{i:05d}-of-01357.parquet" for i in range(50)]
present = [n for n in names if (res_dir / f"{n}.jsonl").exists()]
missing = [n for n in names if n not in present]
print(f"result files: {len(present)}/50")
if missing and not partial:
    print("MISSING:", missing)
    sys.exit(1)

videos, err_videos, secs = set(), {}, {}
n_lines = bad = 0
with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
    for n in present:
        for line in open(res_dir / f"{n}.jsonl"):
            o = json.loads(line)
            v = o["video"]
            videos.add(v)
            if "error" in o:
                err_videos[v] = o["error"]
            elif set(o) == {"video", "second", "objects", "people"}:
                secs.setdefault(v, set()).add(o["second"])
            else:
                bad += 1
            gz.write(line.encode())
            n_lines += 1

gaps = [v for v, s in secs.items() if s != set(range(max(s) + 1))]
hours = sum(max(s) + 1 for s in secs.values()) / 3600
print(f"lines: {n_lines} | videos: {len(videos)} | videos with an error line: {len(err_videos)} | "
      f"badly formed lines: {bad} | videos with missing seconds: {len(gaps)}")
print(f"video covered: {hours:.1f} hours")
for v, e in list(err_videos.items())[:10]:
    print("  error:", v, "->", e)
print(f"written: {out} ({out.stat().st_size/1e6:.1f} MB)")

# ---- timings from the driver log (times in the log are UTC; PKT = UTC+5) ----
log = res_dir / "full_run_log.txt"
if log.exists():
    fmt = "%Y-%m-%d %H:%M:%S"
    done, start = {}, None
    for line in open(log):
        m = re.match(r"^(\S+ \S+) (train-\d+-of-01357\.parquet): download (\d+)s \| run (\d+)s", line)
        if m:
            done[m.group(2)] = (datetime.strptime(m.group(1), fmt), int(m.group(3)), int(m.group(4)))
        m = re.match(r"^(\S+ \S+) start train-00000-of-01357\.parquet", line)
        if m:
            start = datetime.strptime(m.group(1), fmt)
    if done and start:
        end = max(t for t, _, _ in done.values())
        pkt = lambda t: (t + timedelta(hours=5)).strftime(fmt) + " PKT"
        dl = sum(d for _, d, _ in done.values())
        run = sum(r for _, _, r in done.values())
        h = lambda s: f"{s/3600:.2f} h ({s/60:.0f} min)"
        print(f"\nfiles done per log: {len(done)}")
        print(f"start: {pkt(start)} | end: {pkt(end)} | total start to finish: {h((end - start).total_seconds())}")
        print(f"downloading: {h(dl)} | inside run.py: {h(run)}")
