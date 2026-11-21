"""Compare a results file with a reference using the brief's 3 metrics."""
import argparse, json
from collections import defaultdict

NAMES = ["person", "car", "dog", "laptop", "cell phone"]

def load(path):
    lines, errors = {}, {}
    for line in open(path):
        o = json.loads(line)
        if "error" in o:
            errors[o["video"]] = o["error"]
        else:
            lines[(o["video"], o["second"])] = (set(o["objects"]), o["people"])
    return lines, errors

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--sub", required=True)
a = ap.parse_args()

ref, ref_err = load(a.ref)
sub, sub_err = load(a.sub)
ref_videos = {v for v, _ in ref}

tp = fp = fn = 0
cls = defaultdict(lambda: [0, 0, 0])          # per-class tp, fp, fn
exact = within1 = missing = 0
for key, (r_obj, r_ppl) in ref.items():
    s_obj, s_ppl = sub.get(key, (set(), None))
    if key not in sub:
        missing += 1
    tp += len(r_obj & s_obj); fp += len(s_obj - r_obj); fn += len(r_obj - s_obj)
    for n in NAMES:
        cls[n][0] += (n in r_obj and n in s_obj)
        cls[n][1] += (n in s_obj and n not in r_obj)
        cls[n][2] += (n in r_obj and n not in s_obj)
    exact += (s_ppl == r_ppl)
    within1 += (s_ppl is not None and abs(s_ppl - r_ppl) <= 1)

extra = [k for k in sub if k[0] in ref_videos and k not in ref]   # seconds we wrote that the reference doesn't have
for k in extra:
    fp += len(sub[k][0])

f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
N = len(ref)
print(f"seconds in reference: {N} | missing in sub: {missing} | extra in sub: {len(extra)}")
print(f"Object F1        : {f1:.4f}   (need >= 0.95)")
print(f"People exact     : {exact/N:.4f}   (need >= 0.85)")
print(f"People within +-1: {within1/N:.4f}   (need >= 0.95)")
ok = f1 >= 0.95 and exact / N >= 0.85 and within1 / N >= 0.95
print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
print("\nper-class F1:")
for n in NAMES:
    t, p, q = cls[n]
    print(f"  {n:11s} F1={2*t/(2*t+p+q) if (t+p+q) else 1:.3f}  (ref positives {t+q})")
