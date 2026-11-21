"""Simulate sampling strategies offline from per-frame YOLO counts (output of dump_frames.py)."""
import json, math, statistics, sys
from fractions import Fraction

NAMES = ["person", "car", "dog", "laptop", "cell phone"]

def load(path):
    vids = []
    for line in open(path):
        o = json.loads(line)
        tb, p0 = Fraction(*o["tb"]), o["pts"][0]
        rels = [(p - p0) * tb for p in o["pts"]]
        secs = [max(0, math.floor(r)) for r in rels]
        vids.append((o["video"], rels, secs, o["counts"]))
    return vids

def sample(rels, secs, k):
    """Same rule as run.py: first frame in each 1/k part of each second. k=0 means every frame."""
    if k == 0:
        return list(range(len(rels)))
    out, last = [], None
    for i, (r, s) in enumerate(zip(rels, secs)):
        slot = (s, math.floor((r - s) * k))
        if slot != last:
            out.append(i)
            last = slot
    return out

def group(idx, secs, counts):
    by = {}
    for i in idx:
        by.setdefault(secs[i], []).append(counts[i])
    return by

def aggregate(idx, secs, counts):
    """Brief's rules: object present if in >= half the frames; people = floor(median)."""
    res = {}
    for s, cs in group(idx, secs, counts).items():
        n = len(cs)
        objs = frozenset(NAMES[j] for j in range(5) if sum(c[j] > 0 for c in cs) * 2 >= n)
        res[s] = (objs, math.floor(statistics.median(c[0] for c in cs)))
    return res

def uncertain(idx, secs, counts):
    """Seconds where the sampled frames disagree: different people counts, or an object in some frames but not all."""
    bad = set()
    for s, cs in group(idx, secs, counts).items():
        if len({c[0] for c in cs}) > 1 or any(0 < sum(c[j] > 0 for c in cs) < len(cs) for j in range(1, 5)):
            bad.add(s)
    return bad

def choose(st, rels, secs, counts):
    if st[0] == "k":
        return sample(rels, secs, st[1])
    _, k1, k2 = st                                    # adaptive: k1 frames/s, more frames (k2) only where unsure
    base = sample(rels, secs, k1)
    bad = uncertain(base, secs, counts)
    extra = [i for i in sample(rels, secs, k2) if secs[i] in bad]
    return sorted(set(base) | set(extra))

def label(st):
    if st[0] == "k":
        return "every frame" if st[1] == 0 else f"{st[1]} / s"
    return f"adapt {st[1]}->{'all' if st[2] == 0 else st[2]}"

vids = load(sys.argv[1])
total = sum(len(v[1]) for v in vids)
strategies = [("k", 0), ("k", 2), ("k", 3), ("k", 4), ("k", 5), ("k", 6),
              ("ad", 2, 0), ("ad", 2, 6), ("ad", 3, 0), ("ad", 3, 6)]

ref_pos = [0] * 5
print(f"{len(vids)} videos, {total} frames\n")
print(f"{'strategy':>12} | YOLO frames | F1 (>=.95) | exact (>=.85) | +-1 (>=.95) | car    dog    laptop phone")
for st in strategies:
    tp = fp = fn = ex = w1 = N = used = 0
    cls = [[0, 0, 0] for _ in NAMES]
    for key, rels, secs, counts in vids:
        ref = aggregate(range(len(rels)), secs, counts)
        idx = choose(st, rels, secs, counts)
        used += len(idx)
        sub = aggregate(idx, secs, counts)
        for s in range(max(secs) + 1):
            ro, rp = ref.get(s, (frozenset(), 0))
            so, sp = sub.get(s, (frozenset(), 0))
            N += 1
            tp += len(ro & so); fp += len(so - ro); fn += len(ro - so)
            ex += sp == rp; w1 += abs(sp - rp) <= 1
            for j, n in enumerate(NAMES):
                cls[j][0] += n in ro and n in so
                cls[j][1] += n in so and n not in ro
                cls[j][2] += n in ro and n not in so
                if st == ("k", 0):
                    ref_pos[j] += n in ro
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    pc = "  ".join(f"{2*t/(2*t+p+q) if (t+p+q) else 1:.3f}" for t, p, q in cls[1:])
    print(f"{label(st):>12} | {used:>7} {used/total*100:4.1f}% | {f1:.4f}     | {ex/N:.4f}        | {w1/N:.4f}      | {pc}")

print(f"\nseconds: {N}   reference positive seconds -> " + ", ".join(f"{n}: {p}" for n, p in zip(NAMES, ref_pos)))
