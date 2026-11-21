"""Colab driver (not part of run.py): process the 50 files one by one, resumable.
download -> frozen run.py -> copy result to Drive -> delete the parquet."""
import os, shutil, subprocess, sys, time, traceback

DRIVE = "/content/drive/MyDrive/alethia/results"
FROZEN = "/content/frozen"
DATA = "/content/finevideo/data"
VENV_PY = "/content/venv_test/bin/python"          # Python 3.11 + requirements.txt (same as the graders)
PY = VENV_PY if os.path.exists(VENV_PY) else sys.executable
os.makedirs(DRIVE, exist_ok=True)
files = [f"train-{i:05d}-of-01357.parquet" for i in range(50)]

log = open(f"{DRIVE}/full_run_log.txt", "a")
def logp(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    log.write(line + "\n"); log.flush()

logp(f"=== start / resume | python: {PY} ===")
t_all = time.time()
try:
    for name in files:
        out = f"{DRIVE}/{name}.jsonl"
        if os.path.exists(out):
            continue
        logp(f"start {name}")
        t0 = time.time()
        try:
            subprocess.run(["hf", "download", "HuggingFaceFV/finevideo", "--repo-type", "dataset",
                            "--include", f"data/{name}", "--local-dir", "/content/finevideo"],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logp(f"DOWNLOAD FAILED {name}: {e.stderr[-300:]}")
            continue
        t1 = time.time()

        one = f"/content/one/{name}"               # folder with only this one file
        os.makedirs(one, exist_ok=True)
        link = f"{one}/{name}"
        if not os.path.exists(link):
            os.symlink(f"{DATA}/{name}", link)
        tmp = f"/content/one/{name}.jsonl"
        r = subprocess.run([PY, f"{FROZEN}/run.py", "--input", one, "--output", tmp],
                           cwd=FROZEN, capture_output=True, text=True)
        t2 = time.time()
        if r.returncode != 0:
            logp(f"RUN FAILED {name} (code {r.returncode}): {r.stderr[-500:]}")
            continue
        shutil.copy(tmp, out + ".part")
        os.replace(out + ".part", out)             # result appears on Drive only when complete
        os.remove(f"{DATA}/{name}")
        shutil.rmtree(one, ignore_errors=True)
        os.remove(tmp)
        done_line = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        logp(f"{name}: download {t1-t0:.0f}s | run {t2-t1:.0f}s | {done_line}")
    logp(f"=== finished this session in {time.time()-t_all:.0f}s ===")
except BaseException:
    logp("DRIVER CRASHED:\n" + traceback.format_exc())
    raise
