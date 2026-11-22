"""Kaggle driver (not part of run.py): process the files not done yet, one file per GPU at a time.
Usage: python kaggle_run.py "<comma-separated file numbers already done elsewhere>" """
import os, shutil, subprocess, sys, threading, time

DONE = {int(x) for x in sys.argv[1].split(",") if x}
OUT = "/kaggle/working/results"
REPO, PY = "/tmp/repo", "/tmp/venv/bin/python"
os.makedirs(OUT, exist_ok=True)
todo = [i for i in range(50) if i not in DONE and
        not os.path.exists(f"{OUT}/train-{i:05d}-of-01357.parquet.jsonl")]
failed = []
n_gpus = max(1, subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
lock = threading.Lock()
log = open(f"{OUT}/kaggle_run_log.txt", "a")

def logp(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with lock:
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

def worker(gpu):
    data = f"/tmp/data{gpu}"
    while True:
        with lock:
            if not todo:
                return
            i = todo.pop(0)
        name = f"train-{i:05d}-of-01357.parquet"
        logp(f"start {name} on GPU {gpu}")
        t0 = time.time()
        r = subprocess.run(["hf", "download", "HuggingFaceFV/finevideo", "--repo-type", "dataset",
                            "--include", f"data/{name}", "--local-dir", data], capture_output=True, text=True)
        if r.returncode != 0:
            logp(f"DOWNLOAD FAILED {name}: {r.stderr[-300:]}")
            failed.append(i); continue
        t1 = time.time()
        one = f"/tmp/one{gpu}"
        shutil.rmtree(one, ignore_errors=True); os.makedirs(one)
        os.symlink(f"{data}/data/{name}", f"{one}/{name}")
        tmp = f"/tmp/out{gpu}.jsonl"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        r = subprocess.run([PY, f"{REPO}/run.py", "--input", one, "--output", tmp],
                           cwd=REPO, env=env, capture_output=True, text=True)
        t2 = time.time()
        if r.returncode != 0:
            logp(f"RUN FAILED {name}: {r.stderr[-500:]}")
            failed.append(i); continue
        shutil.copy(tmp, f"{OUT}/{name}.jsonl")
        os.remove(f"{data}/data/{name}")
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        logp(f"{name}: GPU {gpu} | download {t1-t0:.0f}s | run {t2-t1:.0f}s | {last}")

def run_all():
    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()

logp(f"=== start: {len(todo)} files on {n_gpus} GPUs ===")
t_all = time.time()
run_all()
if failed:
    logp(f"retrying failed files once: {failed}")
    todo.extend(failed); failed.clear()
    run_all()
done_now = sorted(f for f in os.listdir(OUT) if f.endswith(".parquet.jsonl"))
logp(f"=== finished in {time.time()-t_all:.0f}s | result files here: {len(done_now)} | still failed: {failed} ===")
