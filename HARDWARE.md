# Hardware used for the full run

The full run over the first 50 FineVideo Parquet files (`data/train-00000-of-01357.parquet` to
`data/train-00049-of-01357.parquet`) was done on **two free cloud machines**, because the free Colab GPU
quota ran out after 15 files.

## Machine 1: Google Colab free tier (files 0 to 14)

| | |
|---|---|
| GPU | NVIDIA Tesla T4, 15 GB memory (15360 MiB), driver 580.82.07 |
| CPU | Intel Xeon @ 2.00 GHz, 2 vCPUs (1 physical core, 2 threads) |
| RAM | 12 GB |
| Disk | SSD (cloud VM disk, 113 GB) |
| Operating system | Ubuntu 24.04.4 LTS (kernel 6.6) |
| Software | Python 3.11.18 virtual environment from `requirements.txt` (torch 2.14.1+cu126, ultralytics 8.4.216, av 18.1.1) |
| Script | first committed `run.py` (one decoder thread), driver `tools/full_run.py` |
| Time | 2026-11-21 18:50:57 to 20:38:59 PKT = about 108 min (2 min downloading, 106 min inside `run.py`) |

## Machine 2: Kaggle notebook (files 15 to 49)

| | |
|---|---|
| GPU | 2 x NVIDIA Tesla T4, 15 GB memory each (one file per GPU at a time) |
| CPU | 4 vCPUs (model not recorded) |
| RAM | 31 GB |
| Disk, OS | not recorded |
| Software | fresh `git clone` of this repository (commit b6c3aa2), Python 3.11 virtual environment from `requirements.txt`, then `python prepare.py` |
| Script | final `run.py` (parallel decoder code; 1 decoder thread on 4 vCPUs), driver `tools/kaggle_run.py` |
| Time | 2026-11-22 02:01:54 to 03:59:25 PKT = about 118 min (2 min downloading, 214 min inside `run.py` summed over the 2 GPUs) |

## Total

| | |
|---|---|
| Files / videos / errors | 50 files, 1650 videos, 0 error lines |
| Video covered | 128.3 hours (461,757 output lines) |
| Total run time, start to finish | about 3 h 45 min (108 min on Colab + 118 min on Kaggle). The two machines ran one after the other, with a pause in between while I switched from Colab to Kaggle. |

## What went wrong along the way

- A first Colab attempt (started 18:00 PKT) was killed after about 4 minutes when the machine ran out of memory,
  because I was running other tests at the same time. The Colab run above restarted from the beginning.
- On Kaggle I stopped the first run by mistake at about 03:03 PKT, while files 34 and 35 were in progress.
  Files 15 to 33 were already saved; files 34 to 49 were run again from the start (03:09:59 to 03:59:25 PKT).
- Colab and Kaggle used different `run.py` versions. The final version adds parallel decoder threads; on test
  videos and on the damaged-file tests I checked that both give identical results (same lines after sorting).
- The full per-file logs are in `tools/full_run_log.txt` (Colab) and `tools/kaggle_run_log.txt` (Kaggle).
