# Hardware used for the full run

The full run over the first 50 FineVideo Parquet files (`data/train-00000-of-01357.parquet` to
`data/train-00049-of-01357.parquet`) was done on **Google Colab (free tier)**.

| | |
|---|---|
| Platform | Google Colab free tier (cloud virtual machine) |
| GPU | NVIDIA Tesla T4, 15 GB memory (15360 MiB), driver 580.82.07 |
| CPU | Intel Xeon @ 2.00 GHz, 2 vCPUs (1 physical core, 2 threads) |
| RAM | 12 GB |
| Disk | SSD (cloud VM disk, 113 GB) |
| Operating system | Ubuntu 24.04.4 LTS (kernel 6.6) |
| Software | Python 3.11.18 virtual environment installed from `requirements.txt`: torch 2.14.1+cu126, ultralytics 8.4.216, av 18.1.1 |

## How long the full run took

| | |
|---|---|
| Start (first file) | 2026-11-21 18:50:57 PKT |
| End (last file) | TODO |
| **Total, start to finish** | **TODO** |
| Of which: downloading the Parquet files | TODO |
| Of which: inside `run.py` | TODO |
| Videos / errors | TODO |

The Colab disk could not comfortably hold all 25 GB next to everything else, so the files were processed
one at a time with `tools/full_run.py`: download one Parquet file, run
`python run.py --input <folder with that file> --output <file>.jsonl`, save the result, delete the
Parquet file. The 50 result files were then joined into `results.jsonl.gz`.

An earlier attempt (started 18:00 PKT) was stopped after about 4 minutes when the Colab machine ran out
of memory, because I was running other tests at the same time. The run above restarted from the
beginning and is the one reported here.

The full run used the first committed version of `run.py` (one decoder thread). The final `run.py` adds parallel decoder threads; on this 2-thread machine it also uses one decoder thread, and I checked on test videos and on the damaged-file tests that both versions give identical results (same lines after sorting).
