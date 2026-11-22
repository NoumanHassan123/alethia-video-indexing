# Write-up

## 1. My timings

My first working version ran YOLO on every frame. I tested it on 3 short videos but never ran it on all 50 files. From a 13-video test, I estimate it would take about 38 hours. My final version took about 3 hours 45 minutes of active run time for all 50 files (108 minutes on Colab, 118 minutes on Kaggle). The change that saved the most time was frame sampling. YOLO looks at 2 frames per second, and up to 6 only when they disagree, so only about 10% of frames reach YOLO.

## 2. My pipeline

1. A reader thread reads one row at a time from the Parquet file, taking only the mp4 column.
2. Decoder threads decode each video from memory with PyAV (CPU threads ÷ 4, between 1 and 8).
3. A frame's second is its time since the first frame, rounded down.
4. I take 2 frames per second. If they disagree on people or objects, I take up to 6 frames for that second.
5. Only the chosen frames are converted to BGR and rotated if needed.
6. The main thread runs YOLO11l in FP16, one video per batch, with size 640, confidence 0.4, IoU 0.7 and only the five classes.
7. An object counts in a second if it is in at least half of the chosen frames. People is the median, rounded down.
8. I write a line for every second. A failed video keeps its lines and gets an error line.

## 3. Where the time goes

I never profiled the final version stage by stage, so part of this is an estimate. I measured two speeds on Colab: decoding with BGR conversion ran at 752 frames/s, and YOLO in FP16 ran at 109 frames/s. For an every-frame run, YOLO was the slowest part. With sampling, only about 10% of frames reach YOLO, but every frame is still decoded, so decoding is now a big share. Two signs support this: lowering the sampling rate did not cut the time in proportion, and run.py used about 143% CPU while the GPU worked in bursts. I got these numbers by timing runs with different settings, not with a profiler.

## 4. Checking correctness

I made a reference by running FP32 on every frame of 13 videos, and checked each change against it with check.py, which computes the three grading metrics. For file 00001 I ran FP16 on every frame (272,410 frames) and saved the counts, so I could test sampling strategies without the GPU. The real run matched that simulation (F1 0.987). For speed-only changes (pipelining, parallel decoding, offline mode) I checked that the output was identical. Last, a fresh Python 3.11 offline run gave F1 0.979, exact 0.948, within 1 0.998.

## 5. Working with AI

The AI wrote my requirements.in with av 19.0.2, copied from Colab's Python 3.13. I found this was wrong when I installed it in a fresh Python 3.11 environment, like the grader's. The install failed, because av 19 needs Python 3.12 or newer. I changed the pin to av 18.1.1 and the test passed. For the same reason I also switched to the CUDA 12.6 torch build, since Colab's CUDA 13 build may not run on a CUDA 12 driver.

## 6. Next steps

First I would profile the final version stage by stage, because I never measured it properly. Then I would try GPU video decoding, since decoding every frame is now a big part of the time. I would also try TensorRT, and move the letterbox resize off the CPU, where it runs in the main thread. Finally, I would time the parallel decoder on a 32-thread machine, which I could not do, and check accuracy on more than the 46 videos I used.