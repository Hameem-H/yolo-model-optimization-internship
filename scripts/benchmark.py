
import time
import cv2
import torch
import GPUtil
import pandas as pd
from ultralytics import YOLO

# ============================================
# CONFIG
# ============================================

VIDEO_PATH = "videos/benchmark.mp4"

EXPERIMENTS = [

    {
        "name": "PyTorch_FP32_B1",
        "model": "models/best.pt",
        "half": False,
        "batch": 1,
        "imgsz": 640,
    },

    {
        "name": "PyTorch_FP16_B1",
        "model": "models/best.pt",
        "half": True,
        "batch": 1,
        "imgsz": 640,
    },

    {
        "name": "PyTorch_FP32_B8",
        "model": "models/best.pt",
        "half": False,
        "batch": 8,
        "imgsz": 640,
    },

    {
        "name": "PyTorch_FP16_B8",
        "model": "models/best.pt",
        "half": True,
        "batch": 8,
        "imgsz": 640,
    },

    {
        "name": "ONNX_FP32_B8",
        "model": "models/best.onnx",
        "half": False,
        "batch": 8,
        "imgsz": 640,
    },

    # REMOVED BROKEN ONNX FP16

    {
        "name": "TensorRT_FP16_B8",
        "model": "models/best.engine",
        "half": False,
        "batch": 8,
        "imgsz": 640,
    },
]

# ============================================
# BENCHMARK
# ============================================

all_results = []

for exp in EXPERIMENTS:

    print("\n" + "=" * 60)
    print(f"Running Experiment: {exp['name']}")
    print("=" * 60)

    run_results = []

    try:

        for run in range(3):

            print(f"\nRun {run + 1}/3")

            # ----------------------------------------
            # LOAD MODEL
            # ----------------------------------------

            model = YOLO(exp["model"], task="detect")

            # ----------------------------------------
            # OPEN VIDEO
            # ----------------------------------------

            cap = cv2.VideoCapture(VIDEO_PATH)

            if not cap.isOpened():
                print("ERROR: Cannot open video")
                continue

            # ----------------------------------------
            # WARMUP
            # ----------------------------------------

            ret, warmup_frame = cap.read()

            if not ret:
                print("ERROR: Cannot read video frame")
                continue

            dummy_batch = [warmup_frame] * exp["batch"]

            print("Warmup...")

            for _ in range(20):

                _ = model.predict(
                    dummy_batch,
                    imgsz=exp["imgsz"],
                    device=0,
                    half=exp["half"],
                    verbose=False
                )

            torch.cuda.synchronize()

            # Reset video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # ----------------------------------------
            # RESET GPU STATS
            # ----------------------------------------

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # ----------------------------------------
            # BENCHMARK START
            # ----------------------------------------

            frames = []
            frame_count = 0

            start_time = time.time()

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frames.append(frame)

                if len(frames) == exp["batch"]:

                    _ = model.predict(
                        frames,
                        imgsz=exp["imgsz"],
                        device=0,
                        half=exp["half"],
                        verbose=False
                    )

                    frame_count += len(frames)

                    frames = []

            # ----------------------------------------
            # LEFTOVER FRAMES
            # ----------------------------------------

            if len(frames) > 0:

                _ = model.predict(
                    frames,
                    imgsz=exp["imgsz"],
                    device=0,
                    half=exp["half"],
                    verbose=False
                )

                frame_count += len(frames)

            # ----------------------------------------
            # CUDA SYNC
            # ----------------------------------------

            torch.cuda.synchronize()

            end_time = time.time()

            cap.release()

            # ----------------------------------------
            # METRICS
            # ----------------------------------------

            total_time = end_time - start_time

            fps = frame_count / total_time

            latency = (total_time / frame_count) * 1000

            throughput = frame_count / total_time

            vram = torch.cuda.max_memory_allocated() / 1024**2

            gpu = GPUtil.getGPUs()[0]

            gpu_util = gpu.load * 100

            print(f"Frames Processed : {frame_count}")
            print(f"Total Time       : {total_time:.2f} sec")
            print(f"FPS              : {fps:.2f}")
            print(f"Latency          : {latency:.2f} ms")
            print(f"Throughput       : {throughput:.2f} img/sec")
            print(f"VRAM Usage       : {vram:.2f} MB")
            print(f"GPU Utilization  : {gpu_util:.2f}%")

            run_results.append({
                "FPS": fps,
                "Latency": latency,
                "Throughput": throughput,
                "VRAM": vram,
                "GPU_Util": gpu_util
            })

        # ============================================
        # AVERAGE RESULTS
        # ============================================

        if len(run_results) == 0:
            continue

        avg_fps = sum(r["FPS"] for r in run_results) / len(run_results)

        avg_latency = sum(r["Latency"] for r in run_results) / len(run_results)

        avg_throughput = sum(r["Throughput"] for r in run_results) / len(run_results)

        avg_vram = sum(r["VRAM"] for r in run_results) / len(run_results)

        avg_gpu = sum(r["GPU_Util"] for r in run_results) / len(run_results)

        all_results.append({
            "Experiment": exp["name"],
            "Batch": exp["batch"],
            "Image_Size": exp["imgsz"],
            "FPS": round(avg_fps, 2),
            "Latency_ms": round(avg_latency, 2),
            "Throughput_img_sec": round(avg_throughput, 2),
            "VRAM_MB": round(avg_vram, 2),
            "GPU_Utilization_%": round(avg_gpu, 2)
        })

        # ============================================
        # SAVE AFTER EVERY EXPERIMENT
        # ============================================

        temp_df = pd.DataFrame(all_results)

        temp_df.to_csv(
            "results/benchmark_results.csv",
            index=False
        )

        print("\nIntermediate results saved.")

    except Exception as e:

        print(f"\nERROR in {exp['name']}")
        print(e)

        continue

# ============================================
# FINAL SAVE
# ============================================

df = pd.DataFrame(all_results)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(df)

df.to_csv(
    "results/benchmark_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/benchmark_results.csv")

