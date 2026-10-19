# Deployment-Aware Comparative Evaluation of YOLO11m Inference Frameworks

This research investigates how the choice of inference framework affects the speed and resource usage of a trained YOLO11m object-detection model. The study compares four deployment configurations on the same NVIDIA GPU to understand which approach is most suitable for real-time computer vision applications.

## Research Motivation

Object-detection models are used in applications such as intelligent transportation, surveillance, robotics, and industrial automation. In these applications, detection accuracy is important, but the time required to process each image and the hardware resources consumed during inference also affect practical deployment.

A trained model can be executed through different frameworks and numerical precision modes. This study evaluates these deployment choices under a common experimental setup, allowing their inference performance to be compared directly.

## Research Objective

The main objective is to compare the inference performance of four YOLO11m deployment configurations while keeping the trained model and experimental environment consistent. The research examines detection quality alongside throughput, latency, GPU memory consumption, and GPU utilization.

## Configurations Compared

- **PyTorch FP32:** The baseline implementation using 32-bit floating-point precision.
- **PyTorch FP16:** PyTorch inference using 16-bit floating-point precision.
- **ONNX Runtime GPU:** Inference using an exported ONNX model with GPU execution.
- **TensorRT FP16:** Inference using a TensorRT engine with 16-bit floating-point precision.

All configurations use the same trained YOLO11m checkpoint, `best.pt`. The study focuses on inference and deployment rather than training a new model.

## Dataset

The experiments use the **Nogor Bahon** urban vehicle and road-scene object-detection dataset.

| Dataset property | Details |
|---|---|
| Task | Object detection |
| Total images | 6,950 |
| Classes | 24 |
| Annotated objects | 20,801 |
| Training images | 5,654 |
| Validation images | 864 |
| Test images | 432 |
| Image resolution | 640 × 640 pixels |
| Annotation format | YOLO |

The validation split is used to assess detection performance. A 627-frame video is used for the inference benchmark.

## Experimental Environment

The experiments were conducted on a workstation with an AMD Ryzen 7 5700X CPU, 16 GB DDR4 RAM, and an NVIDIA GeForce RTX 4060 GPU with 8 GB of memory. The software environment includes Ubuntu 22.04 LTS, Python 3.10, PyTorch 2.5.1, CUDA Toolkit 12.1, ONNX Runtime GPU 1.24.4, TensorRT 11.0, and Ultralytics YOLO 8.3 or later.

The benchmark uses a 640 × 640 input resolution and a batch size of 8. GPU warm-up and CUDA synchronization are enabled. Each benchmark is run three times, and the reported deployment measurements are averages.

## Evaluation Metrics

Detection quality is evaluated using:

- **Precision:** The proportion of predicted detections that are correct.
- **Recall:** The proportion of relevant objects that the model successfully detects.
- **mAP@0.5:** Mean average precision at an intersection-over-union threshold of 0.5.
- **mAP@0.5:0.95:** Mean average precision averaged across intersection-over-union thresholds from 0.5 to 0.95.
- **F1-score:** The harmonic mean of precision and recall.

Deployment performance is evaluated using:

- **Throughput (FPS):** The number of frames processed per second.
- **Inference latency:** The reported time associated with processing an image.
- **GPU memory consumption:** The amount of GPU memory used during inference.
- **GPU utilization:** The measured percentage of GPU activity during the benchmark.

## Main Results

The deployment benchmark produced the following average results:

| Configuration | Throughput | Reported latency | GPU memory | Average GPU utilization |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 90.27 FPS | 11.08 ms | 918.75 MB | 56.67% |
| PyTorch FP16 | 119.87 FPS | 8.34 ms | 377.75 MB | 39.33% |
| ONNX Runtime GPU | 80.69 FPS | 12.39 ms | 252.34 MB | 64.00% |
| TensorRT FP16 | 138.16 FPS | 7.24 ms | 430.78 MB | 37.67% |

TensorRT FP16 achieved the highest throughput and lowest reported latency in this benchmark. Its throughput was approximately 53.1% higher than the PyTorch FP32 baseline. PyTorch FP16 also improved performance over the FP32 baseline, reaching 119.87 FPS while using substantially less GPU memory. ONNX Runtime GPU recorded the lowest GPU memory consumption among the four configurations, while its throughput was the lowest in this particular experiment.

These results illustrate that inference framework selection involves a trade-off between processing speed and resource consumption. The most suitable configuration depends on the requirements of the intended deployment environment.

## Research Contribution

The study provides a controlled comparison of four commonly used YOLO inference configurations on the same model, dataset context, hardware platform, and input settings. By examining detection metrics together with deployment measurements, it offers a practical basis for choosing an inference framework for GPU-based real-time object detection.

## Repository Contents

The repository contains the benchmark results, validation results, and figures used to present the experimental comparison. The figures visualize detection metrics and the relationships between throughput, latency, GPU memory consumption, and GPU utilization.
