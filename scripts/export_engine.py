from ultralytics import YOLO

model = YOLO("../models/best.pt")

model.export(
    format="engine",
    imgsz=640,
    device=0,
    half=True,
    dynamic=True,
    batch=8,
    simplify=True
)

print("TensorRT engine export completed")