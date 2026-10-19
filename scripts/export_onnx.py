from ultralytics import YOLO

model = YOLO("../models/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    dynamic=True,
    simplify=True,
    batch=8
)

print("ONNX export completed")