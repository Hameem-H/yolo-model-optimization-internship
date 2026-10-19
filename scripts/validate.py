
from ultralytics import YOLO
import pandas as pd

MODELS = [
    ("PyTorch", "models/best.pt"),
    ("ONNX", "models/best.onnx"),
    ("TensorRT", "models/best.engine"),
]


def run_validation():

    all_results = []

    for name, path in MODELS:

        print("\n" + "=" * 60)
        print(f"Validating: {name}")
        print("=" * 60)

        model = YOLO(path)

        metrics = model.val(
            data="dataset/data.yaml",
            imgsz=640,
            batch=8,
            device=0,
            half=True,
            workers=0
        )

        map50 = metrics.box.map50
        map5095 = metrics.box.map
        precision = metrics.box.mp
        recall = metrics.box.mr

        print(f"\n{name} Results")
        print(f"mAP50      : {map50:.4f}")
        print(f"mAP50-95   : {map5095:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")

        all_results.append({
            "Model": name,
            "mAP50": round(map50, 4),
            "mAP50-95": round(map5095, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4)
        })

    # ============================================
    # SAVE CSV
    # ============================================

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    print(df)

    df.to_csv("results/validation_results.csv", index=False)

    print("\nValidation results saved to:")
    print("results/validation_results.csv")


if __name__ == "__main__":
    run_validation()

