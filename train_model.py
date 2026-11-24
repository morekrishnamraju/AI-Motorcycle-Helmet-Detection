"""
AI-Based Motorcycle Helmet Detection and Rider Safety Monitoring System
Model Training Script (Compatible with Local GPU / Google Colab / Kaggle)
"""

import os
import sys
from pathlib import Path

# Check ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("[!] Error: 'ultralytics' is not installed. Run: pip install ultralytics")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_helmet_model(data_yaml_path="data.yaml", epochs=50, imgsz=640, batch=16, base_model="yolov8n.pt"):
    """
    Trains a YOLOv8 model for Motorcycle Helmet Detection.
    Args:
        data_yaml_path: Path to dataset configuration YAML
        epochs: Number of training epochs (recommended: 50-100)
        imgsz: Input image resolution (recommended: 640)
        batch: Batch size (recommended: 16 on 16GB GPU, 8 on 4-6GB GPU)
        base_model: Pre-trained backbone (yolov8n.pt or yolov8s.pt)
    """
    print("=" * 75)
    print("  AI-Based Motorcycle Helmet Detection: YOLOv8 Model Training")
    print("=" * 75)

    if not Path(data_yaml_path).exists():
        print(f"[!] Dataset config '{data_yaml_path}' not found.")
        print("[!] Generating sample data.yaml template...")
        create_sample_yaml(data_yaml_path)

    print(f"[*] Initializing base model: {base_model}")
    model = YOLO(base_model)

    print(f"[*] Starting training for {epochs} epochs at {imgsz}x{imgsz} resolution...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        save=True,
        plots=True,
        project="helmet_training_runs",
        name="helmet_yolov8n_experiment"
    )

    print("\n[✓] Training complete!")
    best_weights = Path("helmet_training_runs/helmet_yolov8n_experiment/weights/best.pt")
    target_weights = MODELS_DIR / "helmet_yolov8n.pt"

    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"[✓] Best weights copied to application models directory: {target_weights}")


def create_sample_yaml(output_path="data.yaml"):
    """Generates standard data.yaml template for dataset splits"""
    content = """# Dataset configuration for Motorcycle Helmet Detection
path: ./dataset  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images    # val images (relative to 'path')
test: test/images    # test images (optional)

# Classes
nc: 4
names: ['Helmet', 'Without Helmet', 'Motorcycle', 'Rider']
"""
    with open(output_path, "w") as f:
        f.write(content.strip())
    print(f"[✓] Created '{output_path}'. Update directory paths before starting training.")


if __name__ == "__main__":
    train_helmet_model()
