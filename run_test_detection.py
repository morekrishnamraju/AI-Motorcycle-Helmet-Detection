import json
from pathlib import Path
from modules.detector import HelmetDetector

# Path to the test image
image_path = Path(r"C:/Users/krish/.gemini/antigravity/brain/5037d7ba-28be-45e6-9681-6f2bba3c8101/.user_uploaded/media_1795158156596_5b7a4ca1.jpg")

# Initialize detector (loads models)
detector = HelmetDetector()

# Primary model raw predictions
raw_primary = []
results_primary = detector.model.predict(source=str(image_path), conf=0.4, iou=0.45, verbose=False)
if results_primary:
    for box in results_primary[0].boxes:
        raw_primary.append({
            "cls_id": int(box.cls[0].item()),
            "conf": float(box.conf[0].item()),
            "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist()
        })

# Secondary model raw predictions (if available)
raw_secondary = []
if detector.secondary_model:
    results_sec = detector.secondary_model.predict(source=str(image_path), conf=0.4, iou=0.45, verbose=False)
    for box in results_sec[0].boxes:
        raw_secondary.append({
            "cls_id": int(box.cls[0].item()),
            "conf": float(box.conf[0].item()),
            "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist()
        })

# Final detection after deduplication and association logic inside detector
_, detections, stats = detector.detect_image(str(image_path), conf_thresh=0.4, iou_thresh=0.45, save_violations=False)

# Prepare output (exclude non‑serializable crops)
output = {
    "raw_primary": raw_primary,
    "raw_secondary": raw_secondary,
    "final_detections": detections,
    "summary_stats": {k: v for k, v in stats.items() if k != "crops"}
}
print(json.dumps(output, indent=2))
