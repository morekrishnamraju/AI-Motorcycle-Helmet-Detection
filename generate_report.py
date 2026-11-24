import json
from pathlib import Path
from modules.detector import HelmetDetector
from modules.report_generator import ReportGenerator

# test image path (same as run_test_detection)
image_path = Path(r"C:/Users/krish/.gemini/antigravity/brain/5037d7ba-28be-45e6-9681-6f2bba3c8101/.user_uploaded/media_1795158156596_5b7a4ca1.jpg")

detector = HelmetDetector()
# Run detection
annotated, detections, stats = detector.detect_image(str(image_path), conf_thresh=0.4, iou_thresh=0.45, save_violations=False)
# Generate PDF report
pdf_path, pdf_bytes = ReportGenerator.generate_pdf_report(detections, stats, source_name=str(image_path.name))
print(f"PDF generated: {pdf_path}")
# Optionally write bytes (already saved)
