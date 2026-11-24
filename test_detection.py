import sys, os
from pathlib import Path
import cv2
from PIL import Image
sys.path.append(str(Path(__file__).resolve().parent))
from modules.detector import HelmetDetector

# Load sample image uploaded by user
img_path = Path(r'C:/Users/krish/.gemini/antigravity/brain/5037d7ba-28be-45e6-9681-6f2bba3c8101/.user_uploaded/media_1795155614725_258ee69c.jpg')
if not img_path.exists():
    print('Image not found')
    sys.exit(1)
pil_img = Image.open(img_path).convert('RGB')

detector = HelmetDetector()
annotated_rgb, detections, stats = detector.detect_image(pil_img, conf_thresh=0.4, iou_thresh=0.45, save_violations=False)
# Add class_name alias for consistency
for d in detections:
    d["class_name"] = d.get("class")
print('Detections with class_name:', detections)
print('Stats:', stats)
# Show whether crops are numpy arrays
if isinstance(stats.get('crops'), list):
    print('Crops count:', len(stats['crops']))
    print('First crop type:', type(stats['crops'][0].get('crop')))
