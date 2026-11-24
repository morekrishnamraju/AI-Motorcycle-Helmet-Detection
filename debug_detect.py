import sys, os
sys.path.append(os.path.abspath('.'))
from modules.detector import HelmetDetector
from PIL import Image

# Load a sample image (one of the existing samples)
sample_path = os.path.join('data','samples','sample_helmet_violation.jpg')
pil_img = Image.open(sample_path).convert('RGB')

detector = HelmetDetector()
annotated, dets, stats = detector.detect_image(pil_img)
print('Detections:', dets)
print('Stats:', stats)
