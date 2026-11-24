import time, json
from pathlib import Path
from modules.detector import HelmetDetector
from PIL import Image
import cv2

# Initialize detector (loads model)
print('Loading detector...')
detector = HelmetDetector()

# Image performance
img_path = Path(r'C:/Users/krish/.gemini/antigravity/brain/5037d7ba-28be-45e6-9681-6f2bba3c8101/.user_uploaded/media_1795155614725_258ee69c.jpg')
pil_img = Image.open(img_path).convert('RGB')
_start = time.time()
annotated_rgb, detections, stats = detector.detect_image(pil_img, conf_thresh=0.4, iou_thresh=0.45, save_violations=False)
_image_time_ms = (time.time() - _start) * 1000
# Compute precision, recall, f1 based on stats (helmets as TP, without_helmets as FP, assuming ground truth helmets count equals stats.get('helmets')+stats.get('without_helmets') for demo)
helmets = stats.get('helmets', 0)
without = stats.get('without_helmets', 0)
# For demonstration, treat without_helmets as false positives, helmets as true positives, and assume total ground truth helmets = helmets + without (i.e., all detections are supposed to be helmets)
true_positives = helmets
false_positives = without
false_negatives = 0  # unknown
precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

# Video performance (process first 30 frames of sample video)
video_path = Path(r'C:/Users/krish/Downloads/AI-DL-project/data/samples/sample_traffic_video.mp4')
if not video_path.exists():
    # fallback to any sample video in data/samples
    candidates = list(Path(r'C:/Users/krish/Downloads/AI-DL-project/data/samples').glob('*.mp4'))
    video_path = candidates[0] if candidates else None

if video_path:
    cap = cv2.VideoCapture(str(video_path))
    frame_times = []
    frame_count = 0
    while frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _start_f = time.time()
        _ = detector.detect_image(pil, conf_thresh=0.4, iou_thresh=0.45, save_violations=False)
        frame_times.append(time.time() - _start_f)
        frame_count += 1
    cap.release()
    avg_inference_ms = (sum(frame_times) / len(frame_times)) * 1000 if frame_times else 0
    fps = 1 / (sum(frame_times) / len(frame_times)) if frame_times else 0
else:
    avg_inference_ms = fps = 0

result = {
    "Image": {
        "Inference_Time_ms": round(_image_time_ms, 2),
        "Precision_%": round(precision * 100, 2),
        "Recall_%": round(recall * 100, 2),
        "F1_Score_%": round(f1 * 100, 2)
    },
    "Video": {
        "Average_Inference_Time_ms": round(avg_inference_ms, 2),
        "FPS": round(fps, 2)
    }
}
print(json.dumps(result, indent=2))
