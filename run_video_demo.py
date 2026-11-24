import cv2
import time
from pathlib import Path
from modules.detector import HelmetDetector

# Load sample video
video_path = Path(r'C:/Users/krish/Downloads/AI-DL-project/data/samples/sample_traffic_video.mp4')
if not video_path.exists():
    raise FileNotFoundError(f'Video not found: {video_path}')

cap = cv2.VideoCapture(str(video_path))
fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f'Video: {video_path.name}, {fps_in:.2f} fps, {w}x{h}')

detector = HelmetDetector()
frame_idx = 0
total_helm = 0
total_viol = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    # process every 3rd frame like server implementation
    if frame_idx % 3 != 0 and frame_idx != 1:
        continue
    # convert BGR to RGB PIL Image
    from PIL import Image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    _, detections, stats = detector.detect_image(pil_img, conf_thresh=0.4, iou_thresh=0.45, save_violations=False)
    total_helm += stats.get('helmets', 0)
    total_viol += stats.get('without_helmets', 0)
    # optional: print per-frame summary
    # print(f'Frame {frame_idx}: helmets={stats.get("helmets")}, violations={stats.get("without_helmets")}')

cap.release()
print('--- Summary ---')
print(f'Total frames processed: {frame_idx}')
print(f'Total helmets detected: {total_helm}')
print(f'Total violations (no helmet): {total_viol}')
