# detect_media_v2.py
"""
Enhanced detection script for image & video:
- Image: count each helmet and each rider‑without‑helmet once.
- Video: track unique riders across frames using simple IoU‑based association.
  * Maintains a dict of rider IDs -> {'bbox':..., 'has_helmet':bool}
  * Updates counts while processing frames.
- Output displays only Helmet Count and Violations Count.

The existing trained YOLO model (HelmetDetector) is reused unchanged.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, List

import cv2
from PIL import Image

from modules.detector import HelmetDetector

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------

def iou(boxA: List[int], boxB: List[int]) -> float:
    """Intersection‑over‑Union for two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


def associate_riders(prev_riders: Dict[int, Dict], cur_riders: List[Dict]) -> Tuple[Dict[int, Dict], List[int]]:
    """Match current rider detections to previously seen riders.
    Returns updated rider dict and a list of newly‑seen rider IDs.
    """
    # Each rider dict: {'bbox': [...], 'has_helmet': bool}
    next_id = max(prev_riders.keys(), default=0) + 1
    new_ids = []
    for cur in cur_riders:
        matched = None
        for rid, data in prev_riders.items():
            if iou(cur['bbox'], data['bbox']) > 0.3:  # generous overlap
                matched = rid
                break
        if matched is not None:
            # Update bbox (average) and keep helmet flag if already true
            prev = prev_riders[matched]
            # simple bbox averaging (optional, keep first for stability)
            prev_riders[matched]['bbox'] = cur['bbox']
            # helmet flag will be updated by the caller after association
        else:
            # New rider
            prev_riders[next_id] = {'bbox': cur['bbox'], 'has_helmet': False}
            new_ids.append(next_id)
            next_id += 1
    return prev_riders, new_ids

# ---------------------------------------------------------------------
# Image processing (counts once per detection)
# ---------------------------------------------------------------------

def process_image(detector: HelmetDetector, img_path: Path) -> Tuple[int, int]:
    """Detect on a single image and return the *summary* helmet/violation counts.
    The underlying detector already computes correct stats, so we simply reuse
    ``stats['helmets']`` and ``stats['without_helmets']`` which count each rider
    once regardless of multiple detections.
    """
    pil_img = Image.open(img_path).convert('RGB')
    _, _, stats = detector.detect_image(
        pil_img, conf_thresh=0.4, iou_thresh=0.45, save_violations=False
    )
    helmets = int(stats.get('helmets', 0))
    violations = int(stats.get('without_helmets', 0))
    return helmets, violations
    """Detect on a single image and count each helmet and each rider‑without‑helmet once.
    The detector may return multiple detections; we simply count class occurrences.
    """
    pil_img = Image.open(img_path).convert('RGB')
    _, detections, _ = detector.detect_image(
        pil_img, conf_thresh=0.4, iou_thresh=0.45, save_violations=False
    )
    # Count unique helmets and violations directly from detections
    helmets = sum(1 for d in detections if d.get('class') == 'Helmet')
    violations = sum(1 for d in detections if d.get('class') == 'Without Helmet')
    return helmets, violations
    pil_img = Image.open(img_path).convert('RGB')
    _, _, stats = detector.detect_image(pil_img, conf_thresh=0.4, iou_thresh=0.45, save_violations=False)
    helmets = stats.get('helmets', 0)
    violations = stats.get('without_helmets', 0)
    return helmets, violations

# ---------------------------------------------------------------------
# Video processing with simple rider tracking
# ---------------------------------------------------------------------

def process_video(detector: HelmetDetector, vid_path: Path) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video {vid_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f'▶️  Video {vid_path.name}: {fps:.1f} fps, {total_frames} frames')

    rider_db: Dict[int, Dict] = {}  # id -> {'bbox': [...], 'has_helmet': bool}
    frame_idx = 0

    while True:
    start_loop = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    # Process first frame and then every 3rd frame (mirrors API)
    if frame_idx % 3 != 0 and frame_idx != 1:
        continue
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    _, detections, _ = detector.detect_image(
        pil_img, conf_thresh=0.4, iou_thresh=0.45, save_violations=False
    )
    # ... existing processing ...
    # Timing for this processed frame
    elapsed = time.time() - start_loop
    total_time += elapsed
    processed_frames += 1
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # Process first frame and then every 3rd frame (mirrors API)
        if frame_idx % 3 != 0 and frame_idx != 1:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, detections, _ = detector.detect_image(
            pil_img, conf_thresh=0.4, iou_thresh=0.45, save_violations=False
        )

        # Separate rider and helmet detections
        cur_riders = [d for d in detections if d['class'] == 'Rider']
        cur_helmets = [d for d in detections if d['class'] == 'Helmet']

        # Associate current riders with existing IDs
        rider_db, _ = associate_riders(rider_db, cur_riders)

        # For each rider, check overlap with any helmet detection
        for rid, data in rider_db.items():
            rider_box = data['bbox']
            # If already marked as having a helmet, keep it
            if data['has_helmet']:
                continue
            for helm in cur_helmets:
                if iou(rider_box, helm['bbox']) > 0.3:
                    rider_db[rid]['has_helmet'] = True
                    break

        # OPTIONAL: live display of current totals while video plays
        helmets_cnt = sum(1 for r in rider_db.values() if r['has_helmet'])
        violations_cnt = sum(1 for r in rider_db.values() if not r['has_helmet'])
        print(f'Frame {frame_idx:03d}: Helmets={helmets_cnt}, Violations={violations_cnt}', end='\r')

    cap.release()
    # Final counts after video finished
    helmets = sum(1 for r in rider_db.values() if r['has_helmet'])
    violations = sum(1 for r in rider_db.values() if not r['has_helmet'])
    return helmets, violations

# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python detect_media_v2.py <path-to-image-or-video>')
        sys.exit(1)
    media_path = Path(sys.argv[1])
    if not media_path.exists():
        print(f'File not found: {media_path}')
        sys.exit(1)

    detector = HelmetDetector()  # re‑use existing trained model

    if media_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
        helmets, violations = process_image(detector, media_path)
        print('\n--- Image Result ---')
    else:
        helmets, violations = process_video(detector, media_path)
        print('\n--- Video Result ---')

    print(f'Helmet Count   : {helmets}')
    print(f'Violations Count: {violations}')

if __name__ == '__main__':
    main()
