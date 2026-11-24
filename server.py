"""
AI Helmet Detection - Flask Web Server
Runs on http://localhost:5000
"""
import os, sys, io, time, base64, csv, json, threading
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.detector import HelmetDetector
from modules.analytics import AnalyticsEngine
from modules.report_generator import ReportGenerator
from modules.config import (
    MODELS_DIR, SAMPLES_DIR, OUTPUTS_DIR, VIOLATIONS_DIR,
    REPORTS_DIR, LOGS_DIR, DEFAULT_MODEL_NAME, MODEL_PATH
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # 64 MB upload limit

# ── Singletons (loaded once) ──────────────────────────────────────────────────
print("[SERVER] Loading AI model …")
detector   = HelmetDetector()
analytics  = AnalyticsEngine()
report_gen = ReportGenerator()
print("[SERVER] Model ready →", detector.model_mode)

# ── Webcam streaming state ────────────────────────────────────────────────────
_cam_lock   = threading.Lock()
_cam_active = False
_cam_cap    = None
_cam_stats  = {"helmets": 0, "without_helmets": 0, "compliance_rate": 100.0,
               "fps": 0.0, "frames": 0}

# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           model_mode=detector.model_mode,
                           model_name=DEFAULT_MODEL_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# API: IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/detect/image", methods=["POST"])
def detect_image():
    conf  = float(request.form.get("conf",  0.40))
    iou   = float(request.form.get("iou",   0.45))
    file  = request.files.get("image")

    if file is None:
        # sample name passed instead of file
        sample = request.form.get("sample", "")
        sample_path = SAMPLES_DIR / sample
        if not sample_path.exists():
            return jsonify({"error": f"Sample '{sample}' not found"}), 400
        pil_img = Image.open(sample_path).convert("RGB")
    else:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Measure inference time for image detection
    import time as _time_img
    _start_img = _time_img.time()
    annotated_rgb, detections, stats = detector.detect_image(
        pil_img, conf_thresh=conf, iou_thresh=iou, save_violations=True
    )
    _inference_ms = (_time_img.time() - _start_img) * 1000
    # Compute image performance metrics
    total_helmets_img = stats.get("helmets", 0)
    total_violations_img = stats.get("without_helmets", 0)
    total_detections_img = total_helmets_img + total_violations_img
    model_size_bytes_img = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    performance_img = {
        "inference_time_ms": round(_inference_ms, 2),
        "total_detections": total_detections_img,
        "helmet_count": total_helmets_img,
        "no_helmet_count": total_violations_img,
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "model_size_bytes": model_size_bytes_img
    }
    # Add friendly alias for front‑end JavaScript and ensure no colour field (image API)
    for d in detections:
        d["class_name"] = d.get("class")
        if "color" in d:
            del d["color"]
    analytics.log_detections(detections, source_type="image",
                             source_name=file.filename if file else "sample")

    # encode result image to base64
    result_pil = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Prepare stats without non-serializable crops
    stats_no_crops = {k: v for k, v in stats.items() if k != "crops"}
    if not detections:
        # No detections: return minimal info with performance metrics
        performance = {
            "inference_time_ms": round(_inference_ms, 2),
            "total_detections": 0,
            "helmet_count": 0,
            "no_helmet_count": 0,
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "model_size_bytes": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        }
        return jsonify({
            "error": "No detections found.",
            "image_b64": "data:image/jpeg;base64," + img_b64,
            "stats": stats_no_crops,
            "performance": performance
        }), 200

    # Add friendly alias for front‑end JavaScript
    for d in detections:
        d["class_name"] = d.get("class")
        # Remove colour field for image API (only needed for webcam UI)
        if "color" in d:
            del d["color"]
    return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })



# ─────────────────────────────────────────────────────────────────────────────
# API: VIDEO DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/detect/video", methods=["POST"])
def detect_video():
    import time, os
    from modules.config import MODEL_PATH
    start_total = time.time()
    processed_frames = 0
    total_proc_time = 0.0
    import time
    from modules.config import MODEL_PATH
    start_total = time.time()
    processed_frames = 0
    total_proc_time = 0.0
    conf   = float(request.form.get("conf",  0.40))
    iou    = float(request.form.get("iou",   0.45))
    file   = request.files.get("video")
    sample = request.form.get("sample", "")

    if file:
        tmp_path = OUTPUTS_DIR / f"tmp_upload_{int(time.time())}.mp4"
        file.save(str(tmp_path))
        vid_path = tmp_path
    elif sample:
        vid_path = SAMPLES_DIR / sample
        if not vid_path.exists():
            return jsonify({"error": "Sample video not found"}), 400
    else:
        return jsonify({"error": "No video provided"}), 400

    cap = cv2.VideoCapture(str(vid_path))
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = OUTPUTS_DIR / f"analyzed_{vid_path.stem}_{int(time.time())}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w, h))

    all_stats   = []
    frame_idx   = 0
    total_helm  = 0
    total_viol  = 0

    all_detections = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 3 != 0 and frame_idx != 1:
            writer.write(frame)
            continue

        # Start timer for inference on this frame
        frame_start_time = time.time()
        annotated, dets, stats = detector.detect_image(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), conf_thresh=conf, iou_thresh=iou, save_violations=False
        )
        # Record inference time
        proc_time = time.time() - frame_start_time
        total_proc_time += proc_time
        processed_frames += 1

        # collect detections for the response (clean dicts)
        clean_dets = [
            {
                "class": d["class"],
                "bbox": d["bbox"],
                "confidence": round(d["confidence"], 4),
                "is_violation": d.get("is_violation", False),
                "class_name": d["class"]
            }
            for d in dets
            if not d.get("color")
        ]
        all_detections.extend(clean_dets)
        bgr_out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(bgr_out)
        total_helm += stats.get("helmets", 0)
        total_viol += stats.get("without_helmets", 0)
        all_stats.append({
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps_in, 2),
            **stats
        })
        analytics.log_detections(dets, source_type="video", source_name=vid_path.name)

        # --- Performance timing for each processed frame ---
        # Only time frames that go through detection (skip every 3rd frame)
        frame_start_time = None
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 3 != 0 and frame_idx != 1:
            writer.write(frame)
            continue

        # Start timer for inference on this frame
        frame_start_time = time.time()
        annotated, dets, stats = detector.detect_image(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), conf_thresh=conf, iou_thresh=iou, save_violations=False
        )
        # Record inference time
        proc_time = time.time() - frame_start_time
        total_proc_time += proc_time
        processed_frames += 1
        # --- Performance timing for each processed frame ---
        frame_start_time = None  # will be set when we actually run detection
        # Start timer for this frame (only when we actually run detection)
        frame_start_time = None

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 3 != 0 and frame_idx != 1:
            writer.write(frame)
            continue

        annotated, dets, stats = detector.detect_image(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), conf_thresh=conf, iou_thresh=iou, save_violations=False
        )
        # collect detections for the response (clean dicts)
        clean_dets = [
            {
                "class": d["class"],
                "bbox": d["bbox"],
                "confidence": round(d["confidence"], 4),
                "is_violation": d.get("is_violation", False),
                "class_name": d["class"]
            }
            for d in dets
            if not d.get("color")  # ensure no colour field in video API
        ]
        all_detections.extend(clean_dets)
        bgr_out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(bgr_out)
        total_helm += stats.get("helmets", 0)
        total_viol += stats.get("without_helmets", 0)
        all_stats.append({
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps_in, 2),
            **stats
        })
        analytics.log_detections(dets, source_type="video",
                                 source_name=vid_path.name)


    cap.release()
    writer.release()
    if file and tmp_path.exists():
        tmp_path.unlink()

    # Compute performance metrics
    avg_fps = processed_frames / total_proc_time if total_proc_time > 0 else 0
    avg_inference_ms = (total_proc_time / processed_frames * 1000) if processed_frames > 0 else 0
    model_size_bytes = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    performance = {
        "average_fps": round(avg_fps, 2),
        "average_inference_ms": round(avg_inference_ms, 2),
        "processed_frames": processed_frames,
        "total_frames": frame_idx,
        "total_helmets": total_helm,
        "total_violations": total_viol,
        "total_detections": total_helm + total_viol,
        "model_size_bytes": model_size_bytes,
        "confidence_threshold": conf,
        "iou_threshold": iou
    }

    chart_data = [{"t": s["time_sec"], "comp": s.get("compliance_rate", 100)} for s in all_stats]

    
    # After processing video, include a concise detections summary for the client
    detections_summary = {
        "total_helmets": total_helm,
        "total_violations": total_viol,
        "avg_compliance": round(
            sum(s.get("compliance_rate", 100) for s in all_stats) / max(len(all_stats), 1), 1
        ),
        "frame_count": frame_idx
    }

    # Build a list of per‑frame detection summaries (optional for UI)
    frame_summaries = []
    for i, s in enumerate(all_stats, start=1):
        frame_summaries.append({
            "frame": i,
            "helmets": s.get("helmets", 0),
            "violations": s.get("without_helmets", 0),
            "compliance": s.get("compliance_rate", 100)
        })

    # After processing video, include a concise detections summary for the client
    detections_summary = {
        "total_helmets": total_helm,
        "total_violations": total_viol,
        "avg_compliance": round(
            sum(s.get("compliance_rate", 100) for s in all_stats) / max(len(all_stats), 1), 1
        ),
        "frame_count": frame_idx
    }

    # Build a list of per‑frame detection summaries (optional for UI)
    frame_summaries = []
    for i, s in enumerate(all_stats, start=1):
        frame_summaries.append({
            "frame": i,
            "helmets": s.get("helmets", 0),
            "violations": s.get("without_helmets", 0),
            "compliance": s.get("compliance_rate", 100)
        })

    # Guard: if no detections at all, return a helpful message with performance
    if total_helm == 0 and total_viol == 0:
        return jsonify({
            "error": "No detections found in the video.",
            "out_video": out_path.name,
            "total_frames": frame_idx,
            "total_helmets": total_helm,
            "total_violations": total_viol,
            "performance": performance
        }), 200

    # Successful video processing response
    return jsonify({
        "out_video": out_path.name,
        "total_frames": frame_idx,
        "total_helmets": total_helm,
        "total_violations": total_viol,
        "avg_compliance": round(
            sum(s.get("compliance_rate", 100) for s in all_stats) / max(len(all_stats), 1), 1),
        "chart_data": chart_data,
        "detections_summary": detections_summary,
        "frame_summaries": frame_summaries,
        "all_detections": all_detections,
        "performance": performance
    })


# ─────────────────────────────────────────────────────────────────────────────
# API: WEBCAM FRAME (single-frame POST from browser)
# ─────────────────────────────────────────────────────────────────────────────



    """Receive a base64 JPEG frame from the browser webcam, run detection."""
    data = request.get_json(force=True)
    b64 = data.get("frame", "")
    conf = float(data.get("conf", 0.40))
    iou = float(data.get("iou", 0.45))

    if not b64:
        return jsonify({"error": "No frame data"}), 400

    # decode base64 → PIL
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Measure inference time for frame detection
    import time as _time_f
    _start_f = _time_f.time()
    annotated_rgb, detections, stats = detector.detect_image(
        pil_img, conf_thresh=conf, iou_thresh=iou, save_violations=False
    )
    _inference_ms = (_time_f.time() - _start_f) * 1000
    performance = {
        "inference_time_ms": round(_inference_ms, 2),
        "total_detections": stats.get("helmets", 0) + stats.get("without_helmets", 0),
        "helmet_count": stats.get("helmets", 0),
        "no_helmet_count": stats.get("without_helmets", 0),
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "model_size_bytes": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    }
    analytics.log_detections(detections, source_type="webcam", source_name="live_camera")

    result_pil = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Prepare stats without non-serializable crops
    stats_no_crops = {k: v for k, v in stats.items() if k != "crops"}
    if not detections:
        return jsonify({
            "error": "No detections found.",
            "image_b64": "data:image/jpeg;base64," + img_b64,
            "stats": stats_no_crops,
            "performance": performance
        }), 200

    # Add friendly alias for front‑end JavaScript
    for d in detections:
        d["class_name"] = d.get("class")
        if "color" in d:
            del d["color"]
    return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance
    })

    # decode base64 → PIL
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    annotated_rgb, detections, stats = detector.detect_image(
        pil_img, conf_thresh=conf, iou_thresh=iou, save_violations=False
    )
    analytics.log_detections(detections, source_type="webcam",
                             source_name="live_camera")

    result_pil = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Prepare stats without non-serializable crops
    stats_no_crops = {k: v for k, v in stats.items() if k != "crops"}
    if not detections:
        # No detections: return minimal info with performance metrics
        performance = {
            "inference_time_ms": round(_inference_ms, 2),
            "total_detections": 0,
            "helmet_count": 0,
            "no_helmet_count": 0,
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "model_size_bytes": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        }
        return jsonify({
            "error": "No detections found.",
            "image_b64": "data:image/jpeg;base64," + img_b64,
            "stats": stats_no_crops,
            "performance": performance
        }), 200

    # Add friendly alias for front‑end JavaScript
    for d in detections:
        d["class_name"] = d.get("class")
        # Remove colour field for image API (only needed for webcam UI)
        if "color" in d:
            del d["color"]
    return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })



# ─────────────────────────────────────────────────────────────────────────────
# API: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/analytics")
def get_analytics():
    log_csv = LOGS_DIR / "detection_history.csv"
    if not log_csv.exists():
        return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })"rows": [], "summary": {}})

    rows = []
    with open(log_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_helmets    = sum(int(r.get("helmets", 0)) for r in rows)
    total_violations = sum(int(r.get("without_helmets", 0)) for r in rows)
    total_riders     = sum(int(r.get("riders", 0)) for r in rows)
    avg_comp = round(
        sum(float(r.get("compliance_rate", 100)) for r in rows) / max(len(rows), 1), 1)

    return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })
        "rows": rows[-200:],
        "summary": {
            "total_sessions": len(rows),
            "total_helmets": total_helmets,
            "total_violations": total_violations,
            "total_riders": total_riders,
            "avg_compliance": avg_comp
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# API: REPORTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/report/generate", methods=["POST"])
def generate_report():
    fmt   = request.json.get("format", "pdf")
    title = request.json.get("title",  "Helmet Safety Audit Report")
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_csv = LOGS_DIR / "detection_history.csv"
    rows = []
    if log_csv.exists():
        with open(log_csv, newline="") as f:
            rows = list(csv.DictReader(f))

    if fmt == "csv":
        out_path = REPORTS_DIR / f"helmet_safety_report_{ts}.csv"
        with open(out_path, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })"file": out_path.name, "path": str(out_path)})

    # PDF
    try:
        pdf_path = report_gen.generate_pdf_report(
            detections=rows, report_title=title,
            output_path=str(REPORTS_DIR / f"helmet_safety_audit_{ts}.pdf")
        )
        return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })"file": Path(pdf_path).name, "path": pdf_path})
    except Exception as e:
        return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })"error": str(e)}), 500


@app.route("/api/report/download/<filename>")
def download_report(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })"error": "File not found"}), 404
    return send_file(str(path), as_attachment=True)


@app.route("/api/video/download/<filename>")
def download_video(filename):
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })"error": "File not found"}), 404
    return send_file(str(path), as_attachment=True)


# ─────────────────────────────────────────────────────────────────────────────
# API: SYSTEM INFO
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/system")
def system_info():
    import torch
    models = [f.name for f in MODELS_DIR.glob("*.pt")]
    samples_img = [f.name for f in SAMPLES_DIR.glob("*.jpg")] + \
                  [f.name for f in SAMPLES_DIR.glob("*.png")]
    samples_vid = [f.name for f in SAMPLES_DIR.glob("*.mp4")]
    return jsonify({
        "image_b64": "data:image/jpeg;base64," + img_b64,
        "stats": stats_no_crops,
        "detections": detections,
        "performance": performance_img
    })
        "model_mode": detector.model_mode,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "models_available": models,
        "sample_images": samples_img,
        "sample_videos": samples_vid,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Helmet Detection System  —  Flask Server")
    print("  Open:  http://localhost:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
