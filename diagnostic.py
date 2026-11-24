# diagnostic.py – step‑by‑step verification of the HelmetAI pipeline

import sys, os, subprocess, time, json, logging, traceback
from pathlib import Path

# ------------------------------------------------------------
# Setup logger (prints to console and also to a file)
# ------------------------------------------------------------
log_file = Path(__file__).parent / "diagnostic.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")],
)
log = logging.getLogger(__name__)

def ensure_server_running():
    """Check if Flask API is reachable; if not, start it as a background process.
    Returns the subprocess.Popen object (or None if already running)."""
    import socket
    def is_port_open(host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False
        finally:
            s.close()

    if is_port_open("127.0.0.1", 5000):
        log.info("Flask server already listening on http://127.0.0.1:5000")
        return None
    # start server via the existing entry point (server.py) using the virtual env python
    venv_py = Path(".venv/Scripts/python.exe")
    if not venv_py.exists():
        log.error("Virtual‑env python not found at %s", venv_py)
        return None
    log.info("Starting Flask server in background…")
    # Use CREATE_NEW_PROCESS_GROUP to allow later termination via .terminate()
    proc = subprocess.Popen([
        str(venv_py), "server.py"], cwd=Path.cwd(), creationflags=0x00000200
    )
    # wait a few seconds for it to bind
    for i in range(10):
        if is_port_open("127.0.0.1", 5000):
            log.info("Flask server is now up after %d sec", i+1)
            return proc
        time.sleep(1)
    log.error("Flask server failed to start within timeout")
    return proc

def test_direct_detector():
    """Load the detector class directly and run it on a sample image.
    This bypasses the Flask layer to verify model loading and inference."""
    try:
        from modules.detector import HelmetDetector
    except Exception as e:
        log.exception("Failed to import HelmetDetector: %s", e)
        return
    # Verify model paths from config
    from modules.config import MODEL_PATH, SECONDARY_MODEL_PATH, FALLBACK_MODEL_PATH
    log.info("Configured model paths: %s, %s, %s", MODEL_PATH, SECONDARY_MODEL_PATH, FALLBACK_MODEL_PATH)
    # Initialise detector (should load best available model)
    try:
        detector = HelmetDetector()
        log.info("Detector initialised – model_mode: %s", detector.model_mode)
    except Exception as e:
        log.exception("Error during HelmetDetector.__init__: %s", e)
        return
    # Choose a sample image present in data/samples
    sample_img = Path("data/samples/sample_helmet_violation.jpg")
    if not sample_img.exists():
        log.error("Sample image not found: %s", sample_img)
        return
    from PIL import Image
    pil_img = Image.open(sample_img).convert("RGB")
    try:
        annotated, detections, stats = detector.detect_image(pil_img)
        log.info("Detection completed. Detections count: %d", len(detections))
        log.debug("Detections: %s", json.dumps(detections, indent=2))
        log.info("Stats: %s", json.dumps(stats, indent=2))
    except Exception as e:
        log.exception("Error during detection on image %s: %s", sample_img, e)

def test_api_image():
    """POST to /api/detect/image using the running Flask API and log the response."""
    import requests
    url = "http://127.0.0.1:5000/api/detect/image"
    data = {"sample": "sample_helmet_violation.jpg"}
    try:
        resp = requests.post(url, data=data, timeout=20)
        log.info("POST /api/detect/image status: %s", resp.status_code)
        try:
            js = resp.json()
            log.debug("Response JSON (truncated): %s", json.dumps(js)[:500])
        except Exception as e:
            log.error("Response is not valid JSON: %s", e)
            log.debug("Raw response text: %s", resp.text)
    except Exception as e:
        log.exception("Request to image API failed: %s", e)

def test_api_video():
    """POST to /api/detect/video with a sample video file."""
    import requests
    url = "http://127.0.0.1:5000/api/detect/video"
    data = {"sample": "sample_traffic_video.mp4"}
    try:
        resp = requests.post(url, data=data, timeout=60)
        log.info("POST /api/detect/video status: %s", resp.status_code)
        try:
            js = resp.json()
            log.debug("Video response JSON (truncated): %s", json.dumps(js)[:500])
        except Exception as e:
            log.error("Video response not valid JSON: %s", e)
            log.debug("Raw response text: %s", resp.text)
    except Exception as e:
        log.exception("Request to video API failed: %s", e)

def test_api_webcam():
    """Simulate a webcam frame by sending a base64‑encoded sample image to /api/detect/frame."""
    import requests, base64
    from PIL import Image
    sample_img = Path("data/samples/sample_helmet_violation.jpg")
    if not sample_img.exists():
        log.error("Sample image for webcam test not found: %s", sample_img)
        return
    with open(sample_img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {"frame": f"data:image/jpeg;base64,{b64}", "conf": 0.4, "iou": 0.45}
    url = "http://127.0.0.1:5000/api/detect/frame"
    try:
        resp = requests.post(url, json=payload, timeout=20)
        log.info("POST /api/detect/frame status: %s", resp.status_code)
        try:
            js = resp.json()
            log.debug("Webcam response JSON (truncated): %s", json.dumps(js)[:500])
        except Exception as e:
            log.error("Webcam response not valid JSON: %s", e)
            log.debug("Raw response text: %s", resp.text)
    except Exception as e:
        log.exception("Request to webcam API failed: %s", e)

if __name__ == "__main__":
    log.info("--- Diagnostic run start ---")
    proc = ensure_server_running()
    # Direct detector test (bypasses Flask)
    test_direct_detector()
    # API tests (require server)
    test_api_image()
    test_api_video()
    test_api_webcam()
    # Cleanup: if we started a server process, terminate it
    if proc:
        log.info("Terminating Flask server started for diagnostics…")
        proc.terminate()
        proc.wait(timeout=5)
    log.info("--- Diagnostic run end ---")
