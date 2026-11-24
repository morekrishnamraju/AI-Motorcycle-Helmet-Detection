# HelmetAI – Streamlit Front‑End (matches the dark‑theme screenshot)

import streamlit as st
import requests, base64
from pathlib import Path

# ----------------------------------------------------------------
# Configuration – must match Flask backend
# ----------------------------------------------------------------
API_BASE = "http://127.0.0.1:5000/api"

# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
# Helper functions
def call_api(endpoint: str, files: dict):
    """POST multipart files to Flask endpoint and return JSON response."""
    url = f"{API_BASE}/{endpoint}"
    resp = requests.post(url, files=files)
    resp.raise_for_status()
    return resp.json()

def show_b64_image(b64_str: str):
    """Decode a data‑url base64 JPEG and display it with Streamlit.
    Handles both "data:image/jpeg;base64,<data>" and raw base64 strings.
    """
    if "," in b64_str:
        _, data = b64_str.split(",", 1)
    else:
        data = b64_str
    img_bytes = base64.b64decode(data)
    st.image(img_bytes, use_column_width=True)

    """POST multipart files to Flask endpoint and return JSON response."""
    url = f"{API_BASE}/{endpoint}"
    resp = requests.post(url, files=files)
    resp.raise_for_status()
    return resp.json()

def show_b64_image(b64_str: str):
    """Decode a data‑url base64 JPEG and display it with Streamlit."""
    _, data = b64_str.split(",", 1)
    img_bytes = base64.b64decode(data)
    st.image(img_bytes, use_column_width=True)

# ----------------------------------------------------------------
# Page setup – dark theme already defined in .streamlit/config.toml
# ----------------------------------------------------------------
st.set_page_config(page_title="HelmetAI", layout="wide")
st.sidebar.title("🚀 HelmetAI")
st.sidebar.markdown("---")

# ----------------------------------------------------------------
# Navigation – order matches the screenshot you provided
# ----------------------------------------------------------------
pages = [
    "Dashboard",
    "Image Detection",
    "Traffic Video",
    "Live Webcam",
    "Analytics",
    "Reports",
    "Viva Q&A",
]
selected = st.sidebar.radio("Navigate", pages)

# ----------------------------------------------------------------
# UI for each tab (no Model Evaluation, no extra metrics)
# ----------------------------------------------------------------
if selected == "Dashboard":
    # Hero banner (same styling as screenshot)
    st.markdown(
        """
        <div class='hero-banner'>
            <div class='hero-title'>AI‑Based Motorcycle Helmet Detection & Rider Safety System</div>
            <div class='hero-subtitle'>Deep‑Learning Traffic Safety Surveillance Powered by YOLOv8</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # KPI cards – three columns, dark colors
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """<div class='metric-card'>
                 <div class='metric-title'>Total Detections</div>
                 <div class='metric-value metric-val-blue'>0</div>
               </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class='metric-card'>
                 <div class='metric-title'>Compliance Rate</div>
                 <div class='metric-value metric-val-green'>0%</div>
               </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class='metric-card'>
                 <div class='metric-title'>Violations</div>
                 <div class='metric-value metric-val-red'>0</div>
               </div>""",
            unsafe_allow_html=True,
        )
    st.info("Dashboard will update after you run detections.")

# Image Detection UI – show only helmet and violation counts
elif selected == "Image Detection":
    st.header("🖼️ Image Detection")
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        # Read the file bytes for sending and later for optional local preview
        with st.spinner("Running detection …"):
            result = call_api("detect/image", {"image": uploaded})
        # Show annotated image (base64) – the helper now handles plain base64 strings as well
        show_b64_image(result["image_b64"])
        # Extract needed stats safely
        stats = result.get("stats", {})
        helmet_cnt = stats.get("helmets", 0)
        viol_cnt = stats.get("without_helmets", 0)
        # Display counts as simple metrics
        col1, col2 = st.columns(2)
        col1.metric("Helmet Count", helmet_cnt)
        col2.metric("Violations Count", viol_cnt)
        # Optionally show full detections table for debugging (can be hidden)
        if st.checkbox("Show raw detections"):
            st.subheader("Detections")
            st.dataframe(result.get("detections", []))
        # Show raw stats JSON for reference (hidden by default)
        if st.checkbox("Show raw stats JSON"):
            st.subheader("Raw Stats")
            st.json(stats)

    st.header("🖼️ Image Detection")
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        with st.spinner("Running detection …"):
            result = call_api("detect/image", {"image": uploaded})
        show_b64_image(result["image_b64"])
        if result.get("detections"):
            st.subheader("Detections")
            st.dataframe(result["detections"])
        else:
            st.warning(result.get("error", "No detections"))
        st.subheader("Stats")
        st.json(result.get("stats", {}))

# Video Detection UI – show only helmet and violation counts
elif selected == "Traffic Video":
    st.header("🚦 Traffic Video")
    video = st.file_uploader("Upload a video (MP4/AVI)", type=["mp4", "avi"])
    if video:
        with st.spinner("Processing video …"):
            result = call_api("detect/video", {"video": video})
        # Show original video preview
        st.video(video)
        # Extract summary stats (the API already returns total_helmets and total_violations)
        helmets = result.get("total_helmets", 0)
        violations = result.get("total_violations", 0)
        col1, col2 = st.columns(2)
        col1.metric("Helmet Count", helmets)
        col2.metric("Violations Count", violations)
        # Optional raw summary JSON for debugging
        if st.checkbox("Show raw summary JSON"):
            st.subheader("Summary")
            st.json(result)

    st.header("🚦 Traffic Video")
    video = st.file_uploader("Upload a video (MP4/AVI)", type=["mp4", "avi"]))
    if video:
        with st.spinner("Processing video …"):
            result = call_api("detect/video", {"video": video})
        st.success("Video processed")
        st.video(video)
        st.subheader("Summary")
        st.json(result.get("summary", {}))

elif selected == "Live Webcam":
    st.header("📹 Live Webcam")
    stream_url = f"{API_BASE}/detect/webcam"
    st.image(stream_url, caption="Live detection stream", use_column_width=True)

elif selected == "Analytics":
    st.header("📈 Analytics")
    try:
        analytics = requests.get(f"{API_BASE}/analytics").json()
        st.json(analytics)
    except Exception as e:
        st.error(f"Failed to fetch analytics: {e}")

elif selected == "Reports":
    st.header("📄 Reports")
    if st.button("Generate PDF Report"):
        with st.spinner("Generating report …"):
            resp = requests.get(f"{API_BASE}/report")
        if resp.status_code == 200:
            pdf_path = Path("HelmetAI_report.pdf")
            pdf_path.write_bytes(resp.content)
            st.success("Report ready")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="HelmetAI_report.pdf")
        else:
            st.error("Report generation failed")

elif selected == "Viva Q&A":
    st.header("❓ Viva Q&A")
    if "history" not in st.session_state:
        st.session_state.history = []
    question = st.text_input("Ask a question about the analysis …")
    if st.button("Send") and question:
        with st.spinner("Thinking …"):
            answer = call_api("qna", {"question": (question,)})
        st.session_state.history.append((question, answer.get("answer", "")))
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
