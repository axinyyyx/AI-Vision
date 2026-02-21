import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="A.I Object Detection", layout="wide", initial_sidebar_state="expanded")

# --- PREMIUM DARK UI CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0d11; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161a21 !important; border-right: 1px solid #00cc66; }
    .main-title { font-size: 30px; font-weight: 800; color: #00cc66; margin-bottom: 10px; }
    .card { background-color: #1c2128; padding: 20px; border-radius: 15px; border: 1px solid #2d333b; margin-bottom: 20px; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: 600; height: 50px; background-color: #1e1e26; color: white; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ACCURATE MODEL ---
@st.cache_resource
def load_yolo():
    # 'yolov8s.pt' is more accurate than 'yolov8n.pt'
    return YOLO('yolov8s.pt') 
model = load_yolo()

# --- NIGHT VISION ENGINE ---
def apply_night_vision(img_array):
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    pil_img = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
    return np.array(ImageEnhance.Brightness(pil_img).enhance(1.5))

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## ⚙️ Accuracy Settings")
    st.divider()
    night_mode = st.toggle("🌙 Night Enhancement", value=True)
    # High confidence default for 90% accuracy goal
    conf_level = st.slider("Confidence (Accuracy)", 0.1, 1.0, 0.50, help="Higher = More Accurate but fewer detections")
    st.divider()
    st.info("Model: YOLOv8 Small (High Accuracy Mode)")

# --- MAIN HEADER ---
st.markdown("<div class='main-title'>🔍 A.I Object Detection</div>", unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["📤 Upload Image", "📸 Take Snapshot", "🎥 Live Stream"])

processed_img = None
detected_boxes = []

# --- DETECTION FUNCTION ---
def get_prediction(img):
    # iou=0.5 reduces double boxes, agnostic_nms removes class overlaps
    return model.predict(img, conf=conf_level, iou=0.5, imgsz=640, agnostic_nms=True, verbose=False)

# --- TAB 1: UPLOAD ---
with t1:
    up = st.file_uploader("Choose a photo", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    if up:
        img_np = np.array(Image.open(up))
        if night_mode: img_np = apply_night_vision(img_np)
        res = get_prediction(img_np)
        processed_img, detected_boxes = res[0].plot(), res[0].boxes

# --- TAB 2: SNAPSHOT ---
with t2:
    cam = st.camera_input("Snapshot")
    if cam:
        img_np = np.array(Image.open(cam))
        if night_mode: img_np = apply_night_vision(img_np)
        res = get_prediction(img_np)
        processed_img, detected_boxes = res[0].plot(), res[0].boxes

# --- TAB 3: LIVE STREAM ---
with t3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        if night_mode:
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        
        # Balance speed and accuracy for live
        results = model.predict(img, conf=conf_level, iou=0.4, imgsz=320, verbose=False)
        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

    webrtc_streamer(
        key="yolo_pro_stream", 
        video_frame_callback=video_frame_callback, 
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- RESULTS SECTION ---
if processed_img is not None:
    st.divider()
    left, right = st.columns([2, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(processed_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### 📋 Precision Report")
        items = []
        for box in detected_boxes:
            label = model.names[int(box.cls[0])]
            acc = float(box.conf[0])
            items.append({"Object": label.capitalize(), "Accuracy": f"{round(acc*100,1)}%"})
        
        if items:
            df = pd.DataFrame(items)
            st.table(df)
            st.success(f"Average Precision: {round(df['Object'].count(), 1)} items found")
        else:
            st.warning("Low confidence detections hidden. Adjust slider for more results.")
        st.markdown("</div>", unsafe_allow_html=True)
