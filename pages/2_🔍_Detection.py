import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Object Detection", layout="wide", initial_sidebar_state="expanded")

# --- PREMIUM DARK UI ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0d11; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161a21 !important; border-right: 1px solid #00cc66; }
    .main-title { font-size: 30px; font-weight: 800; color: #00cc66; margin-bottom: 20px; }
    .card { background-color: #1c2128; padding: 20px; border-radius: 15px; border: 1px solid #2d333b; margin-bottom: 20px; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: 600; height: 50px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt') 
model = load_yolo()

# --- NIGHT VISION ENGINE ---
def apply_night_vision(img_array):
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    pil_img = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
    return np.array(ImageEnhance.Brightness(pil_img).enhance(2.0))

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## ⚙️ Vision Settings")
    st.divider()
    # NIGHT MODE DEFAULT ON
    night_mode = st.toggle("🌙 Night Enhancement", value=True)
    conf_level = st.slider("Confidence Threshold", 0.1, 1.0, 0.35)
    st.divider()
    st.info("System: YOLOv8 Engine Active")

# --- HEADER ---
st.markdown("<div class='main-title'>🔍 AI Object Detection</div>", unsafe_allow_html=True)

# --- NAVIGATION ---
t1, t2, t3 = st.tabs(["📤 Upload Image", "📸 Take Snapshot", "🎥 Live Stream"])

processed_img = None
detected_boxes = []

# --- TAB 1: UPLOAD ---
with t1:
    up = st.file_uploader("Choose a photo", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    if up:
        img_np = np.array(Image.open(up))
        if night_mode: img_np = apply_night_vision(img_np)
        res = model.predict(img_np, conf=conf_level, imgsz=640, verbose=False)
        processed_img, detected_boxes = res[0].plot(), res[0].boxes

# --- TAB 2: SNAPSHOT ---
with t2:
    cam = st.camera_input("Snapshot")
    if cam:
        img_np = np.array(Image.open(cam))
        if night_mode: img_np = apply_night_vision(img_np)
        res = model.predict(img_np, conf=conf_level, imgsz=640, verbose=False)
        processed_img, detected_boxes = res[0].plot(), res[0].boxes

# --- TAB 3: LIVE STREAM ---
with t3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Fast STUN servers for smooth connection
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        if night_mode:
            # High-speed enhancement for Live
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
        
        # Lower imgsz for higher FPS in live mode
        results = model.predict(img, conf=conf_level, imgsz=320, verbose=False)
        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

    webrtc_streamer(
        key="yolo_live_pro", 
        video_frame_callback=video_frame_callback, 
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True # Isse lag kam hota hai
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- RESULTS SECTION ---
if processed_img is not None:
    st.divider()
    left, right = st.columns([2, 1])
    
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(processed_img, use_container_width=True, caption="Detection Result")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### 📋 Detected Items")
        items = []
        for box in detected_boxes:
            label = model.names[int(box.cls[0])]
            items.append({"Object": label.capitalize(), "Score": f"{round(float(box.conf[0])*100,1)}%"})
        
        if items:
            st.table(pd.DataFrame(items))
        else:
            st.write("No objects detected.")
        st.markdown("</div>", unsafe_allow_html=True)
