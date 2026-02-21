import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Vision Hub", layout="wide", initial_sidebar_state="collapsed")

# --- CLEAN UI CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #121216; color: white; }
    
    /* Title Fix */
    .main-title { 
        font-size: 28px; 
        font-weight: 700; 
        margin-bottom: 20px; 
        color: #00cc66; 
        padding-top: 10px;
    }

    /* Zero Gap Buttons */
    div.stButton > button {
        width: 100%;
        background-color: #1e1e26;
        color: white;
        border: 1px solid #333;
        height: 48px;
        border-radius: 5px;
    }
    
    /* Card Design */
    .card {
        background-color: #1e1e26;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2a2a35;
        margin-top: 10px;
    }

    /* Eliminate Vertical Gaps */
    [data-testid="stVerticalBlock"] > div { gap: 0rem !important; }
    
    /* File Uploader styling to look clean */
    .uploadedFile { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt') 
model = load_yolo()

# --- HEADER ---
st.markdown("<div class='main-title'>🔍 AI Object Detection Hub</div>", unsafe_allow_html=True)

# --- NAVIGATION TOOLBAR ---
col1, col2, col3 = st.columns(3)

with col1:
    btn_snap = st.button("📸 Capture Snap")
with col2:
    # Browse logic: Standard file uploader but with minimal label
    uploaded_file = st.file_uploader("📁 Browse Photo", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
with col3:
    btn_live = st.button("🎥 Live Stream")

# --- STATE LOGIC ---
if 'mode' not in st.session_state: st.session_state.mode = None
if btn_snap: st.session_state.mode = "Snap"
if uploaded_file: st.session_state.mode = "Browse"
if btn_live: st.session_state.mode = "Live"

processed_img = None
detected_boxes = []

# --- ENGINES ---
if st.session_state.mode == "Snap":
    cam_img = st.camera_input("Take a photo")
    if cam_img:
        img = Image.open(cam_img)
        res = model.predict(img, conf=0.3, imgsz=480)
        processed_img, detected_boxes = res[0].plot(), res[0].boxes

elif st.session_state.mode == "Browse" and uploaded_file:
    img = Image.open(uploaded_file)
    res = model.predict(img, conf=0.3, imgsz=480)
    processed_img, detected_boxes = res[0].plot(), res[0].boxes

elif st.session_state.mode == "Live":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.3, imgsz=320, verbose=False)
        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

    webrtc_streamer(key="yolo_live", video_frame_callback=video_frame_callback, 
                    rtc_configuration=RTC_CONFIG,
                    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False})
    st.markdown("</div>", unsafe_allow_html=True)

# --- CONDITIONAL RESULTS (Sirf tab dikhega jab photo hogi) ---
if processed_img is not None:
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    left, mid, right = st.columns([1, 2, 1])
    
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### ⚙️ Settings")
        conf = st.slider("Confidence", 0.1, 1.0, 0.35)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with mid:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(processed_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### 📋 Result")
        table_data = []
        for box in detected_boxes:
            label = model.names[int(box.cls[0])]
            table_data.append({"Object": label, "Conf": f"{round(float(box.conf[0])*100,1)}%"})
        if table_data:
            st.table(table_data)
        else:
            st.write("No items found.")
        st.markdown("</div>", unsafe_allow_html=True)