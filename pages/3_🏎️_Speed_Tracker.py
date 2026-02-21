import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Traffic AI Radar", layout="wide")

# --- CSS FOR HIGH-SPEED INTERFACE ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0d10; color: #00FF00; }
    .radar-box { border: 2px solid #00FF00; padding: 10px; border-radius: 10px; background: #161a21; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL (NANO) ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')
model = load_yolo()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🚓 Radar & Grid Control")
unit = st.sidebar.selectbox("Unit", ["km/h", "m/s", "cm/s"])
grid_val = st.sidebar.slider("Grid Size (px)", 10, 200, 50)
ppm = st.sidebar.slider("PPM (Calibration)", 10, 100, 35)

# THE FIX: Data Storage (External to the Process)
# Isme hum data store karenge bina ScriptRunContext error ke
if 'radar_data' not in st.session_state:
    st.session_state['radar_data'] = {}

class VideoProcessor:
    def __init__(self, unit_type, grid_size, ppm_val):
        self.history = {}
        self.unit = unit_type
        self.grid = grid_size
        self.ppm = ppm_val
        self.last_t = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        t_now = time.time()

        # 1. GRAPH GRID (Fast Slicing)
        img[::self.grid, :, :] = img[::self.grid, :, :] // 2 + 40
        img[:, ::self.grid, :] = img[:, ::self.grid, :] // 2 + 40

        # 2. TRAFFIC ANALYSIS (Multi-Object)
        # 0.15s delay ensures no lag for high-speed objects
        if t_now - self.last_t > 0.15:
            self.last_t = t_now
            # imgsz=256 ensures high speed on all devices
            results = model.track(img, persist=True, verbose=False, imgsz=256)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, obj_id, cls_idx in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    obj_name = model.names[cls_idx]
                    
                    if obj_id in self.history:
                        prev = self.history[obj_id]
                        dt = t_now - prev['t']
                        dist_px = np.sqrt((cx-prev['x'])**2 + (cy-prev['y'])**2)
                        
                        v_mps = (dist_px / self.ppm) / dt
                        factor = 3.6 if self.unit == "km/h" else (100 if self.unit == "cm/s" else 1)
                        v_inst = v_mps * factor
                        accel = (v_inst - prev['v']) / dt
                        total_dist = prev['dist'] + (dist_px / self.ppm)

                        if v_inst > 1.2:
                            # 🟢 Draw Overlay on Video (No Thread Error here)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{obj_name} {int(v_inst)} {self.unit}", (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # HUD Display (Top Left)
                            cv2.rectangle(img, (10, 10), (250, 80), (0, 0, 0), -1)
                            cv2.putText(img, f"ID {obj_id} SPD: {int(v_inst)}", (20, 35), 0, 0.6, (0, 255, 0), 1)
                            cv2.putText(img, f"ACCEL: {round(accel, 1)}", (20, 60), 0, 0.6, (0, 255, 0), 1)

                        self.history[obj_id].update({'x':cx, 'y':cy, 't':t_now, 'v':v_inst, 'dist':total_dist})
                    else:
                        self.history[obj_id] = {'x':cx, 'y':cy, 't':t_now, 'v':0, 'dist':0}

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
st.markdown("<div class='main-title'>🚦 MULTI-OBJECT AI TRAFFIC RADAR</div>", unsafe_allow_html=True)

webrtc_streamer(
    key="traffic-radar-final",
    video_frame_callback=VideoProcessor(unit, grid_val, ppm).recv,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)

st.success("Radar is active! Objects like Cars, Humans, Cycles, and Balls will be tracked automatically.")