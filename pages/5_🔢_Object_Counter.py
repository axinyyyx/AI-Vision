import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import pandas as pd
import time

st.set_page_config(page_title="Object Counter", layout="wide")

# --- NIGHT VISION ENGINE ---
def apply_night_vision(img):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. Brightness & Sharpness boost using PIL
    enhanced_img = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(enhanced_img)
    enhanced_img = enhancer.enhance(1.5) # Increase brightness by 50%
    
    return enhanced_img

# --- APP UI ---
st.title("🔢 Object Counter")
st.sidebar.header("🌙 Vision Settings")
night_mode = st.sidebar.toggle("Enable Night Vision Mode", value=False)

if night_mode:
    st.sidebar.info("Night Vision Active: Enhancing low-light visibility.")

# Load Model
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

if 'count_history' not in st.session_state:
    st.session_state.count_history = []

def process_and_count(img):
    # Apply Night Vision if toggled
    if night_mode:
        img = apply_night_vision(img)
    
    img_arr = np.array(img)
    results = model(img_arr)
    
    counts = {}
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        counts[label] = counts.get(label, 0) + 1
    
    summary = []
    current_time = time.strftime("%H:%M:%S")
    for obj, qty in counts.items():
        data = {"Time": current_time, "Object": obj.capitalize(), "Count": qty, "Mode": "Night" if night_mode else "Day"}
        summary.append(data)
        st.session_state.count_history.append(data)
        
    return results[0].plot(), summary, img

# --- TABS ---
t1, t2 = st.tabs(["🖼️ Image Upload", "📸 Live Capture"])

with t1:
    up_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
    if up_file:
        img_raw = Image.open(up_file)
        annotated_img, summary, processed_raw = process_and_count(img_raw)
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(processed_raw, caption="Enhanced Image (Night Vision)")
        with c2:
            st.image(annotated_img, caption="AI Detection")
        st.table(pd.DataFrame(summary))

with t2:
    p = st.camera_input("Snapshot")
    if p:
        img_raw = Image.open(p)
        annotated_img, summary, processed_raw = process_and_count(img_raw)
        st.image(annotated_img)
        st.table(pd.DataFrame(summary))

# --- EXCEL LOG ---
if st.session_state.count_history:
    st.divider()
    df = pd.DataFrame(st.session_state.count_history)
    st.download_button("📥 Export Report", df.to_csv(index=False), "count_report.csv")
