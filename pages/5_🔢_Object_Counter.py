import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time

st.set_page_config(page_title="AI Object Counter", layout="wide")

# --- APP HEADER ---
st.title("🔢 Smart AI Object Counter")
st.markdown("""
**Description:** Yeh tool photo ya live camera se objects ko count karta hai aur unki list banata hai. 
Aap ise inventory check karne ya crowd counting ke liye use kar sakte hain.
""")

# Load Model
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# Session State for History
if 'count_history' not in st.session_state:
    st.session_state.count_history = []

def process_and_count(img):
    img_arr = np.array(img)
    results = model(img_arr)
    
    # Class wise counting logic
    counts = {}
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        counts[label] = counts.get(label, 0) + 1
    
    # Prepare data for Table/Excel
    current_time = time.strftime("%H:%M:%S")
    summary = []
    for obj, qty in counts.items():
        data = {"Time": current_time, "Object Name": obj.capitalize(), "Quantity": qty}
        summary.append(data)
        st.session_state.count_history.append(data)
        
    return results[0].plot(), summary

# --- TABS ---
t1, t2 = st.tabs(["🖼️ Image Upload", "📸 Live Snapshot Counter"])

with t1:
    up_file = st.file_uploader("Upload Image to Count", type=['jpg','png','jpeg'])
    if up_file:
        img = Image.open(up_file)
        annotated_img, summary_data = process_and_count(img)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(annotated_img, caption="Detection View")
        with col2:
            st.write("### 📊 Count Summary")
            st.table(pd.DataFrame(summary_data))

with t2:
    p = st.camera_input("Take a Photo to Count Objects")
    if p:
        img = Image.open(p)
        annotated_img, summary_data = process_and_count(img)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(annotated_img)
        with col2:
            st.write("### 📊 Count Summary")
            st.table(pd.DataFrame(summary_data))

# --- MASTER EXCEL LOG ---
st.divider()
st.subheader("📋 Master Inventory Log (Excel Export)")
if st.session_state.count_history:
    df = pd.DataFrame(st.session_state.count_history)
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Counting Report", csv, "object_counts.csv", "text/csv")
    
    if st.button("Clear Log"):
        st.session_state.count_history = []
        st.rerun()
