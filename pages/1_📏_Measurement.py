import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import math
import pandas as pd
import time

st.set_page_config(page_title="AI Precision Lab", layout="wide")

# --- USER GUIDE (Instructions) ---
def show_instructions():
    with st.expander("📖 Accuracy Guide: 90% + Results Kaise Paayein?", expanded=True):
        st.markdown("""
        ### ✅ Accurate Measurement ke liye Rules:
        1. **ATM Card Reference:** Frame mein object ke bilkul bagal mein ek **ATM Card/Credit Card** rakhein. (AI ise scale ki tarah use karega).
        2. **90-Degree Angle:** Camera ko hamesha object ke **theek upar (Top-View)** rakhein. Agar camera tedha hoga, toh measurement galat aayegi.
        3. **Distance:** Card aur Object dono camera se **ek hi doori (Height)** par hone chahiye.
        4. **Lighting:** Room mein achhi roshni honi chahiye taaki AI edges ko sahi se detect kare.
        """)

# --- MODEL LOAD ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# --- CALCULATION ENGINE ---
def calculate_metrics(label, w_px, h_px, p2u, unit):
    w, h = w_px * p2u, h_px * p2u
    res = {
        "ID": f"AI-{int(time.time() % 10000)}",
        "Object": label.capitalize(),
        f"Length ({unit})": round(w, 2),
        f"Breadth ({unit})": round(h, 2),
        f"Perimeter ({unit})": round(2*(w+h), 2),
        f"Area ({unit}²)": round(w*h, 2)
    }
    
    # 3D/Shape Specific
    if label in ['sports ball', 'orange', 'apple']:
        r = w / 2
        res[f"Radius ({unit})"] = round(r, 2)
        res[f"Volume ({unit}³)"] = round((4/3)*math.pi*(r**3), 3)
    elif label in ['bottle', 'cup']:
        r = w / 2
        res[f"Volume ({unit}³)"] = round(math.pi*(r**2)*h, 3)
    else:
        res[f"Volume ({unit}³)"] = round(w*h*((w+h)/2), 3) # Est. Volume
    return res

# --- APP LAYOUT ---
st.title("📏 Measurement Lab")
show_instructions()

unit_choice = st.sidebar.selectbox("Select Unit", ["cm", "m", "ft", "inch"])
p2u_manual = st.sidebar.number_input("Manual Calibration (if no card)", value=0.0264, format="%.5f")

if 'history' not in st.session_state:
    st.session_state.history = []

def process_frame(img):
    img_arr = np.array(img)
    results = model(img_arr)
    
    # Calibration Step
    p2u = p2u_manual
    found_ref = False
    
    # YOLO typically detects cards as 'book' or 'cell phone'
    # We will use 'cell phone' as a standard 15cm x 7cm reference if found
    for box in results[0].boxes:
        if model.names[int(box.cls[0])] in ['cell phone', 'book']:
            b = box.xyxy[0].cpu().numpy()
            ref_px_w = b[2] - b[0]
            # Standard Card Width = 8.56 cm
            p2u = 8.56 / ref_px_w 
            found_ref = True
            break
            
    if found_ref: st.success("🎯 Reference Object Detected! Accuracy Optimized.")
    else: st.warning("⚠️ No reference card found. Using manual calibration.")

    # Unit Conversion
    if unit_choice == "m": p2u /= 100
    elif unit_choice == "ft": p2u /= 30.48
    elif unit_choice == "inch": p2u /= 2.54

    final_data = []
    for i, box in enumerate(results[0].boxes):
        label = model.names[int(box.cls[0])]
        b = box.xyxy[0].cpu().numpy()
        data = calculate_metrics(label, b[2]-b[0], b[3]-b[1], p2u, unit_choice)
        final_data.append(data)
        st.session_state.history.append(data)
        
    return results[0].plot(), final_data

# --- TABS ---
t1, t2 = st.tabs(["📤 Image Upload", "📸 Real-time Capture"])

with t1:
    f = st.file_uploader("Upload", type=['jpg','png','jpeg'])
    if f:
        img = Image.open(f)
        annotated_img, data = process_frame(img)
        st.image(annotated_img)
        st.dataframe(pd.DataFrame(data))

with t2:
    p = st.camera_input("Take Photo (Keep ATM Card in frame)")
    if p:
        img = Image.open(p)
        annotated_img, data = process_frame(img)
        st.image(annotated_img)
        st.dataframe(pd.DataFrame(data))

# --- EXCEL EXPORT ---
if st.session_state.history:
    st.divider()
    st.subheader("📊 Session History (Excel)")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
    st.download_button("📥 Download Excel Sheet", df.to_csv(index=False), "measurements.csv", "text/csv")
