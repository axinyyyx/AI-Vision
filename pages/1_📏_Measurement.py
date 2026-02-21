import streamlit as st
import cv2
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Precision Scale", layout="wide")

st.markdown("<h2 style='text-align:center; color:#FF4B4B;'>📏 AI AUTO-SCALE MEASUREMENT</h2>", unsafe_allow_html=True)

# --- SIDEBAR: ONE-TIME CALIBRATION ---
st.sidebar.header("⚙️ Smart Calibration")
# Agar screen par 100 pixels = 5cm hai, toh 1cm = 20px. 
# Is slider se aap ek baar calibrate kar lo, fir card ki zarurat nahi padegi.
calibration_factor = st.sidebar.slider("Fine Tune (Pixels per CM)", 10.0, 100.0, 37.8) 

st.sidebar.info("💡 Tip: Ek baar scale rakh kar check karein ki ye sahi bata raha hai ya nahi, fir bina scale ke use karein.")

img_file = st.camera_input("Object ki Photo lein")

if img_file:
    # Convert image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    orig = img.copy()
    h_img, w_img, _ = img.shape

    # --- IMAGE PROCESSING ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Edge detection for precision
    edged = cv2.Canny(blur, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find Contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small noise
    objects = [c for c in cnts if cv2.contourArea(c) > 2000]

    res_table = []

    # --- AUTO REFERENCE OVERLAY (RED BOX) ---
    # Hum ek imaginary card ka box draw karenge top corner mein reference ke liye
    ref_w_px, ref_h_px = int(8.5 * calibration_factor), int(5.4 * calibration_factor)
    cv2.rectangle(img, (20, 20), (20 + ref_w_px, 20 + ref_h_px), (0, 0, 255), 3)
    cv2.putText(img, "AUTO-REF (8.5cm x 5.4cm)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for i, c in enumerate(objects):
        # Calculate Bounding Box
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Dimensions in CM based on calibration_factor
        (x, y), (w_px, h_px), angle = rect
        width_cm = w_px / calibration_factor
        height_cm = h_px / calibration_factor
        
        # Drawing
        cv2.drawContours(img, [box], 0, (0, 255, 0), 4)
        
        # Labeling
        cv2.putText(img, f"{round(width_cm, 1)}cm", (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        res_table.append({
            "Object ID": f"Item {i+1}",
            "Width (cm)": f"{round(width_cm, 2)} cm",
            "Height (cm)": f"{round(height_cm, 2)} cm",
            "Area": f"{round(width_cm * height_cm, 2)} cm²"
        })

    # --- FINAL DISPLAY ---
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    if res_table:
        st.subheader("📊 Measurement Results")
        st.table(res_table)
    else:
        st.warning("⚠️ Koi object detect nahi hua. Background saaf rakhein (Plain Background).")