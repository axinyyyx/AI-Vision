import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd

# --- CRASH PREVENTING IMPORTS ---
try:
    from deepface import DeepFace
    from yt_dlp import YoutubeDL
except Exception as e:
    st.error(f"Library Missing: Run 'pip install deepface yt-dlp'. Error: {e}")

# --- PAGE SETTINGS ---
st.set_page_config(page_title="MoodSync AI", layout="wide")

# UI CSS
st.markdown("""
    <style>
    .video-card { border-radius: 10px; background-color: #1e2129; padding: 15px; margin-bottom: 10px; border: 1px solid #333; }
    .stMetric { background: #1e2129; border-radius: 10px; padding: 10px; border: 1px solid #00cc66; }
    </style>
""", unsafe_allow_html=True)

st.title("🎭 MoodSync AI")
st.write("Next-Gen Mood & Entertainment Intelligence")

# --- FUNCTIONS ---
def analyze_now(img):
    try:
        # Convert PIL to BGR
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Optimized Analysis (Actions ek saath mangwaye hain)
        results = DeepFace.analyze(img_bgr, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        return results[0]
    except Exception as e:
        st.error(f"DeepFace Analysis Error: {e}")
        return None

def fetch_videos(query):
    ydl_opts = {'quiet': True, 'extract_flat': True, 'force_generic_extractor': True}
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch8:{query}", download=False)
            return info['entries']
    except Exception as e:
        st.warning(f"YouTube Fetch Failed: {e}")
        return []

# --- MAIN UI ---
col_in, col_set = st.columns([2, 1])

with col_in:
    cam = st.camera_input("Snapshot for Analysis")
    
with col_set:
    st.subheader("Settings")
    cat = st.radio("Vibe Type", ["🎵 Music", "🎬 Movies"])
    use_filters = st.multiselect("AI Filters", ["Mood", "Age", "Gender"], default=["Mood"])

if cam:
    input_img = Image.open(cam)
    
    with st.spinner('Deep Learning Models are thinking...'):
        res = analyze_now(input_img)
    
    if res:
        mood = res['dominant_emotion']
        age = int(res['age'])
        gender = res['dominant_gender']

        # Results Display
        m1, m2, m3 = st.columns(3)
        m1.metric("Mood", mood.capitalize())
        m2.metric("Gender", gender)
        m3.metric("Age", f"{age} Yrs")

        # --- SMART LOGIC ---
        q_parts = []
        if age <= 5:
            q_parts.append("kids nursery rhymes cartoons")
        elif 6 <= age <= 12:
            q_parts.append("science fun learning for kids")
        else:
            if "Mood" in use_filters: q_parts.append(f"{mood} mood")
            if "Gender" in use_filters: q_parts.append(f"for {gender.lower()}")
            q_parts.append("trending 2026")

        final_query = " ".join(q_parts) + (" songs" if cat == "🎵 Music" else " movie trailers")
        
        st.divider()
        st.subheader("📺 Recommendations")

        # YT-DLP RESULTS
        vids = fetch_videos(final_query)
        if vids:
            for v in vids:
                url = f"https://www.youtube.com/watch?v={v['id']}"
                st.markdown(f"""
                <div class="video-card">
                    <a href="{url}" target="_blank" style="color: #00cc66; text-decoration: none; font-size: 18px; font-weight: bold;">▶ {v['title']}</a>
                    <p style="color: #888; margin-top: 5px;">Channel: {v.get('uploader', 'YouTube')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("No videos found. Check your internet connection.")

    else:
        st.warning("Could not detect face. Try again with better lighting.")
