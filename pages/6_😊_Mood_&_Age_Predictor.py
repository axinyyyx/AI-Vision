import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import pandas as pd
import time
from yt_dlp import YoutubeDL

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Mood Based Music & Movie Recommendations", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS FOR CLEAN UI ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .video-card {
        border-radius: 12px;
        background-color: #1e2129;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #3e4149;
        transition: 0.3s;
    }
    .video-card:hover {
        border-color: #ff4b4b;
        background-color: #252932;
    }
    .stMetric {
        background-color: #1e2129;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #3e4149;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE FUNCTIONS ---
@st.cache_resource
def get_yt_dlp():
    return YoutubeDL({'quiet': True, 'extract_flat': True, 'force_generic_extractor': True})

def analyze_face(img):
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    try:
        results = DeepFace.analyze(img_bgr, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        return results[0]
    except:
        return None

def get_ytdlp_results(query):
    ydl = get_yt_dlp()
    try:
        info = ydl.extract_info(f"ytsearch10:{query}", download=False)
        return info['entries']
    except:
        return []

# --- UI & LOGIC ---
st.title("🎬 MoodSync AI")

if 'face_history' not in st.session_state:
    st.session_state.face_history = []

col_input, col_ctrl = st.columns([2, 1])

with col_input:
    tab1, tab2 = st.tabs(["📤 Upload", "🤳 Camera"])
    input_img = None
    with tab1:
        up = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
        if up: input_img = Image.open(up)
    with tab2:
        cam = st.camera_input("Snapshot", label_visibility="collapsed")
        if cam: input_img = Image.open(cam)

with col_ctrl:
    st.write("### Settings")
    category = st.selectbox("What to play?", ["🎵 Music", "🎬 Movies"])
    filter_choice = st.multiselect("Filters:", ["Mood", "Age", "Gender"], default=["Mood"])

# --- PROCESSING ---
if input_img:
    with st.spinner('Processing...'):
        res = analyze_face(input_img)
    
    if res:
        mood = res['dominant_emotion'].lower()
        age = int(res['age'])
        gender = res['dominant_gender']
        
        # Simple Metrics
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Mood", mood.capitalize())
        m2.metric("Gender", gender)
        m3.metric("Age", f"{age} Years")
        
        # --- HIDDEN SMART LOGIC (No Labels on UI) ---
        query_parts = []
        if age <= 5:
            query_parts.append("nursery rhymes kids cartoons")
        elif 6 <= age <= 10:
            query_parts.append("educational science fun learning kids")
        elif 11 <= age <= 17:
            query_parts.append(f"{mood} safe trending content")
        else:
            if "Mood" in filter_choice: query_parts.append(f"{mood} mood")
            if "Gender" in filter_choice: query_parts.append(f"for {gender.lower()}")
            query_parts.append("latest 2026")

        final_query = " ".join(query_parts) + (" songs" if category == "🎵 Music" else " movie trailers")
        
        st.divider()
        st.subheader("📺 Recommendations")
        
        with st.spinner('Loading...'):
            results = get_ytdlp_results(final_query)
            if results:
                for vid in results:
                    vid_id = vid.get('id')
                    base_url = "https://music.youtube.com" if category == "🎵 Music" else "https://www.youtube.com"
                    play_link = f"{base_url}/watch?v={vid_id}"
                    thumb = vid.get('thumbnails', [{}])[0].get('url', 'https://via.placeholder.com/320x180')
                    
                    st.markdown(f"""
                    <div class="video-card">
                        <div style="display: flex; gap: 15px; align-items: center;">
                            <img src="{thumb}" style="width: 180px; border-radius: 8px;">
                            <div style="flex: 1;">
                                <h4 style="margin: 0;"><a href="{play_link}" target="_blank" style="color: #ff4b4b; text-decoration: none;">{vid.get('title')[:70]}</a></h4>
                                <p style="color: #888; margin: 5px 0; font-size: 14px;">{vid.get('uploader', 'YouTube')}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # History
        st.session_state.face_history.append({"Time": time.strftime("%H:%M:%S"), "Mood": mood, "Age": age, "Gender": gender})

# --- EXCEL DOWNLOAD ---
if st.session_state.face_history:
    st.divider()
    with st.expander("📑 History Log"):
        df = pd.DataFrame(st.session_state.face_history)
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download Excel", df.to_csv(index=False), "log.csv")
