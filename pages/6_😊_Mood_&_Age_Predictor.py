import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import time
import pandas as pd
from datetime import datetime

# --- CRASH PREVENTING IMPORTS ---
try:
    from deepface import DeepFace
    from yt_dlp import YoutubeDL
except Exception as e:
    st.error(f"Library Missing: Run 'pip install deepface yt-dlp'. Error: {e}")

# --- PAGE SETTINGS ---
st.set_page_config(page_title="MoodSync AI", layout="wide")

# --- UI CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    .video-card { 
        border-radius: 15px; 
        background-color: #1e2129; 
        padding: 0px; 
        margin-bottom: 20px; 
        border: 1px solid #333; 
        overflow: hidden;
        transition: 0.3s;
    }
    .video-card:hover { 
        border-color: #00cc66; 
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 204, 102, 0.2);
    }
    .video-info { padding: 15px; }
    .stMetric { background: #1e2129; border-radius: 10px; padding: 15px; border: 1px solid #00cc66; }
    .thumb-img { width: 100%; height: 180px; object-fit: cover; }
    </style>
""", unsafe_allow_html=True)

# --- AUTO-YEAR LOGIC ---
current_year = datetime.now().year
next_year = current_year + 1
year_string = f"{current_year}-{next_year}"

# --- NIGHT VISION ENGINE ---
def enhance_face_for_low_light(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    enhanced_pil = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(enhanced_pil)
    return enhancer.enhance(1.8)

# --- CORE FUNCTIONS ---
def analyze_now(img, use_night_mode):
    try:
        if use_night_mode:
            img = enhance_face_for_low_light(img)
            st.sidebar.image(img, caption="AI Enhanced View", width=150)

        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = DeepFace.analyze(img_bgr, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        return results[0]
    except Exception as e:
        st.error(f"AI Analysis Error: {e}")
        return None

def fetch_videos(query):
    # Thumbnail nikalne ke liye extract_flat ko False ya detailed info chahiye
    ydl_opts = {'quiet': True, 'extract_flat': True, 'force_generic_extractor': True}
    try:
        with YoutubeDL(ydl_opts) as ydl:
            # ytsearch8 se top 8 results
            info = ydl.extract_info(f"ytsearch8:{query}", download=False)
            return info['entries']
    except Exception as e:
        return []

# --- MAIN UI ---
st.title("🎭 MoodSync AI")
# --- SIDEBAR SETTINGS ---
st.sidebar.header("🌙 Vision Control")
night_vision = st.sidebar.toggle("Enable Night Enhancement", value=False)

st.sidebar.divider()
st.sidebar.header("⚙️ Personalization")
cat = st.sidebar.radio("Entertainment Category", ["🎵 Music", "🎬 Movies"])
use_filters = st.sidebar.multiselect("AI Filters", ["Mood", "Age", "Gender"], default=["Mood"])

# --- INPUT TABS ---
t1, t2 = st.tabs(["📤 Upload Image", "📸 Take Snapshot"])
input_img = None

with t1:
    up = st.file_uploader("Choose a photo...", type=['jpg', 'png', 'jpeg'])
    if up: input_img = Image.open(up)

with t2:
    cam = st.camera_input("Smile for the AI!")
    if cam: input_img = Image.open(cam)

# --- PROCESSING ---
if input_img:
    with st.spinner('Neural Network is processing...'):
        res = analyze_now(input_img, night_vision)
    
    if res:
        mood = res['dominant_emotion']
        age = int(res['age'])
        gender = res['dominant_gender']

        # Results Display
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Mood Detected", mood.capitalize())
        m2.metric("Gender", gender)
        m3.metric("Estimated Age", f"{age} Years")

        # --- SMART LOGIC WITH AUTO-YEAR ---
        q_parts = []
        if age <= 5:
            q_parts.append("kids nursery rhymes cartoons")
        elif 6 <= age <= 12:
            q_parts.append("educational learning fun kids")
        else:
            if "Mood" in use_filters: q_parts.append(f"{mood} mood songs")
            if "Gender" in use_filters: q_parts.append(f"for {gender.lower()}")
            # AUTO-YEAR Dynamic Query
            q_parts.append(f"trending hits {year_string}")

        final_query = " ".join(q_parts) + (" music" if cat == "🎵 Music" else " movie trailers")
        
        st.divider()
        st.subheader(f"📺 Top Mood Based Recommendations for {year_string}")

        with st.spinner('Syncing with YouTube...'):
            vids = fetch_videos(final_query)
            if vids:
                # 2 columns for better visual layout
                cols = st.columns(2)
                for idx, v in enumerate(vids):
                    url = f"https://www.youtube.com/watch?v={v['id']}"
                    if cat == "🎵 Music": url = url.replace("www.youtube.com", "music.youtube.com")
                    
                    # Thumbnail fetch (yt-dlp provides a thumbnails list)
                    thumb_url = f"https://img.youtube.com/vi/{v['id']}/mqdefault.jpg"
                    
                    with cols[idx % 2]:
                        st.markdown(f"""
                        <div class="video-card">
                            <img src="{thumb_url}" class="thumb-img">
                            <div class="video-info">
                                <a href="{url}" target="_blank" style="color: #00cc66; text-decoration: none; font-size: 16px; font-weight: bold;">
                                    {v['title'][:60]}...
                                </a>
                                <p style="color: #888; margin-top: 10px; font-size: 12px;">📺 {v.get('uploader', 'YouTube')}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found.")

# --- HISTORY ---
if 'mood_history' not in st.session_state: st.session_state.mood_history = []
if input_img and res:
    if len(st.session_state.mood_history) == 0 or st.session_state.mood_history[-1]["Time"] != time.strftime("%H:%M:%S"):
        st.session_state.mood_history.append({"Time": time.strftime("%H:%M:%S"), "Mood": mood, "Age": age, "Year": year_string})
    
    st.divider()
    with st.expander("📑 History Log"):
        df_hist = pd.DataFrame(st.session_state.mood_history)
        st.dataframe(df_hist, use_container_width=True)
        st.download_button("📥 Export CSV", df_hist.to_csv(index=False), "mood_log.csv")
