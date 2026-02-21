import streamlit as st

#  --- REMOVE WATERMARK CONFIG ---
st.markdown("""
    <style>
    /* 1. Upar wala Fork/GitHub Button hatane ke liye (Sidebar button ko bachate hue) */
    [data-testid="stHeader"] .stAppDeployButton, 
    [data-testid="stHeader"] a[href*="github.com"],
    header svg[viewBox="0 0 24 24"] {
        display: none !important;
    }

    /* 2. Niche wala "Hosted with Streamlit" aur "GitHub ID" hatane ke liye */
    footer {
        display: none !important;
    }
    
    /* 3. Sabse niche wala profile/status container jo aapne bataya */
    [data-testid="stStatusWidget"],
    div[class*="profileContainer"],
    div[class*="viewerBadge"] {
        display: none !important;
    }

    /* 4. Agar koi toolbar abhi bhi dikh raha hai */
    div[data-testid="stToolbar"] {
        display: none !important;
    }

    /* 5. Pure page ko clean karne ke liye */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)



# --- PAGE CONFIG ---
st.set_page_config(page_title="A.I Vision", page_icon="👁️", layout="wide")

# --- ADVANCED UI CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .main-card {
        background: linear-gradient(135deg, #1e2130 0%, #11141d 100%);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid #00cc66;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .feature-box {
        background: #1e2130;
        padding: 20px;
        border-radius: 15px;
        border-top: 4px solid #00cc66;
        height: 100%;
        transition: 0.3s;
    }
    .feature-box:hover {
        transform: translateY(-10px);
        background: #25293d;
        border-top: 4px solid #ffffff;
    }
    .logic-tag {
        background: rgba(0, 204, 102, 0.1);
        color: #00cc66;
        padding: 4px 10px;
        border-radius: 5px;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown(f"""
    <div class='main-card'>
        <h1 style='color: #00cc66; margin-bottom: 10px;'>A.I Vision</h1>
        <p style='font-size: 20px; color: #bbb;'>Developed by <b style='color: white;'>Rishabh Kumar</b></p>
        <div style='margin-top: 15px;'>
            <a href='https://instagram.com/rishabhsahill' target='_blank' style='color: #00cc66; text-decoration: none;'>📸 Instagram</a> | 
            <a href='https://rishabhsahil.in' style='color: white; text-decoration: none;'>🚀 Portfolio</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- FEATURES SECTION ---
st.write("## 🛠️ Active Modules & Neural Logic")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class='feature-box'>
            <h3>📏 Precision Scale</h3>
            <span class='logic-tag'>OpenCV + Contour Analysis</span>
            <p style='color: #aaa; margin-top: 10px; font-size: 14px;'>
                <b>Logic:</b> Object boundaries detect karke pixels-to-metric ratio calculate karta hai.<br>
                <b>Use Case:</b> Industrial part measurement.
            </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='feature-box'>
            <h3>🚦 Traffic Radar</h3>
            <span class='logic-tag'>YOLOv8 + Kalman Tracking</span>
            <p style='color: #aaa; margin-top: 10px; font-size: 14px;'>
                <b>Logic:</b> Time-space displacement (dx/dt) se speed detect karta hai.<br>
                <b>Use Case:</b> Speed violation detection.
            </p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class='feature-box'>
            <h3>🔍 Neural Detection</h3>
            <span class='logic-tag'>CNN + YOLOv8 Nano</span>
            <p style='color: #aaa; margin-top: 10px; font-size: 14px;'>
                <b>Logic:</b> COCO dataset par trained weights se 80+ objects classify karta hai.<br>
                <b>Use Case:</b> General surveillance.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- NEW MODULES ADDED ---
st.write(" ") # Spacer
col4, col5 = st.columns(2)

with col4:
    st.markdown("""
        <div class='feature-box'>
            <h3>🔢 Smart Counter</h3>
            <span class='logic-tag'>Object Crossing + Line Logic</span>
            <p style='color: #aaa; margin-top: 10px; font-size: 14px;'>
                <b>How it works:</b><br>
                1. <b>Region of Interest (ROI):</b> Frame par ek virtual line draw hoti hai.<br>
                2. <b>Centroid Tracking:</b> Jab object ka center point line cross karta hai, counter <code>+1</code> ho jata hai.<br>
                3. <b>Directional Logic:</b> In/Out movement ko alag-alag track karta hai.
            </p>
        </div>
        """, unsafe_allow_html=True)

with col5:
    st.markdown("""
        <div class='feature-box'>
            <h3>AI Mood Based Music & Movie Recommendations</h3>
            <span class='logic-tag'>DeepFace + Sentiment Analysis</span>
            <p style='color: #aaa; margin-top: 10px; font-size: 14px;'>
                <b>How it works:</b><br>
                1. <b>Biometric Extraction:</b> Face landmarks se Age, Gender aur Mood predict karta hai.<br>
                2. <b>Safe-Search Filter:</b> Age ke base par content restrictions (Kids/Adult) apply karta hai.<br>
                3. <b>YT-DLP API:</b> Mood ke according real-time music and movie recommendations load karta hai.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- SYSTEM ARCHITECTURE DIAGRAM ---
st.write("---")
st.write("### 🏗️ Neural Pipeline Architecture")


st.info("👈 **Sidebar open karein aur feature select karein.** Har module backend par Deep Learning models (Weights) use karta hai.")





