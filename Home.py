import streamlit as st

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
    }
    .logic-tag {
        background: rgba(0, 204, 102, 0.1);
        color: #00cc66;
        padding: 4px 10px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown(f"""
    <div class='main-card'>
        <h1 style='color: #00cc66; margin-bottom: 10px;'>👁️ A.I Vision</h1>
        <p style='font-size: 20px; color: #bbb;'>Architected by <b style='color: white;'>Rishabh Kumar</b></p>
        <div style='margin-top: 15px;'>
            <a href='https://instagram.com/rishabhsahill' target='_blank' style='color: #00cc66; text-decoration: none;'>📸 @rishabhsahill</a>
        </div>
        <p style='margin-top: 20px; font-size: 16px; max-width: 800px; margin-left: auto; margin-right: auto;'>
            Welcome to the future of Computer Vision. This hub integrates <b>YOLOv8</b> and <b>OpenCV</b> 
            to solve real-world problems like high-speed tracking, precision measurement, and instant object recognition.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- FEATURES & LOGIC SECTION ---
st.write("## 🛠️ Integrated Technologies")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class='feature-box'>
            <h3>📏 Precision Scale</h3>
            <span class='logic-tag'>OPENCV + CONTOUR ANALYSIS</span>
            <p style='color: #aaa; margin-top: 10px;'>
                <b>How it works:</b><br>
                1. <b>Canny Edge Detection:</b> Background se object ko alag karta hai.<br>
                2. <b>Auto-Calibration:</b> Virtual reference point ke base par Pixels ko Centimeters mein convert karta hai.<br>
                3. <b>Geometric Math:</b> Object ke bounding box ka Euclidean distance nikaal kar area calculate karta hai.
            </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='feature-box'>
            <h3>🚦 Traffic AI Radar</h3>
            <span class='logic-tag'>YOLOV8 + KALMAN FILTER</span>
            <p style='color: #aaa; margin-top: 10px;'>
                <b>How it works:</b><br>
                1. <b>Object Tracking:</b> Har object ko ek unique ID di jati hai jo frame-to-frame follow hoti hai.<br>
                2. <b>Velocity Vector:</b> Do frames ke beech ka distance aur time gap (dt) nikaal kar speed (km/h) calculate hoti hai.<br>
                3. <b>Integration:</b> Speed ka change acceleration (a = dv/dt) ke roop mein dikhta hai.
            </p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class='feature-box'>
            <h3>🔍 Neural Detection</h3>
            <span class='logic-tag'>DEEP LEARNING (CNN)</span>
            <p style='color: #aaa; margin-top: 10px;'>
                <b>How it works:</b><br>
                1. <b>YOLOv8 Nano:</b> Ek light-weight neural network jo real-time mein objects ko classify karta hai.<br>
                2. <b>Confidence Score:</b> Har object ke liye AI ek probability batata hai (0.0 to 1.0).<br>
                3. <b>Live Stream:</b> WebRTC ka use karke camera stream ko server par process kiya jata hai.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- SYSTEM ARCHITECTURE DIAGRAM ---


st.write("---")
# --- FOOTER ---
st.info("👈 **Sidebar open karein aur feature select karein.** Har module live processing par chalta hai.")