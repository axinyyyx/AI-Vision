import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Rishabh Kumar | Portfolio", page_icon="🚀", layout="centered")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .dev-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 30px;
        border: 1px solid rgba(0, 255, 0, 0.2);
        text-align: center;
        transition: 0.3s;
    }
    .dev-card:hover {
        border: 1px solid #00FF00;
        transform: translateY(-5px);
    }
    .name-title {
        color: #00FF00;
        font-family: 'Courier New', monospace;
        font-size: 35px;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .role-subtitle {
        color: #888;
        font-size: 18px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .bio-text {
        font-size: 16px;
        line-height: 1.6;
        color: #e0e0e0;
        margin-top: 20px;
    }
    .social-btn {
        display: inline-block;
        padding: 10px 20px;
        background-color: transparent;
        color: #00FF00;
        border: 1px solid #00FF00;
        border-radius: 20px;
        text-decoration: none;
        font-weight: bold;
        margin-top: 15px;
        transition: 0.3s;
    }
    .social-btn:hover {
        background-color: #00FF00;
        color: #000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- PROFILE LAYOUT ---
st.markdown("""
    <div class="dev-card">
        <p class="name-title">RISHABH KUMAR</p>
        <p class="role-subtitle">🚀 Full Stack Developer & AI Vision Lead</p>
        <hr style="border-color: rgba(0,255,0,0.1);">
        <p class="bio-text">
            I bridge the gap between <b>High-Performance Web Applications</b> and <b>Cutting-Edge Computer Vision</b>. 
            Specializing in building scalable Full Stack solutions with integrated AI intelligence. 
            From pixels to production, I build systems that see, think, and scale.
        </p>
        <div style="margin-top: 25px;">
            <span style="color: #00FF00; font-weight: bold;">SKILLS:</span><br>
            Python • React • OpenCV • YOLOv8 • Node.js • Streamlit • Cloud Architecture
        </div>
        <br>
        <a href="https://instagram.com/rishabhsahill" class="social-btn" target="_blank">
            📸 Follow on Instagram @rishabhsahill
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- CONTACT INFO ---
st.write("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Projects", "50+")
with col2:
    st.metric("Tea & Coffee Consumed", "∞")
with col3:
    st.metric("Experience", "5+ Years")

st.info("💡 Open for collaborations in AI Vision & Web Innovation.")