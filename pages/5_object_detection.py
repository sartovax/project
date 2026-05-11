import streamlit as st
import cv2
import numpy as np
import pyttsx3
import threading
import time

st.set_page_config(page_title="Object Detection", page_icon="🦾", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-title {
        text-align: center; padding: 18px 0 8px 0;
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #3b82f6 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .page-subtitle {
        text-align: center; color: #9ca3af; font-size: 0.95rem;
        margin-top: -8px; margin-bottom: 30px;
    }
    .styled-divider {
        height: 2px; border: none; border-radius: 2px;
        background: linear-gradient(90deg, transparent, #10b981, #3b82f6, transparent);
        margin: 0 auto 28px auto; width: 60%; opacity: 0.5;
    }

    /* ── Settings panel ───────────────────────────────── */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #10b981, #3b82f6) !important;
    }

    /* ── Buttons ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #10b981, #06b6d4) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
    }

    /* ── Checkbox ──────────────────────────────────────── */
    .stCheckbox > label > span:first-child {
        border-radius: 6px;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* ── Pulsing dot indicator ─────────────────────────── */
    .live-dot {
        display: inline-block;
        width: 10px; height: 10px;
        background: #ef4444;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.4s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ─── Speech System ──────────────────────────────────────────────
if "last_speech" not in st.session_state:
    st.session_state.last_speech = 0

def speak(text):
    current_time = time.time()
    if current_time - st.session_state.last_speech > 3:
        def _say():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except: pass
        threading.Thread(target=_say).start()
        st.session_state.last_speech = current_time

# ─── Model Loading ──────────────────────────────────────────────
@st.cache_resource
def load_net():
    try:
        return cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
    except: return None

net = load_net()
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# ─── UI ──────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🦾 Object Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Real-time object & face detection with audio feedback</p>', unsafe_allow_html=True)
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

if st.button("🏠 Back to Home"):
    st.switch_page("app.py")

col1, col2 = st.columns([1, 4])
with col1:
    st.subheader("⚙️ Settings")
    conf_thresh = st.slider("Sensitivity", 0.1, 1.0, 0.4)
    audio_on = st.toggle("🔊 Audio Feedback", True)

with col2:
    run = st.checkbox("▶️  Start Live Detection")
    window = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while run:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        objects_found = []

        # 1. DNN Detection (requires numpy)
        if net:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                if detections[0, 0, i, 2] > conf_thresh:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (sX, sY, eX, eY) = box.astype("int")
                    
                    label = CLASSES[idx]
                    objects_found.append(label)
                    cv2.rectangle(display, (sX, sY), (eX, eY), (0, 255, 0), 2)
                    cv2.putText(display, label, (sX, sY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Face Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, wf, hf) in faces:
            cv2.rectangle(display, (x, y), (x+wf, y+hf), (255, 0, 0), 2)
            cv2.putText(display, "human", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            objects_found.append("human")

        # 3. Speech
        if audio_on and objects_found:
            speak(f"Detected {', '.join(list(set(objects_found)))}")

        window.image(display)
    cap.release()
