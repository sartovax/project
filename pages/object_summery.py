import streamlit as st
import cv2
import numpy as np
import pyttsx3
import threading
import time

st.set_page_config(page_title="🦾 Full AI Detector", page_icon="🦾", layout="wide")

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
st.title("🦾 Advanced Object Detector")

if st.button("🏠 Back to Home"):
    st.switch_page("app.py")

col1, col2 = st.columns([1, 4])
with col1:
    st.subheader("Settings")
    conf_thresh = st.slider("Sensitivity", 0.1, 1.0, 0.4)
    audio_on = st.toggle("Audio Feedback", True)
    
    # Face Registration removed per user request

with col2:
    run = st.checkbox("Start Live Detection")
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
