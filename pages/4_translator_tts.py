import streamlit as st
from groq import Groq
import pyttsx3
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Translator & TTS", page_icon="🌐", layout="centered")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-title {
        text-align: center; padding: 18px 0 8px 0;
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
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
        background: linear-gradient(90deg, transparent, #a855f7, #3b82f6, transparent);
        margin: 0 auto 28px auto; width: 60%; opacity: 0.5;
    }

    /* ── Card-style containers ─────────────────────────── */
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15) !important;
    }

    /* ── Buttons ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7, #6366f1) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 600 !important;
        padding: 12px 28px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.45) !important;
    }

    /* ── Expander ──────────────────────────────────────── */
    .streamlit-expanderHeader {
        border-radius: 10px;
        font-weight: 600;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="page-title">🌐 Translator & TTS</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Translate text and hear it spoken aloud — powered by Sartova</p>', unsafe_allow_html=True)
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

user_text = st.text_area("Enter your text to translate:", height=150)

with st.expander("⚙️ Options"):
    language = st.selectbox(
        "Select output language",
        [
            "English",
            "French",
            "Spanish",
            "German",
            "Italian",
            "Portuguese",
        ]
    )
    # Only list voice names; don't init engine on every rerun
    try:
        _engine = pyttsx3.init()
        _voices = _engine.getProperty("voices")
        voice_names = [v.name for v in _voices]
        voice_ids = {v.name: v.id for v in _voices}
        _engine.stop()
    except Exception:
        voice_names = ["Default"]
        voice_ids = {}

    voice_name = st.selectbox("Choose a voice", voice_names)

if st.button("🔊  Translate & Speak"):
    if user_text.strip():
        with st.spinner("Translating..."):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": f"You are a native translator. Translate the user's text into {language}. Only output the translation — no explanations, no markdown, no emojis, no extra text. If the input has mistakes, silently correct them in the translation."},
                        {"role": "user", "content": user_text}
                    ],
                    model="llama-3.3-70b-versatile"
                )

                response = chat_completion.choices[0].message.content
                st.markdown("**Translation:**")
                st.write(response)

                # Speak the translation
                try:
                    engine = pyttsx3.init()
                    if voice_name in voice_ids:
                        engine.setProperty("voice", voice_ids[voice_name])
                    engine.say(response)
                    engine.runAndWait()
                    engine.stop()
                except Exception as tts_err:
                    st.warning(f"TTS playback failed: {tts_err}")

            except Exception as e:
                st.error(f"Translation error: {e}")
    else:
        st.warning("Please enter some text first.")
