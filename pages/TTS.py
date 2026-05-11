import streamlit as st
from groq import Groq
import pyttsx3
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Translator & TTS", layout="centered")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="chat-title">AI Translator & TTS</p>', unsafe_allow_html=True)

user_text = st.text_area("Enter your text to translate:", height=150)

with st.expander("Options"):
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

if st.button("Translate & Speak"):
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