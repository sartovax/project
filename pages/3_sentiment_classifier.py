import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Sentiment Classifier", page_icon="🎯", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-title {
        text-align: center; padding: 18px 0 8px 0;
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 50%, #ec4899 100%);
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
        background: linear-gradient(90deg, transparent, #f59e0b, #ec4899, transparent);
        margin: 0 auto 28px auto; width: 60%; opacity: 0.5;
    }
    .stChatMessage {
        border-radius: 16px; margin-bottom: 10px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(245, 158, 11, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(245, 158, 11, 0.12);
    }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        color: white; border: none; border-radius: 10px;
        font-weight: 600; padding: 10px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.45);
    }
    .stChatInput > div {
        border-radius: 14px !important;
        border: 1px solid rgba(245, 158, 11, 0.25) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stChatInput > div:focus-within {
        border-color: #f59e0b !important;
        box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

with st.sidebar:
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.review_messages = []
        st.rerun()

selected_model = "llama-3.3-70b-versatile"

st.markdown('<p class="page-title">🎯 Sentiment Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Classify review sentiment as positive or negative</p>', unsafe_allow_html=True)
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

if "review_messages" not in st.session_state:
    st.session_state.review_messages = []

for message in st.session_state.review_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Paste a review to classify its sentiment...")

if user_input:
    st.session_state.review_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Classifying..."):
            try:
                chat_completion = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": """You are a strict binary sentiment classifier. 

Task: Evaluate the input text and determine if the sentiment is positive or negative.

Constraints:
- You must output exactly one word: either "positive" or "negative".
- Do not output any punctuation, capitalization, explanations, or additional text."""},
                        *st.session_state.review_messages,
                    ],
                )
                response = chat_completion.choices[0].message.content
                st.markdown(response)
                st.session_state.review_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
