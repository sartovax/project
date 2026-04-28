import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Sentiment Classifier", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }

    .chat-title {
        text-align: center;
        padding: 10px 0 20px 0;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .chat-subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-top: -15px;
        margin-bottom: 25px;
    }

    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

with st.sidebar:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.review_messages = []
        st.rerun()

selected_model = "llama-3.3-70b-versatile"

st.markdown('<p class="chat-title">🎯 Sentiment Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="chat-subtitle">Classify review sentiment as positive or negative</p>', unsafe_allow_html=True)


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
