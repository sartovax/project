import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Chatbot", layout="centered")

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
        st.session_state.chatbot_messages = []
        st.rerun()

selected_model = "llama-3.3-70b-versatile"

st.markdown('<p class="chat-title">📊 Data Extraction Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="chat-subtitle">Summarize calls and reviews — powered by Sartova</p>', unsafe_allow_html=True)


if "chatbot_messages" not in st.session_state:
    st.session_state.chatbot_messages = []


for message in st.session_state.chatbot_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Paste a call transcript or review to summarize...")

if user_input:

    st.session_state.chatbot_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chat_completion = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": """You are an analytical summarization assistant. Your task is to process call transcripts or customer reviews, extract the core technical and operational information, and explicitly deduce and state the client's dominant feeling.

Constraints:
- Output strictly as formatted markdown text. Do NOT use JSON.
- Maintain strict objectivity for the summary, but deeply analyze tone, context, and word choice to determine the client's feeling.
- Ignore casual banter and focus entirely on actionable data.
- Format the output exactly as specified below. Do not include conversational filler, introductions, or conclusions.

Required Output Structure:
**Input Type:** [Call or Review]
**Summary:** [2-3 sentence overview of the interaction]
**Client Feeling:** [Explicitly state the emotion, e.g., Highly Anxious, Relieved, Frustrated, Satisfied]
**Feeling Justification:** [One brief sentence or direct quote supporting the deduced emotion]
**Key Takeaways:** - [Point 1]
- [Point 2]
**Action Items / Next Steps:**
- [Action 1 or "None required"]"""},
                        *st.session_state.chatbot_messages,
                    ],
                )
                response = chat_completion.choices[0].message.content
                st.markdown(response)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
