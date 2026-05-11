import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Data Extraction Engine", page_icon="📊", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Animated gradient title ────────────────────────── */
    .page-title {
        text-align: center;
        padding: 18px 0 8px 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .page-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: -8px;
        margin-bottom: 30px;
        letter-spacing: 0.3px;
    }

    /* ── Divider ────────────────────────────────────────── */
    .styled-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        border: none;
        border-radius: 2px;
        margin: 0 auto 28px auto;
        width: 60%;
        opacity: 0.5;
    }

    /* ── Chat bubbles ──────────────────────────────────── */
    .stChatMessage {
        border-radius: 16px;
        margin-bottom: 10px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(102, 126, 234, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(102, 126, 234, 0.12);
    }

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.45);
    }

    /* ── Input box polish ──────────────────────────────── */
    .stChatInput > div {
        border-radius: 14px !important;
        border: 1px solid rgba(102, 126, 234, 0.25) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stChatInput > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }

    /* ── Spinner override ──────────────────────────────── */
    .stSpinner > div {
        border-top-color: #764ba2 !important;
    }
</style>
""", unsafe_allow_html=True)

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

with st.sidebar:
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.chatbot_messages = []
        st.rerun()

selected_model = "llama-3.3-70b-versatile"

st.markdown('<p class="page-title">📊 Data Extraction Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Summarize calls and reviews — powered by Sartova</p>', unsafe_allow_html=True)
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


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
