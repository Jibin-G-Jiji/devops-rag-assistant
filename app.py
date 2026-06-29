import streamlit as st
import os
from rag_pipeline import load_vectorstore, build_vectorstore, get_chain, answer_question

st.set_page_config(
    page_title="DevOps RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .stChatMessage { background-color: #1a1d27 !important; border-radius: 12px; }
    .stChatInput input { background-color: #1a1d27 !important; color: #ffffff !important; border-radius: 12px !important; }
    h1 { color: #ffffff !important; }
    p { color: #a0a0b0 !important; }
    .stDivider { border-color: #2a2d3a !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 DevOps RAG Assistant")
st.markdown("Ask me anything about **Docker, Kubernetes, Jenkins, Terraform, or ArgoCD**")
st.divider()

@st.cache_resource(show_spinner="Loading knowledge base...")
def initialize():
    CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(CHROMA_DIR):
        vectorstore = build_vectorstore()
    else:
        vectorstore = load_vectorstore()
    chain, retriever = get_chain(vectorstore)
    return chain, retriever

chain, retriever = initialize()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a DevOps question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question, chain, retriever)
        st.markdown(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
