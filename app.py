import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import speech_recognition as sr
import os

st.set_page_config(page_title="My Private AI", layout="centered")
st.title("🚀 My Private ChatGPT (RAG)")

# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

# Load fast LLM only once
@st.cache_resource
def load_qa():
    return pipeline("text2text-generation", model="google/flan-t5-small")

qa = load_qa()

# Upload multiple PDFs
pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if pdfs:
    paths = []
    for i, pdf in enumerate(pdfs):
        path = f"doc_{i}.pdf"
        with open(path, "wb") as f:
            f.write(pdf.read())
        paths.append(path)

    if "db_built" not in st.session_state:
        from rag import build_vector_db
        build_vector_db(paths)
        st.session_state.db_built = True
        st.success("Documents Indexed Successfully")

# Voice input
question = st.text_input("Ask your question")

if st.button("🎤 Speak"):
    r = sr.Recognizer()
    with sr.Microphone() as src:
        audio = r.listen(src)
        question = r.recognize_google(audio)
        st.write("You:", question)

if question:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

    docs = db.similarity_search(question, k=1)
    context = docs[0].page_content

    prompt = f"""
Chat History:
{st.session_state.chat_history}

Document:
{context}

Question: {question}
Answer:
"""

    result = qa(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    st.session_state.chat_history += f"\nUser: {question}\nAI: {result}"
    st.success(result)
