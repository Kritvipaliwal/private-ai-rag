import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

st.set_page_config(page_title="Private AI RAG", layout="centered")
st.title("🔐 My Private ChatGPT (RAG)")

# Load embedding model once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load QA model once
@st.cache_resource
def load_qa():
    return pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

embeddings = load_embeddings()
qa_model = load_qa()

# Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    with open("doc.pdf", "wb") as f:
        f.write(pdf.read())

    reader = PdfReader("doc.pdf")
    raw_text = ""

    for page in reader.pages:
        raw_text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local("vector_db")

    st.success("PDF Indexed Successfully!")

# Ask question
question = st.text_input("Ask your question")

if question:
    db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    answer = qa_model(context, max_length=256)[0]["generated_text"]
    st.success(answer)
