import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fake import FakeEmbeddings

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.title("My Private ChatGPT (RAG)")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    with open("doc.pdf","wb") as f:
        f.write(pdf.read())
    from rag import build_vector_db
    build_vector_db("doc.pdf")
    st.success("PDF Indexed")

question = st.text_input("Ask your question")

if question:
    embeddings = FakeEmbeddings(size=384)
    db = Chroma(persist_directory="vector_db", embedding_function=embeddings)
    docs = db.similarity_search(question, k=3)

    context = "\n".join(d.page_content for d in docs)

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(context + "\n\nQuestion: " + question)

    st.success(response.text)

