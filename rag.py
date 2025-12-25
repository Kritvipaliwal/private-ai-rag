from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

def build_vector_db(path):
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # FakeEmbeddings are python-only and cloud safe
    embeddings = FakeEmbeddings(size=384)

    db = Chroma.from_documents(chunks, embeddings, persist_directory="vector_db")
    db.persist()
