from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def build_vector_db(files):
    all_text = ""

    for file in files:
        loader = PyPDFLoader(file)
        docs = loader.load()
        for doc in docs:
            if doc.page_content.strip():
                all_text += doc.page_content + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(all_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("vector_db")

    return "Vector DB Created Successfully"
