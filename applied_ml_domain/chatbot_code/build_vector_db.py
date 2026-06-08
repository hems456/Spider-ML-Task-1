
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FOLDER = os.path.join(BASE_DIR, "papers")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

print("BASE_DIR:", BASE_DIR)
print("PDF_FOLDER:", PDF_FOLDER)
print("PDF_FOLDER EXISTS:", os.path.exists(PDF_FOLDER))


documents = []

print("Loading PDFs...")

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, file)

        print(f"Reading: {file}")

        loader = PyPDFLoader(path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = file

        documents.extend(docs)

print(f"Total Pages Loaded: {len(documents)}")

# ==========================================
# Split into chunks
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")

# ==========================================
# Embedding Model
# ==========================================

print("Loading Sentence Transformer...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# Create Chroma Vector DB
# ==========================================

print("Creating Chroma Database...")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_DIR
)

vectordb.persist()

print("Vector Database Created Successfully!")
print("Saved in folder: vector_db/")
