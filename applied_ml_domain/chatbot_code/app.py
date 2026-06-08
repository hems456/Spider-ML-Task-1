
import os
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")

# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# ==========================================
# Load Embeddings
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# Load Vector DB
# ==========================================

vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 4}
)

# ==========================================
# Load LLM
# ==========================================

llm = Ollama(model="llama3")

# ==========================================
# Streamlit UI
# ==========================================

st.title("Research Paper RAG Assistant")

question = st.text_input(
    "Ask a question about the papers"
)

if question:

    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a research assistant.

Answer ONLY from the context below.

If the answer is not available,
say:

'I could not find enough information in the provided papers.'

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response)

    st.subheader("Sources")

    sources = list(
        set(
            doc.metadata.get(
                "source_file",
                "Unknown"
            )
            for doc in docs
        )
    )

    for source in sources:
        st.write(source)

