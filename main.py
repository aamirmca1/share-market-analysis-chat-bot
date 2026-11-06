import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

FAISS_PATH = "faiss_index"

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="RockyBot",
    page_icon="📈"
)

st.title("RockyBot: News Research Tool 📈")

st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

FAISS_PATH = "faiss_index"

# --------------------------------------------------
# Embedding Model
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Groq LLM Model
# -----------------------------
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

# --------------------------------------------------
# Process URLs
# --------------------------------------------------
if process_url_clicked:

    valid_urls = [url for url in urls if url.strip()]

    if not valid_urls:
        st.error("Please enter at least one URL.")
        st.stop()

    try:

        # Load URLs
        loader = UnstructuredURLLoader(
            urls=valid_urls
        )

        main_placeholder.text(
            "Loading data... ✅"
        )

        data = loader.load()

        # Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ","]
        )

        docs = text_splitter.split_documents(data)

        main_placeholder.text(
            "Creating embeddings... ✅"
        )

        # Create Vector Store
        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        # Save Locally
        vectorstore.save_local(FAISS_PATH)

        time.sleep(1)

        main_placeholder.success(
            "Knowledge Base Created Successfully ✅"
        )

    except Exception as e:
        st.error(f"Error: {e}")


# -----------------------------------
# Ask Question
# -----------------------------------

query = st.text_input(
    "Ask a question based on the articles"
)

if query:

    if not os.path.exists(FAISS_PATH):

        st.warning(
            "Please process URLs first."
        )

    else:

        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = vectorstore.similarity_search(query, k=4)

        context = "\n".join([d.page_content for d in docs])

        response = llm.invoke(
            f"""
            Answer the question based on the context.

            Context:
            {context}

            Question:
            {query}
            """
        )

        st.header("Answer")

        st.write(response.content)

        