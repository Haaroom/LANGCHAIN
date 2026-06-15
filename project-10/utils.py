from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import os
load_dotenv(find_dotenv())
os.makedirs("chromadb", exist_ok=True)
def csv_documenter(file):
    data = pd.read_csv(file)
    documents = []
    for i, row in data.iterrows():
        row_text = "\n".join([f"{col}: {val}" for col, val in row.items()])
        documents.append(Document(page_content=row_text,metadata={"row_index": i}))
    return documents
def chunkker(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    return splitter.split_documents(documents)
def embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
def db_creator(chunks):
    embedding_model = embedder()
    if os.path.exists("chromadb"):
        import shutil
        shutil.rmtree("chromadb")
    os.makedirs("chromadb", exist_ok=True)
    return Chroma.from_documents(documents=chunks,embedding=embedding_model,persist_directory="chromadb")
def load_db():
    embedding_model = embedder()
    return Chroma(persist_directory="chromadb",embedding_function=embedding_model)
def retriever():
    return load_db().as_retriever(search_kwargs={"k": 5})
def llm_creator():
    return ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"))