from langchain_core.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv , find_dotenv
load_dotenv(find_dotenv)
def read_multiplepdfs(*pdfs):
    docs=[]
    for i in pdfs:
        docs.extend(PyPDFLoader(i).load())
    return docs
        
def db_initializer(docs):
    db = Chroma(RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=50).split_documents(docs),
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    return db 
def retriever_initailizer(db):
    return db.as_retriever()
def llm_creator():
    return ChatGroq(model="llama-3.3-70b-versatile",api_key=os.getenv("GROQ_API_KEY"))

