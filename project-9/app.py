from langchain_community.document_loaders import SitemapLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os 
from dotenv import find_dotenv,load_dotenv
load_dotenv(find_dotenv)
@st.cache_resource
def build_retriever():
    docs = SitemapLoader("https://python.langchain.com/sitemap.xml").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(chunks, embedder).as_retriever()
llm =ChatGroq(model="llama-3.3-70b-versatile",api_key= os.environ.get("GROQ_API_KEY"))
retriever= build_retriever()
st.set_page_config(page_title="Langchain Chatbot")
st.title("Ask Questions About Langchain")
prompt=ChatPromptTemplate.from_template("""
You are a helpful LangChain assistant.
Answer only from the provided context.
Context:
{context}
Question:
{question}
""") 
chain=prompt | llm 
query = st.text_input("Enter your question")
if "memory" not in st.session_state:
    st.session_state.memory = []
if st.button("Search"):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({
        "context": context,
        "question": query})
    content = response.content
    st.session_state.memory.append({
        "question": query,
        "answer": content})
    st.write(content)