import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Embedding Similarity Model")
@st.cache_resource
def load_db():
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    docs = CSVLoader("data.csv").load()   
    return FAISS.from_documents(docs, embedder)
db = load_db()
st.title("Similar Words")
st.header("Find Similar Words")
prompt = st.text_input("Enter a word to find similar words")
submit = st.button("Find similar words")
if submit and prompt:
    results = db.similarity_search(prompt, k=6)
    st.subheader("Best matches")
    shown = 0
    for doc in results:
        word = doc.page_content.split(":", 1)[-1].strip()
        if word.lower() == prompt.strip().lower():
            continue
        st.text(word)
        shown += 1
        if shown == 5:
            break
