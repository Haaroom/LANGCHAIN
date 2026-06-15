import streamlit as st 
import uuid 
from utils import db_initializer,llm_creator,retriever_initailizer,read_multiplepdfs


st.set_page_config(page_title="resume screening")
st.title("HR-RESUME SCREENING ASSISTANCE")
st.subheader("I can help you with finding the best match for your job description and role")
job_desc = st.text_area("Enter your job description")
n_doc = st.number_input("enter how many documents to return")
resumes =st.file_uploader("Upload your resumes for analysis",type=['pdf'],accept_multiple_files=True)
if st.button("help me with analysis"):
    with st.spinner("Anlysing"):
        st.session_state['id']=uuid.uuid64.hex
        docs=read_multiplepdfs(resumes)
        st.write("pdf uploaded succesfully")
        db=db_initializer(docs)
        retriever = retriever_initailizer(db)
        st.wrte("retriever with the memory of these pdfs initialized succesfully")
        

        