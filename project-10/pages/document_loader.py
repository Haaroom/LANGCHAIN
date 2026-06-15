import streamlit as st 
import pandas as pd  
from utils import chunkker,embedder,db_creator,llm_creator,csv_documenter
st.set_page_config(page_title="Document loader ")
tabs = ['LOAD FILES','PREVIEW FILES'] 
tab_load,tab_preview=st.tabs(tabs)
with tab_load: 
    docs=st.file_uploader("upload your csv file",type="csv") 
    if docs is not None: 
        with st.spinner("uploading ..."): 
            data=csv_documenter(docs)
            st.write("documented the csv file") 
            chunks=chunkker(data) 
            st.write("chunked the csv file ") 
            embedder=embedder() 
            st.write("embedded the csv file") 
            db=db_creator(chunks,embedder) 
            st.write("file uploaded to data base(recommended : check preview )")
            st.session_state["df"] = pd.read_csv(docs)
with tab_preview:
    if "df" not in st.session_state:
        st.info("Upload a CSV first")
    else:
        df = st.session_state["df"]
        col1, col2 = st.columns(2)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        st.subheader("Preview")
        st.dataframe(df.head(10))
        st.subheader("Columns")
        st.write(df.columns.tolist())
