import os 
import pandas as pd 
from dotenv import find_dotenv, load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import streamlit as st
load_dotenv(find_dotenv())
def query(file, llm, prompt):
    df = pd.read_csv(file)
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        agent_type="tool-calling",
        allow_dangerous_code=True
    )
    response = agent.invoke({"input": prompt})
    return response.get("output", "No response generated.")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ.get("GROQ_API_KEY")
)
st.set_page_config(page_title="csv analyzer")
st.title("CSV DATA ANALYZER")
st.subheader("Enter your csv file here")
file = st.file_uploader("csv file", type=["csv"])
prompt = st.text_area("enter your prompt")
if file is not None and prompt.strip() != "":
    with st.spinner("Analyzing data..."):
        try:
            response = query(file, llm, prompt)
            st.write("Analysis Result:")
            st.write(response)
        except Exception as e:
              st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file and enter a prompt to begin.")
