import os 
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
load_dotenv(find_dotenv())
st.set_page_config(page_title="script writer")
st.title("Script Writer For Youtube")
st.sidebar.title("GROQ-APIKEY")
def create_script(topic, temperature, video_length, api_key):
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature, api_key=api_key)
    title_prompt = PromptTemplate(
        input_variables=["topic"],
        template="You are an expert YouTube content creator. Generate 5 catchy, high-click-through-rate (CTR) video titles about the following topic: {topic}"
    )
    script_prompt = PromptTemplate(
        input_variables=["title", "duration", "search_data"],
        template="""You are a professional YouTube scriptwriter. 
        Write a compelling video script based on the title: '{title}'.
        The estimated video length should be roughly: {duration} minutes.
        Incorporate relevant facts and context from this recent research data:
        {search_data}
        Structure the script clearly with an engaging Hook, Introduction, Body Points, and a Call to Action (CTA)."""
    )
    search = DuckDuckGoSearchResults()
    raw_search_data = search.run(topic)
    title_chain = title_prompt | llm 
    script_chain = script_prompt | llm 
    suggested_titles = title_chain.invoke({"topic": topic})
    script = script_chain.invoke({"title": topic, "duration": video_length, "search_data": raw_search_data})
    return suggested_titles.content, script.content, raw_search_data
topic = st.text_input("Enter the topic in which you want your script")
api_key = st.sidebar.text_input("enter your api key", type="password")
video_length = st.number_input("video length (in mins)", min_value=1, max_value=60, value=5)
temperature = st.slider("Creativity level (for technical topics low creativity is suggested)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
button = st.button("generate")
if button: 
    if not topic:
        st.warning("Please enter a topic first!")
    elif not api_key and not os.environ.get("GROQ_API_KEY"):
        st.error("Please provide a GROQ API Key in the sidebar or via .env file.")
    else:
        with st.spinner("Generating your titles and script..."):
            title, script, search_data = create_script(topic, temperature, video_length, api_key)
            st.subheader("Suggested Titles")
            st.write(title)
            st.subheader("Script")
            st.write(script)
            with st.expander("The sources For The Script"):
                st.write(search_data)
