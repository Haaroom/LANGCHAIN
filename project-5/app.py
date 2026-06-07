import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage,HumanMessage 
import streamlit as st 
load_dotenv(find_dotenv())
st.set_page_config(page_title="AI-Chat bot")
st.title("AI Model")
st.spinner("Model Loading ...")
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])
summary_prompt = ChatPromptTemplate.from_template(
"""
Summarize the following conversation.
Conversation:
{chat_history}
Provide:
1. Main Topics
2. Important Information
3. Decisions Taken
4. Action Items
Keep it concise.
"""
)
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
user_input = st.chat_input("Ask anything ...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    langchain_history = [
        (HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])) 
        for msg in st.session_state.messages[:-1] 
][-10:]
    chain = prompt_template | llm 
    response = chain.invoke({
        "history": langchain_history,
        "input": user_input
    })
    st.session_state.messages.append({"role": "ai", "content": response.content})
    with st.chat_message("ai"):
        st.markdown(response.content)
with st.sidebar:
    st.write("create a summary of chats ")
    chain = summary_prompt | llm | StrOutputParser()
    res=chain.invoke({
        "chat_history":[i["role"]+i["content"] for i in st.session_state.messages]
    })
    st.session_state.summary=res
    if not st.session_state.summary:
        st.session_state.summary=" "
    with st.sidebar:
        st.text_area("Summary",value=st.session_state.summary,height=300)
    if st.session_state.summary:
        st.download_button(
        label="⬇ Download Summary",
        data=st.session_state.summary,
        file_name="chat_summary.txt",
        mime="text/plain"
    )
    if st.button("Clear Summary"):
        st.session_state.summary = " "
        st.rerun()



