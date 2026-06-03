import os

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


st.set_page_config(page_title="Chat with GPT-3.5 Turbo", layout="wide")
st.title("Chat with GPT-3.5 Turbo")

# API key from sidebar (or OPENAI_API_KEY env var). Get one at
# https://platform.openai.com/api-keys
api_key = st.sidebar.text_input(
    "OpenAI API key",
    type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
)
if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to start chatting.")
    st.stop()

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, api_key=api_key)
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant that translates English to French.")
    ]
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
if user_input := st.chat_input("Enter your message:"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        response = chat.invoke(st.session_state.messages)
        st.write(response.content)
    st.session_state.messages.append(AIMessage(content=response.content))
