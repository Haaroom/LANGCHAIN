import os

import streamlit as st
from huggingface_hub import HfApi
from langchain_huggingface import HuggingFaceEndpoint

st.set_page_config(page_title="Hugging Face Hub Integration", layout="wide")
st.title("Hugging Face Hub Integration with LangChain")

# --- Configuration (do NOT hardcode your API key in the source file) ---
# Get a token at https://huggingface.co/settings/tokens
token = st.sidebar.text_input(
    "Hugging Face API token",
    type="password",
    value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
)
if token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

repo_id = st.sidebar.text_input("Model repo id", value="microsoft/Phi-3-mini-4k-instruct")

# --- Model information (uses the huggingface_hub API, not LangChain) ---
st.subheader("Model Information")
try:
    model_info = HfApi().model_info(repo_id)
    st.write(f"**Model Name:** {model_info.modelId}")
    st.write(f"**Model Type:** {model_info.pipeline_tag}")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not fetch model info: {exc}")

# --- Text generation via LangChain ---
st.subheader("Generate Text")
prompt = st.text_input("Enter a prompt to generate text:")
if prompt:
    if not token:
        st.error("Please enter your Hugging Face API token in the sidebar first.")
    else:
        with st.spinner("Generating text..."):
            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                task="text-generation",
                max_new_tokens=50,
            )
            generated_text = llm.invoke(prompt)
        st.subheader("Generated Text")
        st.write(generated_text)
