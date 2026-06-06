import os
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from dotenv import find_dotenv,load_dotenv
load_dotenv(find_dotenv())
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)
qadataset = [
  {
    "query": "Name: Alex Rivera, Role: Senior Full-Stack Engineer, Skills: [React, Node.js, AWS], Achievement: Led a team of 5 to rebuild a legacy app, improving load times by 40%., CTA: Let's connect!",
    "answer": "HEADLINE: Senior Full-Stack Engineer | React & Node.js Expert\n\nABOUT: I am a Senior Full-Stack Engineer with a passion for designing scalable web architectures..."
  },
  {
    "query": "Name: Sarah Jenkins, Role: B2B SaaS Growth Marketer, Skills: [SEO, HubSpot, Paid Ads], Achievement: Scaled MRR from $20k to $100k in 12 months., CTA: DM me for a free audit.",
    "answer": "HEADLINE: B2B SaaS Growth Marketer | Scale Specialist\n\nABOUT: I don't just chase clicks; I build predictable revenue pipelines for B2B SaaS companies..."
  }
] 
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{query}"),
    ("ai", "{answer}")
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=qadataset,
)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert LinkedIn copywriter. Generate professional profiles using the exact structural format shown in the examples."),
    few_shot_prompt,
    ("human", "{query}")
])
chain = final_prompt | llm 
st.set_page_config(page_title="LinkedIn Description Creator")
st.title("RAG for LinkedIn")
st.subheader("Summary of You!")
if "num_inputs" not in st.session_state:
    st.session_state.num_inputs = 1
col1, col2 = st.columns(2)
with col1:
    if st.button("Add Skill Block"):
        st.session_state.num_inputs += 1
        st.rerun()
with col2:
    if st.button("Delete Skill Block") and st.session_state.num_inputs > 1:
        st.session_state.num_inputs -= 1
        st.rerun()
with st.form(key="user_inputs"):
    st.write("User Information")
    name = st.text_input("Enter your name")
    role = st.text_input("Enter your role")
    skills = []
    for i in range(st.session_state.num_inputs):
        user_input = st.text_input(f"Skill #{i+1}", key=f"input_{i}")
        if user_input:
            skills.append(user_input)
    achievement = st.text_input("Enter your achievements")
    CTA = st.text_input("Enter CTA")
    submit = st.form_submit_button("Generate Profile")
if submit:
    if not name or not role:
        st.error("Please provide at least a Name and Role to generate a profile.")
    else:
        with st.spinner("Generating your LinkedIn Profile..."):
            user_query = f"Name: {name}, Role: {role}, Skills: {skills}, Achievement: {achievement}, CTA: {CTA}"
            result = chain.invoke({"query": user_query})
            st.success("Generated Profile successfully!")
            st.text_area("Your Generated Content", value=result.content, height=350)

