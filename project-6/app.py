import os
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import ResponseSchema,StructuredOutputParser
import streamlit as st 
from pinecone import Pinecone
load_dotenv(find_dotenv())
def build_db(pdf):
    data = PyPDFLoader(pdf).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30).split_documents(data)
    embedder=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    return FAISS.from_documents(chunks,embedder),chunks
def build_retriever(db,k=2):
    return db.as_retriever(search_kwargs={"k":k})
def chat():
    pass
def summary(chunks):
    summary_type = st.selectbox(
        "Summary Type",
        ["Brief", "Crisp", "Technical"]
    )
    instructions = {
        "Brief": "Summarize in one concise paragraph.",
        "Crisp": "Summarize in 5-10 bullet points.",
        "Technical": """
Provide a detailed technical summary.
Include:
- Main concepts
- Methodology
- Findings
- Conclusions
"""
    }
    if st.button("Generate Summary"):
        batch_size = 10
        batch_summaries = []
        prompt = ChatPromptTemplate.from_template("""
You are an expert document analyst.
{instruction}
Document Content:
{context}
""")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            context = "\n\n".join(chunk.page_content for chunk in batch)
            formatted_prompt = prompt.invoke({"instruction": instructions[summary_type],"context": context})
            response = llm.invoke(formatted_prompt)
            batch_summaries.append(response.content)
        combined_summary = "\n\n".join(batch_summaries)
        final_prompt = f"""
Create a final {summary_type.lower()} summary
from the following partial summaries.
{combined_summary}
"""
        final_response = llm.invoke(final_prompt)
        st.subheader("Summary")
        st.write(final_response.content)
def mcqs(retriever,chunks):
    n = st.number_input("Enter no. of mcqs to generate",min_value=1,max_value=50,value=5)
    nO = st.number_input("Enter number of options a question should have ",min_value=2,max_value=7,value=4)
    difficulty=st.radio("enter difficulty",["easy","medium","hard"])
    type=st.selectbox("topic/full pdf",["topic","fullPDF"])
    if type=="topic":
        topic =st.text_input("Enter the topic you want to generate mcqs")
        if st.button("generate mcq's"):
            res=retriever.invoke(topic)
            context= "\n\n".join(i.page_content for i in res)
    elif type == "fullPDF":
        if st.button("generate mcq's"):
            context = "\n\n".join(chunk.page_content for chunk in chunks)
    response_schemas = [
    ResponseSchema(
        name="mcqs",
        description=f"""
Generate {n} MCQs.

Each MCQ must contain:
- question (string)
- options (dictionary with keys A,B,C,D... and option text as values)
- answer (single option key such as A,B,C,D)

Return all MCQs as a list.
"""
    )
]
    format_instruction=StructuredOutputParser.from_response_schemas(response_schemas).get_format_instructions()
    prompt = ChatPromptTemplate.from_template("""
You are an expert educational content creator.

Generate {n} multiple choice questions.

Requirements:
- Difficulty: {difficulty}
- Number of options per question: {num_options}
- Questions must come only from the provided context.
- Do not create duplicate questions.
- Exactly one correct answer.
- Store answer as option label (A,B,C,D...)

Context:
{context}

{format_instructions}
""")
    format_prompt=prompt.invoke({
        "n": n,
    "difficulty": difficulty,
    "num_options": nO,
    "context": context,
    "format_instructions": format_instruction
    }
    )
    response = llm.invoke(format_prompt)
    parsed = StructuredOutputParser.from_response_schema(response_schemas).parse(response.content)
    st.write(parsed)
def research(retriever):
    topic = st.text_input("enter your topic")
    context ="\n\n".join(i.page_content for i in retriever.invoke(topic))
    response_schemas = [
    ResponseSchema(
        name="main_concepts",
        description="Main concepts related to the topic"),
    ResponseSchema(
        name="key_findings",
        description="Important findings and insights"
    ),
    ResponseSchema(
        name="terminology",
        description="Important technical terms"
    ),
    ResponseSchema(
        name="applications",
        description="Applications of the topic"
    ),
    ResponseSchema(
        name="conclusion",
        description="Final conclusion")
]
    format_instruction=StructuredOutputParser.from_response_schemas(response_schemas).get_format_instuctions()
    prompt = ChatPromptTemplate.from_template("""
You are an expert researcher.
Analyze the topic using only the supplied context.
Topic:
{topic}
Context:
{context}
{format_instructions}
""").invoke({
    "topics":topic,
    "context":context,
    "format_instruction":format_instruction
})
    response=llm.invoke(prompt)
    parsed = StructuredOutputParser.from_response_schemas(response_schemas).parse(response.content)
    st.subheader("Main Concepts")
    st.write(parsed["main_concepts"])
    st.subheader("Key Findings")
    st.write(parsed["key_findings"])
    st.subheader("Terminology")
    st.write(parsed["terminology"])
    st.subheader("Applications")
    st.write(parsed["applications"])
    st.subheader("Conclusion")
    st.write(parsed["conclusion"])

def question():
    query = st.text_input("Enter your question")
    if query:
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"""
        Answer the question using only the provided context.
        Context:
        {context}
        Question:
        {query}
        """
        response = llm.invoke(prompt)
        st.write(response)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ.get("GROQ_API_KEY")
)
st.set_page_config(page_title="pdf analyzer")
st.title("PDF ANALYZER")
st.sidebar.title("PDF")
file = st.sidebar.file_uploader(type=["pdf"])
if file :
    db,chunks=build_db(file)
    retriever = build_retriever(db)

action=st.selectbox("choose action",["Question","Summary","Mcq's","Chat","Research"])
actions=["question","summary","mcqs","chat","research"]
if action == "Question":
    question(retriever,llm)
elif action == "Summary":
    summary(db,chunks)
elif action == "MCQs":
    mcqs()
elif action == "Chat":
    chat()
elif action == "Research":
    research()



