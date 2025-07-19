#  Activate env : conda activate venv/

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with OpenAI"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a smart question answer system designed to answer user queries"),
        ("user","This is the user question : {question}")
    ]
)

def generate_response(question,llm,temperature,api_key,max_tokens):
    openai.api_key = api_key
    print(f"API Key : {api_key}")
    # llm = ChatOpenAI(model=llm)
    llm = ChatOpenAI(
        model=llm,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key  # This is the correct way to pass the key
    )
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer

st.title("Enhanced Q&A Chatbot with OpenAI")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key:",type="password")
llm = st.sidebar.selectbox("Select an OpenAI model",["gpt-4o","gpt-4-turbo","gpt-4"])

temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input,llm,temperature,api_key,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")