import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import validators
# from langchain_openai import OpenAI
import os
from langchain_huggingface import HuggingFaceEndpoint

st.set_page_config(page_title="LangChain: Summarize YT Content", page_icon="🦜")
st.title("🦜 LangChain: Summarize YT Content (Hardcoded)")
st.subheader("Summarizing hardcoded YouTube URL")

with st.sidebar:
    hf_api_key = st.text_input("Hugging Face API Key", value="", type="password")

# Hardcoded values

generic_url = st.text_input("Enter YouTube URL")
# generic_url = "https://www.youtube.com/watch?v=FjJAjrGLEMc"



prompt_template = """
Provide a summary of the following content in about 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key


if st.button("Summarize"):
    try:
        hf_token = os.getenv("HF_TOKEN")
        print(hf_token)
        st.write(f"HF_TOKEN: {hf_token} ")
        if not hf_api_key.strip() or not generic_url.strip():
            st.error("Please provide both a Groq API Key and a valid URL.")
        elif not validators.url(generic_url):
            st.error("Please enter a valid YouTube URL.")
        st.info("Extracting video ID...")
        parsed_url = urlparse(generic_url)
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]

        if not video_id:
            st.error("Unable to extract video ID. Please check the URL format.")
        else:
            st.info("Fetching transcript...")
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t["text"] for t in transcript]) 
            text = text[:50]
            st.success("Transcript loaded successfully.")
            docs = [Document(page_content=text)]
            
            st.info("Initializing language model...")
            
            repo_id = "meta-llama/Llama-3.1-8B-Instruct"
            # llm = HuggingFaceEndpoint(
            #         repo_id=repo_id,
            #         task="conversational", 
            #         huggingfacehub_api_token=hf_token,
            # )
            llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation", 
            huggingfacehub_api_token=hf_token
            )
            st.write(f"LLM initialized {llm}")
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=True)
            st.write(f"loaded summarize chain")
            output_summary = chain.run(docs)
            st.write(f"Writing output summary")
            st.write(repo_id)
            st.write(output_summary)
            st.subheader("📄 Summary:")
            st.success(output_summary)

    except Exception as e:
        st.error(f"An error occurred: {e}")
