import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Libraries for vector db
from langchain_chroma import Chroma

# Libraries for chat histories
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Libraries for prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Libraries for LLM and Embeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_core.runnables.history import RunnableWithMessageHistory

# Libraries for loading PDF and splitting document chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Difference between Q&A Chatbot and Q&A Chatbot with History
# query -> retriever
# (query,history) -> LLM -> rephrased_query -> retreiver


load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational Q&A Chatbot")
st.write("Upload PDF's and chat with their content")

api_key = st.text_input("Enter your GROQ API Key:",type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")
    # Chat interface
    session_id = st.text_input("Session ID",value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=True)
    if uploaded_files:
        documents= []
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp_pdf"
            with open(temp_pdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            print(f"Loading PDF")
            loader = PyPDFLoader(temp_pdf)
            print("Created PDF loader 2")
            docs = loader.load()
            print(f"docs: {docs}")
            documents.extend(docs)
        print('Updated')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())
        retriever = vector_store.as_retriever()

        contextualized_system_prompt = ("Given a chat history and the latest user question, create a standalone question which can be understood without chat history. Dont answer the question, just formulate it if needed or return as it is")

        contextualized_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualized_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualized_prompt)

        # This wraps the retriever with query rephrasing logic:

        # It first uses the LLM + contextualized_prompt to generate a standalone version of the question.

        # Then sends that query to the retriever (which queries the vector database).

        # Answer prompt 
        system_prompt = ("You are an assistant for question answering tasks. Use the following information retrieved to answer, if you don;t know the answer say you don't know. Keep the answer precise. content: {context}")
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                print(ChatMessageHistory())
                st.session_state.store[session_id]=ChatMessageHistory()
            print(f"session_id : {st.session_state.store[session_id]}---")
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable": {"session_id":session_id}
                    },
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response["answer"])
            st.write("Chat History:",session_history.messages)
else:
    st.warning("Please enter the Groq API key")