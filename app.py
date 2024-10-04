import os
import requests
import streamlit as st
import faiss
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_ai21 import AI21Embeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize error storage
if "error_logs" not in st.session_state:
    st.session_state.error_logs = []

# Set up API keys from environment variables
ai21_api_key = os.getenv("AI21_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# List of URLs to load
urls = [
    "https://www.maplelabs.com/performance-engineering.html"
]

@st.cache_data
def load_data(urls):
    """Load data from all provided URLs."""
    combined_data = []
    for url in urls:
        try:
            loaders = UnstructuredURLLoader(urls=[url])
            data = loaders.load()
            combined_data.extend(data)
            logger.info(f"Data loaded successfully from {url}")
        except Exception as e:
            error_message = f"Error loading data from {url}: {e}"
            logger.error(error_message)
            st.session_state.error_logs.append(error_message)
    return combined_data

@st.cache_data
def split_text(_data):
    """Splits text into chunks using a text splitter."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(_data)
        logger.info("Text split into chunks successfully")
        return docs
    except Exception as e:
        error_message = f"Error splitting documents: {e}"
        logger.error(error_message)
        st.session_state.error_logs.append(error_message)
        return None

@st.cache_resource
def initialize_embeddings():
    try:
        embeddings = AI21Embeddings(api_key=ai21_api_key)
        logger.info("Embeddings initialized successfully")
        return embeddings
    except Exception as e:
        error_message = f"Error initializing embeddings: {e}"
        logger.error(error_message)
        st.session_state.error_logs.append(error_message)
        return None

@st.cache_resource
def create_vector_index(_docs, _embeddings):
    """Creates a vector index from documents and embeddings."""
    try:
        vector_index = FAISS.from_documents(_docs, _embeddings)
        file_path = "vector_index.faiss"
        faiss.write_index(vector_index.index, file_path)
        logger.info("Vector index created successfully")
        return vector_index
    except Exception as e:
        error_message = f"Error creating vector index: {e}"
        logger.error(error_message)
        st.session_state.error_logs.append(error_message)
        return None

@st.cache_resource
def setup_chat_model():
    try:
        chat = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            api_key=groq_api_key
        )
        logger.info("Chat model set up successfully")
        return chat
    except Exception as e:
        error_message = f"Error setting up chat model: {e}"
        logger.error(error_message)
        st.session_state.error_logs.append(error_message)
        return None

@st.cache_resource
def create_qa_chain(_chat):
    """Creates a QA chain from the chat model."""
    try:
        qa_chain = load_qa_with_sources_chain(_chat, chain_type="stuff")
        logger.info("QA chain created successfully")
        return qa_chain
    except Exception as e:
        error_message = f"Error creating QA chain: {e}"
        logger.error(error_message)
        st.session_state.error_logs.append(error_message)
        return None

def query_serpapi(query, search_type="search"):
    """Queries SerpApi for the given query."""
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "api_key": serpapi_api_key
        }
        
        # Adjust parameters based on the search type
        if search_type == "youtube":
            params["tbm"] = "vid"  # YouTube video search

        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        logger.info(f"SerpApi response: {result}")
        
        # Return appropriate results based on search type
        if search_type == "youtube":
            return result.get("video_results", [])  # Adjust as needed for YouTube results
        else:
            return result.get("organic_results", [])  # Default to organic search results
    except Exception as e:
        error_message = f"Error querying SerpApi: {e}"
        logger.error(error_message)
        st.session_state.error_logs.append(error_message)
        return []

# Load and process data
data = load_data(urls)
if data:
    docs = split_text(data)
    if docs:
        embeddings = initialize_embeddings()
        if embeddings:
            vector_index = create_vector_index(docs, embeddings)
            if vector_index:
                chat = setup_chat_model()
                if chat:
                    qa_chain = create_qa_chain(chat)
                else:
                    qa_chain = None
            else:
                qa_chain = None
        else:
            qa_chain = None
    else:
        qa_chain = None
else:
    qa_chain = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the secretary:"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Define the retriever
        if vector_index:
            retriever = VectorStoreRetriever(vectorstore=vector_index)
            relevant_docs = retriever.get_relevant_documents(prompt)
            logger.info(f"Retrieved relevant documents: {relevant_docs[:2]}")

            # Query the SerpApi
            search_results = query_serpapi(prompt, search_type="search")  # or "youtube" for YouTube search
            serpapi_answers = " ".join(result.get("snippet", "") for result in search_results[:3])  # Adjust as needed
            
            # Get answer from QA chain
            qa_result = qa_chain({"question": prompt, "input_documents": relevant_docs})
            qa_answer = qa_result.get('output_text', "I don't know").strip()
            if qa_chain:
               
                  # Default to "I don't know" if no answer
                
                if qa_answer.lower() ==  qa_answer:
                    # Fallback to SerpApi answers if QA chain answer is "I don't know"
                    answer = f"Seri answer: {serpapi_answers}"
                else:
                    answer = f"Seri answers: {serpapi_answers}"
            else:
                answer = f"QA Chain answer: {qa_answer}"

            logger.info(f"Combined output: {answer}")

        else:
            answer = "Vector index not available."

    except Exception as e:
        answer = f"Error retrieving or processing answer: {e}"
        logger.error(answer)

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})