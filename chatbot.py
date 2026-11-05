"""
===============================================================================
Interview Handbook AI Assistant - RAG Chatbot Application
===============================================================================
This Streamlit application implements a Retrieval-Augmented Generation (RAG) 
system using LangChain, FAISS vector store, HuggingFace embeddings, and Groq LLMs.

Key Features:
  - Multi-source Document Ingestion: PDF, TXT, DOCX, and live Web Scraping.
  - Text Chunking: Recursive character text splitting for optimal context capture.
  - Vector Embeddings: HuggingFace sentence-transformers (all-MiniLM-L6-v2).
  - Vector Search: FAISS similarity search for top-k document retrieval.
  - LLM Integration: ChatGroq (Llama-3.3-70b-versatile, Llama-3.1-8b-instant).
  - Streamlit UI: Interactive chat interface with session state message persistence 
    and collapsible source context inspection.
===============================================================================
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import requests

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE & ENVIRONMENT CONFIGURATION
# -----------------------------------------------------------------------------
# Set page title and browser tab icon
st.set_page_config(page_title="Python Handbook QA", page_icon="🤖")
st.title("😉 Interview Handbook AI Assistant")
st.write("Ask questions based on the loaded knowledge sources (Python PDF, Java TXT, React DOCX and about DCL Website).")

# Load environment variables (e.g., GROQ_API_KEY from .env file)
load_dotenv()

# Initialize session state array for storing chat message history across UI reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# 2. SIDEBAR - SETTINGS & MODEL SELECTION
# -----------------------------------------------------------------------------
with st.sidebar:
    # Display branding image logo and sidebar header
    st.image("images/icon_dcl.png", width=40)
    st.title("Settings")
    
    # Dropdown widget allowing the user to select the Groq LLM architecture
    model_name = st.selectbox(
        "Choose Model",
        [
           "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
        ],
        help="Select the Groq LLM model architecture used to generate answers."
    )

# -----------------------------------------------------------------------------
# 3. KNOWLEDGE BASE & RETRIEVER INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_retriever():
    """
    Loads, processes, chunks, and indexes all knowledge sources into a FAISS vector database.
    
    Cached using `@st.cache_resource` so that document ingestion, splitting, and vector 
    embedding generation run only once during initial application startup.

    Returns:
        VectorStoreRetriever: A retriever instance configured to fetch top 5 matching text chunks.
    """
    all_docs = []

    # Source 1: PDF File (Python Handbook)
    pdf_loader = PyPDFLoader("data/pythonHandbook.pdf")
    all_docs.extend(pdf_loader.load())

    # Source 2: Text File (Java Concepts Data)
    with open("data/java_data.txt", "r", encoding="utf-8", errors="ignore") as f:
        all_docs.append(Document(page_content=f.read(), metadata={"source": "java_data.txt"}))

    # Source 3: Word Document (React Concepts Data)
    doc = DocxDocument("data/react_data.docx")
    doc_text = "\n".join(p.text for p in doc.paragraphs)
    all_docs.append(Document(page_content=doc_text, metadata={"source": "react_data.docx"}))

    # Source 4: Live Website Scraping (Dhee Coding Lab)
    try:
        response = requests.get("https://dheecodinglab.com/", timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        web_text = soup.get_text(separator="\n", strip=True)
        all_docs.append(Document(page_content=web_text, metadata={"source": "website"}))
    except Exception as e:
        # Gracefully log warning so local file functionality remains intact if web fetch fails
        print(f"Warning: Could not fetch website data: {e}")

    # Text Chunking: Split large document texts into smaller segments for optimal search
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_docs)

    # Vector Embeddings: Convert text chunks into 384-dimensional dense vectors
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store: Index embedded chunks inside a FAISS vector database
    vectordb = FAISS.from_documents(chunks, embeddings)

    # Return retriever object set to fetch the top k=5 most similar document chunks
    return vectordb.as_retriever(search_kwargs={"k": 5})

# Execute cached retriever initialization (displays error box and halts if execution fails)
try:
    retriever = initialize_retriever()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 4. LLM & PROMPT TEMPLATE SETUP
# -----------------------------------------------------------------------------
# Instantiate ChatGroq model using the model selected in the sidebar
llm = ChatGroq(model_name=model_name)

# Define prompt template to strictly restrict LLM responses to retrieved context
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, reply exactly:

Sorry, I could not get the answer.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------------------------------------------------------
# 5. CHAT INTERFACE & CONVERSATION HANDLING
# -----------------------------------------------------------------------------
# Render historical messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input Box
question = st.chat_input("Ask your question...")

if question:
    # Append user question to session state and render in current chat window
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display AI Assistant response using the RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            # Step 1: Retrieve top matching chunks based on user query
            docs = retriever.invoke(question)

            # Step 2: Combine retrieved chunk texts into a formatted context string
            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # Step 3: Populate prompt template with retrieved context and user question
            final_prompt = prompt.format(
                context=context,
                question=question
            )

            # Step 4: Call Groq LLM to generate the final response
            response = llm.invoke(final_prompt)
            answer = response.content

            # Step 5: Render generated answer in UI
            st.markdown(answer)

            # Step 6: Render collapsible expander displaying retrieved context chunks
            with st.expander("Retrieved Context"):
                for i, d in enumerate(docs, 1):
                    source = d.metadata.get("source", "Unknown")
                    st.markdown(f"**Chunk {i}** *(Source: {source})*")
                    st.write(d.page_content[:1000])

    # Save assistant response to session state history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
