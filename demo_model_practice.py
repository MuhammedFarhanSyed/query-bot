import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from docx import Document
from bs4 import BeautifulSoup
import requests
# 1. Page Configuration & Setup
st.set_page_config(page_title="Python Handbook QA",  )
st.title(" Python Handbook AI Assistant")

st.write("Ask questions directly based on the Python Handbook PDF.")

##side bar


load_dotenv()
with st.sidebar:
    st.image("images/icon_dcl.png", width=35)
    st.title("Chat history")
    model =st.selectbox(
        "choose any model", {
           "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
        }
    )





# 2. Cached Initialization (Prevents reloading the PDF on every user interaction)
@st.cache_resource
def initialize_retriever():



    # Step 1: Load Document
    loader = PyPDFLoader('data/pythonHandbook.pdf')
    documents = loader.load()



        # loading from text file 

    with open("data/java_data.txt", "r" ,encoding="utf-8", errors="ignore") as file:
        react_info = file.read()

        # loading from docxs 
    doc = Document("data/react_data.docx")

    docs_data = "\n".join([para.text for para in doc.paragraphs])

    #  loading from website 
    url = "https://dheecodinglab.com/"

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    web_data = soup.get_text()


    # Step 2: Chunking (Consider increasing chunk_size slightly if answers feel cut off)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=150, 
        chunk_overlap=30
    )
    chunkData = chunks.split_documents(documents)+chunks.split_documents(react_info)+chunks.split_documents(docs_data)+ chunks.split_documents(web_data)

    # Step 3: Embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Step 4: Vector DB
    vector_dbs = FAISS.from_documents(chunkData, embeddings)

    # Step 5: Retriever
    return vector_dbs.as_retriever(search_kwargs={'k': 3})

# Initialize everything once
try:
    retriever = initialize_retriever()
except Exception as e:
    st.error(f"Failed to load PDF. Check if 'data/pythonHandbook.pdf' exists. Error: {e}")
    st.stop()

# 3. Initialize LLM & Prompt (No need to cache these, they are fast)
llm = ChatGroq(model_name='llama-3.3-70b-versatile')

# Note: Fixed a small case typo in your template key ('quesTion' -> 'question')
prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template="""You are an AI assistant.
Answer only from the document with the provided context. If the answer is not available then say 'Sorry, I could not get the answer'.

Context: {context}
Question: {question}
Answer:"""
)

# 4. Streamlit Chat Interface
# Initialize a chat history in session state so the conversation stays on screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in the chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input via the chat bar
if user_question := st.chat_input("What would you like to know about Python?"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching handbook..."):
            # RAG Retrieval Process
            docs = retriever.invoke(user_question)
            context = '\n\n'.join([doc.page_content for doc in docs])
            
            # Format and call LLM
            final_prompt = prompt.format(context=context, question=user_question)
            response = llm.invoke(final_prompt)
            
            output_text = response.content
            st.markdown(output_text)
            
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": output_text})



    